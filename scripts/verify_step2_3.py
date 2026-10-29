import hashlib
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from skimage.metrics import structural_similarity as ssim_fn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
probe_path = os.path.join(ROOT, "attacks", "r1_pixel", "probe_qwen.pt")

print("=" * 80)
print("VERIFICATION OF PROBE_QWEN.PT HASH AND METADATA")
print("=" * 80)
with open(probe_path, "rb") as f:
    probe_bytes = f.read()
sha256 = hashlib.sha256(probe_bytes).hexdigest()
mtime = os.path.getmtime(probe_path)
mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
print(f"Path:      {probe_path}")
print(f"Size:      {len(probe_bytes)} bytes")
print(f"Modified:  {mtime_str}")
print(f"SHA-256:   {sha256}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
probe_data = torch.load(probe_path, map_location=device)
print(f"Probe keys: {list(probe_data.keys())}")
if "metadata" in probe_data:
    print(f"Probe metadata field: {probe_data['metadata']}")
print(f"Probe class names in file: {probe_data.get('classes', 'NOT FOUND')}")
print(f"Linear weight shape: {probe_data['state_dict']['weight'].shape}")
print(f"Linear bias: {probe_data['state_dict']['bias']}")

CLASSES = ["summarise_document", "send_slack_message", "query_database"]
print(f"\nEvaluating against CLASSES: {CLASSES}")
print(f"Index 0: {CLASSES[0]}")
print(f"Index 1: {CLASSES[1]}")
print(f"Index 2: {CLASSES[2]} (Target tool: query_database)")

snapshot_dir = "C:/Users/chiranth/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
config_path = os.path.join(snapshot_dir, "config.json")
shard1_path = os.path.join(snapshot_dir, "model-00001-of-00005.safetensors")

with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

v_cfg = Qwen2_5_VLVisionConfig(**config_dict["vision_config"])
v_model = Qwen2_5_VisionTransformerPretrainedModel(v_cfg)
state_dict = load_file(shard1_path)
visual_weights = {k[len("visual."):]: v for k, v in state_dict.items() if k.startswith("visual.")}
v_model.load_state_dict(visual_weights, strict=True)
v_model.to(device).half().eval()

probe = torch.nn.Linear(1280, 3).to(device)
probe.load_state_dict(probe_data["state_dict"])
probe.eval()

processor = AutoProcessor.from_pretrained(snapshot_dir)

def eval_via_processor(img):
    inputs = processor.image_processor(images=[img], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device).half()
    grid_thw = inputs["image_grid_thw"].to(device)
    with torch.no_grad():
        out = v_model(pixel_values, grid_thw=grid_thw)
        feat = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
        pooled = feat.mean(dim=0, keepdim=True).float()
        logits = probe(pooled).squeeze(0)
        probs = F.softmax(logits, dim=-1)
    return logits.cpu().numpy(), probs.cpu().numpy()

# Also test surrogate forward (bilinear resize & manual patch)
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

def eval_via_surrogate(img_tensor):
    resized = F.interpolate(img_tensor, size=(476, 644), mode="bilinear", align_corners=False)
    norm_img = (resized - mean) / std
    b, c, h, w = norm_img.shape
    gh, gw = h // 14, w // 14
    patches = norm_img.reshape(b, c, gh // 2, 2, 14, gw // 2, 2, 14).permute(0, 2, 5, 3, 6, 1, 4, 7)
    pv = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, 2, -1, -1).reshape(b, gh * gw, c * 2 * 14 * 14).squeeze(0).half()
    grid_thw = torch.tensor([[1, gh, gw]], device=device)
    with torch.no_grad():
        out = v_model(pv, grid_thw=grid_thw)
        feat = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
        pooled = feat.mean(dim=0, keepdim=True).float()
        logits = probe(pooled).squeeze(0)
        probs = F.softmax(logits, dim=-1)
    return logits.cpu().numpy(), probs.cpu().numpy()

print("\n" + "=" * 80)
print("EVALUATION ON SYNTHETIC_01: CLEAN vs FGSM vs PGD-10")
print("=" * 80)

clean_path = os.path.join(ROOT, "adversarial_dataset", "synthetic", "synthetic_01_clean.png")
fgsm_path = os.path.join(ROOT, "adversarial_dataset", "adversarial_qwen", "synthetic_01_fgsm_eps4.png")
pgd_path = os.path.join(ROOT, "adversarial_dataset", "adversarial_qwen_pgd-10_eps4", "synthetic_01_pgd-10_eps4.png")

pil_clean = Image.open(clean_path).convert("RGB")
clean_tensor = torch.from_numpy(np.array(pil_clean, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

print(f"1. Clean Image Path: {clean_path}")
c_log_proc, c_prob_proc = eval_via_processor(pil_clean)
c_log_surr, c_prob_surr = eval_via_surrogate(clean_tensor)
print(f"   [Via Processor] Logits: {c_log_proc}")
print(f"   [Via Processor] Probs:  {c_prob_proc} -> Pred: {CLASSES[np.argmax(c_prob_proc)]}")
print(f"   [Via Surrogate] Logits: {c_log_surr}")
print(f"   [Via Surrogate] Probs:  {c_prob_surr} -> Pred: {CLASSES[np.argmax(c_prob_surr)]}")

if os.path.exists(fgsm_path):
    print(f"\n2. FGSM eps=4 Image Path: {fgsm_path}")
    pil_fgsm = Image.open(fgsm_path).convert("RGB")
    fgsm_tensor = torch.from_numpy(np.array(pil_fgsm, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    f_log_proc, f_prob_proc = eval_via_processor(pil_fgsm)
    f_log_surr, f_prob_surr = eval_via_surrogate(fgsm_tensor)
    ssim_fgsm = ssim_fn(np.array(pil_clean)/255.0, np.array(pil_fgsm)/255.0, data_range=1.0, channel_axis=2)
    print(f"   SSIM vs clean: {ssim_fgsm:.5f}")
    print(f"   [Via Processor] Logits: {f_log_proc}")
    print(f"   [Via Processor] Probs:  {f_prob_proc} -> Pred: {CLASSES[np.argmax(f_prob_proc)]}")
    print(f"   [Via Surrogate] Logits: {f_log_surr}")
    print(f"   [Via Surrogate] Probs:  {f_prob_surr} -> Pred: {CLASSES[np.argmax(f_prob_surr)]}")

if os.path.exists(pgd_path):
    print(f"\n3. PGD-10 eps=4 Image Path: {pgd_path}")
    pil_pgd = Image.open(pgd_path).convert("RGB")
    pgd_tensor = torch.from_numpy(np.array(pil_pgd, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    p_log_proc, p_prob_proc = eval_via_processor(pil_pgd)
    p_log_surr, p_prob_surr = eval_via_surrogate(pgd_tensor)
    ssim_pgd = ssim_fn(np.array(pil_clean)/255.0, np.array(pil_pgd)/255.0, data_range=1.0, channel_axis=2)
    print(f"   SSIM vs clean: {ssim_pgd:.5f}")
    print(f"   [Via Processor] Logits: {p_log_proc}")
    print(f"   [Via Processor] Probs:  {p_prob_proc} -> Pred: {CLASSES[np.argmax(p_prob_proc)]}")
    print(f"   [Via Surrogate] Logits: {p_log_surr}")
    print(f"   [Via Surrogate] Probs:  {p_prob_surr} -> Pred: {CLASSES[np.argmax(p_prob_surr)]}")

    # Check pixel difference between Clean and PGD-10
    diff = np.abs(np.array(pil_pgd, dtype=np.float32) - np.array(pil_clean, dtype=np.float32))
    print(f"   Pixel L_inf diff: {diff.max()} / 255.0 (max allowed eps=4: {diff.max() <= 4})")
    print(f"   Pixel L_1 mean diff: {diff.mean():.4f}")
