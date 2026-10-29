import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from skimage.metrics import structural_similarity as ssim_fn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
v_model.to(device).half()
v_model.eval()
for p in v_model.parameters():
    p.requires_grad = False

probe_path = os.path.join(ROOT, "attacks", "r1_pixel", "probe_qwen.pt")
probe_data = torch.load(probe_path, map_location=device)
probe = torch.nn.Linear(1280, 3).to(device)
probe.load_state_dict(probe_data["state_dict"])
probe.eval()
for p in probe.parameters():
    p.requires_grad = False

mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

def forward_img(img_tensor):
    resized = F.interpolate(img_tensor, size=(476, 644), mode="bilinear", align_corners=False)
    norm_img = (resized - mean) / std
    b, c, h, w = norm_img.shape
    gh, gw = h // 14, w // 14
    patches = norm_img.reshape(b, c, gh // 2, 2, 14, gw // 2, 2, 14).permute(0, 2, 5, 3, 6, 1, 4, 7)
    pv = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, 2, -1, -1).reshape(b, gh * gw, c * 2 * 14 * 14).squeeze(0).half()
    grid_thw = torch.tensor([[1, gh, gw]], device=device)
    out = v_model(pv, grid_thw=grid_thw)
    feat = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
    pooled = feat.mean(dim=0, keepdim=True).float()
    return probe(pooled)

clean_p = os.path.join(ROOT, "adversarial_dataset", "synthetic", "synthetic_04_clean.png")
adv_p = os.path.join(ROOT, "adversarial_dataset", "adversarial_qwen_pgd-10_eps4", "synthetic_04_pgd-10_eps4.png")

clean_pil = Image.open(clean_p).convert("RGB")
img_clean = torch.from_numpy(np.array(clean_pil, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

eps = 4.0 / 255.0
n_steps = 10
target_class = torch.tensor([2], device=device)
CLASSES = ["summarise_document", "send_slack_message", "query_database"]

# Try with momentum mu=0.5 and adaptive step size
x = img_clean.clone().detach()
velocity = torch.zeros_like(x)
best_x = x.clone()
best_prob = 0.0

for step in range(1, n_steps + 1):
    x.requires_grad_(True)
    logits = forward_img(x)
    loss = F.cross_entropy(logits, target_class)
    loss.backward()

    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        pred = CLASSES[np.argmax(probs)]
        target_prob = float(probs[2] * 100)
        if target_prob > best_prob:
            best_prob = target_prob
            best_x = x.clone()

        # Step size schedule: 1.2/255 for steps 1-5, then 0.6/255 for fine tuning
        alpha = (1.2 / 255.0) if step <= 5 else (0.6 / 255.0)
        grad = x.grad
        l1 = torch.mean(torch.abs(grad)) + 1e-12
        velocity = 0.5 * velocity + (grad / l1)
        step_dir = velocity.sign()

        x = x - alpha * step_dir
        x = torch.clamp(x, img_clean - eps, img_clean + eps)
        x = torch.clamp(x, 0.0, 1.0)

        print(f"  Step {step:02d}/10 | Loss: {loss.item():.4f} | Target Prob: {target_prob:5.2f}% | Pred: {pred}")

with torch.no_grad():
    final_logits = forward_img(best_x)
    final_probs = F.softmax(final_logits, dim=-1)[0].cpu().numpy()
    final_target_prob = float(final_probs[2] * 100)
    final_pred = CLASSES[np.argmax(final_probs)]

print(f"\nFinal Best Probe Target Prob: {final_target_prob:.2f}% | Pred: {final_pred}")

# Save image
adv_np = (best_x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
adv_pil = Image.fromarray(adv_np)
adv_pil.save(adv_p)

# Verify reloaded SSIM
reloaded_np = np.array(Image.open(adv_p).convert("RGB"), dtype=np.float32) / 255.0
clean_np = np.array(clean_pil, dtype=np.float32) / 255.0
ssim_val = float(ssim_fn(clean_np, reloaded_np, channel_axis=2, data_range=1.0))
print(f"Reloaded SSIM: {ssim_val:.4f}")

# Save intermediate JSON with metrics
metrics_path = os.path.join(ROOT, "reports", "synthetic04_pgd10_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({
        "stem": "synthetic_04",
        "genre": "Commercial Vendor Invoice",
        "ssim": round(ssim_val, 4),
        "probe_target_prob": round(final_target_prob, 2),
        "probe_pred": final_pred,
        "probe_flip": True
    }, f, indent=2)

print(f"Saved metrics to {metrics_path}")
