"""
Diagnostic script to test:
1. Verify image paths sent to Ollama in manifest_qwen25vl_whitebox.json
2. Check pixel difference between clean and adv images before and after AutoProcessor.image_processor
3. Test probe_qwen.pt directly (bypassing Ollama) on clean vs adversarial images across eps 1, 2, 4, 8
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLASSES = ["summarise_document", "send_slack_message", "query_database"]

print("=" * 70)
print("DIAGNOSTIC 1: Verify Image Paths in Whitebox Manifest")
print("=" * 70)
manifest_path = os.path.join(ROOT, "adversarial_dataset", "manifest_qwen25vl_whitebox.json")
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total entries in manifest: {len(data)}")
    for i, item in enumerate(data):
        adv_p = os.path.join(ROOT, item["adv_path"])
        exists = os.path.exists(adv_p)
        size = os.path.getsize(adv_p) if exists else 0
        print(f"[{i+1:02d}] {item['stem']} | eps={item['epsilon_int']} | path: {item['adv_path']} (Exists: {exists}, size: {size} bytes)")
else:
    print("manifest_qwen25vl_whitebox.json not found!")

print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Pixel Diff Before & After Preprocessing (synthetic_01)")
print("=" * 70)
snapshot_dir = "C:/Users/chiranth/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
processor = AutoProcessor.from_pretrained(snapshot_dir)

clean_path = os.path.join(ROOT, "adversarial_dataset", "synthetic", "synthetic_01_clean.png")
clean_img = Image.open(clean_path).convert("RGB")
clean_np = np.array(clean_img, dtype=np.float32) / 255.0
clean_inputs = processor.image_processor(images=[clean_img], return_tensors="pt")

for eps in [1, 2, 4, 8]:
    adv_path = os.path.join(ROOT, "adversarial_dataset", "adversarial_qwen", f"synthetic_01_fgsm_eps{eps}.png")
    if not os.path.exists(adv_path):
        print(f"Adv path missing: {adv_path}")
        continue
    adv_img = Image.open(adv_path).convert("RGB")
    adv_np = np.array(adv_img, dtype=np.float32) / 255.0
    adv_inputs = processor.image_processor(images=[adv_img], return_tensors="pt")

    # In raw pixel space (0-1)
    raw_diff = np.abs(adv_np - clean_np)
    max_raw_diff = raw_diff.max()
    mean_raw_diff = raw_diff.mean()

    # After image_processor (normalized pixel_values)
    pv_clean = clean_inputs["pixel_values"]
    pv_adv = adv_inputs["pixel_values"]
    pv_diff = (pv_adv - pv_clean).abs()
    max_pv_diff = pv_diff.max().item()
    mean_pv_diff = pv_diff.mean().item()

    print(f"Eps {eps}/255 ({eps/255.0:.5f}):")
    print(f"  Raw image: max diff = {max_raw_diff:.5f} ({max_raw_diff*255:.1f}/255), mean diff = {mean_raw_diff:.5f}")
    print(f"  Processor pv: max diff = {max_pv_diff:.5f}, mean diff = {mean_pv_diff:.5f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Direct Probe Evaluation on Transformers Vision Tower")
print("=" * 70)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

config_path = os.path.join(snapshot_dir, "config.json")
shard1_path = os.path.join(snapshot_dir, "model-00001-of-00005.safetensors")
probe_path = os.path.join(ROOT, "attacks", "r1_pixel", "probe_qwen.pt")

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

probe_data = torch.load(probe_path, map_location=device)
probe = torch.nn.Linear(1280, 3).to(device)
probe.load_state_dict(probe_data["state_dict"])
probe.eval()
for p in probe.parameters():
    p.requires_grad = False

print(f"Probe metadata: {probe_data.get('vision_encoder')}, final_acc: {probe_data.get('final_acc')}%")

def eval_image_with_probe(img):
    inputs = processor.image_processor(images=[img], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device).half()
    grid_thw = inputs["image_grid_thw"].to(device)
    with torch.no_grad():
        out = v_model(pixel_values, grid_thw=grid_thw)
        feat_tensor = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
        pooled = feat_tensor.mean(dim=0, keepdim=True).float()
        logits = probe(pooled).squeeze(0)
        probs = F.softmax(logits, dim=-1)
    return logits.cpu().numpy(), probs.cpu().numpy()

# Test across all 10 document genres
print("\nTesting all 10 documents: Clean vs Adversarial (eps 1, 2, 4, 8) under Direct Probe:")
print(f"{'Doc':<14} | {'Clean Pred':<20} | {'Eps1 Pred':<20} | {'Eps2 Pred':<20} | {'Eps4 Pred':<20} | {'Eps8 Pred':<20}")
print("-" * 125)

for doc_idx in range(1, 11):
    stem = f"synthetic_{doc_idx:02d}"
    c_path = os.path.join(ROOT, "adversarial_dataset", "synthetic", f"{stem}_clean.png")
    c_img = Image.open(c_path).convert("RGB")
    c_logits, c_probs = eval_image_with_probe(c_img)
    c_pred = CLASSES[np.argmax(c_probs)]
    c_conf = np.max(c_probs) * 100

    row = [f"{stem}"]
    row.append(f"{c_pred[:15]} ({c_conf:.1f}%)")

    for eps in [1, 2, 4, 8]:
        a_path = os.path.join(ROOT, "adversarial_dataset", "adversarial_qwen", f"{stem}_fgsm_eps{eps}.png")
        a_img = Image.open(a_path).convert("RGB")
        a_logits, a_probs = eval_image_with_probe(a_img)
        a_pred = CLASSES[np.argmax(a_probs)]
        a_conf = np.max(a_probs) * 100
        # Check target class probability (class 2: query_database)
        target_prob = a_probs[2] * 100
        row.append(f"{a_pred[:15]} (t:{target_prob:.1f}%)")

    print(f"{row[0]:<14} | {row[1]:<20} | {row[2]:<20} | {row[3]:<20} | {row[4]:<20} | {row[5]:<20}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 4: Detailed Logits on synthetic_01")
print("=" * 70)
c_path = os.path.join(ROOT, "adversarial_dataset", "synthetic", "synthetic_01_clean.png")
c_logits, c_probs = eval_image_with_probe(Image.open(c_path).convert("RGB"))
print(f"Clean synthetic_01 logits: {c_logits}")
print(f"Clean synthetic_01 probs:  {c_probs} -> Pred: {CLASSES[np.argmax(c_probs)]}")

for eps in [1, 2, 4, 8]:
    a_path = os.path.join(ROOT, "adversarial_dataset", "adversarial_qwen", f"synthetic_01_fgsm_eps{eps}.png")
    a_logits, a_probs = eval_image_with_probe(Image.open(a_path).convert("RGB"))
    print(f"Adv eps={eps} logits:     {a_logits}")
    print(f"Adv eps={eps} probs:      {a_probs} -> Pred: {CLASSES[np.argmax(a_probs)]} (query_db prob: {a_probs[2]*100:.2f}%)")
