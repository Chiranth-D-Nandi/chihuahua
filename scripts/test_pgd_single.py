import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

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

probe_data = torch.load("attacks/r1_pixel/probe_qwen.pt", map_location=device)
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

pil_img = Image.open("adversarial_dataset/synthetic/synthetic_01_clean.png").convert("RGB")
img_clean = torch.from_numpy(np.array(pil_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

eps = 8.0 / 255.0
n_steps = 20
alpha = eps / 10.0
target_class = torch.tensor([2], device=device)

x = img_clean.clone().detach()

print(f"\nRunning test PGD-{n_steps} (eps={eps*255:.1f}/255, alpha={alpha*255:.2f}/255) on synthetic_01:")
CLASSES = ["summarise_document", "send_slack_message", "query_database"]

t0 = time.time()
for step in range(1, n_steps + 1):
    x.requires_grad_(True)
    logits = forward_img(x)
    loss = F.cross_entropy(logits, target_class)
    loss.backward()

    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        pred = CLASSES[np.argmax(probs)]
        target_prob = probs[2] * 100

        # Gradient descent step to minimize CE w.r.t target class
        x = x - alpha * x.grad.sign()
        # Project back into L_inf ball around clean image
        x = torch.clamp(x, img_clean - eps, img_clean + eps)
        # Project into valid pixel range
        x = torch.clamp(x, 0.0, 1.0)

        print(f"Step {step:02d}/{n_steps:02d} | Loss: {loss.item():.4f} | Target Prob: {target_prob:5.2f}% | Pred: {pred} ({np.max(probs)*100:.1f}%)")

elapsed = time.time() - t0
print(f"Completed in {elapsed:.2f}s ({elapsed/n_steps:.2f}s/step)")
