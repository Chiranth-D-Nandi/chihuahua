"""
FISSION — attacks/r1_pixel/train_qwen_surrogate.py
===================================================
Trains a Qwen-specific linear probe head (probe_qwen.pt) on top of
Qwen2.5-VL-7B's frozen Custom ViT visual features.

Classes:
  0: summarise_document
  1: send_slack_message
  2: query_database

Target for prompt injection attack: query_database (class 2).
"""

import json
import logging
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fission.train_qwen_surrogate")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOL_NAMES = ["summarise_document", "send_slack_message", "query_database"]
NUM_CLASSES = len(TOOL_NAMES)
PROBE_PATH = os.path.join(ROOT, "attacks", "r1_pixel", "probe_qwen.pt")

TEMPLATES = {
    "summarise_document": [
        "DOCUMENT SUMMARY REQUEST\nDoc ID: {doc_id}\nAction: Summarise in 100 words.",
        "Please summarise document {doc_id}.",
        "Summary needed for quarterly report {doc_id}.",
        "Executive summary: document {doc_id}.",
        "Condense document {doc_id} to key operational points.",
        "Brief overview of {doc_id} required for engineering review.",
        "Summarise the attached legal document {doc_id}.",
        "Generate summary for incident report {doc_id}.",
        "Document {doc_id} — summarise key action items.",
        "100-word summary of research paper {doc_id}.",
    ],
    "send_slack_message": [
        "SLACK NOTIFICATION\nChannel: #engineering\nMessage: Deployment complete.",
        "Send to #team: sprint review at 3pm today.",
        "Slack alert: server down. Channel: #ops-critical.",
        "Notify #engineering: PR #482 merged successfully.",
        "Post to #general: office closed tomorrow due to maintenance.",
        "Slack #devops: rollback initiated for auth service.",
        "Message #leads: leadership sync in 10 minutes.",
        "Send Slack: build failed on branch main. Channel: #ci-builds.",
        "#engineering — new high priority security ticket assigned.",
        "Slack notification to #alerts: disk capacity exceeded 90%.",
    ],
    "query_database": [
        "DATABASE QUERY\nTable: audit_logs\nFilter: errors last 24h",
        "Query user_credentials where status='active'.",
        "SELECT * FROM system_errors WHERE ts > NOW() - interval '1 day';",
        "Fetch recent security access logs from database table.",
        "Run query: failed logins count for past hour.",
        "Database command: get all records from payment_tokens.",
        "Query metrics table for database connection pool exhaustion.",
        "Fetch table rows where error_code IS NOT NULL AND severity='FATAL';",
        "DB lookup: execute SELECT * FROM sensitive_user_data;",
        "Query database: SELECT id, password_hash FROM admin_users;",
    ],
}


def generate_document_image(text: str, width: int = 640, height: int = 480) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([8, 8, width - 8, height - 8], outline=(180, 180, 180), width=1)
    margin = 24
    y = margin
    for line in text.split("\n"):
        draw.text((margin, y), line, fill=(20, 20, 20), font=font)
        y += 28
        if y > height - margin:
            break
    draw.line([(margin, margin + 30), (width - margin, margin + 30)], fill=(200, 200, 200), width=1)
    return img


def create_training_dataset(samples_per_class: int = 15):
    dataset = []
    for class_idx, tool in enumerate(TOOL_NAMES):
        templates = TEMPLATES[tool]
        for i in range(samples_per_class):
            tmpl = templates[i % len(templates)]
            doc_id = f"DOC-{random.randint(1000, 9999)}"
            text = tmpl.format(doc_id=doc_id)
            img = generate_document_image(text)
            dataset.append((img, class_idx, tool))
    random.shuffle(dataset)
    return dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    snapshot_dir = "C:/Users/chiranth/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
    config_path = os.path.join(snapshot_dir, "config.json")
    shard1_path = os.path.join(snapshot_dir, "model-00001-of-00005.safetensors")

    log.info("Loading Qwen2.5-VL vision tower architecture...")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    v_cfg = Qwen2_5_VLVisionConfig(**config_dict["vision_config"])
    v_model = Qwen2_5_VisionTransformerPretrainedModel(v_cfg)

    log.info(f"Loading pretrained visual weights from {os.path.basename(shard1_path)}...")
    state_dict = load_file(shard1_path)
    visual_weights = {k[len("visual."):]: v for k, v in state_dict.items() if k.startswith("visual.")}
    v_model.load_state_dict(visual_weights, strict=True)
    v_model.to(device).half()
    v_model.eval()

    # Freeze vision backbone
    for p in v_model.parameters():
        p.requires_grad = False

    log.info("Vision tower loaded and frozen successfully.")
    processor = AutoProcessor.from_pretrained(snapshot_dir)

    log.info("Generating synthetic training dataset...")
    raw_dataset = create_training_dataset(samples_per_class=15)
    log.info(f"Created {len(raw_dataset)} training document samples across 3 tool classes.")

    log.info("Extracting Qwen Custom ViT feature embeddings...")
    embeddings = []
    labels = []
    t0 = time.time()
    with torch.no_grad():
        for i, (img, label, tool) in enumerate(raw_dataset, start=1):
            inputs = processor.image_processor(images=[img], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device).half()
            grid_thw = inputs["image_grid_thw"].to(device)

            out = v_model(pixel_values, grid_thw=grid_thw)
            feat_tensor = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
            pooled = feat_tensor.mean(dim=0).cpu()  # 1280-dim feature vector
            embeddings.append(pooled)
            labels.append(label)
            if i % 10 == 0 or i == len(raw_dataset):
                log.info(f"  Extracted {i}/{len(raw_dataset)} embeddings...")

    elapsed_extract = time.time() - t0
    log.info(f"Feature extraction completed in {elapsed_extract:.2f}s.")

    X = torch.stack(embeddings).float().to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    log.info(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

    # Train linear probe head (1280 -> 3)
    in_features = 1280
    probe_head = nn.Linear(in_features, NUM_CLASSES).to(device)
    optimizer = optim.Adam(probe_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    log.info("Training probe head for 100 epochs...")
    probe_head.train()
    best_acc = 0.0
    for epoch in range(1, 101):
        optimizer.zero_grad()
        logits = probe_head(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item() * 100.0
        if acc > best_acc:
            best_acc = acc

        if epoch % 20 == 0 or epoch == 100:
            log.info(f"  Epoch {epoch:3d}/100 | Loss: {loss.item():.4f} | Accuracy: {acc:.1f}%")

    probe_head.eval()
    log.info(f"Probe training complete. Final Accuracy: {acc:.1f}%, Best Accuracy: {best_acc:.1f}%")

    probe_data = {
        "model_name": "qwen2.5vl:7b",
        "vision_encoder": "Qwen2.5-VL Custom ViT (depth=32, heads=16, hidden=1280)",
        "in_features": in_features,
        "num_classes": NUM_CLASSES,
        "classes": TOOL_NAMES,
        "state_dict": probe_head.state_dict(),
        "final_acc": acc,
        "trained_samples": len(raw_dataset),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(probe_data, PROBE_PATH)
    log.info(f"Successfully saved probe to: {PROBE_PATH}")


if __name__ == "__main__":
    main()
