import time
from huggingface_hub import hf_hub_download

print("Starting download of Qwen2.5-VL-7B shard 1 (contains visual encoder weights)...")
t0 = time.time()
path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    filename="model-00001-of-00005.safetensors"
)
elapsed = time.time() - t0
print(f"Downloaded model-00001-of-00005.safetensors in {elapsed:.1f}s to: {path}")
