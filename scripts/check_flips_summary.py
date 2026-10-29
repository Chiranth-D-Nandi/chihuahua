import json

with open("reports/qwen_ladder_intermediate_state.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"{'Config':<20} | {'Phase':<20} | {'Flips':<10} | {'Mean SSIM'}")
print("-" * 65)
for d in data:
    print(f"{d['config']:<20} | {d['phase']:<20} | {d['flips']:<10} | {d['mean_ssim']:.4f}")
