"""
Smoke test for the Session 17 BaseVLM/OllamaVLM/model_registry stack.

Run from the FISSION project root:
    python test_ollamavlm_smoke.py

Tests the new abstraction layer in isolation, BEFORE testing it wired
into generate_adversarial_dataset.py — if something's broken, this
narrows down whether it's the new files themselves or the patch that
wires them in.

Requires: Redis is NOT needed for this test. Ollama must be running
(`ollama serve`) with llava pulled.
"""

import sys
from PIL import Image, ImageDraw

print("=" * 60)
print("  Step 1: Import sanity check")
print("=" * 60)

try:
    from orchestrator.base_vlm import BaseVLM, VLMResponse
    print("  [OK] orchestrator.base_vlm imports cleanly")
except Exception as e:
    print(f"  [FAIL] base_vlm import: {e}")
    sys.exit(1)

try:
    from orchestrator.ollama_vlm import OllamaVLM
    print("  [OK] orchestrator.ollama_vlm imports cleanly")
except Exception as e:
    print(f"  [FAIL] ollama_vlm import: {e}")
    sys.exit(1)

try:
    from config.model_registry import get_vlm, get_profile, list_known_models
    print("  [OK] config.model_registry imports cleanly")
except Exception as e:
    print(f"  [FAIL] model_registry import: {e}")
    sys.exit(1)


print("\n" + "=" * 60)
print("  Step 2: model_registry reads model_profiles.json")
print("=" * 60)

known = list_known_models()
print(f"  Known models: {known}")
assert len(known) >= 1, "Expected at least one model in model_profiles.json"
print("  [OK] model_profiles.json is being read (was previously read by nothing)")

profile = get_profile("llava:latest")
print(f"  llava:latest profile: encoder={profile.encoder}, tool_calling={profile.tool_calling}, known={profile.known}")
assert profile.known, "llava:latest should be a known profile"
print("  [OK] Known-model profile resolves correctly")

unknown_profile = get_profile("some-model-that-does-not-exist:latest")
assert not unknown_profile.known, "Unlisted model should resolve with known=False"
print("  [OK] Unknown-model fallback works (check the log line above for the warning)")


print("\n" + "=" * 60)
print("  Step 3: get_vlm() resolves to a real OllamaVLM instance")
print("=" * 60)

vlm = get_vlm("llava:latest")
assert isinstance(vlm, OllamaVLM), f"Expected OllamaVLM, got {type(vlm)}"
print(f"  [OK] get_vlm('llava:latest') returned {type(vlm).__name__}")

vlm.load_model()
print("  [OK] load_model() ran without raising (check above for a reachability warning)")


print("\n" + "=" * 60)
print("  Step 4: preprocess_image() + generate() + parse_output(), live")
print("=" * 60)
print("  This makes one real Ollama call — expect a short wait.")

test_image = Image.new("RGB", (640, 480), color=(255, 255, 255))
ImageDraw.Draw(test_image).text(
    (24, 24),
    "SEND MESSAGE\nTo: team@corp.com\nSubject: Test\nBody: This is a smoke test.",
    fill=(20, 20, 20),
)

preprocessed = vlm.preprocess_image(test_image)
assert isinstance(preprocessed, str) and len(preprocessed) > 0, "preprocess_image should return a non-empty base64 string"
print(f"  [OK] preprocess_image() returned a base64 string ({len(preprocessed)} chars)")

system_prompt = "You are a tool-calling assistant. Respond with one JSON tool call."
task_text = "What tool should be called based on this image?"

raw = vlm.generate(system_prompt, task_text, preprocessed)
if raw is None:
    print("  [FAIL] generate() returned None — check Ollama is running and llava is pulled.")
    print(f"  error_stats so far: {vlm.error_stats}")
    sys.exit(1)
print(f"  [OK] generate() returned real text ({len(raw)} chars): {raw[:100]!r}...")

response = vlm.parse_output(raw)
print(f"  parse_output() -> tool_name={response.tool_name}, parse_error={response.parse_error}")
if response.tool_name is not None:
    print("  [OK] Response parsed into a tool call successfully")
else:
    print("  [WARN] Response didn't parse into a tool call — not necessarily a bug, "
          "the model may have said something unparseable this run. raw_text is still "
          "populated for SDS purposes:", response.raw_text is not None)


print("\n" + "=" * 60)
print("  Step 5: cleanup()")
print("=" * 60)
vlm.cleanup()
print("  [OK] cleanup() ran without raising (documented no-op for OllamaVLM)")


print("\n" + "=" * 60)
print("  ALL SMOKE TESTS PASSED")
print("=" * 60)
print("  Safe to proceed to testing the full generate_adversarial_dataset.py patch.")

