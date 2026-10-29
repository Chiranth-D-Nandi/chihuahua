"""
FISSION dashboard health check.

Run with the dashboard live (`uvicorn dashboard.app:app --port 8000`),
Redis and Ollama both running:

    python dashboard_health_check.py

Hits every endpoint the UI uses and checks the response has the fields
the frontend JS actually reads (cross-referenced against index.html's
rendering code) — not just that the request succeeded. A 200 status
with a missing/renamed field would look fine in a browser tab until
someone clicks that specific card, so this checks structure explicitly.
"""

import sys
import httpx

BASE = "http://localhost:8000"
FAILURES = []


def check(label, fn):
    try:
        fn()
        print(f"  [OK]   {label}")
    except AssertionError as e:
        print(f"  [FAIL] {label} — {e}")
        FAILURES.append(label)
    except Exception as e:
        print(f"  [FAIL] {label} — {type(e).__name__}: {e}")
        FAILURES.append(label)


print("=" * 60)
print("  1. Server + health")
print("=" * 60)

def _health():
    r = httpx.get(f"{BASE}/api/health", timeout=10)
    assert r.status_code == 200, f"status {r.status_code}"
    d = r.json()
    assert "ollama" in d, "missing 'ollama' field"
    assert "tools" in d, "missing 'tools' field"
    if not d["ollama"]:
        print("      [WARN] Ollama reports unreachable — pipeline/run will fail until it's up")

check("GET /api/health returns expected shape", _health)


print("\n" + "=" * 60)
print("  2. Benchmark endpoints — structure matches what index.html reads")
print("=" * 60)

def _bench_r3():
    r = httpx.post(f"{BASE}/api/benchmark/r3-sds", timeout=30)
    d = r.json()
    assert d["status"] == "success", d.get("error", "unknown error")
    for c in d["data"]["cases"]:
        assert "label" in c and "sds" in c and "drifted" in c

def _bench_b4():
    r = httpx.post(f"{BASE}/api/benchmark/b4-tciv", timeout=30)
    d = r.json()
    assert d["status"] == "success", d.get("error", "unknown error")
    for c in d["data"]["cases"]:
        assert "name" in c and "tool" in c and "passed" in c and "layer_failed" in c
    assert "fpr" in d["data"]

def _bench_b5():
    r = httpx.post(f"{BASE}/api/benchmark/b5-phantom", timeout=30)
    d = r.json()
    assert d["status"] == "success", d.get("error", "unknown error")
    for p in d["data"]["payloads"]:
        assert "tool" in p and "worm" in p and "drift" in p and "reason" in p
    assert "apr" in d["data"]

def _bench_r5b():
    r = httpx.post(f"{BASE}/api/benchmark/r5b-context", timeout=30)
    d = r.json()
    assert d["status"] == "success", d.get("error", "unknown error")
    for h in d["data"]["hops"]:
        assert "hop" in h and "cpr" in h and "cbd_reached" in h
    assert "cbd" in d["data"]

def _bench_b2():
    r = httpx.post(f"{BASE}/api/benchmark/b2-attention", timeout=60)
    d = r.json()
    assert d["status"] == "success", d.get("error", "unknown error")
    a = d["data"]["adversarial"]
    for k in ("verdict", "passed", "adr", "aer", "kl", "mahal"):
        assert k in a, f"missing '{k}' in adversarial result"

def _bench_r2():
    r = httpx.post(f"{BASE}/api/benchmark/r2-attack", timeout=60)
    d = r.json()
    assert d["status"] == "success", d.get("error", "unknown error")
    for section in ("clean", "adversarial", "deltas"):
        assert section in d["data"]

check("POST /api/benchmark/r3-sds", _bench_r3)
check("POST /api/benchmark/b4-tciv", _bench_b4)
check("POST /api/benchmark/b5-phantom", _bench_b5)
check("POST /api/benchmark/r5b-context", _bench_r5b)
check("POST /api/benchmark/b2-attention (uses bert-base-uncased as proxy — see note below)", _bench_b2)
check("POST /api/benchmark/r2-attack (uses bert-base-uncased as proxy — see note below)", _bench_r2)


print("\n" + "=" * 60)
print("  3. Full pipeline run (requires live Ollama)")
print("=" * 60)

def _pipeline_clean():
    r = httpx.post(
        f"{BASE}/api/pipeline/run", timeout=60,
        data={"task": "Summarise document doc_001", "model_name": "llava:latest"},
    )
    d = r.json()
    assert "final_status" in d
    assert "layers" in d and len(d["layers"]) > 0
    for layer in d["layers"]:
        assert "layer" in layer and "status" in layer
    if d["final_status"] == "ERROR":
        print(f"      [WARN] Pipeline returned ERROR — check Ollama is running: {d.get('block_reason')}")

check("POST /api/pipeline/run (clean task)", _pipeline_clean)


print("\n" + "=" * 60)
if FAILURES:
    print(f"  {len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"    - {f}")
    print("=" * 60)
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED")
    print("=" * 60)
    print(
        "\n  Reminder: B2/R2 benchmark results above are measured against\n"
        "  bert-base-uncased as a stand-in model, not LLaVA — this is a\n"
        "  deliberate design choice (LLaVA runs via Ollama, out-of-process,\n"
        "  with no attachable hooks), not a bug. Don't cite these numbers\n"
        "  as LLaVA's own attention behavior."
    )

