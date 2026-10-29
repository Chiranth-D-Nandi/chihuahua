"""
FISSION — tests/test_tciv.py
==============================
Unit tests for B4 TCIV.
Run with: python tests/test_tciv.py
All tests should PASS before B4 TCIV is considered done.

ABHYUDAYA — B4 TCIV tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from defenses.b4_tciv.tciv import TCIV, TCIVVerdict


def run_tests():
    tciv    = TCIV()
    passed  = 0
    failed  = 0

    def test(name, result, expect_pass, expect_layer=None):
        nonlocal passed, failed
        ok = (result.passed == expect_pass)
        if expect_layer and not result.passed:
            ok = ok and (result.layer_failed == expect_layer)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        reason = f" — {result.block_reason}" if result.block_reason else ""
        print(f"  [{status}] {name}{reason}")

    # ── Helper: make a valid signed call ────────────────────────
    def make_valid_call():
        tool_name   = "summarise_document"
        arguments   = {"document_id": "doc_001", "max_length": 150}
        intent_text = "Summarise doc_001 and post to Slack"
        task_id     = "task_test_001"
        sig = tciv.sign(tool_name, arguments, intent_text, task_id)
        return tool_name, arguments, intent_text, task_id, sig

    print("\n── B4 TCIV Unit Tests ──\n")

    print("Layer 1 — Whitelist")
    tool, args, intent, tid, sig = make_valid_call()

    # Valid call
    r = tciv.verify(tool, args, "executor", intent, tid, sig)
    test("Valid call passes", r, expect_pass=True)

    # Unknown tool
    r = tciv.verify("delete_all_files", args, "executor", intent, tid, sig)
    test("Unknown tool is blocked", r, expect_pass=False, expect_layer="whitelist")

    # Wrong role
    r = tciv.verify(tool, args, "orchestrator", intent, tid, sig)
    test("Wrong agent role is blocked", r, expect_pass=False, expect_layer="whitelist")

    # Empty tool name
    r = tciv.verify("", args, "executor", intent, tid, sig)
    test("Empty tool name is blocked", r, expect_pass=False, expect_layer="whitelist")

    print("\nLayer 2 — Schema")
    tool, args, intent, tid, sig = make_valid_call()

    # Missing required argument
    bad_args = {"document_id": "doc_001"}   # max_length missing
    r = tciv.verify(tool, bad_args, "executor", intent, tid,
                    tciv.sign(tool, bad_args, intent, tid))
    test("Missing required arg is blocked", r, expect_pass=False, expect_layer="schema")

    # Wrong argument type (int instead of str)
    bad_args = {"document_id": 12345, "max_length": 150}
    r = tciv.verify(tool, bad_args, "executor", intent, tid,
                    tciv.sign(tool, bad_args, intent, tid))
    test("Wrong arg type (int for str) is blocked", r, expect_pass=False, expect_layer="schema")

    # Wrong argument type (str instead of int)
    bad_args = {"document_id": "doc_001", "max_length": "one hundred"}
    r = tciv.verify(tool, bad_args, "executor", intent, tid,
                    tciv.sign(tool, bad_args, intent, tid))
    test("Wrong arg type (str for int) is blocked", r, expect_pass=False, expect_layer="schema")

    # Extra argument — should warn but not block
    extra_args = {"document_id": "doc_001", "max_length": 150, "extra_field": "surprise"}
    r = tciv.verify(tool, extra_args, "executor", intent, tid,
                    tciv.sign(tool, extra_args, intent, tid))
    test("Extra argument passes (warn only)", r, expect_pass=True)

    print("\nLayer 3 — HMAC")
    tool, args, intent, tid, sig = make_valid_call()

    # Tampered HMAC
    tampered_sig = "0" * 64
    r = tciv.verify(tool, args, "executor", intent, tid, tampered_sig)
    test("Tampered HMAC is blocked", r, expect_pass=False, expect_layer="hmac")

    # Correct call, wrong intent text
    wrong_intent = "Do something completely different"
    r = tciv.verify(tool, args, "executor", wrong_intent, tid, sig)
    test("Wrong intent text breaks HMAC", r, expect_pass=False, expect_layer="hmac")

    # Correct call, wrong task_id
    r = tciv.verify(tool, args, "executor", intent, "task_different", sig)
    test("Wrong task_id breaks HMAC", r, expect_pass=False, expect_layer="hmac")

    # Argument tampered after signing
    tampered_args = {"document_id": "evil_doc", "max_length": 150}
    r = tciv.verify(tool, tampered_args, "executor", intent, tid, sig)
    test("Tampered argument breaks HMAC", r, expect_pass=False, expect_layer="hmac")

    print("\nFalse Positive Rate")
    # Build a set of 20 legitimate calls and check FPR
    legit_calls = []
    for i in range(20):
        t_name = "summarise_document"
        t_args = {"document_id": f"doc_{i:03d}", "max_length": 100 + i}
        t_intent = f"Summarise document doc_{i:03d}"
        t_id = f"task_{i:03d}"
        legit_calls.append({
            "tool_name": t_name,
            "arguments": t_args,
            "intent_text": t_intent,
            "task_id": t_id,
            "agent_role": "executor",
        })

    fpr = tciv.compute_fpr(legit_calls)
    fpr_ok = fpr == 0.0
    print(f"  {'[PASS]' if fpr_ok else '[FAIL]'} FPR on 20 legitimate calls: {fpr:.3f} "
          f"(target <= 0.05)")
    if fpr_ok:
        passed += 1
    else:
        failed += 1

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n── Results: {passed} passed, {failed} failed ──")
    if failed == 0:
        print("  All tests passed. B4 TCIV is working correctly.\n")
    else:
        print("  Some tests failed. Check output above.\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

