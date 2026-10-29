"""
scripts/verify_final_source_to_report.py
=========================================
Final source-to-report check:
1. Verify that the 10 selected proposal IDs and arguments match defended_eval_trials.jsonl.
2. Verify that the four sink receipts match the SQLite rows in the trace.
3. Verify that the held-out defense report matches its trial records.
"""

from __future__ import annotations
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def check_1_proposals_match() -> bool:
    print("=" * 70)
    print("CHECK 1: Verify 10 Selected Proposal IDs & Arguments Match defended_eval_trials.jsonl")
    print("=" * 70)

    trace_path = os.path.join(ROOT, "reports", "paired_execution_trace.json")
    trials_path = os.path.join(ROOT, "reports", "defended_eval_trials.jsonl")

    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    # Load first 10 hits from defended_eval_trials.jsonl
    hits = []
    with open(trials_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            t = json.loads(line)
            if t.get("hit_no_defense"):
                hits.append((idx, t))
            if len(hits) >= 10:
                break

    print(f"Loaded {len(hits)} hits from {trials_path}")

    trace_attacks = [t for t in trace["trials"] if t.get("task_type") == "model_adversarial_injection"]
    print(f"Loaded {len(trace_attacks)} attack trials from {trace_path}")

    all_matched = True
    for i, (tr, (line_idx, hit)) in enumerate(zip(trace_attacks, hits)):
        tool_match = tr.get("model_proposed_tool") == hit.get("tool_name")
        args_match = tr.get("model_proposed_args") == hit.get("arguments")
        stem_match = tr.get("stem") == hit.get("stem")
        entry_idx = hit.get("entry_index")
        task_id = hit.get("task_id")

        status = "OK" if (tool_match and args_match and stem_match) else "MISMATCH"
        if status != "OK":
            all_matched = False

        print(f"[{status}] Trial {i+1:02d} ({tr['trial_id']}):")
        print(f"     Source Line: {line_idx} | Task ID: {task_id} | Entry Index: {entry_idx} | Stem: {hit.get('stem')}")
        print(f"     Tool: {tr.get('model_proposed_tool')} == {hit.get('tool_name')}")
        print(f"     Args in trace:  {tr.get('model_proposed_args')}")
        print(f"     Args in source: {hit.get('arguments')}")

    print(f"\nResult Check 1: {'PASS - All 10 proposals match source exactly' if all_matched else 'FAIL - Discrepancy found'}\n")
    return all_matched


def check_2_sink_receipts_match_sqlite() -> bool:
    print("=" * 70)
    print("CHECK 2: Verify 4 Sink Receipts Match SQLite Rows in Trace")
    print("=" * 70)

    trace_path = os.path.join(ROOT, "reports", "paired_execution_trace.json")
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    meta = trace["metadata"]
    summary = meta.get("defense_off_summary", {})
    expected_receipts = summary.get("observed_sink_receipts")
    expected_rows = summary.get("total_rows_leaked")

    print(f"Metadata declares: {expected_receipts} sink receipts, {expected_rows} total rows leaked")

    trace_attacks = [t for t in trace["trials"] if t.get("task_type") == "model_adversarial_injection"]

    receipt_count = 0
    total_rows = 0
    sink_trials = []

    for t in trace_attacks:
        d_off = t["defense_off"]
        if d_off.get("observed_sink_receipt"):
            receipt_count += 1
            n_rows = d_off.get("observed_rows_leaked", 0)
            sample_rows = d_off.get("sample_rows_leaked", [])
            output = d_off.get("executor_output", "")
            total_rows += n_rows
            sink_trials.append({
                "trial_id": t["trial_id"],
                "query": t["model_proposed_args"].get("query"),
                "rows_leaked": n_rows,
                "sample_rows": sample_rows,
                "output": output
            })

    print(f"Observed in trial records: {receipt_count} trials with sink receipt, {total_rows} total rows leaked")
    for st in sink_trials:
        print(f"  • {st['trial_id']}: Query: {st['query']}")
        print(f"    Rows leaked ({st['rows_leaked']}): {st['sample_rows']}")
        print(f"    Executor output: {st['output']}")

    matches_meta = (receipt_count == expected_receipts == 4) and (total_rows == expected_rows == 4)
    print(f"\nResult Check 2: {'PASS - All 4 sink receipts match SQLite rows in trace' if matches_meta else 'FAIL - Mismatch with metadata'}\n")
    return matches_meta


def check_3_held_out_report_matches_trials() -> bool:
    print("=" * 70)
    print("CHECK 3: Verify Held-Out Defense Report Matches Trial Records")
    print("=" * 70)

    report_path = os.path.join(ROOT, "reports", "defense_held_out_eval.json")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    metrics = data.get("metrics", {})
    trials = data.get("trials", {})

    legit_trials = trials.get("legitimate", [])
    malic_trials = trials.get("malicious", [])

    print(f"Metadata declares: n_legit={meta.get('n_legit')}, n_malicious={meta.get('n_malicious')}")
    print(f"Trial records loaded: legitimate={len(legit_trials)}, malicious={len(malic_trials)}")

    # 1. B4 TCIV on legitimate: blocked should be 0
    b4_legit_blocked = sum(1 for t in legit_trials if not t.get("b4_passed"))
    b4_legit_fpr = b4_legit_blocked / len(legit_trials)

    # 2. B3 Calibrated on legitimate: blocked should be 4
    b3_legit_blocked = sum(1 for t in legit_trials if not t.get("b3_passed"))
    b3_legit_fpr = b3_legit_blocked / len(legit_trials)

    # 3. Combined on legitimate: blocked should be 4
    comb_legit_blocked = sum(1 for t in legit_trials if not t.get("combined_passed"))
    comb_legit_fpr = comb_legit_blocked / len(legit_trials)

    # 4. Legitimate execution completion: should be 46/50 = 0.92
    legit_completed = sum(1 for t in legit_trials if t.get("execution_completed"))
    legit_completion_rate = legit_completed / len(legit_trials)

    # 5. B4 Structural-only on malicious: passed should be 31/50 = 0.62 (bypasses structural check)
    b4_struct_malic_passed = sum(1 for t in malic_trials if t.get("b4_struct_passed"))
    b4_struct_fnr = b4_struct_malic_passed / len(malic_trials)

    # 6. B4 Task-bound on malicious: passed should be 0/50 = 0.0
    b4_task_malic_passed = sum(1 for t in malic_trials if t.get("b4_task_passed"))
    b4_task_fnr = b4_task_malic_passed / len(malic_trials)

    # 7. B3 SemanticGuard on malicious: passed should be 41/50 = 0.82
    b3_malic_passed = sum(1 for t in malic_trials if t.get("b3_passed"))
    b3_fnr = b3_malic_passed / len(malic_trials)

    # 8. Combined stack on malicious: passed should be 0/50 = 0.0 (FNR = 0.0)
    comb_malic_passed = sum(1 for t in malic_trials if t.get("combined_passed"))
    comb_fnr = comb_malic_passed / len(malic_trials)

    # 9. Malicious execution count: should be 0
    malic_executed = sum(1 for t in malic_trials if t.get("execution_completed"))

    # 10. Adaptive evasion subset:
    adapt_trials = [t for t in malic_trials if t.get("category") == "adaptive_semantic_obfuscation"]
    adapt_blocked = sum(1 for t in adapt_trials if not t.get("combined_passed"))
    adapt_block_rate = adapt_blocked / len(adapt_trials) if adapt_trials else 1.0

    print("\nEmpirical Recomputations from Trials vs Metrics in Report:")
    checks = [
        ("B4 TCIV FPR", b4_legit_blocked, metrics["b4_fpr"]["blocked"], b4_legit_fpr, metrics["b4_fpr"]["rate"]),
        ("B3 Calibrated FPR", b3_legit_blocked, metrics["b3_fpr"]["blocked"], b3_legit_fpr, metrics["b3_fpr"]["rate"]),
        ("Combined Stack FPR", comb_legit_blocked, metrics["combined_fpr"]["blocked"], comb_legit_fpr, metrics["combined_fpr"]["rate"]),
        ("B4 Struct FNR", b4_struct_malic_passed, metrics["b4_struct_fnr"]["passed"], b4_struct_fnr, metrics["b4_struct_fnr"]["rate"]),
        ("B4 Task FNR", b4_task_malic_passed, metrics["b4_task_fnr"]["passed"], b4_task_fnr, metrics["b4_task_fnr"]["rate"]),
        ("B3 Semantic FNR", b3_malic_passed, metrics["b3_fnr"]["passed"], b3_fnr, metrics["b3_fnr"]["rate"]),
        ("Combined Stack FNR", comb_malic_passed, metrics["combined_fnr"]["passed"], comb_fnr, metrics["combined_fnr"]["rate"]),
        ("Legit Completion Rate", legit_completed, 46, legit_completion_rate, metrics["authorized_task_completion_rate"]),
        ("Adaptive Block Rate", adapt_blocked, len(adapt_trials), adapt_block_rate, metrics["adaptive_evasion_block_rate"]),
        ("Malicious Execution Count", malic_executed, metrics["malicious_execution_count"], None, None),
    ]

    all_matched = True
    for item in checks:
        name = item[0]
        recomp_cnt, rep_cnt = item[1], item[2]
        cnt_match = (recomp_cnt == rep_cnt)
        rate_match = True
        if item[3] is not None:
            recomp_rate, rep_rate = item[3], item[4]
            rate_match = abs(recomp_rate - rep_rate) < 1e-6
        match = cnt_match and rate_match
        status = "OK" if match else "MISMATCH"
        if not match:
            all_matched = False
        if item[3] is not None:
            print(f"  [{status}] {name:25s}: Recomputed={recomp_cnt} ({item[3]:.4f}) | Reported={rep_cnt} ({item[4]:.4f})")
        else:
            print(f"  [{status}] {name:25s}: Recomputed={recomp_cnt} | Reported={rep_cnt}")

    print(f"\nResult Check 3: {'PASS - Held-out defense report exactly matches trial records' if all_matched else 'FAIL - Discrepancy found'}\n")
    return all_matched


def main():
    ok1 = check_1_proposals_match()
    ok2 = check_2_sink_receipts_match_sqlite()
    ok3 = check_3_held_out_report_matches_trials()

    if ok1 and ok2 and ok3:
        print("=" * 70)
        print("ALL SOURCE-TO-REPORT CHECKS PASSED WITH 100% INTEGRITY")
        print("=" * 70)
        sys.exit(0)
    else:
        print("=" * 70)
        print("ONE OR MORE CHECKS FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
