import os
import sys
import uuid
import math
sys.path.insert(0, os.path.abspath("."))

from defenses.b3_semantic.semantic_guard import SemanticGuard
from defenses.b4_tciv.tciv import TCIV
from scripts.defense_eval import LEGIT_CALLS, MALICIOUS_CALLS, wilson_ci
from config.config import cfg

guard = SemanticGuard()
tciv = TCIV()

print(f"Total unique legit: {len(LEGIT_CALLS)}, malicious: {len(MALICIOUS_CALLS)}")

# Evaluate Unique Test Suite (N=20)
print("\n--- UNIQUE TEST SUITE (N=20) ---")

# 1. Legitimate calls
b3_legit_blocks = 0
b4_legit_blocks = 0
comb_legit_blocks = 0

for c in LEGIT_CALLS:
    tid = str(uuid.uuid4())
    # B3
    b3_res = guard.check(c["tool_name"], c["arguments"], c["intent_text"])
    b3_block = not b3_res.passed
    if b3_block:
        b3_legit_blocks += 1
    # B4 (orchestrator signs legitimate call)
    sig = tciv.sign(c["tool_name"], c["arguments"], c["intent_text"], tid)
    b4_res = tciv.verify(c["tool_name"], c["arguments"], cfg.mcp.role_executor, c["intent_text"], tid, sig)
    b4_block = not b4_res.passed
    if b4_block:
        b4_legit_blocks += 1
    if b3_block or b4_block:
        comb_legit_blocks += 1

print(f"Legit (N=20):")
print(f"  B4 alone: {b4_legit_blocks}/20 blocked (FPR = {b4_legit_blocks/20:.4f}, Wilson: {wilson_ci(b4_legit_blocks, 20)})")
print(f"  B3 alone: {b3_legit_blocks}/20 blocked (FPR = {b3_legit_blocks/20:.4f}, Wilson: {wilson_ci(b3_legit_blocks, 20)})")
print(f"  Combined: {comb_legit_blocks}/20 blocked (FPR = {comb_legit_blocks/20:.4f}, Wilson: {wilson_ci(comb_legit_blocks, 20)})")

# 2. Malicious calls
# A: Cross-Modal Injection / Unauthorized Dispatch (orchestrator signed benign task)
b3_mal_blocks = 0
b4_unauth_blocks = 0
comb_unauth_blocks = 0

# B: Structural-Only (attacker signed/bypassed HMAC, testing whitelist + schema)
b4_struct_blocks = 0
comb_struct_blocks = 0

for c in MALICIOUS_CALLS:
    tid = str(uuid.uuid4())
    # B3
    b3_res = guard.check(c["tool_name"], c["arguments"], c["intent_text"])
    b3_block = not b3_res.passed
    if b3_block:
        b3_mal_blocks += 1

    # B4 Model A: Unauthorized Dispatch (benign task signed)
    benign_sig = tciv.sign("summarise_document", {"document_id": "doc_001", "max_length": 150}, "Summarise document doc_001", tid)
    b4_unauth_res = tciv.verify(c["tool_name"], c["arguments"], cfg.mcp.role_executor, "Summarise document doc_001", tid, benign_sig)
    b4_unauth_block = not b4_unauth_res.passed
    if b4_unauth_block:
        b4_unauth_blocks += 1
    if b3_block or b4_unauth_block:
        comb_unauth_blocks += 1

    # B4 Model B: Structural Only (pre-signed with call parameters, testing whitelist & schema)
    sig_for_call = tciv.sign(c["tool_name"], c["arguments"], c["intent_text"], tid)
    b4_struct_res = tciv.verify(c["tool_name"], c["arguments"], cfg.mcp.role_executor, c["intent_text"], tid, sig_for_call)
    b4_struct_block = not b4_struct_res.passed
    if b4_struct_block:
        b4_struct_blocks += 1
    if b3_block or b4_struct_block:
        comb_struct_blocks += 1

print(f"\nMalicious (N=20):")
print(f"  B3 alone: blocked {b3_mal_blocks}/20 (FNR = {(20-b3_mal_blocks)/20:.4f}, Wilson: {wilson_ci(20-b3_mal_blocks, 20)})")
print(f"  B4 alone (Unauthorized Injection): blocked {b4_unauth_blocks}/20 (FNR = {(20-b4_unauth_blocks)/20:.4f}, Wilson: {wilson_ci(20-b4_unauth_blocks, 20)})")
print(f"  B4 alone (Structural-Only / Pre-signed): blocked {b4_struct_blocks}/20 (FNR = {(20-b4_struct_blocks)/20:.4f}, Wilson: {wilson_ci(20-b4_struct_blocks, 20)})")
print(f"  Combined (Unauthorized Injection): blocked {comb_unauth_blocks}/20 (FNR = {(20-comb_unauth_blocks)/20:.4f}, Wilson: {wilson_ci(20-comb_unauth_blocks, 20)})")
print(f"  Combined (Structural-Only + B3): blocked {comb_struct_blocks}/20 (FNR = {(20-comb_struct_blocks)/20:.4f}, Wilson: {wilson_ci(20-comb_struct_blocks, 20)})")
