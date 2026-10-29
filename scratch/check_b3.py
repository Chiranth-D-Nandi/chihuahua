import os
import sys
sys.path.insert(0, os.path.abspath("."))

from defenses.b3_semantic.semantic_guard import SemanticGuard
from scripts.defense_eval import LEGIT_CALLS

g = SemanticGuard()
for i, c in enumerate(LEGIT_CALLS):
    res = g.check(c['tool_name'], c['arguments'], c['intent_text'])
    print(f"[{i:2d}] tool={c['tool_name']:<20} passed={str(res.passed):<5} layer={str(res.layer_failed):<5} ics={res.ics_score:.4f} reason={res.block_reason}")
