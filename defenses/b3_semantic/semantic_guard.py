"""
FISSION — B3 Semantic Anomaly Guard

Detects and blocks tool calls that are structurally valid (would pass B4)
but semantically hijacked — i.e. the stated intent no longer matches the
tool being invoked, or an argument's actual content doesn't match its
declared semantic role.

Runs BEFORE B4 TCIV in Executor.handle() (see tciv.py's own docstring:
"B3 runs before B4 to close this gap") — TCIV can only verify a call is
correctly *shaped*, not that it means what it claims to mean. Neither B4
(structural-only) nor B5 (runs on a decoy honeypot copy, not the real
executor path) can substitute for this.

Three layers, run in order, first failure wins:

  Layer 1 — ICS  (Intent-Call Similarity)
            Does intent_text semantically match the tool being called?
            Embeds intent_text against "{tool_name}: {tool description}"
            and blocks if cosine similarity < ics_threshold.

  Layer 2 — ASV  (Argument Semantic Validation)
            Does each argument's value semantically match its declared
            arg_semantics description for that tool? Flags any argument
            whose cosine similarity to its declared meaning falls below
            asv_threshold; blocks if the flagged fraction exceeds
            asv_block_ratio.

  Layer 3 — BSD  (Baseline Score Drift)
            Does this call's ICS score sit above the configured
            percentile of a running baseline built from confirmed-clean
            calls? Non-blocking — this escalates (logs) rather than
            blocks, since it's an early-warning signal on an evolving
            baseline, not a hard-evidence gate.

Reuses the same MiniLM sentence-embedding + cosine-distance primitive
already proven in sds_measure.py and honeypot_agent.py — no new model,
no new dependency.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from config.config import cfg

log = logging.getLogger("fission.b3_semantic")

_model = None


# GPU audit fix (Findings 2/3): shared across sds_measure.py,
# context_poisoner.py, and honeypot_agent.py — see shared/embedding_model.py.
def _get_model():
    from shared.embedding_model import get_embedding_model
    return get_embedding_model()


def _cosine_sim(text_a: str, text_b: str) -> float:
    model = _get_model()
    emb = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


class GuardVerdict(Enum):
    PASS    = "PASS"
    BLOCKED = "BLOCKED"


@dataclass
class GuardResult:
    """Mirrors TCIVResult's shape so Executor.handle() can treat B3 and
    B4 results uniformly."""
    verdict:      GuardVerdict
    block_reason: Optional[str] = None
    layer_failed: Optional[str] = None    # "ics" | "asv" | None
    ics_score:    Optional[float] = None
    escalated:    bool = False            # True if Layer 3 flagged (non-blocking)
    flagged_args: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == GuardVerdict.PASS


class SemanticGuard:
    """
    B3 Semantic Anomaly Guard — three-layer semantic check on tool calls.

    Usage:
        guard = SemanticGuard()
        result = guard.check(
            tool_name    = "summarise_document",
            arguments    = {"document_id": "doc_001", "max_length": 150},
            intent_text  = "Summarise doc_001 and post to Slack",
        )
        if not result.passed:
            log.warning(result.block_reason)
    """

    def __init__(self):
        self._registry        = cfg.mcp.registered_tools
        self.ics_threshold    = cfg.defense.ics_threshold
        self.asv_threshold    = cfg.defense.asv_threshold
        self.asv_block_ratio  = cfg.defense.asv_block_ratio
        self.bsd_percentile   = cfg.defense.bsd_percentile

        # Layer 3 baseline — running ICS scores from confirmed-clean calls.
        # In-memory for now; see roadmap's "Persistent B3 baseline store"
        # future enhancement for a Redis-backed version that survives
        # process restarts.
        self._clean_ics_baseline: List[float] = []

    # ── Layer 1: Intent-Call Similarity ─────────────────────────────

    def _check_ics(self, tool_name: str, intent_text: str) -> Tuple[float, bool]:
        tool = self._registry.get(tool_name, {})
        description = tool.get("description", "")
        tool_string = f"{tool_name}: {description}"
        score = _cosine_sim(intent_text, tool_string)
        return score, score < self.ics_threshold

    # ── Layer 2: Argument Semantic Validation ───────────────────────

    def _check_asv(self, tool_name: str, arguments: dict) -> Tuple[List[str], bool]:
        tool = self._registry.get(tool_name, {})
        arg_semantics = tool.get("arg_semantics", {})
        if not arg_semantics or not arguments:
            return [], False

        flagged = []
        checked = 0  # only count args that were actually semantically evaluated
        for arg_name, arg_value in arguments.items():
            expected_meaning = arg_semantics.get(arg_name)
            if expected_meaning is None:
                continue  # no semantic spec registered for this arg — skip

            # Skip numeric / boolean args: comparing str(100) or str(True)
            # against a natural-language description always fails in embedding
            # space because digits/booleans carry no semantic meaning as text.
            # Only string arguments have content that can be compared.
            if not isinstance(arg_value, str):
                continue

            # Skip identifiers (IDs like "doc_001" are identifiers,
            # not semantic content — comparing them against descriptions
            # produces noise).
            if arg_name.endswith("_id") or "identifier" in expected_meaning.lower():
                continue

            score = _cosine_sim(str(arg_value), expected_meaning)
            checked += 1
            if score < self.asv_threshold:
                flagged.append(arg_name)

        # If no args were semantically evaluable, ASV passes by default.
        if checked == 0:
            return [], False

        # >= , not > : asv_block_ratio=0.50 should mean "half or more of the
        # *evaluated* arguments violate", not "strictly more than half".
        block = (len(flagged) / checked) >= self.asv_block_ratio
        return flagged, block

    # ── Layer 3: Baseline Score Drift (non-blocking) ────────────────

    def _check_bsd(self, ics_score: float) -> bool:
        """Returns True (escalate) if ics_score is an outlier relative to
        the clean baseline. Requires at least 10 baseline points before
        it will flag anything — too little data to judge otherwise."""
        if len(self._clean_ics_baseline) < 10:
            return False
        threshold = np.percentile(self._clean_ics_baseline, self.bsd_percentile)
        return ics_score > threshold

    def _record_clean(self, ics_score: float):
        self._clean_ics_baseline.append(ics_score)

    # ── Public entry point ───────────────────────────────────────────

    def check(self, tool_name: str, arguments: dict, intent_text: str) -> GuardResult:
        ics_score, ics_block = self._check_ics(tool_name, intent_text)

        if ics_block:
            log.warning(
                f"B3 Layer 1 (ICS) BLOCKED: intent/tool mismatch "
                f"(score={ics_score:.4f} < {self.ics_threshold})"
            )
            return GuardResult(
                verdict=GuardVerdict.BLOCKED,
                block_reason=f"Intent-call semantic mismatch (ICS={ics_score:.4f})",
                layer_failed="ics",
                ics_score=ics_score,
            )

        flagged_args, asv_block = self._check_asv(tool_name, arguments)

        if asv_block:
            log.warning(
                f"B3 Layer 2 (ASV) BLOCKED: {len(flagged_args)}/{len(arguments)} "
                f"arguments semantically inconsistent: {flagged_args}"
            )
            return GuardResult(
                verdict=GuardVerdict.BLOCKED,
                block_reason=f"Argument semantics violated: {flagged_args}",
                layer_failed="asv",
                ics_score=ics_score,
                flagged_args=flagged_args,
            )

        escalated = self._check_bsd(ics_score)
        if escalated:
            log.warning(
                f"B3 Layer 3 (BSD) ESCALATED (non-blocking): ICS={ics_score:.4f} "
                f"exceeds {self.bsd_percentile}th percentile of clean baseline"
            )
        else:
            self._record_clean(ics_score)

        return GuardResult(
            verdict=GuardVerdict.PASS,
            ics_score=ics_score,
            escalated=escalated,
            flagged_args=flagged_args,
        )


