"""
FISSION — config/model_registry.py
=====================================
Resolves a model name string to a (BaseVLM instance, profile dict) pair.

This is the file that finally gives config/model_profiles.json a reason
to exist — per R1_Model_Agnostic_Architecture_Analysis.md, Section 1:
"grepping the full repo dump found zero references to this file anywhere
else — it's declared but never loaded by any script." This closes that
gap.

Session 17 (R1 generalization, Step 3 of the practical plan). Only the
Ollama backend is wired up in this pass — every profile in
model_profiles.json today (llava, qwen2.5vl, internvl2) is Ollama-served,
so this covers all three without needing HFVLM yet. Requesting a model
not present in model_profiles.json still works (falls back to treating
it as an Ollama-servable name with a generic profile) but logs a warning,
since none of that model's per-model characteristics (encoder, expected
tool-calling style) are known.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from orchestrator.base_vlm import BaseVLM
from orchestrator.ollama_vlm import OllamaVLM

log = logging.getLogger("fission.model_registry")

_PROFILES_PATH = os.path.join(os.path.dirname(__file__), "model_profiles.json")

_UNKNOWN_MODEL_PROFILE = {
    "display_name": None,
    "encoder":       None,
    "backbone":      None,
    "vram_gb_4bit":  None,
    "tool_calling":  "unknown",
    "safety_level":  "unknown",
    "backend":       "ollama",   # default assumption: try Ollama
    "vision":        False,
    "owner":         None,
}


@dataclass
class ModelProfile:
    name:          str
    display_name:  Optional[str]
    encoder:       Optional[str]
    backbone:      Optional[str]
    vram_gb_4bit:  Optional[int]
    tool_calling:  str
    safety_level:  str
    backend:       str    # "ollama" | "hf_inference"
    vision:        bool
    owner:         Optional[str]
    known:         bool   # False if this model wasn't in model_profiles.json


def _load_profiles() -> dict:
    if not os.path.exists(_PROFILES_PATH):
        log.warning(f"model_profiles.json not found at {_PROFILES_PATH} — all models will use the unknown-model fallback profile.")
        return {}
    with open(_PROFILES_PATH) as f:
        return json.load(f)


_PROFILES = _load_profiles()


def get_profile(model_name: str) -> ModelProfile:
    """
    Look up a model's profile. Falls back to a generic "unknown" profile
    (rather than raising) for any model not in model_profiles.json.
    """
    raw = _PROFILES.get(model_name)
    if raw is None:
        log.warning(
            f"'{model_name}' is not in model_profiles.json — proceeding with "
            "a generic profile. Per-model parser fixups and encoder/backbone "
            "metadata are unavailable for this model."
        )
        raw = _UNKNOWN_MODEL_PROFILE
        known = False
    else:
        known = True

    return ModelProfile(
        name=model_name,
        display_name=raw.get("display_name"),
        encoder=raw.get("encoder"),
        backbone=raw.get("backbone"),
        vram_gb_4bit=raw.get("vram_gb_4bit"),
        tool_calling=raw.get("tool_calling", "unknown"),
        safety_level=raw.get("safety_level", "unknown"),
        backend=raw.get("backend", "ollama"),
        vision=raw.get("vision", False),
        owner=raw.get("owner"),
        known=known,
    )


def get_vlm(model_name: str) -> BaseVLM:
    """
    Resolve a model name to a live BaseVLM instance.

    Routes to HFInferenceVLM for models with backend="hf_inference"
    (runs on HF's servers, no local download needed), and OllamaVLM
    for everything else.
    """
    from config.config import cfg
    profile = get_profile(model_name)

    if profile.backend == "hf_inference":
        from orchestrator.hf_inference_vlm import HFInferenceVLM
        log.info(
            f"Resolved '{model_name}' -> HFInferenceVLM "
            f"(known_profile={profile.known}, vision={profile.vision})"
        )
        return HFInferenceVLM(
            model_id=model_name,
            token=cfg.model.hf_token,
            timeout=cfg.model.hf_inference_timeout,
        )

    # Default: Ollama
    log.info(
        f"Resolved '{model_name}' -> OllamaVLM "
        f"(known_profile={profile.known}, tool_calling={profile.tool_calling})"
    )
    return OllamaVLM(model_name=model_name)


def list_known_models() -> list:
    """All models currently declared in model_profiles.json."""
    return list(_PROFILES.keys())


def list_ollama_models() -> list:
    """Models that use the Ollama backend (need to be pulled locally)."""
    return [k for k, v in _PROFILES.items() if v.get("backend", "ollama") == "ollama"]


def list_hf_models() -> list:
    """Models that run on HuggingFace Inference API (no local download)."""
    return [k for k, v in _PROFILES.items() if v.get("backend") == "hf_inference"]


