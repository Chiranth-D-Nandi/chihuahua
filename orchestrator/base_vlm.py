"""
FISSION — orchestrator/base_vlm.py
====================================
The single abstraction every model-talking code path should depend on,
instead of each having its own way of calling a model.

Two things this replaces:
  1. generate_adversarial_dataset.py's private call_ollama() — a
     hand-rolled requests.post duplicating VLMClient.
  2. The implicit assumption that "the VLM" always means LLaVA via Ollama.

One implementation per BACKEND FAMILY, not per model — e.g. one
OllamaVLM handles llava/qwen2.5vl/internvl2 (any Ollama-servable model).
Per-model differences (image resolution, parser quirks) come from
config/model_profiles.json via config/model_registry.py, not from a
new BaseVLM subclass per model.

Session 17 (R1 generalization, Steps 1-3 of the practical plan in
R1_Model_Agnostic_Architecture_Analysis.md). Only OllamaVLM is built in
this pass — HFVLM (in-process HuggingFace models) is deliberately not
built yet, per the document's own recommendation to cover "any
Ollama-servable model" first without touching gradient code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image


@dataclass
class VLMResponse:
    """
    Result of a full generate() + parse_output() round trip.

    raw_text is populated whenever the model said *anything*, even if
    parsing failed afterward — this matters for R3's SDS measurement,
    which needs the actual text the model produced independent of
    whether that text happened to parse into a clean tool call.
    """
    raw_text:    Optional[str]
    tool_name:   Optional[str]   # None if parsing failed or raw_text is None
    arguments:   Optional[dict]
    parse_error: Optional[str]   # None on success; ResponseParser.last_error otherwise


class BaseVLM(ABC):
    """
    One implementation per backend family. Per-model differences (image
    resolution, which parser fixups apply) are driven by
    config/model_profiles.json, not by subclassing.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load weights / establish connection. Idempotent — safe to call
        more than once. For a server-backed model (Ollama) this is a
        connectivity check, not a weight load."""

    @abstractmethod
    def preprocess_image(self, image: Image.Image) -> Any:
        """
        Model-specific preprocessing. Returns whatever generate() expects
        for this backend — e.g. a file path or base64 string for Ollama,
        a tensor for a local in-process model.
        """

    @abstractmethod
    def generate(self, system_prompt: str, task_text: str, image: Any) -> Optional[str]:
        """
        Returns the raw text response, or None on failure. Must NOT raise
        on a model/network failure — return None and let the caller decide
        how to log/tally it, matching the existing call_ollama() contract
        this replaces.
        """

    @abstractmethod
    def parse_output(self, raw_text: Optional[str]) -> VLMResponse:
        """
        Applies this model's registered parser fixups, then the shared
        JSON-extraction core, and wraps the result as a VLMResponse.
        If raw_text is None (generate() failed), returns a VLMResponse
        with everything None rather than raising.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """
        Free resources (VRAM for an in-process model). Matters far more
        for a locally-loaded HF model than for Ollama, which manages its
        own model lifecycle server-side — but every backend must
        implement this explicitly (even as a documented no-op) so a
        future HF backend can't be silently skipped by code that assumes
        cleanup() is always a no-op.
        """

    # NOT abstract — only needed by attacks that require real gradients
    # (a white-box path analogous to runner.py's LlavaWhiteBoxGradient).
    # Default returns None so black-box-only backends (like OllamaVLM,
    # which never has an in-process model to differentiate through) can
    # still fully implement BaseVLM without pretending to support
    # white-box attacks they structurally cannot.
    def get_gradient_source(self) -> Optional["Any"]:
        return None

