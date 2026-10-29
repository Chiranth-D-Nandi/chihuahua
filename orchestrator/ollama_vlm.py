"""
FISSION — orchestrator/ollama_vlm.py
======================================
BaseVLM implementation for any Ollama-servable model (llava, qwen2.5vl,
internvl2 — anything model_profiles.json lists, and anything else Ollama
can serve).

This wraps the EXISTING VLMClient (orchestrator/vlm_client.py) rather
than re-implementing raw HTTP — that duplication (generate_adversarial_
dataset.py's private call_ollama() vs. VLMClient) was the first finding
in R1_Model_Agnostic_Architecture_Analysis.md, Section 1. Adopting
OllamaVLM in the benchmark script (see the accompanying patch) deletes
that duplicate path as a side effect rather than as separate cleanup work.

Known, documented tradeoff versus the old call_ollama(): the Session-10
error-category granularity (ollama_timeout / ollama_connection_error /
ollama_http_error / ollama_malformed_response / ollama_other_error) is
narrowed here to two buckets, because VLMClient.query() catches a broad
Exception internally rather than exposing the specific requests
exception type. This class distinguishes "Ollama unreachable at all"
(checked via VLMClient.is_available() when a query fails) from "query
failed for some other reason" — coarser than before, but not silent. If
the fine-grained categories turn out to matter for the pipeline crash
reports, the fix is a small enhancement to VLMClient.query()'s except
block, not a reason to bypass VLMClient again.
"""

import base64
import io
import logging
from typing import Any, Optional

from PIL import Image

from orchestrator.base_vlm import BaseVLM, VLMResponse
from orchestrator.vlm_client import VLMClient
from orchestrator.response_parser import ResponseParser

log = logging.getLogger("fission.ollama_vlm")


class OllamaVLM(BaseVLM):
    """
    Args:
        model_name: an Ollama-servable model name, e.g. "llava:latest",
            "qwen2.5vl:7b", "internvl2:8b" — anything in model_profiles.json
            or anything else `ollama pull`-able.
        parser: a ResponseParser instance. Currently always applies the
            full fixup set (curly quotes, SQL-semicolon leakage, raw
            newlines, string concatenation) regardless of model — the
            per-model fixup registry described in the architecture
            analysis (Section 6, "ResponseParserStrategy") is a
            follow-on step, not part of this pass. Documented here so
            it isn't silently assumed to already exist.
    """

    def __init__(self, model_name: str, parser: Optional[ResponseParser] = None):
        self.model_name = model_name
        self._client = VLMClient(model=model_name)
        self._parser = parser or ResponseParser()

        # Coarser two-bucket error tracking — see module docstring for
        # why this is narrower than the old call_ollama()'s five buckets.
        self.error_stats = {
            "ollama_unreachable": 0,
            "ollama_query_failed": 0,
        }

    def load_model(self) -> None:
        """For Ollama, 'loading' means confirming the server is up and
        the model is pulled — Ollama owns the actual weight lifecycle."""
        if not self._client.is_available():
            log.warning(
                f"Ollama not reachable or no models pulled — "
                f"'{self.model_name}' may fail on first generate() call. "
                "Start it with `ollama serve` and `ollama pull {self.model_name}` "
                "if this is unexpected."
            )

    def preprocess_image(self, image: Image.Image) -> str:
        """
        Returns a base64-encoded PNG string. VLMClient.query() accepts
        either a file path or (per the ollama Python client's own
        handling) a base64 string in the images list — base64 is used
        here rather than round-tripping through a temp file, since the
        benchmark script already has images in memory as PIL objects and
        this avoids adding filesystem I/O to the hot path.
        """
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate(self, system_prompt: str, task_text: str, image: Any) -> Optional[str]:
        """
        image: whatever preprocess_image() returned (a base64 string).
        Delegates to VLMClient.query(), which handles system-prompt +
        image-message composition internally. Returns None on failure —
        never raises — matching the contract call_ollama() already had.
        """
        result = self._client.query(
            system_prompt=system_prompt,
            user_message=task_text,
            image_path=image,   # VLMClient accepts a base64 string here too
        )

        if result is None:
            if not self._client.is_available():
                self.error_stats["ollama_unreachable"] += 1
                log.error(f"[{self.model_name}] Ollama unreachable")
            else:
                self.error_stats["ollama_query_failed"] += 1
                log.error(f"[{self.model_name}] Query failed (Ollama reachable, cause unclear)")
            return None

        return result

    def parse_output(self, raw_text: Optional[str]) -> VLMResponse:
        if raw_text is None:
            return VLMResponse(
                raw_text=None, tool_name=None, arguments=None,
                parse_error="generate_failed_no_text",
            )

        parsed = self._parser.parse(raw_text)
        if parsed is None:
            return VLMResponse(
                raw_text=raw_text, tool_name=None, arguments=None,
                parse_error=self._parser.last_error,
            )

        return VLMResponse(
            raw_text=raw_text,
            tool_name=parsed.tool_name,
            arguments=parsed.arguments,
            parse_error=None,
        )

    def cleanup(self) -> None:
        """Documented no-op — Ollama manages the model's lifecycle
        server-side; there is nothing in this process to free."""
        pass

    # get_gradient_source() intentionally not overridden — Ollama never
    # exposes an in-process model to differentiate through, so the
    # BaseVLM default (returns None) is exactly correct here, not a gap.

