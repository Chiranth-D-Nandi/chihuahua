"""
FISSION — orchestrator/hf_inference_vlm.py
============================================
BaseVLM implementation that calls HuggingFace's hosted Inference API.

Models run on HF's servers — no local download needed.
Uses HF's OpenAI-compatible /v1/chat/completions endpoint so the
request/response format is identical to what the rest of the pipeline
already handles.

Requires:
  - A HuggingFace account + free API token (hf_...)
  - The model must be available on the HF Inference API
    (most popular VLMs like Qwen2.5-VL, LLaVA-Next are hosted)

Set token via:
  export HF_TOKEN=hf_...
  OR pass directly to HFInferenceVLM(token="hf_...")
  OR set FISSION_HF_TOKEN env var

Rate limits on free tier: ~1000 requests/day per model.
"""

import base64
import io
import logging
import os
from typing import Any, Optional

import httpx
from PIL import Image

from orchestrator.base_vlm import BaseVLM, VLMResponse
from orchestrator.response_parser import ResponseParser

log = logging.getLogger("fission.hf_inference_vlm")

# HF's OpenAI-compatible inference endpoint
# router.huggingface.co requires a provider segment — free serverless tier = hf-inference
_HF_API_BASE = "https://router.huggingface.co/hf-inference/v1"

# Models that support vision (multimodal) on HF Inference API
_VISION_CAPABLE = {
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "OpenGVLab/InternVL2-8B",
    "microsoft/Phi-3.5-vision-instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
}


class HFInferenceVLM(BaseVLM):
    """
    Calls HuggingFace Inference API — no local weights needed.

    Args:
        model_id:  Full HF model ID, e.g. "Qwen/Qwen2.5-VL-7B-Instruct"
        token:     HF API token (hf_...). Falls back to FISSION_HF_TOKEN
                   then HF_TOKEN environment variables.
        timeout:   HTTP timeout in seconds (default 120 — HF cold starts
                   can be slow on free tier when a model is warming up).
        parser:    ResponseParser instance. Shared singleton is fine.
    """

    def __init__(
        self,
        model_id: str,
        token: Optional[str] = None,
        timeout: int = 120,
        parser: Optional[ResponseParser] = None,
    ):
        self.model_id = model_id
        self.timeout = timeout
        self._parser = parser or ResponseParser()
        self._token = (
            token
            or os.environ.get("FISSION_HF_TOKEN")
            or os.environ.get("HF_TOKEN")
            or ""
        ).strip()  # strip trailing spaces/newlines from env file loading
        self._vision = model_id in _VISION_CAPABLE
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        self.error_stats = {
            "hf_auth_error": 0,
            "hf_model_loading": 0,
            "hf_rate_limit": 0,
            "hf_query_failed": 0,
        }

    # ── BaseVLM interface ────────────────────────────────────────

    def load_model(self) -> None:
        """
        For HF Inference API, 'loading' is just a connectivity + auth
        check — HF manages model lifecycle server-side, just like Ollama.
        """
        if not self._token:
            log.warning(
                "No HF token set — anonymous inference is severely rate-limited. "
                "Set FISSION_HF_TOKEN=hf_... for reliable access."
            )
            return
        try:
            r = httpx.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
            if r.status_code == 200:
                username = r.json().get("name", "?")
                log.info(f"HF token valid — authenticated as '{username}'")
            elif r.status_code == 401:
                log.error("HF token is invalid or expired (HTTP 401)")
            else:
                log.warning(f"HF auth check returned HTTP {r.status_code}")
        except Exception as e:
            log.warning(f"Could not verify HF token: {e}")

    def preprocess_image(self, image: Image.Image) -> str:
        """Returns a base64-encoded PNG data URI for embedding in the API payload."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def generate(self, system_prompt: str, task_text: str, image: Any) -> Optional[str]:
        """
        image: either None, a base64 data URI string (from preprocess_image),
               or a raw base64 string (from the pipeline's existing image
               handling in app.py which passes b64 without the data: prefix).

        HF's /v1/chat/completions accepts OpenAI-style messages.
        Vision models accept content arrays with image_url items.
        """
        if not self._token:
            log.warning("No HF token — attempting anonymous request (may be rate-limited)")

        # Normalise image to a data URI if it's a raw base64 blob
        image_uri: Optional[str] = None
        if image:
            if isinstance(image, str):
                if image.startswith("data:"):
                    image_uri = image
                else:
                    image_uri = f"data:image/png;base64,{image}"

        # Build message content
        if image_uri and self._vision:
            content = [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": f"{system_prompt}\n\nTask: {task_text}"},
            ]
        else:
            if image_uri and not self._vision:
                log.warning(
                    f"'{self.model_id}' is not in the vision-capable set — "
                    "sending text-only. Add it to _VISION_CAPABLE if it supports images."
                )
            content = f"{system_prompt}\n\nTask: {task_text}"

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
            "temperature": 0.1,
            "stream": False,
        }

        url = f"{_HF_API_BASE}/chat/completions"
        try:
            r = httpx.post(url, json=payload, headers=self._headers, timeout=self.timeout)
            body = ""
            try:
                body = r.json()
            except Exception:
                body = r.text[:300]

            if r.status_code == 200:
                return body["choices"][0]["message"]["content"]

            if r.status_code == 401:
                self.error_stats["hf_auth_error"] += 1
                log.error(f"[{self.model_id}] HF 401 — invalid/expired token. Body: {body}")
                self.last_error = "hf_auth_error"
                return None

            if r.status_code == 403:
                self.error_stats["hf_auth_error"] += 1
                log.error(f"[{self.model_id}] HF 403 — access denied (model may be gated). Body: {body}")
                self.last_error = "hf_gated"
                return None

            if r.status_code == 404:
                self.error_stats["hf_query_failed"] += 1
                log.error(f"[{self.model_id}] HF 404 — model not found on Inference API. Body: {body}")
                self.last_error = "hf_not_found"
                return None

            if r.status_code == 422:
                self.error_stats["hf_query_failed"] += 1
                log.error(f"[{self.model_id}] HF 422 — request format rejected. Body: {body}")
                self.last_error = "hf_format_error"
                return None

            if r.status_code == 429:
                self.error_stats["hf_rate_limit"] += 1
                log.error(f"[{self.model_id}] HF 429 — rate limit hit. Body: {body}")
                self.last_error = "hf_rate_limit"
                return None

            if r.status_code == 503:
                self.error_stats["hf_model_loading"] += 1
                estimated = body.get("estimated_time", "?") if isinstance(body, dict) else "?"
                log.warning(f"[{self.model_id}] HF 503 — model loading, est. {estimated}s. Body: {body}")
                self.last_error = "hf_model_loading"
                return None

            # Any other non-200
            self.error_stats["hf_query_failed"] += 1
            log.error(f"[{self.model_id}] HF HTTP {r.status_code}. Body: {body}")
            self.last_error = f"http_{r.status_code}"
            return None

        except httpx.TimeoutException:
            self.error_stats["hf_query_failed"] += 1
            self.last_error = "timeout"
            log.error(f"[{self.model_id}] HF API request timed out after {self.timeout}s")
            return None
        except Exception as e:
            self.error_stats["hf_query_failed"] += 1
            self.last_error = str(e)
            log.error(f"[{self.model_id}] HF API request failed: {e}")
            return None

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
        """No-op — HF manages the model lifecycle server-side."""
        pass

    def is_token_set(self) -> bool:
        return bool(self._token)
