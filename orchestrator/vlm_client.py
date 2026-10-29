"""
FISSION — orchestrator/vlm_client.py
======================================
Talks to the Ollama VLM server.
Handles text-only and image-attached queries.

ABHYUDAYA — orchestrator side.
"""

import logging
import time
from pathlib import Path
from typing import Optional

try:
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent
    ollama = None

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import cfg

log = logging.getLogger("fission.vlm_client")


class VLMClient:

    def __init__(self, model: str = None):
        self.model = model or cfg.model.default
        self.client = None if ollama is None else ollama.Client(host=cfg.model.ollama_host)
        if ollama is None:
            log.warning(
                "ollama package is not installed; VLMClient is unavailable until dependencies are installed"
            )
        else:
            log.info(f"VLMClient initialised — model: {self.model} @ {cfg.model.ollama_host}")

    def query(
        self,
        system_prompt: str,
        user_message:  str,
        image_path:    Optional[str] = None,
    ) -> Optional[str]:
        """
        Send a query to the VLM and return the raw text response.
        Returns None if the query fails.

        image_path: EITHER a filesystem path OR a base64-encoded image
            string. Ollama's `images` field accepts both, so any truthy
            value here is attached — we no longer gate attachment on
            Path(image_path).exists(). That gate was a real bug: for a
            multi-KB base64 string, .exists() raises OSError("File name
            too long") on Linux/macOS, and can silently return False on
            other platforms. Either way, the image was never attached
            and the model got queried text-only — which is why models
            sometimes replied things like "It is not possible to
            provide a response based on the given image" despite an
            image clearly being intended.
        """
        if self.client is None:
            log.error("VLM query failed: ollama package is not installed")
            return None

        try:
            if image_path:
                is_file = self._looks_like_existing_file(image_path)
                log.info(
                    f"Querying VLM with image file: {image_path}" if is_file
                    else f"Querying VLM with inline image data ({len(image_path)} chars)"
                )
                messages = [{
                    "role":    "user",
                    "content": user_message,
                    "images":  [image_path],
                }]
                # Note: when image is attached, system prompt goes in user message
                # because Ollama multimodal does not support system role with images
                messages[0]["content"] = (
                    f"{system_prompt}\n\nTask: {user_message}"
                )
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ]

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"num_predict": 128},
            )
            return response["message"]["content"]

        except Exception as e:
            log.error(f"VLM query failed: {e}")
            return None

    @staticmethod
    def _looks_like_existing_file(image_path: str) -> bool:
        """
        Best-effort check used only for logging clarity (file path vs.
        inline base64 data) — NOT used to decide whether to attach the
        image; that decision is made purely on image_path being truthy.
        Guards against Path.exists() raising OSError on inputs that are
        far too long to be a real path (e.g. base64 blobs).
        """
        if len(image_path) > 260:
            return False
        try:
            return Path(image_path).exists()
        except OSError:
            return False

    def is_available(self, retries: int = 3, delay: int = 3) -> bool:
        """Check if the Ollama server is running and the model is available."""
        import urllib.request
        import json

        for attempt in range(retries):
            try:
                url = f"{cfg.model.ollama_host}/api/tags"
                with urllib.request.urlopen(url, timeout=5) as r:
                    data   = json.loads(r.read())
                    models = [m["name"] for m in data.get("models", [])]
                    if models:
                        log.info(f"Ollama available. Models: {', '.join(models)}")
                        return True
                    log.warning(f"Ollama running but no models pulled ({attempt+1}/{retries})")
            except Exception as e:
                log.warning(f"Ollama not reachable ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)

        return False

