"""
FISSION — shared/embedding_model.py
====================================
Shared SentenceTransformer embedding model singleton.

All modules that need text embeddings (B3 Semantic Guard, B5 PHANTOM,
L5b Context Poisoning) use this single shared instance. Loading
all-MiniLM-L6-v2 once and sharing it avoids the GPU OOM that would
result from each module loading its own copy.

Usage
-----
    from shared.embedding_model import get_embedding_model

    model = get_embedding_model()   # returns the singleton, loads on first call
    embedding = model.encode("some text", convert_to_numpy=True)

The singleton is process-global (module-level). It is NOT
thread-safe for concurrent encode() calls — if you need parallelism,
use a thread pool with the model replicated, or use the
sentence-transformers pipeline's built-in batching.

GPU audit note: this singleton IS loaded on every import of any module
that transitively imports it (B3, B5, L5b). If you import all three
modules in the same process, there is still exactly ONE copy of the
model in GPU memory, not three.

ABHYUDAYA — shared infrastructure.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("fission.shared.embedding")

# Model name — all-MiniLM-L6-v2 is fast (~18ms/review on CPU, ~2ms on GPU)
# and well-suited for short text similarity tasks (intent vs tool, tool vs
# argument semantics). Larger models (all-mpnet-base-v2) would improve
# quality at the cost of speed; swap here if quality > speed.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy-loaded singleton — loaded once on first get_embedding_model() call
_model_instance: Optional["EmbeddingModel"] = None
_load_lock = threading.Lock()


class EmbeddingModel:
    """
    Thin wrapper around sentence-transformers SentenceTransformer.

    Provides:
      - Lazy loading (first encode call triggers load)
      - encode() passthrough with sensible defaults
      - device property (cuda/cpu)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model_name = model_name
        self._model = None
        self._device = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        if self._device is None:
            # Probe CUDA availability after first load
            self._device = "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu"
        return self._device

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {self._model_name} ...")
        self._model = SentenceTransformer(self._model_name)
        logger.info(f"Embedding model loaded on device: {self.device}")

    def encode(self, *args, **kwargs):
        """Passthrough to sentence_transformers.SentenceTransformer.encode().
        First call triggers lazy loading."""
        self._ensure_loaded()
        return self._model.encode(*args, **kwargs)

    def get_device(self):
        """Returns the torch device the underlying model is on."""
        self._ensure_loaded()
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return "cpu"


# ── Module-level singleton getter ─────────────────────────────────────────────

def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    force_reload: bool = False,
) -> EmbeddingModel:
    """
    Return the shared EmbeddingModel singleton.

    The singleton is created and loaded on the first call.
    Subsequent calls return the same instance.

    Args
    ----
        model_name: SentenceTransformer model name. Only used on first call;
                    ignored on subsequent calls. Default: all-MiniLM-L6-v2.
        force_reload: If True, discards the existing singleton and reloads.
                      Useful for testing or swapping models in the same process.

    Returns
    -------
        EmbeddingModel (the shared singleton)
    """
    global _model_instance

    if force_reload:
        with _load_lock:
            logger.warning("Force-reloading embedding model singleton")
            _model_instance = None

    if _model_instance is None:
        with _load_lock:
            # Double-check inside lock
            if _model_instance is None:
                _model_instance = EmbeddingModel(model_name=model_name)
                logger.info(
                    f"Embedding model singleton created "
                    f"(model={model_name}). "
                    f"Load deferred until first encode() call."
                )

    return _model_instance


def preload(model_name: str = EMBEDDING_MODEL_NAME) -> "EmbeddingModel":
    """
    Eagerly create the singleton and load weights into memory.

    Used by dashboard.app lifespan so the first request doesn't pay the
    SentenceTransformer download/load cost. Safe to call multiple times.
    """
    model = get_embedding_model(model_name=model_name)
    # Force the underlying ST weights to load now (not on first encode).
    try:
        model._ensure_loaded()
    except Exception as exc:
        logger.warning("preload: model created but weight load deferred (%s)", exc)
    return model


def unload_embedding_model():
    """
    Explicitly unload the embedding model from GPU/CPU memory.
    Call this at the end of a benchmark script if you need to free
    VRAM before loading a different large model (e.g. the CLIP probe).

    After this call, get_embedding_model() will reload on next use.
    """
    global _model_instance
    with _load_lock:
        if _model_instance is not None and _model_instance._model is not None:
            del _model_instance._model
            _model_instance._model = None
            _model_instance._device = None
            logger.info("Embedding model unloaded from memory")
        _model_instance = None