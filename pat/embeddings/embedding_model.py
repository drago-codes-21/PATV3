"""
Sentence embedding wrapper used by semantic detectors.

This wrapper provides:
- lazy and thread-safe model loading
- consistent embedding dimension inference
- robust path discovery for offline models
- graceful handling of empty inputs
- safe fallback behaviour
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import threading
import logging
import numpy as np

from pat.config import get_settings

LOG = logging.getLogger(__name__)


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding model cannot be loaded or is invalid."""


@dataclass
class EmbeddingModel:
    """Lazy, thread-safe embedding model loader."""

    model_path: Path | None = None
    device: str | None = None

    # internal state (not dataclass fields)
    _model: any = None
    _path: Path | None = None
    _load_lock: threading.Lock = threading.Lock()
    _embedding_dim: int | None = None

    def __post_init__(self) -> None:
        settings = get_settings()
        path = Path(self.model_path or settings.embedding_model_path)

        if not path.exists():
            raise EmbeddingModelError(
                f"Embedding model path does not exist: {path}. "
                "Ensure offline weights have been downloaded."
            )

        resolved = self._resolve_model_dir(path)
        self._path = resolved

        # Auto-detect device if caller did not provide one
        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        LOG.debug("EmbeddingModel initialised. Path=%s Device=%s", self._path, self.device)

    # ----------------------------------------------------------------------
    # Path resolution
    # ----------------------------------------------------------------------
    def _resolve_model_dir(self, base_path: Path) -> Path:
        """
        Resolve the actual model directory that contains config.json.
        Handles structures like:
          - base_path/config.json
          - base_path/<subfolder>/config.json
        """
        config = base_path / "config.json"

        if config.is_file():
            return base_path

        # search for exactly one valid sub-directory
        candidates = [
            d for d in base_path.iterdir()
            if d.is_dir() and (d / "config.json").is_file()
        ]

        if len(candidates) == 1:
            LOG.debug("Resolved embedding model directory: %s", candidates[0])
            return candidates[0]

        if not candidates:
            raise EmbeddingModelError(
                f"No config.json found in '{base_path}'. "
                "EmbeddingModel requires a SentenceTransformer-compatible folder."
            )

        raise EmbeddingModelError(
            f"Multiple possible model folders in '{base_path}': {candidates}. "
            "Specify PAT_EMBEDDING_MODEL_PATH to disambiguate."
        )

    # ----------------------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------------------
    def _load(self) -> None:
        """Thread-safe lazy loader."""
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingModelError(
                    "sentence-transformers is required. Install via: pip install sentence-transformers"
                ) from exc

            LOG.info("Loading sentence-transformer model from %s", self._path)
            LOG.info("Device used for embeddings: %s", self.device)

            try:
                model = SentenceTransformer(str(self._path), device=self.device)
            except Exception as exc:
                raise EmbeddingModelError(
                    f"Failed to load embedding model from {self._path}: {exc}"
                )

            # infer embedding dimension once
            try:
                dim = int(model.get_sentence_embedding_dimension())
            except Exception:
                # fall back to standard SBERT default
                LOG.warning(
                    "Could not infer embedding dimension from model. "
                    "Defaulting to 768."
                )
                dim = 768

            self._embedding_dim = dim
            self._model = model

    # ----------------------------------------------------------------------
    # Public inference API
    # ----------------------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector."""
        if not text:
            return self._empty_vector()

        self._load()
        assert self._model is not None

        vector = self._model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(vector, dtype=np.float32)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts into a matrix."""
        if not texts:
            return self._empty_matrix()

        self._load()
        assert self._model is not None

        vectors = self._model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(vectors, dtype=np.float32)

    # ----------------------------------------------------------------------
    # Utility helpers
    # ----------------------------------------------------------------------
    def _empty_vector(self) -> np.ndarray:
        """Return zero vector matching embedding dimension."""
        self._load()
        assert self._embedding_dim is not None
        return np.zeros(self._embedding_dim, dtype=np.float32)

    def _empty_matrix(self) -> np.ndarray:
        """Return empty matrix of shape (0, dim)."""
        self._load()
        assert self._embedding_dim is not None
        return np.zeros((0, self._embedding_dim), dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension, triggering model load if needed."""
        if self._embedding_dim is None:
            self._load()
        return self._embedding_dim
