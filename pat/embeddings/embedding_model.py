"""Sentence embedding wrapper used by semantic detectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import logging

import numpy as np

from pat.config import get_settings

LOG = logging.getLogger(__name__)


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding model cannot be loaded."""


@dataclass
class EmbeddingModel:
    """Lazy loader for an offline embedding model."""

    model_path: Path | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        settings = get_settings()
        path = Path(self.model_path or settings.embedding_model_path)
        if not path.exists():
            raise EmbeddingModelError(
                f"Embedding model path does not exist: {path}. "
                "Ensure offline weights are available."
            )

        resolved_path = path
        config_file = resolved_path / "config.json"
        if not config_file.is_file():
            candidates = [
                candidate
                for candidate in resolved_path.iterdir()
                if candidate.is_dir() and (candidate / "config.json").is_file()
            ]
            if len(candidates) == 1:
                LOG.debug(
                    "Using model directory '%s' discovered inside '%s'.",
                    candidates[0],
                    resolved_path,
                )
                resolved_path = candidates[0]
                config_file = resolved_path / "config.json"

        if not config_file.is_file():
            raise EmbeddingModelError(
                f"Could not find a valid model at: '{resolved_path}'. "
                "Ensure the directory (or a single sub-directory) contains 'config.json'."
            )
        self._path = resolved_path
        self._model = None
        # Auto-detect device if not explicitly provided.
        if self.device is None:
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise EmbeddingModelError(
                "sentence-transformers is required for embedding operations."
            ) from exc

        LOG.info("Loading embedding model from %s", self._path)
        LOG.info("Using device: '%s' for sentence-transformer model.", self.device)
        self._model = SentenceTransformer(str(self._path), device=self.device)

    def encode(self, text: str) -> np.ndarray:
        """Return embedding vector for a single text string."""

        self._load()
        if not text:
            return np.zeros(self._embedding_dim, dtype=np.float32)
        assert self._model is not None
        vector = self._model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vector, dtype=np.float32)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Return embedding vectors for a batch of texts."""

        self._load()
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        assert self._model is not None
        vectors = self._model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def _embedding_dim(self) -> int:
        if self._model is None:
            self._load()
        if self._model is not None and hasattr(self._model, "get_sentence_embedding_dimension"):
            return int(self._model.get_sentence_embedding_dimension())
        return 768
