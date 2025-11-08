"""Sentence-Transformer embedding helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """Lightweight wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode_texts(
        self,
        texts: Sequence[str] | Iterable[str],
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_texts([text], batch_size=1, show_progress_bar=False)[0]
