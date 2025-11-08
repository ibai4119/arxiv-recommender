"""Thin FAISS wrapper used by both the build script and the API."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray) -> "VectorIndex":
        instance = cls(embeddings.shape[1])
        instance.add(embeddings)
        return instance

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: np.ndarray) -> None:
        vectors = self._normalize(embeddings)
        self.index.add(vectors)

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        queries = self._normalize(query_embeddings)
        scores, indices = self.index.search(queries, k)
        return scores, indices

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found at {path}")
        instance = cls.__new__(cls)
        index = faiss.read_index(str(path))
        instance.index = index
        instance.dimension = index.d
        return instance

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        vectors = np.asarray(array).astype("float32")
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=0)
        faiss.normalize_L2(vectors)
        return vectors
