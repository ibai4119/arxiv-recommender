"""Recommendation service built from persisted artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from arxiv_rec.config import EMBEDDINGS_PATH, INDEX_PATH, METADATA_PATH
from arxiv_rec.models.embed import EmbeddingService
from arxiv_rec.models.index import VectorIndex


class RecommenderService:
    """Serve semantic search and nearest-neighbour recommendations."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        embeddings: np.ndarray,
        index: VectorIndex,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.metadata = metadata
        self.embeddings = embeddings
        self.index = index
        self.embedder = embedder or EmbeddingService()
        self.row_lookup = {
            str(item_id): idx for idx, item_id in enumerate(self.metadata["id"].astype(str))
        }
        if self.index.size != len(self.metadata):
            raise RuntimeError("Index size does not match metadata length.")

    @classmethod
    def from_artifacts(cls) -> "RecommenderService":
        if not METADATA_PATH.exists() or not EMBEDDINGS_PATH.exists():
            raise RuntimeError("Artifacts missing. Run `make embed` first.")

        metadata = pd.read_parquet(METADATA_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        if INDEX_PATH.exists():
            index = VectorIndex.load(INDEX_PATH)
        else:
            index = VectorIndex.from_embeddings(embeddings)
        return cls(metadata=metadata, embeddings=embeddings, index=index)

    def format_results(self, indices: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.metadata):
                continue
            record = self.metadata.iloc[int(idx)]
            results.append(
                {
                    "id": record.get("id", ""),
                    "title": record.get("title", ""),
                    "abstract": record.get("abstract", ""),
                    "categories": record.get("categories", ""),
                    "score": float(score),
                }
            )
        return results

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query_emb = self.embedder.encode_query(query)
        scores, indices = self.index.search(query_emb, k=k)
        return self.format_results(indices[0], scores[0])

    def recommend(self, item_id: str, k: int = 5) -> list[dict[str, Any]]:
        if item_id not in self.row_lookup:
            raise KeyError(f"Item id {item_id} not found")
        row_idx = self.row_lookup[item_id]
        item_emb = self.embeddings[row_idx]
        scores, indices = self.index.search(item_emb, k=k + 1)
        filtered = [
            (idx, score)
            for idx, score in zip(indices[0], scores[0])
            if idx != row_idx and idx != -1
        ][:k]
        return self.format_results(
            np.array([idx for idx, _ in filtered], dtype=int),
            np.array([score for _, score in filtered], dtype=float),
        )
