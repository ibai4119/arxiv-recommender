"""FastAPI service exposing semantic search + recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from arxiv_rec.models.embed import EmbeddingService
from arxiv_rec.models.index import VectorIndex

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
METADATA_PATH = ARTIFACTS_DIR / "metadata.parquet"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"

app = FastAPI(title="arXiv Recommender", version="0.1.0")


class RecommenderState:
    def __init__(self) -> None:
        if not METADATA_PATH.exists() or not EMBEDDINGS_PATH.exists():
            raise RuntimeError("Artifacts missing. Run `make embed` first.")

        self.metadata = pd.read_parquet(METADATA_PATH)
        self.embeddings = np.load(EMBEDDINGS_PATH)
        if INDEX_PATH.exists():
            self.index = VectorIndex.load(INDEX_PATH)
        else:
            self.index = VectorIndex.from_embeddings(self.embeddings)

        self.embedder = EmbeddingService()
        self.row_lookup = {
            str(item_id): idx for idx, item_id in enumerate(self.metadata["id"].astype(str))
        }
        if self.index.size != len(self.metadata):
            raise RuntimeError("Index size does not match metadata length.")

    def _format_results(self, indices: np.ndarray, scores: np.ndarray) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
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

    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        query_emb = self.embedder.encode_query(query)
        scores, indices = self.index.search(query_emb, k=k)
        return self._format_results(indices[0], scores[0])

    def recommend(self, item_id: str, k: int = 5) -> List[Dict[str, str]]:
        if item_id not in self.row_lookup:
            raise KeyError(f"Item id {item_id} not found")
        row_idx = self.row_lookup[item_id]
        item_emb = self.embeddings[row_idx]
        scores, indices = self.index.search(item_emb, k=k + 1)
        flat_indices = indices[0]
        flat_scores = scores[0]
        filtered = [
            (idx, score)
            for idx, score in zip(flat_indices, flat_scores)
            if idx != row_idx and idx != -1
        ][:k]
        formatted = self._format_results(
            np.array([idx for idx, _ in filtered], dtype=int),
            np.array([score for _, score in filtered], dtype=float),
        )
        return formatted


_state: RecommenderState | None = None


def get_state() -> RecommenderState:
    global _state
    if _state is None:
        _state = RecommenderState()
    return _state


@app.get("/search")
def search(q: str = Query(..., min_length=3), k: int = Query(5, ge=1, le=50)):
    state = get_state()
    return {"results": state.search(q, k=k)}


@app.get("/recommend")
def recommend(item_id: str = Query(...), k: int = Query(5, ge=1, le=50)):
    state = get_state()
    try:
        results = state.recommend(item_id, k=k)
    except KeyError as exc:  # pragma: no cover - simple error translation
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"results": results}
