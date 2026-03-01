"""FastAPI service exposing semantic search + recommendations."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from arxiv_rec.services.recommender import RecommenderService

app = FastAPI(title="arXiv Recommender", version="0.1.0")

_service: RecommenderService | None = None


def get_service() -> RecommenderService:
    global _service
    if _service is None:
        _service = RecommenderService.from_artifacts()
    return _service


@app.get("/search")
def search(q: str = Query(..., min_length=3), k: int = Query(5, ge=1, le=50)):
    return {"results": get_service().search(q, k=k)}


@app.get("/recommend")
def recommend(item_id: str = Query(...), k: int = Query(5, ge=1, le=50)):
    try:
        results = get_service().recommend(item_id, k=k)
    except KeyError as exc:  # pragma: no cover - simple error translation
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"results": results}
