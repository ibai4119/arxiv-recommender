from fastapi.testclient import TestClient

from arxiv_rec.api import server


class DummyService:
    def search(self, query, k=5):
        return [{"id": "1", "title": query, "score": 0.9, "abstract": "", "categories": "cs.AI"}]

    def recommend(self, item_id, k=5):
        if item_id == "missing":
            raise KeyError("Item id missing not found")
        return [{"id": "2", "title": "recommended", "score": 0.8, "abstract": "", "categories": "cs.LG"}]


def test_search_endpoint(monkeypatch):
    monkeypatch.setattr(server, "get_service", lambda: DummyService())
    client = TestClient(server.app)

    response = client.get("/search", params={"q": "graph embeddings", "k": 3})

    assert response.status_code == 200
    assert response.json()["results"][0]["title"] == "graph embeddings"


def test_recommend_endpoint(monkeypatch):
    monkeypatch.setattr(server, "get_service", lambda: DummyService())
    client = TestClient(server.app)

    response = client.get("/recommend", params={"item_id": "1234.5678", "k": 2})

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "2"


def test_recommend_returns_404_for_missing_item(monkeypatch):
    monkeypatch.setattr(server, "get_service", lambda: DummyService())
    client = TestClient(server.app)

    response = client.get("/recommend", params={"item_id": "missing", "k": 2})

    assert response.status_code == 404
    assert "missing" in response.json()["detail"]
