import numpy as np

from arxiv_rec.models import embed as embed_module


class DummyModel:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
        self.calls = []

    def encode(
        self,
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        texts_list = list(texts)
        self.calls.append(
            {
                "texts": texts_list,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        size = len(texts_list)
        data = np.ones((size, 3), dtype=np.float64)
        return data


def test_embedding_service_encodes_and_casts(monkeypatch):
    monkeypatch.setattr(embed_module, "SentenceTransformer", DummyModel)
    service = embed_module.EmbeddingService(model_name="dummy", device="mps")
    result = service.encode_texts(["a", "b"], batch_size=2, show_progress_bar=False)

    assert result.shape == (2, 3)
    assert result.dtype == np.float32
    assert service.model.model_name == "dummy"
    assert service.model.device == "mps"


def test_embedding_service_passes_encode_flags(monkeypatch):
    monkeypatch.setattr(embed_module, "SentenceTransformer", DummyModel)
    service = embed_module.EmbeddingService(model_name="dummy", device=None)
    service.encode_texts(["x"], batch_size=8, show_progress_bar=True)

    assert len(service.model.calls) == 1
    call = service.model.calls[0]
    assert call["batch_size"] == 8
    assert call["show_progress_bar"] is True
    assert call["convert_to_numpy"] is True
    assert call["normalize_embeddings"] is True


def test_embedding_service_handles_empty_input(monkeypatch):
    monkeypatch.setattr(embed_module, "SentenceTransformer", DummyModel)
    service = embed_module.EmbeddingService(model_name="dummy")
    result = service.encode_texts([], batch_size=4, show_progress_bar=False)

    assert result.shape == (0, 3)
    assert result.dtype == np.float32


def test_embedding_service_encode_query(monkeypatch):
    monkeypatch.setattr(embed_module, "SentenceTransformer", DummyModel)
    service = embed_module.EmbeddingService(model_name="dummy")
    result = service.encode_query("hola")

    assert result.shape == (3,)
    assert result.dtype == np.float32
    assert service.model.calls[0]["texts"] == ["hola"]
    assert service.model.calls[0]["batch_size"] == 1
    assert service.model.calls[0]["show_progress_bar"] is False
