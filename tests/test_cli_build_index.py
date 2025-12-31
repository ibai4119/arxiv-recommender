from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import arxiv_rec.cli.build_index as build_cli
from arxiv_rec.config import DEFAULT_COLUMNS


class DummyEmbeddingService:
    last_instance = None

    def __init__(self, device=None):
        self.device = device
        self.calls = []
        DummyEmbeddingService.last_instance = self

    def encode_texts(self, texts, batch_size=64):
        self.calls.append({"count": len(texts), "batch_size": batch_size})
        return np.zeros((len(texts), 3), dtype=np.float32)


class DummyVectorIndex:
    last_embeddings = None

    def __init__(self, size):
        self._size = size

    @property
    def size(self):
        return self._size

    @classmethod
    def from_embeddings(cls, embeddings):
        cls.last_embeddings = embeddings
        return cls(len(embeddings))

    def save(self, path):
        path.write_text("index", encoding="utf-8")
        return path


def make_frame(rows=3):
    data = {
        "id": [str(i) for i in range(rows)],
        "title": [f"Title {i}" for i in range(rows)],
        "abstract": [f"Abstract {i}" for i in range(rows)],
        "categories": ["cs.AI"] * rows,
    }
    return pd.DataFrame(data)


def test_build_index_calls_components(monkeypatch, tmp_path):
    calls = {"load": 0, "clean": 0}

    def fake_load(path, columns):
        calls["load"] += 1
        assert list(columns) == list(DEFAULT_COLUMNS)
        return make_frame()

    def fake_prepare(df):
        calls["clean"] += 1
        df = df.copy()
        df["text"] = df["title"] + ". " + df["abstract"]
        return df

    monkeypatch.setattr(build_cli, "load_metadata", fake_load)
    monkeypatch.setattr(build_cli, "prepare_corpus", fake_prepare)
    monkeypatch.setattr(build_cli, "EmbeddingService", DummyEmbeddingService)
    monkeypatch.setattr(build_cli, "VectorIndex", DummyVectorIndex)
    monkeypatch.setattr(
        build_cli,
        "parse_args",
        lambda: SimpleNamespace(
            data_path=tmp_path / "snapshot.parquet",
            artifacts_dir=tmp_path / "artifacts",
            limit=None,
            batch_size=4,
            device=None,
            reuse_embeddings=False,
        ),
    )

    build_cli.main()

    assert calls["load"] == 1
    assert calls["clean"] == 1


def test_build_index_respects_limit(monkeypatch, tmp_path):
    def fake_load(path, columns):
        return make_frame(rows=5)

    def fake_prepare(df):
        df = df.copy()
        df["text"] = df["title"] + ". " + df["abstract"]
        return df

    monkeypatch.setattr(build_cli, "load_metadata", fake_load)
    monkeypatch.setattr(build_cli, "prepare_corpus", fake_prepare)
    monkeypatch.setattr(build_cli, "EmbeddingService", DummyEmbeddingService)
    monkeypatch.setattr(build_cli, "VectorIndex", DummyVectorIndex)
    monkeypatch.setattr(
        build_cli,
        "parse_args",
        lambda: SimpleNamespace(
            data_path=tmp_path / "snapshot.parquet",
            artifacts_dir=tmp_path / "artifacts",
            limit=2,
            batch_size=4,
            device=None,
            reuse_embeddings=False,
        ),
    )

    build_cli.main()

    assert DummyEmbeddingService.last_instance.calls[0]["count"] == 2


def test_build_index_uses_device(monkeypatch, tmp_path):
    def fake_load(path, columns):
        return make_frame(rows=1)

    def fake_prepare(df):
        df = df.copy()
        df["text"] = df["title"] + ". " + df["abstract"]
        return df

    monkeypatch.setattr(build_cli, "load_metadata", fake_load)
    monkeypatch.setattr(build_cli, "prepare_corpus", fake_prepare)
    monkeypatch.setattr(build_cli, "EmbeddingService", DummyEmbeddingService)
    monkeypatch.setattr(build_cli, "VectorIndex", DummyVectorIndex)
    monkeypatch.setattr(
        build_cli,
        "parse_args",
        lambda: SimpleNamespace(
            data_path=tmp_path / "snapshot.parquet",
            artifacts_dir=tmp_path / "artifacts",
            limit=None,
            batch_size=4,
            device="mps",
            reuse_embeddings=False,
        ),
    )

    build_cli.main()

    assert DummyEmbeddingService.last_instance.device == "mps"


def test_build_index_reuses_embeddings(monkeypatch, tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    cached = np.zeros((2, 3), dtype=np.float32)
    np.save(artifacts_dir / "embeddings.npy", cached)

    def fake_load(path, columns):
        return make_frame(rows=2)

    def fake_prepare(df):
        df = df.copy()
        df["text"] = df["title"] + ". " + df["abstract"]
        return df

    def fail_encode(*_args, **_kwargs):
        raise AssertionError("encode_texts should not be called when reuse_embeddings is True")

    class ReuseEmbeddingService(DummyEmbeddingService):
        def encode_texts(self, texts, batch_size=64):
            return fail_encode()

    monkeypatch.setattr(build_cli, "load_metadata", fake_load)
    monkeypatch.setattr(build_cli, "prepare_corpus", fake_prepare)
    monkeypatch.setattr(build_cli, "EmbeddingService", ReuseEmbeddingService)
    monkeypatch.setattr(build_cli, "VectorIndex", DummyVectorIndex)
    monkeypatch.setattr(
        build_cli,
        "parse_args",
        lambda: SimpleNamespace(
            data_path=tmp_path / "snapshot.parquet",
            artifacts_dir=artifacts_dir,
            limit=None,
            batch_size=4,
            device=None,
            reuse_embeddings=True,
        ),
    )

    build_cli.main()

    assert DummyVectorIndex.last_embeddings is not None


def test_build_index_writes_artifacts(monkeypatch, tmp_path):
    def fake_load(path, columns):
        return make_frame(rows=1)

    def fake_prepare(df):
        df = df.copy()
        df["text"] = df["title"] + ". " + df["abstract"]
        return df

    monkeypatch.setattr(build_cli, "load_metadata", fake_load)
    monkeypatch.setattr(build_cli, "prepare_corpus", fake_prepare)
    monkeypatch.setattr(build_cli, "EmbeddingService", DummyEmbeddingService)
    monkeypatch.setattr(build_cli, "VectorIndex", DummyVectorIndex)
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        build_cli,
        "parse_args",
        lambda: SimpleNamespace(
            data_path=tmp_path / "snapshot.parquet",
            artifacts_dir=artifacts_dir,
            limit=None,
            batch_size=4,
            device=None,
            reuse_embeddings=False,
        ),
    )

    build_cli.main()

    assert (artifacts_dir / "metadata.parquet").exists()
    assert (artifacts_dir / "embeddings.npy").exists()
    assert (artifacts_dir / "index.faiss").exists()
    assert (artifacts_dir / "metadata_full.parquet").exists()
