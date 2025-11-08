"""Build embeddings + FAISS index from the arXiv metadata snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from arxiv_rec.data.clean import prepare_corpus
from arxiv_rec.data.ingest import load_metadata
from arxiv_rec.models.embed import EmbeddingService
from arxiv_rec.models.index import VectorIndex

DEFAULT_DATA_PATH = Path("data/arxiv-metadata-oai-snapshot.json")
DEFAULT_ARTIFACTS = Path("artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick runs")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data_path
    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {data_path}...")
    df = load_metadata(data_path)
    if args.limit:
        df = df.head(args.limit)

    tidy_df = prepare_corpus(df)
    texts = tidy_df["text"].tolist()

    print("Computing embeddings with SentenceTransformer...")
    embedder = EmbeddingService()
    embeddings = embedder.encode_texts(texts, batch_size=args.batch_size)

    metadata_cols = [col for col in ("id", "title", "abstract", "categories", "text") if col in tidy_df.columns]
    metadata_path = artifacts_dir / "metadata.parquet"
    tidy_df[metadata_cols].to_parquet(metadata_path, index=False)
    print(f"Saved metadata to {metadata_path}")

    embeddings_path = artifacts_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    index = VectorIndex.from_embeddings(embeddings)
    index_path = artifacts_dir / "index.faiss"
    index.save(index_path)
    print(f"FAISS index with {index.size} items saved to {index_path}")


if __name__ == "__main__":
    main()
