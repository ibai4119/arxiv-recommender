"""Project configuration defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SNAPSHOT_JSON = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
SNAPSHOT_PARQUET = DATA_DIR / "arxiv-metadata-oai-snapshot.parquet"
METADATA_PATH = ARTIFACTS_DIR / "metadata.parquet"
FULL_METADATA_PATH = ARTIFACTS_DIR / "metadata_full.parquet"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"

DEFAULT_COLUMNS = ("id", "title", "abstract", "categories")
FULL_COLUMNS = (
    "id",
    "submitter",
    "authors",
    "title",
    "comments",
    "journal-ref",
    "doi",
    "report-no",
    "categories",
    "license",
    "abstract",
    "versions",
    "update_date",
    "authors_parsed",
)
