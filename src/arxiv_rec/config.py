"""Project configuration defaults."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")

SNAPSHOT_JSON = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
SNAPSHOT_PARQUET = DATA_DIR / "arxiv-metadata-oai-snapshot.parquet"

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
