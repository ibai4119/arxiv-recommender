"""Utilities to load the raw arXiv metadata snapshot."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_COLUMNS: Sequence[str] = (
    "id",
    "title",
    "abstract",
    "categories",
    "doi",
    "created",
    "updated",
)


def load_metadata(data_path: str | Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the arXiv metadata snapshot from CSV or JSON (optionally zipped)."""

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found at {path}")

    selected_cols = list(columns) if columns is not None else list(DEFAULT_COLUMNS)

    if path.suffix == ".csv":
        df = pd.read_csv(path, usecols=selected_cols)
        return df

    if path.suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            json_members = [name for name in zf.namelist() if name.endswith(".json")]
            if not json_members:
                raise ValueError(f"No JSON file found inside {path}")
            with zf.open(json_members[0]) as fh:
                buffer = io.TextIOWrapper(fh, encoding="utf-8")
                df = pd.read_json(buffer, lines=True)
    else:
        raise ValueError(f"Unsupported metadata format: {path.suffix}")

    missing = [col for col in selected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns missing in metadata: {missing}")

    return df[selected_cols]
