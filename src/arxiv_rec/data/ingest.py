"""Utilities to load the raw arXiv metadata snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from arxiv_rec.config import FULL_COLUMNS

ProgressCallback = Callable[[int, int], None] | None


def load_metadata(
    data_path: str | Path,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load the arXiv metadata snapshot from Parquet.

    Args:
        data_path: Location of the dataset.
        columns: Optional subset of columns to return.
    """

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found at {path}")

    selected_cols = list(columns) if columns is not None else list(FULL_COLUMNS)

    if path.suffix != ".parquet":
        raise ValueError(f"Unsupported metadata format: {path.suffix}")

    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    missing = [col for col in selected_cols if col not in available]
    if missing:
        raise ValueError(f"Columns missing in metadata: {missing}")

    df = pd.read_parquet(path, columns=selected_cols)

    return df[selected_cols]
