"""Cleaning helpers to prepare text for embedding."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

TEXT_COLUMNS: Iterable[str] = ("id", "title", "abstract", "categories")


def _normalize(value: str | float | None) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text.strip())
    return text


def prepare_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame with combined text ready for embeddings."""

    missing = [col for col in ("id", "title", "abstract") if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tidy = df.copy()
    for column in TEXT_COLUMNS:
        if column in tidy.columns:
            tidy[column] = tidy[column].apply(_normalize)

    tidy["text"] = (
        tidy["title"].where(tidy["title"] != "", other="")
        + ". "
        + tidy["abstract"].where(tidy["abstract"] != "", other="")
    ).str.strip(" .")

    tidy = tidy[tidy["text"] != ""].reset_index(drop=True)
    return tidy
