import pandas as pd
import pytest

from arxiv_rec.config import DEFAULT_COLUMNS
from arxiv_rec.data.ingest import load_metadata


def test_load_metadata_from_parquet(tmp_path):
    data = {
        "id": ["1", "2"],
        "title": ["A", "B"],
        "abstract": ["Text A", "Text B"],
        "categories": ["cs.AI", "cs.LG"],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "snapshot.parquet"
    df.to_parquet(path, index=False)

    loaded = load_metadata(path, columns=DEFAULT_COLUMNS)

    assert list(loaded.columns) == list(DEFAULT_COLUMNS)
    assert len(loaded) == 2


def test_load_metadata_raises_on_missing_columns(tmp_path):
    data = {
        "id": ["1"],
        "title": ["A"],
        "categories": ["cs.AI"],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "snapshot.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Columns missing"):
        load_metadata(path, columns=DEFAULT_COLUMNS)
