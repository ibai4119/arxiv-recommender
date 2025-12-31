from arxiv_rec import config


def test_config_defaults():
    assert config.DATA_DIR.name == "data"
    assert config.ARTIFACTS_DIR.name == "artifacts"
    assert config.SNAPSHOT_JSON.parent == config.DATA_DIR
    assert config.SNAPSHOT_PARQUET.parent == config.DATA_DIR
    assert config.SNAPSHOT_PARQUET.suffix == ".parquet"
    assert set(config.DEFAULT_COLUMNS) == {"id", "title", "abstract", "categories"}
