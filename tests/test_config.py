from arxiv_rec import config


def test_config_defaults():
    assert config.PROJECT_ROOT.name == "arxiv-recommender"
    assert config.DATA_DIR.name == "data"
    assert config.ARTIFACTS_DIR.name == "artifacts"
    assert config.SNAPSHOT_JSON.parent == config.DATA_DIR
    assert config.SNAPSHOT_PARQUET.parent == config.DATA_DIR
    assert config.SNAPSHOT_PARQUET.suffix == ".parquet"
    assert config.METADATA_PATH.parent == config.ARTIFACTS_DIR
    assert config.EMBEDDINGS_PATH.parent == config.ARTIFACTS_DIR
    assert config.INDEX_PATH.parent == config.ARTIFACTS_DIR
    assert set(config.DEFAULT_COLUMNS) == {"id", "title", "abstract", "categories"}
