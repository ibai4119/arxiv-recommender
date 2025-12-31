from types import SimpleNamespace

import arxiv_rec.cli.download_snapshot as download_snapshot
from arxiv_rec.config import SNAPSHOT_PARQUET


def test_skips_download_when_json_exists(monkeypatch, tmp_path):
    mod = download_snapshot
    (tmp_path / mod.TARGET_FILENAME).write_text("{}", encoding="utf-8")

    calls = {"download": 0, "extract": 0, "convert": 0}

    def fake_download(*_args, **_kwargs):
        calls["download"] += 1

    def fake_extract(*_args, **_kwargs):
        calls["extract"] += 1

    def fake_convert(*_args, **_kwargs):
        calls["convert"] += 1

    monkeypatch.setattr(mod, "download_archive", fake_download)
    monkeypatch.setattr(mod, "extract_json", fake_extract)
    monkeypatch.setattr(mod, "convert_to_parquet", fake_convert)
    monkeypatch.setattr(mod, "safe_remove", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "require_kaggle_auth", lambda: "auth")
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: SimpleNamespace(output_dir=tmp_path, keep_json=True, force=False),
    )

    mod.main()

    assert calls["download"] == 0
    assert calls["extract"] == 0
    assert calls["convert"] == 1


def test_skips_conversion_when_parquet_exists(monkeypatch, tmp_path):
    mod = download_snapshot
    json_path = tmp_path / mod.TARGET_FILENAME
    parquet_path = tmp_path / SNAPSHOT_PARQUET.name
    json_path.write_text("{}", encoding="utf-8")
    parquet_path.write_text("", encoding="utf-8")

    called = {"convert": 0}
    removed = []

    def fake_convert(*_args, **_kwargs):
        called["convert"] += 1

    def fake_remove(path):
        removed.append(path)

    monkeypatch.setattr(mod, "convert_to_parquet", fake_convert)
    monkeypatch.setattr(mod, "safe_remove", fake_remove)
    monkeypatch.setattr(mod, "require_kaggle_auth", lambda: "auth")
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: SimpleNamespace(output_dir=tmp_path, keep_json=False, force=False),
    )

    mod.main()

    assert called["convert"] == 0
    assert json_path not in removed


def test_force_redownloads_and_reconverts(monkeypatch, tmp_path):
    mod = download_snapshot
    (tmp_path / mod.TARGET_FILENAME).write_text("{}", encoding="utf-8")
    (tmp_path / SNAPSHOT_PARQUET.name).write_text("", encoding="utf-8")

    calls = {"auth": 0, "download": 0, "extract": 0, "convert": 0}

    def fake_auth():
        calls["auth"] += 1
        return "auth"

    def fake_download(*_args, **_kwargs):
        calls["download"] += 1

    def fake_extract(*_args, **_kwargs):
        calls["extract"] += 1

    def fake_convert(*_args, **_kwargs):
        calls["convert"] += 1

    monkeypatch.setattr(mod, "require_kaggle_auth", fake_auth)
    monkeypatch.setattr(mod, "download_archive", fake_download)
    monkeypatch.setattr(mod, "extract_json", fake_extract)
    monkeypatch.setattr(mod, "convert_to_parquet", fake_convert)
    monkeypatch.setattr(mod, "safe_remove", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: SimpleNamespace(output_dir=tmp_path, keep_json=True, force=True),
    )

    mod.main()

    assert calls["auth"] == 1
    assert calls["download"] == 1
    assert calls["extract"] == 1
    assert calls["convert"] == 1


def test_removes_json_after_conversion_by_default(monkeypatch, tmp_path):
    mod = download_snapshot
    json_path = tmp_path / mod.TARGET_FILENAME
    json_path.write_text("{}", encoding="utf-8")

    removed = []

    def fake_remove(path):
        removed.append(path)

    monkeypatch.setattr(mod, "convert_to_parquet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "safe_remove", fake_remove)
    monkeypatch.setattr(mod, "require_kaggle_auth", lambda: "auth")
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: SimpleNamespace(output_dir=tmp_path, keep_json=False, force=False),
    )

    mod.main()

    assert json_path in removed


def test_keep_json_preserves_source(monkeypatch, tmp_path):
    mod = download_snapshot
    json_path = tmp_path / mod.TARGET_FILENAME
    json_path.write_text("{}", encoding="utf-8")

    removed = []

    def fake_remove(path):
        removed.append(path)

    monkeypatch.setattr(mod, "convert_to_parquet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "safe_remove", fake_remove)
    monkeypatch.setattr(mod, "require_kaggle_auth", lambda: "auth")
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: SimpleNamespace(output_dir=tmp_path, keep_json=True, force=False),
    )

    mod.main()

    assert json_path not in removed
