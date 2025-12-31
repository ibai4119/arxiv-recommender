"""Utility script to download the arXiv metadata snapshot from Kaggle."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.json as paj
import pyarrow.parquet as pq
import requests
from requests.auth import HTTPBasicAuth
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from arxiv_rec.config import DATA_DIR, SNAPSHOT_JSON, SNAPSHOT_PARQUET

DATASET_SLUG = "cornell-university/arxiv"
TARGET_FILENAME = "arxiv-metadata-oai-snapshot.json"
DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}?fileName={TARGET_FILENAME}"
)
CHUNK_SIZE = 1024 * 1024  # 1 MiB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory where the dataset will be stored.",
    )
    parser.add_argument(
        "--keep-json",
        action="store_true",
        help="Keep the JSON file after generating Parquet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload and re-extraction even if files already exist.",
    )
    return parser.parse_args()


def require_kaggle_auth() -> HTTPBasicAuth:
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY environment "
            "variables."
        )
    return HTTPBasicAuth(username, key)


def download_archive(archive_path: Path, auth: HTTPBasicAuth, force: bool) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and not force:
        print(f"Archive already exists at {archive_path}, skipping download.")
        return archive_path

    tmp_path = archive_path.with_suffix(archive_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    print("Downloading snapshot from Kaggle...")
    with requests.get(DOWNLOAD_URL, stream=True, auth=auth, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(show_speed=False),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Downloading", total=total_size or None)
            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress.update(task, completed=downloaded)
    tmp_path.replace(archive_path)
    print(f"Download complete: {archive_path}")
    return archive_path


def extract_json(archive_path: Path, json_path: Path, force: bool) -> Path:
    if json_path.exists() and not force:
        print(f"JSON already exists at {json_path}, skipping extraction.")
        return json_path

    tmp_path = json_path.with_suffix(json_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    if zipfile.is_zipfile(archive_path):
        print("Extracting JSON from archive...")
        with zipfile.ZipFile(archive_path) as zf:
            if TARGET_FILENAME not in zf.namelist():
                raise RuntimeError(f"{TARGET_FILENAME} not found inside {archive_path}")
            info = zf.getinfo(TARGET_FILENAME)
            total_size = info.file_size
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(show_speed=False),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Extracting", total=total_size or None)
                with zf.open(TARGET_FILENAME) as src, tmp_path.open("wb") as dst:
                    while True:
                        chunk = src.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        dst.write(chunk)
                        if total_size:
                            progress.update(task, advance=len(chunk))
    else:
        print("Archive is already JSON, copying...")
        shutil.copyfile(archive_path, tmp_path)

    tmp_path.replace(json_path)
    print(f"JSON ready at {json_path}")
    return json_path


def convert_to_parquet(json_path: Path, parquet_path: Path) -> Path:
    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    print("Converting JSON to Parquet (this can take a while)...")
    total_size = json_path.stat().st_size
    bytes_read = 0

    class ProgressFile:
        def __init__(self, file_obj):
            self._file = file_obj

        def read(self, size=-1):
            nonlocal bytes_read
            data = self._file.read(size)
            bytes_read += len(data)
            return data

        def readinto(self, b):
            nonlocal bytes_read
            n = self._file.readinto(b)
            if n is not None:
                bytes_read += n
            return n

        def __getattr__(self, name):
            return getattr(self._file, name)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(show_speed=False),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Converting", total=total_size or None)
        with json_path.open("rb") as fh:
            reader = paj.open_json(
                ProgressFile(fh),
                read_options=paj.ReadOptions(block_size=8 * 1024 * 1024),
            )

            writer: pq.ParquetWriter | None = None
            try:
                while True:
                    batch = reader.read_next_batch()
                    table = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema, compression="snappy")
                    writer.write_table(table)
                    if total_size:
                        progress.update(task, completed=bytes_read)
            except StopIteration:
                pass
            finally:
                if writer is not None:
                    writer.close()

    tmp_path.replace(parquet_path)
    print(f"Parquet available at {parquet_path}")
    return parquet_path


def safe_remove(path: Path) -> None:
    try:
        path.unlink()
        print(f"Removed {path}")
    except FileNotFoundError:
        pass


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{TARGET_FILENAME}.zip"
    json_path = output_dir / SNAPSHOT_JSON.name
    parquet_path = output_dir / SNAPSHOT_PARQUET.name

    if parquet_path.exists() and not args.force:
        print(f"Parquet already exists at {parquet_path}, skipping download.")
    elif json_path.exists() and not args.force:
        print(f"JSON already exists at {json_path}, skipping download.")
    else:
        auth = require_kaggle_auth()
        download_archive(archive_path, auth=auth, force=args.force)
        extract_json(archive_path, json_path, force=args.force)

    converted = False
    if parquet_path.exists() and not args.force:
        print(f"Parquet already exists at {parquet_path}, skipping conversion.")
    else:
        convert_to_parquet(json_path, parquet_path)
        converted = True

    if converted and not args.keep_json:
        safe_remove(json_path)

    safe_remove(archive_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI helper
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
