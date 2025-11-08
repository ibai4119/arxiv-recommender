"""Utility script to download the arXiv metadata snapshot from Kaggle."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import requests
from requests.auth import HTTPBasicAuth

DATASET_SLUG = "cornell-university/arxiv"
TARGET_FILENAME = "arxiv-metadata-oai-snapshot.json"
CSV_FILENAME = "arxiv-metadata-oai-snapshot.csv"
DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}?fileName={TARGET_FILENAME}"
)
CSV_COLUMNS: Iterable[str] = (
    "id",
    "title",
    "abstract",
    "categories",
    "doi",
    "created",
    "updated",
)
CHUNK_SIZE = 1024 * 1024  # 1 MiB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where the dataset will be stored.",
    )
    parser.add_argument(
        "--convert-csv",
        action="store_true",
        help="Also convert the JSON lines file into CSV.",
    )
    parser.add_argument(
        "--remove-json",
        action="store_true",
        help="Delete the JSON file after finishing (implies you only rely on CSV).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload and re-extraction even if files already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for the CSV conversion (debugging).",
    )
    return parser.parse_args()


def require_kaggle_auth() -> HTTPBasicAuth:
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
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
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = downloaded / total_size * 100
                        sys.stdout.write(f"\r  -> {downloaded / (1024 * 1024):.2f} MiB / {total_size / (1024 * 1024):.2f} MiB ({percent:5.1f}%)")
                        sys.stdout.flush()
        if total_size:
            sys.stdout.write("\n")
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
            with zf.open(TARGET_FILENAME) as src, tmp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK_SIZE)
    else:
        print("Archive is already JSON, copying...")
        shutil.copyfile(archive_path, tmp_path)

    tmp_path.replace(json_path)
    print(f"JSON ready at {json_path}")
    return json_path


def convert_to_csv(json_path: Path, csv_path: Path, limit: int | None = None) -> Path:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    print("Converting JSON to CSV (this can take a while)...")
    with json_path.open("r", encoding="utf-8") as src, tmp_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        writer = csv.DictWriter(dst, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for idx, line in enumerate(src, start=1):
            record = json.loads(line)
            row = {column: record.get(column, "") for column in CSV_COLUMNS}
            writer.writerow(row)
            if limit and idx >= limit:
                break
    tmp_path.replace(csv_path)
    print(f"CSV available at {csv_path}")
    return csv_path


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
    json_path = output_dir / TARGET_FILENAME
    csv_path = output_dir / CSV_FILENAME

    auth = require_kaggle_auth()

    download_archive(archive_path, auth=auth, force=args.force)
    extract_json(archive_path, json_path, force=args.force)

    if args.convert_csv:
        convert_to_csv(json_path, csv_path, limit=args.limit)

    if args.remove_json:
        safe_remove(json_path)

    safe_remove(archive_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI helper
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
