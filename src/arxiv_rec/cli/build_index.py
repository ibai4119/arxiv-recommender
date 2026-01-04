"""Build embeddings + FAISS index from the arXiv metadata snapshot."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import numpy.lib.format as npformat
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from arxiv_rec.config import ARTIFACTS_DIR, DEFAULT_COLUMNS, FULL_COLUMNS, SNAPSHOT_PARQUET
from arxiv_rec.data.clean import prepare_corpus
from arxiv_rec.data.ingest import load_metadata
from arxiv_rec.models.embed import EmbeddingService
from arxiv_rec.models.index import VectorIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=SNAPSHOT_PARQUET)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick runs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--embed-chunk-size",
        type=int,
        default=10000,
        help="Chunk size for incremental embedding writes.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Update progress description every N batches.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index (0-based) when splitting the dataset.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of shards to split the dataset into.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Embedding device override (e.g. cpu, mps, cuda).",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse embeddings if artifacts already exist and match the input size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data_path
    if args.shard_count > 1:
        shard_label = f"shard_{args.shard_index:03d}_of_{args.shard_count:03d}"
        artifacts_dir = args.artifacts_dir / shard_label
    else:
        artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    total_steps = 7

    progress = Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="yellow", complete_style="green"),
        TaskProgressColumn(show_speed=True),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )

    def run_stage(description: str, func):
        progress.update(pipeline_task, description=description)
        result = func()
        progress.advance(pipeline_task)
        return result

    with progress:
        pipeline_task = progress.add_task("Inicializando pipeline...", total=total_steps)

        def apply_shard(frame):
            if args.shard_count <= 1:
                return frame
            if args.shard_index < 0 or args.shard_index >= args.shard_count:
                raise ValueError("--shard-index must be within [0, --shard-count).")
            total = len(frame)
            shard_size = total // args.shard_count
            remainder = total % args.shard_count
            start = args.shard_index * shard_size + min(args.shard_index, remainder)
            end = start + shard_size + (1 if args.shard_index < remainder else 0)
            return frame.iloc[start:end].reset_index(drop=True)

        def load_stage():
            frame = load_metadata(data_path, columns=DEFAULT_COLUMNS)
            if args.limit:
                frame = frame.head(args.limit)
            return apply_shard(frame)

        df = run_stage(f"Cargando metadatos desde {data_path}", load_stage)

        tidy_df = run_stage("Limpiando y preparando corpus", lambda: prepare_corpus(df))

    texts = tidy_df["text"].tolist()

    embedder = EmbeddingService(device=args.device)
    embeddings_path = artifacts_dir / "embeddings.npy"
    embeddings = None
    if args.reuse_embeddings and embeddings_path.exists():
        try:
            cached = np.load(embeddings_path)
            if len(cached) == len(texts):
                embeddings = cached
        except Exception:
            embeddings = None
    if embeddings is None:
        total = len(texts)
        chunk_size = max(1, args.embed_chunk_size)
        dimension = embedder.embedding_dimension()
        embeddings = npformat.open_memmap(
            embeddings_path,
            mode="w+",
            dtype="float32",
            shape=(total, dimension),
        )
        embed_progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="yellow", complete_style="green"),
            TaskProgressColumn(show_speed=True),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )
        with embed_progress:
            task = embed_progress.add_task(
                f"Calculando embeddings (batch={args.batch_size}, chunk={chunk_size})",
                total=total,
            )
            chunk_total = math.ceil(total / chunk_size) if total else 1
            batch_counter = 0
            total_batch_time = 0.0
            for chunk_idx, start in enumerate(range(0, total, chunk_size), start=1):
                end = min(start + chunk_size, total)
                batch_total = math.ceil((end - start) / args.batch_size) if end > start else 1
                for batch_idx, batch_start in enumerate(
                    range(start, end, args.batch_size), start=1
                ):
                    batch_end = min(batch_start + args.batch_size, end)
                    if args.progress_every > 0 and (
                        batch_idx == 1 or batch_idx % args.progress_every == 0
                    ):
                        embed_progress.update(
                            task,
                            description=(
                                "Calculando embeddings "
                                f"(chunk {chunk_idx}/{chunk_total}, "
                                f"batch {batch_idx}/{batch_total})"
                            ),
                        )
                    batch_texts = texts[batch_start:batch_end]
                    start_time = time.monotonic()
                    batch_embeddings = embedder.encode_texts(
                        batch_texts,
                        batch_size=args.batch_size,
                        show_progress_bar=False,
                    )
                    batch_time = time.monotonic() - start_time
                    batch_counter += 1
                    total_batch_time += batch_time
                    embeddings[batch_start:batch_end] = batch_embeddings
                    embed_progress.advance(task, batch_end - batch_start)
                embeddings.flush()

    metadata_cols = [
        col for col in ("id", "title", "abstract", "categories", "text") if col in tidy_df.columns
    ]
    metadata_path = artifacts_dir / "metadata.parquet"
    run_stage(
        "Guardando metadata en Parquet",
        lambda: tidy_df[metadata_cols].to_parquet(metadata_path, index=False),
    )

    def load_full_metadata():
        full_df = load_metadata(data_path, columns=FULL_COLUMNS)
        if args.limit:
            full_df = full_df.head(args.limit)
        return apply_shard(full_df)

    full_metadata = run_stage("Cargando metadata completa", load_full_metadata)
    full_metadata_path = artifacts_dir / "metadata_full.parquet"
    run_stage(
        "Guardando metadata completa en Parquet",
        lambda: full_metadata.to_parquet(full_metadata_path, index=False),
    )

    if not embeddings_path.exists():
        run_stage("Guardando embeddings (NumPy)", lambda: np.save(embeddings_path, embeddings))

    index = run_stage("Construyendo índice FAISS", lambda: VectorIndex.from_embeddings(embeddings))
    index_path = artifacts_dir / "index.faiss"
    run_stage("Persistiendo índice FAISS", lambda: index.save(index_path))

    console.print(f"[cyan]Metadata guardada en[/cyan] {metadata_path}")
    console.print(f"[cyan]Embeddings guardados en[/cyan] {embeddings_path}")
    console.print(f"[cyan]Metadata completa guardada en[/cyan] {full_metadata_path}")
    console.print(
        f"[bold green]Índice FAISS con {index.size} items guardado en {index_path}[/bold green]"
    )
    (artifacts_dir / ".complete").write_text("ok")


if __name__ == "__main__":
    main()
