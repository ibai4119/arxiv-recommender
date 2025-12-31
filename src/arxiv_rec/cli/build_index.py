"""Build embeddings + FAISS index from the arXiv metadata snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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

        def load_stage():
            frame = load_metadata(data_path, columns=DEFAULT_COLUMNS)
            return frame.head(args.limit) if args.limit else frame

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
        embeddings = run_stage(
            f"Calculando embeddings (batch={args.batch_size})",
            lambda: embedder.encode_texts(texts, batch_size=args.batch_size),
        )

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
        return full_df.head(args.limit) if args.limit else full_df

    full_metadata = run_stage("Cargando metadata completa", load_full_metadata)
    full_metadata_path = artifacts_dir / "metadata_full.parquet"
    run_stage(
        "Guardando metadata completa en Parquet",
        lambda: full_metadata.to_parquet(full_metadata_path, index=False),
    )

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


if __name__ == "__main__":
    main()
