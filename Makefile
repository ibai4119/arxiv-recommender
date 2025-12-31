.PHONY: install download embed embed-fast serve test test-all lint fmt help
SHELL := /bin/bash

install:
	poetry install

download:
	@set -a; \
	if [ -f .env ]; then source .env; fi; \
	set +a; \
	poetry run python -m arxiv_rec.cli.download_snapshot

embed:
	poetry run python -m arxiv_rec.cli.build_index --batch-size 256 --device mps

embed-fast:
	poetry run python -m arxiv_rec.cli.build_index --limit 1000 --device mps

serve:
	poetry run uvicorn arxiv_rec.api.server:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest

test-all:
	poetry run pytest
	poetry run ruff check src scripts tests
	poetry run ruff format --check src scripts tests

lint:
	poetry run ruff check src scripts tests

fmt:
	poetry run ruff check --fix src scripts tests
	poetry run ruff format src scripts tests

help:
	@printf "Available targets:\n"
	@printf "\nGeneral:\n"
	@printf "  install    Install dependencies with Poetry.\n"
	@printf "  download   Download snapshot and build Parquet artifacts.\n"
	@printf "  embed      Build embeddings + FAISS index.\n"
	@printf "  embed-fast Quick embed run with limit=1000 and device=mps.\n"
	@printf "  serve      Run the FastAPI server locally.\n"
	@printf "\nQuality:\n"
	@printf "  test       Run test suite.\n"
	@printf "  test-all   Run tests + lint + format checks.\n"
	@printf "  lint       Run ruff lint checks.\n"
	@printf "  fmt        Fix lint and format with ruff.\n"
	@printf "\nOther:\n"
	@printf "  help       Show this help.\n"
