.PHONY: install download embed embed-fast serve test test-all lint fmt help
SHELL := /bin/bash
SHARD_COUNT ?= 25
EMBED_BATCH_SIZE ?= 32
EMBED_DEVICE ?= mps
EMBED_CHUNK_SIZE ?= 20000
PROGRESS_EVERY ?= 1

install:
	uv sync --dev

download:
	@set -a; \
	if [ -f .env ]; then source .env; fi; \
	set +a; \
	uv run python -m arxiv_rec.cli.download_snapshot

embed:
	@for shard in $$(seq 0 $$(( $(SHARD_COUNT) - 1 ))); do \
		shard_label=shard_$$(printf "%03d" $$shard)_of_$$(printf "%03d" $(SHARD_COUNT)); \
		if [ -f artifacts/$$shard_label/.complete ]; then \
			echo "==> Skipping shard $$shard/$(SHARD_COUNT) (already complete)"; \
			continue; \
		fi; \
		echo "==> Embedding shard $$shard/$(SHARD_COUNT)"; \
		uv run python -m arxiv_rec.cli.build_index \
			--batch-size $(EMBED_BATCH_SIZE) \
			--device $(EMBED_DEVICE) \
			--embed-chunk-size $(EMBED_CHUNK_SIZE) \
			--progress-every $(PROGRESS_EVERY) \
			--shard-index $$shard \
			--shard-count $(SHARD_COUNT); \
	done

embed-fast:
	uv run python -m arxiv_rec.cli.build_index \
		--limit 1000 \
		--device $(EMBED_DEVICE) \
		--embed-chunk-size $(EMBED_CHUNK_SIZE) \
		--progress-every $(PROGRESS_EVERY)

serve:
	uv run uvicorn arxiv_rec.api.server:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest

test-all:
	uv run pytest
	uv run ruff check src scripts tests
	uv run ruff format --check src scripts tests

lint:
	uv run ruff check src scripts tests

fmt:
	uv run ruff check --fix src scripts tests
	uv run ruff format src scripts tests

help:
	@printf "Available targets:\n"
	@printf "\nGeneral:\n"
	@printf "  install    Install dependencies with uv.\n"
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
