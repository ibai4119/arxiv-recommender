.PHONY: install download embed serve test fmt
SHELL := /bin/bash

install:
	poetry install

download:
	@set -a; \
	if [ -f .env ]; then source .env; fi; \
	set +a; \
	poetry run python scripts/download_snapshot.py

embed:
	poetry run python scripts/build_index.py

serve:
	poetry run uvicorn arxiv_rec.api.server:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest

fmt:
	poetry run black src scripts tests
	poetry run isort src scripts tests
