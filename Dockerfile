FROM python:3.11-slim AS base
ENV POETRY_VERSION=1.8.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml README.md ./
RUN poetry install --no-root --only main

COPY . .
RUN poetry install --no-root

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "arxiv_rec.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
