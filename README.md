# arxiv-recommender

Recomendador semántico de artículos de arXiv basado en embeddings de Sentence Transformers y búsqueda vectorial con FAISS.

## Índice

1. [Descripción general](#1-descripcion-general)
2. [Requisitos](#2-requisitos)
3. [Instalación rápida](#3-instalacion-rapida)
4. [Descargar el snapshot](#4-descargar-el-snapshot)
5. [Generar embeddings e índice](#5-generar-embeddings-e-indice)
6. [API y búsqueda](#6-api-y-busqueda)
7. [Pruebas y formato](#7-pruebas-y-formato)
8. [Docker](#8-docker)
9. [Estructura del proyecto](#9-estructura-del-proyecto)
10. [Próximos pasos](#10-proximos-pasos)

## 1. Descripción general

1. El proyecto ingiere el snapshot oficial de metadatos de arXiv.
2. Compone título y abstract para cada paper y genera embeddings con `sentence-transformers/all-MiniLM-L6-v2`.
3. Construye un índice FAISS + archivos auxiliares (`metadata.parquet`, `embeddings.npy`, `index.faiss`) dentro de `artifacts/`.
4. Expone un API FastAPI que realiza búsqueda semántica y recomendaciones por item.

## 2. Requisitos

1. Python 3.11.
2. [Poetry](https://python-poetry.org/) 1.5 o superior.
3. Dataset `data/arxiv-metadata-oai-snapshot.json` (se obtiene con `make download`).

## 3. Instalación rápida

1. Instala dependencias con Poetry:

   ```bash
   poetry install
   ```

2. (Opcional) Habilita los hooks de pre-commit:

   ```bash
   poetry run pre-commit install
   ```

3. Recuerda que `poetry.toml` fuerza a crear el entorno virtual dentro del repo (`.venv/`), ideal para Docker/CI.

## 4. Descargar el snapshot

`scripts/download_snapshot.py` baja `arxiv-metadata-oai-snapshot.json` desde Kaggle y deja el JSON listo (sin comprimir) en `data/`.

1. Crea un archivo `.env` con tus credenciales de Kaggle (o exporta las variables manualmente):

   ```bash
   cat <<'EOF' > .env
   KAGGLE_USERNAME=tu_usuario
   KAGGLE_KEY=tu_token
   EOF
   ```

2. Ejecuta la descarga:

   ```bash
   make download
   ```

3. El objetivo lee automáticamente `.env`, detecta si Kaggle entrega ZIP o JSON directo y guarda el resultado en `data/`.

### Opciones útiles

```bash
poetry run python scripts/download_snapshot.py --convert-csv    # genera CSV adicional
poetry run python scripts/download_snapshot.py --remove-json    # borra el JSON tras convertir
poetry run python scripts/download_snapshot.py --limit 1000     # limita filas durante la conversión
```

## 5. Generar embeddings e índice

1. Ejecuta:

   ```bash
   make embed
   ```

2. El comando corre `scripts/build_index.py`, que:
   1. Lee el JSON lineal original (puedes cambiar la ruta con `--data-path`).
   2. Limpia los textos y concatena título + abstract.
   3. Calcula embeddings con el modelo por defecto (configurable).
   4. Persiste `artifacts/metadata.parquet`, `artifacts/embeddings.npy` y `artifacts/index.faiss`.
3. Personaliza el proceso, por ejemplo:

   ```bash
   poetry run python scripts/build_index.py --limit 10000 --batch-size 32
   ```

## 6. API y búsqueda

1. Levanta el servidor:

   ```bash
   make serve
   ```

2. Endpoints disponibles:
   - `GET /search?q=texto&k=5`
   - `GET /recommend?item_id=arXivID&k=5`
3. El servidor usa `uvicorn` en modo recarga (`--reload`) por defecto para acelerar iteraciones locales.

## 7. Pruebas y formato

1. Ejecuta los tests:

   ```bash
   make test
   ```

2. Aplica formato (Black + isort):

   ```bash
   make fmt
   ```

## 8. Docker

1. Construye la imagen:

   ```bash
   docker build -t arxiv-recommender .
   ```

2. Levanta el contenedor mapeando artefactos y puerto:

   ```bash
   docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts arxiv-recommender
   ```

## 9. Estructura del proyecto

```text
├── scripts/build_index.py
├── src/arxiv_rec
│   ├── data/{ingest,clean}.py
│   ├── models/{embed,index}.py
│   └── api/server.py
├── artifacts/
├── tests/
└── Dockerfile
```

## 10. Próximos pasos

1. Añadir lógica de actualización incremental del índice.
2. Automatizar pipelines de ingesta y refresco (Airflow, Prefect, etc.).
