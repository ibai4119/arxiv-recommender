# arxiv-recommender

Recomendador semántico de artículos de arXiv basado en embeddings de Sentence Transformers y búsqueda vectorial con FAISS.

## Índice

1. [Descripción general](#1-descripción-general)
2. [Requisitos](#2-requisitos)
3. [Instalación rápida](#3-instalación-rápida)
4. [Descargar el snapshot](#4-descargar-el-snapshot)
5. [Generar embeddings e índice](#5-generar-embeddings-e-índice)
6. [API y búsqueda](#6-api-y-búsqueda)
7. [Pruebas y formato](#7-pruebas-y-formato)
8. [Docker](#8-docker)
9. [Estructura del proyecto](#9-estructura-del-proyecto)
10. [Próximos pasos](#10-próximos-pasos)

## 1. Descripción general

1. El proyecto ingiere el snapshot oficial de metadatos de arXiv.
2. Compone título y abstract para cada paper y genera embeddings con `sentence-transformers/all-MiniLM-L6-v2`.
3. Construye un índice FAISS + archivos auxiliares (`metadata.parquet`, `metadata_full.parquet`, `embeddings.npy`, `index.faiss`) dentro de `artifacts/`.
4. Expone un API FastAPI que realiza búsqueda semántica y recomendaciones por item.

## 2. Requisitos

1. Python 3.14.
2. [`uv`](https://docs.astral.sh/uv/) instalado.
3. Dataset `data/arxiv-metadata-oai-snapshot.parquet` (se obtiene con `make download`).

## 3. Instalación rápida

1. Instala dependencias con `uv`:

   ```bash
   uv sync --dev
   ```

2. (Opcional) Habilita los hooks de pre-commit:

   ```bash
   uv run pre-commit install
   ```

3. `uv` gestionará el entorno y el lockfile del proyecto (`uv.lock`) desde la raíz del repo.

## 4. Descargar el snapshot

`arxiv_rec.cli.download_snapshot` baja `arxiv-metadata-oai-snapshot.json` desde Kaggle y genera `arxiv-metadata-oai-snapshot.parquet` en `data/`.

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
uv run python -m arxiv_rec.cli.download_snapshot --keep-json    # conserva el JSON tras convertir
uv run python -m arxiv_rec.cli.download_snapshot --force        # fuerza redescarga y reconversión
```

## 5. Generar embeddings e índice

1. Ejecuta:

   ```bash
   make embed
   ```

2. El comando corre `arxiv_rec.cli.build_index`, que:
   1. Lee el Parquet generado (puedes cambiar la ruta con `--data-path`).
   2. Limpia los textos y concatena título + abstract.
   3. Calcula embeddings con el modelo por defecto (configurable).
   4. Persiste `artifacts/metadata.parquet`, `artifacts/metadata_full.parquet`, `artifacts/embeddings.npy` y `artifacts/index.faiss`.
3. Personaliza el proceso, por ejemplo:

   ```bash
   uv run python -m arxiv_rec.cli.build_index --limit 10000 --batch-size 32
   ```

   Para shards:

   ```bash
   uv run python -m arxiv_rec.cli.build_index --shard-index 0 --shard-count 4
   ```

   Los shards guardan artefactos en `artifacts/shard_XXX_of_YYY/` y crean un marcador `.complete` al terminar.

### Notas de rendimiento

- El mayor coste suele ser la generación de embeddings, no la lectura.
- Si quieres guardar embeddings incrementalmente, ajusta `--embed-chunk-size` (por defecto 10000).
- Puedes ajustar la frecuencia de updates con `--progress-every`.
- `metadata_full.parquet` sirve para mostrar detalles completos tras seleccionar un paper.

Variables de `make` (con defaults, pero sobreescribibles):

- `SHARD_COUNT`: numero total de shards; 1 = sin shard.
- `EMBED_BATCH_SIZE`: tamaño de batch para embeddings.
- `EMBED_DEVICE`: dispositivo (`mps`, `cpu`, `cuda`).
- `EMBED_CHUNK_SIZE`: tamaño de chunk para escritura incremental.
- `PROGRESS_EVERY`: cada cuantos batches se actualiza la descripcion.
- Para consultas rápidas de metadatos sin cargar Parquet, puedes usar DuckDB como capa de lectura.

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

2. Ejecuta lint y formato con Ruff:

   ```bash
   make lint
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
├── src/arxiv_rec/cli/{build_index,download_snapshot}.py
├── src/arxiv_rec
│   ├── data/{ingest,clean}.py
│   ├── models/{embed,index}.py
│   ├── services/recommender.py
│   └── api/server.py
├── artifacts/
├── tests/
└── Dockerfile
```

## 10. Próximos pasos

1. Añadir lógica de actualización incremental del índice.
2. Automatizar pipelines de ingesta y refresco (Airflow, Prefect, etc.).
