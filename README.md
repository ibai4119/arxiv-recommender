# arxiv-recommender

Recomendador semántico de artículos de arXiv basado en embeddings de Sentence Transformers y búsqueda vectorial con FAISS.

## Requisitos

- Python 3.11
- [Poetry](https://python-poetry.org/) 1.5+
- Dataset `arxiv-metadata-oai-snapshot.json` (JSON lines) en `data/` (usa `make download` para bajarlo).

## Descargar el snapshot automáticamente

`scripts/download_snapshot.py` descarga `arxiv-metadata-oai-snapshot.json` desde Kaggle y deja el JSON listo en `data/`. La conversión a CSV es opcional.

1. Configura tus credenciales de Kaggle en un archivo `.env` (o exporta las variables manualmente):

   ```bash
   cat <<'EOF' > .env
   KAGGLE_USERNAME=tu_usuario
   KAGGLE_KEY=tu_token
   EOF
   ```

2. Ejecuta:

   ```bash
   make download
   ```
   > El objetivo lee automáticamente las variables de `.env` antes de llamar al script.

Opciones útiles:

```bash
poetry run python scripts/download_snapshot.py --convert-csv    # además genera un CSV
poetry run python scripts/download_snapshot.py --remove-json    # borra el JSON tras descargar
poetry run python scripts/download_snapshot.py --limit 1000     # limita filas al convertir a CSV
```

## Configuración rápida

```bash
poetry install
poetry run pre-commit install  # opcional, para los hooks
```

> Nota: el archivo `poetry.toml` fuerza a Poetry a crear el entorno virtual dentro del proyecto (`.venv/`), facilitando su uso en Docker/CI.

## Generar embeddings e índice

```bash
make embed  # usa data/arxiv-metadata-oai-snapshot.json por defecto
```

El comando ejecuta `scripts/build_index.py`, que:

1. Lee el JSON lineal original (puedes cambiar la ruta con `--data-path`).
2. Limpia y combina título + abstract.
3. Calcula embeddings con `sentence-transformers/all-MiniLM-L6-v2`.
4. Guarda `artifacts/metadata.parquet`, `artifacts/embeddings.npy` y `artifacts/index.faiss`.

### Parámetros opcionales

```bash
poetry run python scripts/build_index.py --limit 10000 --batch-size 32
```

## Ejecutar el API

```bash
make serve
```

Endpoints principales:

- `GET /search?q=texto&k=5`
- `GET /recommend?item_id=arXivID&k=5`

## Pruebas, formato y lint

```bash
make test
make fmt
```

## Docker

```bash
docker build -t arxiv-recommender .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts arxiv-recommender
```

## Estructura

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

## Próximos pasos

- Añadir lógica de actualización incremental del índice.
- Automatizar pipelines (Airflow, Prefect, etc.).
