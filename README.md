# Local RAG Tool

A step-by-step local Retrieval-Augmented Generation (RAG) pipeline that lets you ask
questions against your own documents, fully offline and private — no cloud services,
no API keys.

Pipeline: **load → chunk → embed/store → retrieve → generate → web UI**

| Piece | Technology |
|---|---|
| Embeddings | Ollama · `nomic-embed-text` (768-dim) |
| Vector store | ChromaDB (embedded, on disk) |
| Generation | Ollama · `llama3.2:3b` |
| Web UI | Streamlit |

Everything runs in Docker via `docker compose`: an `ollama` service (with the
models in a named volume) and an `app` service (Streamlit + ChromaDB).

## Prerequisites

- Linux or WSL2 (this project was built on WSL2)
- Docker Engine + the compose plugin — install per the
  [official Docker docs](https://docs.docker.com/engine/install/ubuntu/)
- Internet access **once**, to pull the images and the models

## First-time setup (one-time only)

```bash
# 1. Build the images and start both services
make setup

# 2. Pull the models into the ollama volume (~2.2 GB, happens once)
make pull-llm

# 3. Build the vector index from data/ into the app volume
make index

# 4. Confirm everything is healthy
docker compose ps
```

Then open http://localhost:8501 in your browser.

`make index` loads every file in `data/` (`.txt`, `.md`, `.pdf`), splits it into
chunks, embeds them, and stores the vectors in `chroma_db/`. It deletes and
rebuilds the whole collection, so it is safe to re-run any time.

## Daily use

```bash
make up      # start the stack
make down    # stop it (volumes and models are kept)
make logs    # follow the app logs
```

`docker compose up` starts the same containers every time — the models stay in
the `ollama_data` volume, so there is no re-download.

## Adding new files

1. Copy the files into the `data/` volume, e.g.
   `docker compose cp my-doc.pdf app:/app/data/` (supported: `.txt`, `.md`, `.pdf`).
2. Re-run `make index`.
3. Ask again in the web app — **no Streamlit restart needed**; the app reads the
   database fresh on every question.

Alternatively, you can upload a file directly through the web UI: **sidebar →
Add a document → Index it**. It saves the file into `data/` and rebuilds the
index for you — the file is queryable immediately and persists across restarts.

### Removing files

Files can also be removed through the web UI: **sidebar → Delete a file** → pick
the file → tick "I understand this removes the file from data/ permanently." →
**Delete file**. The file is deleted from the app volume and the index is rebuilt
automatically, so it stops being queryable immediately.

This only affects the volume, not the repository's `data/` folder — seed files
return after a full wipe (`make cleanup`).

## Try it from the CLI (optional)

```bash
docker compose run --rm app python src/generate.py
```

Type a question, e.g. *"How often should I descale?"*, and you get the grounded
answer with source filenames cited, e.g. `according to product-manual.pdf`.

## Pipeline overview

| Phase | File | What it does |
|---|---|---|
| Load | `src/loaders.py` | Reads `.txt`/`.md`/`.pdf` files from `data/` into `{"filename", "text"}` |
| Chunk | `src/chunking.py` | Splits text into sentence-aware ~500-char chunks with 50-char overlap |
| Embed + store | `src/indexing.py` | Embeds each chunk with `nomic-embed-text` and stores vectors + text in ChromaDB |
| Retrieve | `src/retrieval.py` | Embeds the question, finds the k most similar chunks, returns them with distances |
| Generate | `src/generate.py` | Sends the retrieved passages + question to `llama3.2:3b`, returns a grounded answer with citations |
| Web UI | `src/app.py` | Streamlit page wrapping retrieve + generate, showing answer and retrieved passages |

Indexing is the only step that writes to the database; a question only embeds the
question itself and searches existing vectors — no re-indexing per question.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Chunk size / overlap | `src/chunking.py` | 500 / 50 |
| Number of retrieved passages (`k`) | `src/retrieval.py` | 5 |
| Embedding model | `src/indexing.py` | `nomic-embed-text` |
| Generation model | `src/generate.py` | `llama3.2:3b` |
| Generation temperature | `src/generate.py` | 0.2 |
| Similarity metric | `src/indexing.py` | cosine |
| Ollama endpoint | `docker-compose.yml` (`OLLAMA_HOST`) | `http://ollama:11434` |

## Troubleshooting

**The app says "Something went wrong. Is Ollama running?"**
The `app` container waits for the `ollama` service to be healthy before starting.
Check the stack state:

```bash
docker compose ps
docker compose logs ollama
```

If the models were never pulled, run `make pull-llm` (with the stack up).

**The page won't load.**
Check the app container is up and healthy: `docker compose ps`. If it shows
`unhealthy`, look at the logs with `make logs`.

**A file is skipped with "Skipping unsupported file: ..."**
`loaders.py` only handles `.txt`, `.md`, and `.pdf`. Rename/convert the file.

**An uploaded file produces 0 chunks (warning: "produced 0 chunks").**
The file has no extractable text — typically a scanned PDF or one using
outline/vector fonts. See the server console (`make logs`) for per-file details.

**I want to start from scratch (wipe everything).**
```bash
make cleanup   # docker compose down -v — deletes models, index, and uploads too
```

**Note for WSL2:** on a slow `/mnt/c` mount the first `docker compose build` can
take a while; subsequent starts are fast.
