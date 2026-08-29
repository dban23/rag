# Local RAG application

A step-by-step local Retrieval-Augmented Generation (RAG) pipeline that lets you ask
questions against your own documents. Fully offline, private (no cloud services,
no API keys) and free.

Pipeline: **load → chunk → embed/store → retrieve → generate → web UI**

| Piece | Technology |
|---|---|
| Embeddings | Ollama · `nomic-embed-text` (768-dim) |
| Embedding cache | Redis (in-memory, 24 h TTL) |
| Vector store | ChromaDB (embedded, on disk) |
| Generation | Ollama · `llama3.2:3b` |
| Web UI | Streamlit |

Everything runs in Docker via `docker compose`: an `ollama` service (with the
models in a named volume), a `redis` service (embedding cache), and an `app`
service (Streamlit + ChromaDB).

## Prerequisites

- Linux or WSL2 (this project was built on WSL2)
- Docker Engine + the compose plugin — install per the
  [official Docker docs](https://docs.docker.com/engine/install/ubuntu/)
- Internet access **once**, to pull the images and the models

## First-time setup (one-time only)

```bash
# 1. Build the images and start the stack (ollama, redis, app)
make setup

# 2. Pull the models into the ollama volume (~2.2 GB, happens once)
make pull-llm

# 3. Build the vector index from data/ into the app volume
make index

# 4. Confirm everything is healthy
make health
```

Then open http://localhost:8501 in your browser.

`make index` loads every file in `data/` (`.txt`, `.md`, `.pdf`), splits it into
chunks, embeds them, and stores the vectors in `chroma_db/`. It deletes and
rebuilds the whole collection, so it is safe to re-run any time.

## Daily use

```bash
make up      # start the stack
make health  # show container status
make down    # stop it (volumes and models are kept)
make logs    # follow the app logs
make help    # list all available commands
```

`docker compose up` starts the same containers every time — the models stay in
the `ollama_data` volume, so there is no re-download.

## Adding new files

Upload a file directly through the web UI: **sidebar →
Add a document → Index it**. It saves the file into `data/` and rebuilds the
index for you — the file is queryable immediately and persists across restarts.

Alternatively:
1. Copy the files into the `data/` volume, e.g.
   `docker compose cp my-doc.pdf app:/app/data/` (supported: `.txt`, `.md`, `.pdf`).
2. Re-run `make index`.
3. Ask again in the web app — **no Streamlit restart needed**; the app reads the
   database fresh on every question.


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
| Chunk | `src/chunking.py` | Splits text into sentence-aware chunks (defaults ~500 chars, 50 overlap — see `src/config.py`) |
| Embed + store | `src/indexing.py` | Embeds each chunk with `nomic-embed-text` and stores vectors + text in ChromaDB |
| Retrieve | `src/retrieval.py` | Embeds the question, finds the k most similar chunks, returns them with distances |
| Generate | `src/generate.py` | Sends the retrieved passages + question to `llama3.2:3b`, returns a grounded answer with citations |
| Web UI | `src/app.py` | Streamlit page wrapping retrieve + generate, showing answer and retrieved passages |

Indexing is the only step that writes to the database; a question only embeds the
question itself and searches existing vectors — no re-indexing per question.

## Task prefixes (asymmetric search)

`nomic-embed-text` is trained with a task prefix prepended to every input, so it
has to see the same prefix at index time and query time to behave correctly.
Documents are embedded one way, questions another:

| Text | Prefix | Used when |
|---|---|---|
| Document chunks | `search_document: ` | Indexing (`build_index`) |
| User question | `search_query: ` | Retrieval (`retrieve`) |

This is "asymmetric search": documents live in a different region of the vector
space than queries, which embeds them closer to the questions that would retrieve
them. Without the prefixes, the model receives text it was not trained on, and
retrieval quality drops significantly (in testing, the relevant chunk went from
rank 14 to rank 2 after adding them). Both prefixes are configurable in
`src/config.py` (`EMBED_DOC_PREFIX` / `EMBED_QUERY_PREFIX`).

## Redis caching

Embedding is the slowest part of the pipeline — every chunk needs an
`ollama.embed()` call. Redis caches these vectors in RAM so the same text is
never embedded twice.

### How it works

```
Embedding request
  → check Redis for cached vector
    → cache HIT:  return cached vector (sub-millisecond, no ollama call)
    → cache MISS: call ollama.embed(), store result in Redis, return it
```

### Why only embeddings?

| Component | Cacheable? | Why |
|---|---|---|
| Embeddings | Yes (Redis) | Same text = same vector, expensive to compute |
| LLM generation | No | Every query gets a unique response |
| Document loading | No | Already on disk, fast enough |
| ChromaDB queries | No | ChromaDB has its own internal caching |

### Cache details

- **Storage:** In-memory (RAM), sub-millisecond access
- **TTL:** 24 hours — vectors expire automatically, then re-embed on next access
- **Graceful degradation:** If Redis is down, the app works without cache (slower)
- **Key:** SHA-256 hash of the prefixed text (`search_document:` for docs,
  `search_query:` for queries)

## Configuration

All application-level settings live in one place: **`src/config.py`**. Change a
value there and it applies everywhere. Endpoints that are Docker-network 
details stay in `docker-compose.yml`.

| Setting | Where | Name | Default |
|---|---|---|---|
| Chunk size / overlap | `src/config.py` | `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 |
| Number of retrieved passages  | `src/config.py` | `K` | 5 |
| Embedding model | `src/config.py` | `EMBEDDING_MODEL` | `nomic-embed-text` |
| Document / query prefixes | `src/config.py` | `EMBED_DOC_PREFIX` / `EMBED_QUERY_PREFIX` | `search_document: ` / `search_query: ` |
| Generation model | `src/config.py` | `GENERATION_MODEL` | `llama3.2:3b` |
| Generation temperature | `src/config.py` | `GENERATION_TEMPERATURE` | 0.2 |
| Similarity metric | `src/config.py` | `SIMILARITY_METRIC` | cosine |
| Embedding cache TTL | `src/config.py` | `CACHE_TTL` | 86400 (24 hours) |
| Collection name | `src/config.py` | `COLLECTION_NAME` | `documents` |
| Ollama endpoint | `docker-compose.yml` | `OLLAMA_HOST` | `http://ollama:11434` |
| Redis endpoint | `docker-compose.yml` | `REDIS_URL` | `redis://redis:6379` |

Changing `EMBEDDING_MODEL`, `SIMILARITY_METRIC`, or `COLLECTION_NAME` requires
re-running `make index` to rebuild the vector collection.

Because `src/` is baked into the Docker image, any change to `src/config.py`
takes effect only after rebuilding:

```bash
make setup   # rebuild the app image and start the stack
make index   # if the model, metric, or collection changed
```

## Testing

Run the full test suite inside the Docker container:

```bash
make test
```

Tests run against the built image (no local Python or Ollama needed). External
services (Ollama, ChromaDB) are mocked so tests are fast and deterministic.

| Test file | What it covers |
|---|---|
| `tests/test_chunking.py` | Sentence splitting, chunk packing, overlap logic, no mid-word cuts |
| `tests/test_loaders.py` | `.txt`/`.md`/`.pdf` loading, empty/mixed-page PDFs, unsupported formats |
| `tests/test_generate.py` | Citation resolution, Ollama call parameters, context building |
| `tests/test_retrieval.py` | Query validation, Hit object creation, ChromaDB mock integration |
| `tests/test_indexing.py` | Redis caching: cache hit, cache miss, graceful degradation, TTL, cache keys |

## Troubleshooting

**The app says "Something went wrong. Is Ollama running?"**
The `app` container waits for the `ollama` service to be healthy before starting.
Check the stack state:

```bash
make health
docker compose logs ollama
```

If the models were never pulled, run `make pull-llm` (with the stack up).

**The page won't load.**
Check the app container is up and healthy: `make health`. If it shows
`unhealthy`, look at the logs with `make logs`.

**A file is skipped with "Skipping unsupported file: ..."**
`loaders.py` only handles `.txt`, `.md`, and `.pdf`. Rename/convert the file.

**Redis warning in logs ("Redis unavailable — embedding cache disabled")**
Redis is optional — the app works without it. Embeddings will be slower because
every chunk is re-computed on each indexing run. To fix:
```bash
make health                 # check if redis is running
docker compose logs redis   # check for errors
docker compose restart redis
```

**An uploaded file produces 0 chunks (warning: "produced 0 chunks").**
The file has no extractable text — typically a scanned PDF or one using
outline/vector fonts. See the server console (`make logs`) for per-file details.

**I want to start from scratch (wipe everything).**
```bash
make cleanup   # docker compose down -v — deletes models, index, and uploads too
```

**Note for WSL2:** on a slow `/mnt/c` mount the first `docker compose build` can
take a while; subsequent starts are fast.
