"""Central configuration for the RAG pipeline.

All tunable values live here so they can be changed in one place
instead of being scattered across the source modules. After changing
any value below, rebuild/re-index as noted next to each section.
"""

from loaders import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Paths & vector store
# ChromaDB persists the vector index here (a Docker named volume in compose).
# Recreating the collection happens automatically on every `make index`.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# The name under which the vector collection is stored inside ChromaDB.
# Renaming it orphans the old data; `make index` will then build a fresh one.
COLLECTION_NAME = "documents"

# Distance metric used for the HNSW index. Options: "cosine", "l2",
# "ip" (inner product). NOTE: changing this after data exists has no
# effect on old vectors — re-run `make index` to rebuild the collection.
SIMILARITY_METRIC = "cosine"

# ---------------------------------------------------------------------------
# Chunking (src/chunking.py)
# Target size for each text chunk in characters. Smaller chunks give more
# precise retrieval but less context per passage; larger chunks do the
# opposite. Chunks never cut mid-word.
CHUNK_SIZE = 500

# Characters of overlap between consecutive chunks so a sentence split across
# a boundary is still fully present in the next chunk. 0 disables overlap.
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Embeddings (src/indexing.py)
# Ollama model used to turn chunks/questions into 768-dim vectors. Must be
# pulled with `make pull-llm`. If you change it, re-run `make index`.
EMBEDDING_MODEL = "nomic-embed-text"

# Prefix prepended to document chunks before embedding.
EMBED_DOC_PREFIX = "search_document: "

# Prefix prepended to the user's question before embedding.
EMBED_QUERY_PREFIX = "search_query: "

CACHE_TTL = 86400  # seconds = 24 hours

# ---------------------------------------------------------------------------
# Retrieval (src/retrieval.py)
# How many most-similar chunks are retrieved and passed to the LLM as context.
K = 5

# ---------------------------------------------------------------------------
# Generation (src/generate.py)
# Ollama chat model that produces the final answer from the retrieved context.
# Must be pulled with `make pull-llm`.
GENERATION_MODEL = "llama3.2:3b"

GENERATION_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Infra-level settings — NOT here, they live in docker-compose.yml:
#   OLLAMA_HOST = http://ollama:11434   (Docker network name of the ollama service)
#   REDIS_URL   = redis://redis:6379     (Docker network name of the redis service)
# ---------------------------------------------------------------------------

