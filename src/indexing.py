import hashlib
import os
from pathlib import Path

import chromadb
import numpy as np
import ollama
import redis
from chromadb.api.types import EmbeddingFunction

from chunking import chunk_text
from loaders import DATA_DIR, PROJECT_ROOT, load_documents

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "product_manual"

CACHE_TTL = 86400  # 24 hours


class OllamaEmbedder(EmbeddingFunction):
    def __init__(
        self,
        model: str = "nomic-embed-text",
        prefix: str = "search_document: ",
        redis_url: str | None = None,
    ) -> None:
        self.model = model
        self.prefix = prefix
        self._redis = None
        url = redis_url or os.environ.get("REDIS_URL")
        if url:
            try:
                client = redis.Redis.from_url(url, decode_responses=False)
                client.ping()
                self._redis = client
            except redis.RedisError:
                print("[WARNING] Redis unavailable — embedding cache disabled")

    def _cache_key(self, text: str) -> bytes:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embed:{h}".encode()

    def _cache_get(self, key: bytes) -> list[float] | None:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
            if raw is None:
                return None
            return np.frombuffer(raw, dtype=np.float32).tolist()
        except redis.RedisError:
            return None

    def _cache_set(self, key: bytes, vector: list[float]) -> None:
        if self._redis is None:
            return
        try:
            self._redis.set(key, np.array(vector, dtype=np.float32).tobytes(), ex=CACHE_TTL)
        except redis.RedisError:
            pass

    def __call__(self, input: list[str]) -> list[list[float]]:
        prefixed = [self.prefix + t for t in list(input)]

        cached: dict[int, list[float]] = {}
        miss_indices: list[int] = []
        for i, text in enumerate(prefixed):
            key = self._cache_key(text)
            vec = self._cache_get(key)
            if vec is not None:
                cached[i] = vec
            else:
                miss_indices.append(i)

        if miss_indices:
            miss_texts = [prefixed[i] for i in miss_indices]
            result = ollama.embed(model=self.model, input=miss_texts)
            for idx, vec in zip(miss_indices, result.embeddings):
                self._cache_set(self._cache_key(prefixed[idx]), vec)
                cached[idx] = vec

        return [cached[i] for i in range(len(prefixed))]


def build_index(
    data_dir: Path = DATA_DIR,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(persist_dir))

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=OllamaEmbedder(),
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids = []
    empty_files = []
    for doc in load_documents(data_dir):
        chunks = chunk_text(doc["text"])
        if not chunks:
            empty_files.append(doc["filename"])
            print(
                f"[WARNING] {doc['filename']}: 0 chunks extracted "
                f"(empty or unreadable file). It will NOT be indexed."
            )
            continue
        print(f"  indexed {doc['filename']}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": doc["filename"], "chunk_index": i})
            ids.append(f"{doc['filename']}::{i}")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    if empty_files:
        print(
            f"[WARNING] skipped {len(empty_files)} file(s) with no text: "
            f"{', '.join(empty_files)}"
        )
    return collection


if __name__ == "__main__":
    collection = build_index()
    print(f"collection: {collection.name}")
    print(f"chunks stored: {collection.count()}")
    print()

    peek = collection.peek(limit=2)
    for doc, meta in zip(peek["documents"], peek["metadatas"]):
        print(f"--- {meta['source']} :: chunk {meta['chunk_index']} ---")
        print(doc[:80])
        print()

    raw = collection.get(include=["embeddings"])
    first = raw["embeddings"][0]
    print(f"vector dimension: {len(first)}")
    print(f"first 5 numbers:  {first[:5]}")
    print()

    results = collection.query(
        query_texts=["How many cups of coffee can the Aurora Brew 3000 brew?"],
        n_results=3,
    )
    top = results["metadatas"][0][0]
    print("top hit ->", top["source"], "chunk", top["chunk_index"])
    print("distance:", results["distances"][0][0])
    print(results["documents"][0][0])
