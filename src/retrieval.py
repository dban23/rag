from dataclasses import dataclass

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_QUERY_PREFIX, K
from indexing import OllamaEmbedder


@dataclass
class Hit:
    """A single retrieval result from the vector store."""
    text: str
    source: str
    chunk_index: int
    distance: float


def retrieve(query: str, k: int = K) -> list[Hit]:
    if not query.strip():
        return []

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(
        COLLECTION_NAME,
        embedding_function=OllamaEmbedder(prefix=EMBED_QUERY_PREFIX),
    )

    results = collection.query(query_texts=[query], n_results=k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            Hit(
                text=doc,
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                distance=dist,
            )
        )

    return hits


if __name__ == "__main__":
    r = retrieve("How many cups can it brew?")
    print(r)
