import chromadb
import ollama
from chromadb.api.types import EmbeddingFunction

from chunking import chunk_text
from loaders import DATA_DIR, PROJECT_ROOT, load_documents

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "product_manual"


class OllamaEmbedder(EmbeddingFunction):
    def __init__(self, model="nomic-embed-text"):
        self.model = model

    def __call__(self, input):
        result = ollama.embed(model=self.model, input=list(input))
        return result.embeddings


def build_index(
    data_dir=DATA_DIR, persist_dir=CHROMA_DIR, collection_name=COLLECTION_NAME
):
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
    for doc in load_documents(data_dir):
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": doc["filename"], "chunk_index": i})
            ids.append(f"{doc['filename']}::{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
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
