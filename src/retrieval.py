import chromadb

from indexing import OllamaEmbedder, CHROMA_DIR, COLLECTION_NAME


def retrieve(query, k=5):
    if not query.strip():
        return []

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(
        COLLECTION_NAME, embedding_function=OllamaEmbedder(prefix="search_query: ")
    )

    results = collection.query(query_texts=[query], n_results=k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance": dist,
            }
        )

    return hits


if __name__ == "__main__":
    r = retrieve("How many cups can it brew?")
    print(r)
