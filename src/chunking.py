from loaders import load_documents


def chunk_text(text, chunk_size=500, overlap=50):
    if not text.strip():
        return []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    return chunks


if __name__ == "__main__":
    chunk_size, overlap = 500, 50

    text = ""
    for doc in load_documents():
        if doc["filename"] == "product-manual.txt":
            text = doc["text"]
            break

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"{len(text)} characters -> {len(chunks)} chunks\n")

    step = chunk_size - overlap
    for i, chunk in enumerate(chunks):
        start = i * step
        end = min(start + chunk_size, len(text))
        print(f"--- chunk {i} | chars {start}:{end} | {len(chunk)} chars ---")
        print(chunk)
        print()
