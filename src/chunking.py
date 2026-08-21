import re

from loaders import load_documents


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    cur = ""
    for sent in sentences:
        if cur and len(cur) + len(sent) + 1 > chunk_size:
            chunks.append(cur)
            cur = sent
        else:
            cur = (cur + " " + sent).strip() if cur else sent
    if cur:
        chunks.append(cur)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-overlap:]
            overlapped.append((tail + chunks[i]).strip())
        chunks = overlapped

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
