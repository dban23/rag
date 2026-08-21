import re

import ollama
from retrieval import Hit, retrieve

SYSTEM_PROMPT = "You are a helpful assistant. Answer using ONLY the provided context. If the answer is not in the context, say you don't know. When you use a passage, cite it like [1]. Cite only the passages you actually used."


def resolve_citations(answer: str, hits: list[Hit]) -> str:
    def replace(m):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(hits):
            return hits[idx].source
        return m.group(0)

    return re.sub(r"\[(\d+)\]", replace, answer)


def generate(question: str, hits: list[Hit]) -> str:
    passages = []
    for i, hit in enumerate(hits, start=1):
        passages.append(
            f"[{i}] ({hit.source}, chunk {hit.chunk_index})\n{hit.text}"
        )
    context = "\n\n".join(passages)

    user_text = f"Context:\n{context}\n\nQuestion: {question}"

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        options={"temperature": 0.2},
    )

    return resolve_citations(response["message"]["content"], hits)


if __name__ == "__main__":
    question = input("Ask a question: ")
    hits = retrieve(question)
    answer = generate(question, hits)
    print(answer)
