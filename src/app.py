import streamlit as st

from generate import generate
from retrieval import retrieve

st.set_page_config(page_title="Local RAG Demo", layout="centered")
st.title("Local RAG")
st.caption(
    "A local RAG pipeline: nomic-embed-text embeddings, ChromaDB retrieval, "
    "llama3.2:3b generation."
)

with st.form("ask"):
    question = st.text_input(
        "Your question", placeholder="e.g. How often should I descale?"
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    try:
        with st.spinner("Searching and generating..."):
            hits = retrieve(question)
            answer = generate(question, hits)
    except Exception as e:
        st.error(
            f"Something went wrong. Is Ollama running? "
            f"(hint: `setsid nohup ~/.local/bin/ollama serve ...`)\n\n{e}"
        )
    else:
        st.markdown(answer)
        with st.expander(f"Retrieved passages ({len(hits)})"):
            for i, hit in enumerate(hits, start=1):
                st.markdown(
                    f"**{i}. {hit['source']}** — chunk {hit['chunk_index']} "
                    f"· distance {hit['distance']:.3f}"
                )
                st.write(hit["text"])
