from pathlib import Path

import streamlit as st
from chromadb.errors import NotFoundError

from generate import generate
from indexing import build_index
from loaders import DATA_DIR
from retrieval import retrieve

st.set_page_config(page_title="Local RAG Demo", layout="centered")
st.title("Local RAG")
st.caption(
    "A local RAG pipeline: nomic-embed-text embeddings, ChromaDB retrieval, "
    "llama3.2:3b generation."
)

with st.sidebar:
    st.header("Documents")
    uploaded_file = st.file_uploader(
        "Add a document", type=["txt", "md", "pdf"]
    )
    if st.button("Index it") and uploaded_file is not None:
        try:
            with st.spinner("Indexing..."):
                path = DATA_DIR / Path(uploaded_file.name).name
                path.write_bytes(uploaded_file.getvalue())
                collection = build_index()
            indexed = collection.get(include=["metadatas"])
            sources = {m["source"] for m in indexed["metadatas"]}
            if uploaded_file.name in sources:
                st.success(f"Indexed {collection.count()} chunks from {uploaded_file.name}.")
            else:
                st.warning(
                    f"Uploaded {uploaded_file.name}, but it produced 0 chunks. "
                    f"If it is a PDF, it may be scanned or use outline/vector fonts "
                    f"with no extractable text. See the server console for details."
                )
        except Exception as e:
            st.error(f"Indexing failed. Is Ollama running?\n\n{e}")

    st.divider()
    st.markdown("**Indexed files**")
    files = sorted(p.name for p in DATA_DIR.iterdir() if p.is_file())
    for name in files:
        st.write(f"- {name}")

    st.divider()
    st.markdown("**Delete a file**")
    if not files:
        st.write("No files to delete.")
    else:
        file_to_delete = st.selectbox("Choose a file", files)
        confirm_delete = st.checkbox(
            "I understand this removes the file from data/ permanently.",
            key="delete_confirm",
        )
        if st.button("Delete file", disabled=not confirm_delete):
            try:
                with st.spinner("Deleting and re-indexing..."):
                    (DATA_DIR / file_to_delete).unlink()
                    collection = build_index()
                st.session_state["delete_confirm"] = False
                st.success(
                    f"Deleted {file_to_delete}. "
                    f"Index rebuilt: {collection.count()} chunks."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed.\n\n{e}")

with st.form("ask"):
    question = st.text_input(
        "Your question", placeholder="e.g. How often should I descale?"
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    try:
        with st.spinner("Searching and generating..."):
            hits = retrieve(question)
            answer = generate(question, hits) if hits else None
    except NotFoundError:
        st.info(
            "No documents indexed yet. Upload a file in the sidebar "
            "and click **Index it** first."
        )
    except Exception as e:
        st.error(
            f"Something went wrong. Is Ollama running? "
            f"(hint: `setsid nohup ~/.local/bin/ollama serve ...`)\n\n{e}"
        )
    else:
        if not hits:
            st.info(
                "No indexed passages match your question. "
                "Check the documents in the sidebar."
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
