"""
app.py
Streamlit front-end for the RAG Q&A bot.
Retrieves relevant chunks from the FAISS index built by ingest.py, plus any
documents uploaded live in the browser, then asks a free Groq-hosted LLM to
answer using only that context.

Run with:
    streamlit run app.py
"""

import os
import pickle
import tempfile
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_DIR = Path("index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"  # free on Groq; swap if this model is retired
TOP_K = 4  # how many chunks to retrieve per question
DISTANCE_THRESHOLD = 1.2  # chunks with L2 distance above this are considered irrelevant
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        data = pickle.load(f)
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return embed_model, index, data["chunks"], data["sources"], groq_client


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Same chunking logic as ingest.py, kept here so uploads are processed identically."""
    chunks = []
    start = 0
    text = text.replace("\n", " ").strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_text_from_upload(uploaded_file) -> str:
    """Extract text from an in-memory uploaded .txt, .md, or .pdf file."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        reader = PdfReader(tmp_path)
        os.unlink(tmp_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def add_uploaded_files_to_session(uploaded_files, embed_model):
    """Chunk, embed, and store newly uploaded files in session state (not on disk)."""
    if "session_chunks" not in st.session_state:
        st.session_state.session_chunks = []
        st.session_state.session_sources = []
        st.session_state.session_embeddings = None
        st.session_state.processed_filenames = set()

    new_chunks, new_sources = [], []
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_filenames:
            continue  # already processed this session
        text = load_text_from_upload(uploaded_file)
        chunks = chunk_text(text)
        if not chunks:
            st.warning(f"No extractable text found in {uploaded_file.name} (may be a scanned/image-based file).")
            continue
        new_chunks.extend(chunks)
        new_sources.extend([uploaded_file.name] * len(chunks))
        st.session_state.processed_filenames.add(uploaded_file.name)

    if new_chunks:
        new_embeddings = embed_model.encode(new_chunks, convert_to_numpy=True).astype("float32")
        st.session_state.session_chunks.extend(new_chunks)
        st.session_state.session_sources.extend(new_sources)
        if st.session_state.session_embeddings is None:
            st.session_state.session_embeddings = new_embeddings
        else:
            st.session_state.session_embeddings = np.vstack(
                [st.session_state.session_embeddings, new_embeddings]
            )
        st.success(f"Added {len(new_chunks)} chunks from {len(uploaded_files)} file(s) to this session.")


def retrieve(question: str, embed_model, index, chunks, sources, k=TOP_K):
    """Retrieve the top-k chunks from the base index AND any session-uploaded
    files, merge by distance, and keep only ones within DISTANCE_THRESHOLD.
    """
    q_vec = embed_model.encode([question], convert_to_numpy=True).astype("float32")

    candidates = []

    distances, indices = index.search(q_vec, k)
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        candidates.append({"text": chunks[idx], "source": sources[idx], "distance": float(dist)})

    session_embeddings = st.session_state.get("session_embeddings")
    if session_embeddings is not None and len(session_embeddings) > 0:
        session_index = faiss.IndexFlatL2(session_embeddings.shape[1])
        session_index.add(session_embeddings)
        s_distances, s_indices = session_index.search(q_vec, k)
        for dist, idx in zip(s_distances[0], s_indices[0]):
            if idx == -1:
                continue
            candidates.append({
                "text": st.session_state.session_chunks[idx],
                "source": st.session_state.session_sources[idx],
                "distance": float(dist),
            })

    candidates.sort(key=lambda c: c["distance"])
    results = [c for c in candidates if c["distance"] <= DISTANCE_THRESHOLD][:k]
    return results


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    return f"""You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't know — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


def main():
    st.set_page_config(page_title="Notes & Papers Q&A Bot", page_icon="📚")
    st.title("📚 Notes & Papers Q&A Bot")
    st.caption("Ask questions grounded in your own notes and research papers — answered by a free LLM via Groq.")

    if not (INDEX_DIR / "faiss.index").exists():
        st.error("No index found. Run `python ingest.py` first after adding files to the data/ folder.")
        return

    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set. Copy .env.example to .env and add your free Groq API key.")
        return

    embed_model, index, chunks, sources, groq_client = load_resources()

    with st.sidebar:
        st.header("📄 Indexed documents")
        preloaded_files = sorted(set(sources))
        for f in preloaded_files:
            st.markdown(f"- {f}")

        session_files = sorted(st.session_state.get("processed_filenames", set()))
        if session_files:
            st.markdown("**Uploaded this session:**")
            for f in session_files:
                st.markdown(f"- {f}")

        st.divider()
        st.subheader("➕ Add your own files")
        uploaded_files = st.file_uploader(
            "Upload .txt, .md, or .pdf files to query alongside the notes above",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            add_uploaded_files_to_session(uploaded_files, embed_model)

    question = st.text_input("Ask a question about your notes or papers:")

    if question:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve(question, embed_model, index, chunks, sources)

        if not retrieved:
            st.subheader("Answer")
            st.write("I don't know — nothing in your notes or papers looks relevant to this question.")
        else:
            with st.spinner("Generating answer..."):
                prompt = build_prompt(question, retrieved)
                response = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Sources used"):
                for c in retrieved:
                    st.markdown(f"**{c['source']}** (distance: {c['distance']:.3f})")
                    st.caption(c["text"][:300] + "...")


if __name__ == "__main__":
    main()