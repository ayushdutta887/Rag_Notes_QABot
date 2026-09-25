# 📚 Notes & Papers Q&A Bot (RAG)

A Retrieval-Augmented Generation (RAG) app that answers questions grounded in
your own notes and research papers, instead of relying on an LLM's general
knowledge. Embeddings run locally (free), and generation uses Groq's
free-tier LLM API.

## How it works
1. `ingest.py` loads your files from `data/`, splits them into overlapping
   chunks, embeds them locally with `sentence-transformers`, and stores them
   in a FAISS vector index.
2. `app.py` (Streamlit) takes your question, retrieves the most relevant
   chunks from the index, filters out any that are too semantically distant
   to be genuinely relevant, and — only if relevant context exists — sends
   it + your question to a free Groq LLM, which answers using only that
   context. If nothing relevant is found, it says so instead of guessing.
3. Visitors can also upload their own `.txt`, `.md`, or `.pdf` files directly
   in the app sidebar — these are chunked and embedded live, in-session, and
   merged into retrieval alongside the pre-loaded documents, without being
   saved to disk.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get a free Groq API key
# -> https://console.groq.com/keys (no credit card required)

# 4. Set up your .env file
cp .env.example .env
# then edit .env and paste your GROQ_API_KEY

# 5. Add your documents
# Drop .txt, .md, or .pdf files (your notes, research papers, etc.) into data/

# 6. Build the index
python ingest.py

# 7. Run the app
streamlit run app.py
```

## Project structure
```
rag-qa-bot/
├── data/           # put your notes/papers here (.txt, .md, .pdf)
├── index/           # generated: FAISS index + chunk store
├── ingest.py         # document loading, chunking, embedding
├── app.py            # Streamlit UI + retrieval + Groq generation
├── requirements.txt
├── .env.example
└── README.md
```

## Tech stack
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2) — free, runs locally
- **Vector store:** FAISS
- **Generation:** Groq API (openai/gpt-oss-120b) — free tier
- **UI:** Streamlit

## Key design decision: relevance filtering
Early testing showed that FAISS always returns the top-k nearest chunks —
even when none of them are actually relevant to the question. This meant
irrelevant text was still being passed to the LLM as "context." To fix
this, retrieval results are filtered by a distance threshold: chunks beyond
it are discarded, and if nothing passes, the app skips the LLM call
entirely and reports that no relevant information was found, rather than
risking a hallucinated answer.

## Roadmap / Future Improvements
- Swap character-based chunking for sentence/paragraph-aware chunking
- Add a small evaluation set of Q&A pairs to measure retrieval accuracy
- Cite exact source + page number instead of just filename
- Deploy for free on Streamlit Community Cloud or HuggingFace Spaces
- Add OCR support for scanned/image-based PDFs
