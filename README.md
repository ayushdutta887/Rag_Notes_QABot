# 📚 Notes & Papers Q&A Bot (RAG)

[Live Demo](https://ragnotesappbot-3tokhec5jvxenegzaki6as.streamlit.app/)

A Retrieval-Augmented Generation (RAG) app that answers questions grounded in your own notes and research papers, instead of relying on an LLM's general knowledge. Embeddings run locally using Sentence Transformers, while generation uses Groq's free-tier LLM API.

## How it works

1. `ingest.py` loads your files from `data/`, splits them into overlapping chunks, generates local embeddings using `sentence-transformers`, and stores them in a FAISS vector index.

2. `app.py` (Streamlit) takes your question, retrieves the most relevant chunks from the index, and filters out chunks that are too semantically distant to be genuinely relevant. If relevant context exists, it is sent along with the question to a Groq LLM, which generates an answer using only the retrieved context. If nothing relevant is found, the app reports that instead of guessing.

3. Visitors can also upload their own `.txt`, `.md`, or `.pdf` files directly in the app sidebar. These files are chunked and embedded live during the session and merged into retrieval alongside the pre-loaded documents, without being saved to disk.

## Setup

```text
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