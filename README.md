RAG Chat Assistant (Streamlit + Gemini)
=======================================

Overview
--------
Streamlit UI for a Retrieval-Augmented Generation (RAG) assistant that uses Google Gemini and ChromaDB. Upload documents, ingest them into a vector store, and chat with Gemini using retrieved context.

<div align="center">
  <img src="https://raw.githubusercontent.com/langchain-ai/.github/main/profile/langchain-og.png" alt="LangChain" width="420" />
  <img src="https://developers.google.com/static/learn/pathways/guides/gemini-overview/social.png" alt="Gemini" width="420" />
</div>

Features
--------
- Streamlit chat UI with source citations and similarity score display
- Multiple retrieval strategies: basic, multi-query, contextual compression, hybrid (BM25 + vectors)
- Document upload and ingestion (txt, md, pdf, docx)
- Analytics tab with query metrics and response-time trends
- Document explorer with search and basic stats

Prerequisites
-------------
- Python 3.10+ recommended (project uses a `venv/` in the repo)
- Google API key with access to Gemini (`GOOGLE_API_KEY`)
- Optional: set `USE_GEMINI_EMBEDDINGS=true` to use Gemini embeddings instead of HuggingFace

Quick Start
-----------
1) Clone and enter the project:
```
cd "/Users/abhiramrangoon/Desktop/video editing/New_RAG"
```

2) Use the existing virtualenv (or create one):
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Set environment variables (recommended via `.env`):
```
GOOGLE_API_KEY=your_api_key_here
# Optional
USE_GEMINI_EMBEDDINGS=true
```

4) Run Streamlit:
```
venv/bin/python -m streamlit run streamlit_app.py
```

Using the App
-------------
- Sidebar:
  - Select retrieval strategy, `k`, and temperature.
  - Upload documents and click “Ingest Documents”.
  - Click “Initialize RAG Pipeline” (auto-init also happens on first query).
- Chat tab: ask questions; sources appear in an expander.
- Analytics tab: recent queries and response-time chart.
- Documents tab: search and view document stats.

Document Ingestion
------------------
- Supported: `.txt`, `.md`, `.pdf`, `.docx`
- Default search paths: `data/documents/`, `data/sample_document_1.txt`, `data/sample_document_2.md`
- Uploaded files are ingested into Chroma at `vector_store/` by default.

Configuration
-------------
Key settings in `config.py`:
- Models: `LLM_MODEL` (default `gemini-1.5-flash`), `EMBEDDING_MODEL` (`BAAI/bge-small-en-v1.5`)
- Vector store: `VECTOR_STORE_PATH`, `CHROMA_COLLECTION_NAME`
- Retrieval: `RETRIEVAL_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
- Generation: `TEMPERATURE`, `MAX_OUTPUT_TOKENS`, `TOP_P`, `TOP_K`

Troubleshooting
---------------
- Import error for `langchain_community`: ensure venv is active and run `pip install -r requirements.txt`.
- “RAG pipeline not initialized”: click “Initialize RAG Pipeline” in the sidebar; the app now persists the initialized pipeline across reruns.
- Missing `GOOGLE_API_KEY`: add it to `.env` or export before launching Streamlit.

Project Structure
-----------------
- `streamlit_app.py` — Streamlit UI and session handling
- `ingest.py` — document loading, splitting, and vector store creation
- `vectorstore.py` — vector store management and retrievers
- `rag_chain.py` — LCEL-based RAG pipeline with Gemini
- `config.py` — configuration defaults and env helpers
- `data/` — sample documents
- `vector_store/` — persisted Chroma DB (created/used at runtime)

Notes
-----
- Keep the venv active when running Streamlit to ensure all dependencies resolve.
- If you switch retrieval strategies, re-initialize the pipeline for consistency.

