# RAG-Based Customer Support Chatbot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-1C3C3C.svg)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama%203.2-000000.svg?logo=ollama&logoColor=white)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-FFA000.svg)](https://www.trychroma.com/)

A fully local Retrieval-Augmented Generation (RAG) chatbot for customer support, built as a collaborative undergraduate project. Zero API costs, zero cloud dependencies; Everything runs on your laptop, with streamed answers, cited sources, and a text-to-speech voice that reads replies aloud in a male or female voice.

---

## Architecture Overview

```
data/raw/   →  loader.py   →  chunker.py  →  embedder.py  →  ChromaDB
(PDF / MD)     (load docs)    (512 / 64)     (MiniLM-L6)     (persist)
                                                                  │
                                                                  ▼
User query ──► embedder.py ──► vector search ──► top-k chunks (filtered)
                                                                  │
                                                                  ▼
                                              prompt.py (system + context + history)
                                                                  │
                                                                  ▼
                                              ChatOllama → Llama 3.2 latest (local)
                                                                  │
                                                   ┌──────────────┘
                                                   ▼
                                        FastAPI SSE stream
                                      (token / sources / done)
                                                   │
                                                   ▼
                                          Streamlit UI
                                   (st.write_stream + expanders)
```

**Data flow:**

- **Ingestion** — `PyPDFLoader` and `TextLoader` pull documents from `data/raw/`. `RecursiveCharacterTextSplitter` breaks them into 512-char chunks with 64-char overlap, preferring paragraph and sentence boundaries.
- **Embedding & persistence** — `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine-normalised, CPU-friendly) embeds each chunk. ChromaDB persists vectors to `data/chroma_db/` with stable IDs (`<filename>__chunk_<n>`) for idempotent re-ingestion.
- **Retrieval** — At query time the same embedding model encodes the question. ChromaDB returns up to `top_k=4` candidates; any chunk scoring below `similarity_threshold=0.3` is dropped before reaching the LLM.
- **Generation** — Filtered chunks are formatted into a cited context block (`[Source: file.pdf, p.N]`) and injected into a structured system prompt alongside the last 10 conversation turns. Llama 3.2 1B Instruct (via Ollama) is explicitly instructed to refuse if context is insufficient.
- **Streaming API** — FastAPI `POST /chat` runs the LCEL chain with `astream()` and emits Server-Sent Events: `{"type":"token","content":"…"}` per partial token, then `{"type":"sources",…}` and `{"type":"done"}`.
- **Streamlit UI** — `st.write_stream` consumes the SSE generator in real time. After each response, a collapsible expander shows the cited chunks; the sidebar shows raw retrieved context for transparency.
- **Voice** — a 🔊 button next to each assistant reply reads it aloud via the browser's Web Speech API, with a Female/Male voice picker in the sidebar. Runs entirely client-side — no extra Python dependencies or server load.
- **Session memory** — `HumanMessage`/`AIMessage` objects are stored in a per-session in-memory dict, capped at 10 turns, and passed into the prompt's `MessagesPlaceholder` to support multi-turn conversation.

---

## Project Structure

```
├── app/
│   ├── config.py            # Pydantic-settings (reads from .env)
│   ├── session.py           # In-memory conversation history store
│   ├── ingestion/
│   │   ├── loader.py        # PDF + Markdown document loaders
│   │   ├── chunker.py       # RecursiveCharacterTextSplitter wrapper
│   │   ├── embedder.py      # HuggingFaceEmbeddings singleton
│   │   └── pipeline.py      # load → chunk → embed → persist (idempotent)
│   ├── retrieval/
│   │   ├── vector_store.py  # ChromaDB client + add/count helpers
│   │   └── retriever.py     # Similarity search with threshold filtering
│   ├── generation/
│   │   ├── prompt.py        # ChatPromptTemplate (system + history + question)
│   │   └── chain.py         # LCEL chain, streaming + sync generation
│   └── api/
│       ├── models.py        # Pydantic request/response models
│       ├── routes.py        # GET /health, POST /chat, DELETE /chat/{id}
│       └── main.py          # FastAPI app factory + CORS
├── ui/
│   └── streamlit_app.py     # Chat UI: streaming, sources, voice playback, debug sidebar
├── data/
│   ├── raw/                 # Drop your PDFs and Markdown files here
│   └── chroma_db/           # Auto-created by ChromaDB on first ingest
├── tests/
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── scripts/
│   ├── ingest.py            # One-shot CLI ingest
│   └── evaluate.py          # Precision@k + MRR + LLM-as-judge
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```

---

## Setup Guide

### 1 — Install Ollama and pull the model

**Windows / macOS:**
```bash
# Download from https://ollama.com and run the installer, then:
ollama pull llama3.2 latest
ollama serve          # keep this terminal open (or it runs as a service)
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
ollama serve &
```

Verify Ollama is reachable:
```bash
curl http://localhost:11434/api/tags
```

---

### 2 — Create a Python virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

---

### 3 — Install Python dependencies

Install CPU-only PyTorch **first** to avoid the 2 GB GPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

### 4 — Configure environment variables

```bash
cp .env.example .env
# Edit .env if you need to change ports, model name, or paths
```

---

### 5 — Add documents and ingest

Copy your PDFs or Markdown files into `data/raw/`, then run:

```bash
python scripts/ingest.py
```

Expected output:
```
✓ Ingestion complete
  Documents loaded : 12
  Chunks created   : 248
  Chunks stored    : 248
  Total in store   : 248
```

Re-running is safe — existing chunks are replaced, not duplicated.

---

### 6 — Start the FastAPI backend

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` to explore the interactive API docs.

---

### 7 — Start the Streamlit UI (new terminal)

```bash
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

### 8 — Run tests

```bash
pytest -v
```

Tests mock all external services (Ollama, ChromaDB, HuggingFace) so they run without any infrastructure.

---

### 9 — Run evaluation

```bash
python scripts/evaluate.py
```

Add domain-specific questions to `scripts/evaluate.py` → `DEFAULT_QA_PAIRS`, or pass a custom file:

```bash
python scripts/evaluate.py --qa-file my_questions.json
```

Custom file format:
```json
[
  {"question": "What is the return policy?", "expected_keywords": ["return", "refund", "days"]}
]
```

---

### Optional — Docker Compose

```bash
docker compose up --build
```

After containers start, pull the model into the Ollama container:

```bash
docker exec rag_ollama ollama pull llama3.2:1b
```

---

## Troubleshooting

### 1 — Ollama not running / `ConnectionRefusedError`

**Symptom:** `httpx.ConnectError: All connection attempts failed` or the Streamlit sidebar shows "API unreachable".

**Fix:**
```bash
curl http://localhost:11434/api/tags   # verify Ollama responds
ollama serve                           # if not, start it
ollama list                            # confirm llama3.2:1b is pulled
```

In Docker: `docker compose logs ollama` and check the healthcheck status.

---

### 2 — ChromaDB lock error (`database is locked`)

**Symptom:** `sqlite3.OperationalError: database is locked` when the API and an ingest script run simultaneously.

**Fix:** ChromaDB's SQLite backend does not support concurrent writers. Stop uvicorn, run the ingest, then restart.

```bash
Ctrl+C                    # stop the API
python scripts/ingest.py  # ingest
uvicorn app.api.main:app --reload  # restart
```

---

### 3 — Out-of-memory (OOM) during embedding

**Symptom:** `RuntimeError: defaultCPUAllocator: can't allocate memory` during `python scripts/ingest.py`.

**Fix:** Reduce chunk size and overlap in `.env` to lower peak tensor memory:

```env
CHUNK_SIZE=256
CHUNK_OVERLAP=32
```

The MiniLM model itself needs ~90 MB. The bottleneck is holding many chunk tensors in RAM at once - smaller chunks reduce peak usage.

---

### 4 — Slow inference (very low tokens/second)

**Expected:** Llama 3.2 1B on a CPU without AVX-512 produces ~3–8 tokens/second. This is normal.

**Mitigations:**
- Lower `LLM_MAX_TOKENS=256` in `.env` to cap response length.
- Lower `RETRIEVAL_TOP_K=2` — less context = shorter prompt = faster generation.
- Use a machine with AVX-512 support (Intel Ice Lake / AMD Zen 4+); Ollama auto-detects it.
- Benchmark raw speed: `ollama run llama3.2:1b "hello"`.

---

### 5 — Empty retrievals (bot always says "no information")

**Symptom:** The LLM always refuses even for questions clearly present in your documents.

**Diagnosis:**
```bash
# Check chunk count
curl http://localhost:8000/health

# Re-ingest with verbose logging
python scripts/ingest.py --verbose

# Temporarily lower the threshold to see if anything comes back
# In .env:
SIMILARITY_THRESHOLD=0.1
```

**Common root causes:**
- Documents were never ingested (collection count = 0 in `/health`).
- Query language differs from document language (model is English-optimised).
- `CHROMA_PERSIST_DIR` changed between ingestion and retrieval — both must point to the same path.
- Mixing embedding models between ingest and query runs (vectors become incompatible).
