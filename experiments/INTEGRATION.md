# Integration Guide

How to apply experiment winners back into the production app.

---

## Step 1 — Update `.env`

After running the experiments, update the following variables in `.env`
based on the winning configurations from `experiments/findings.md`:

```bash
# Winning embedding model (from embedding_comparison)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2   # replace with winner

# Winning chunking config (from retrieval_ablation / chunking phase)
CHUNK_SIZE=512      # replace with winner
CHUNK_OVERLAP=64    # replace with winner

# Winning retrieval config
RETRIEVAL_TOP_K=4   # update if ablation shows a different k is better
SIMILARITY_THRESHOLD=0.3   # may need tuning based on new embedding space
```

---

## Step 2 — Re-ingest documents

Whenever the **embedding model or chunking parameters** change, you must
re-ingest all documents. The existing ChromaDB index uses the old embedding
space and must be rebuilt:

```bash
# Stop the API server first (SQLite doesn't allow concurrent writers)
# then:
python scripts/ingest.py --verbose
```

The ingestion pipeline is idempotent — running it twice is safe.

---

## Step 3 — Update the prompt template (if prompt ablation recommends a change)

The system prompt lives in `app/generation/prompt.py` → `SYSTEM_TEMPLATE`.

Replace the `SYSTEM_TEMPLATE` string with the winning template body from
`experiments/prompt_experiments/templates.py`.

**Example**: If `few_shot_2` wins, find the system template body in
`templates.py` and copy it into `SYSTEM_TEMPLATE`. For few-shot templates,
also add the example `(human, ai)` message tuples to `build_prompt()` in
`prompt.py`, before the `MessagesPlaceholder`.

```python
# app/generation/prompt.py — example with 2-shot examples
def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", NEW_SYSTEM_TEMPLATE),      # ← from winning template
        ("human", FEW_SHOT_Q1),               # ← add if few-shot wins
        ("ai",    FEW_SHOT_A1),
        ("human", FEW_SHOT_Q2),
        ("ai",    FEW_SHOT_A2),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
```

---

## Step 4 — Update hybrid retrieval (if hybrid_rrf or reranked wins)

The production retriever (`app/retrieval/retriever.py`) currently uses
dense-only retrieval. To add BM25 + RRF hybrid:

### 4a. Add dependencies to `requirements.txt`:
```
rank-bm25>=0.2.2
```

### 4b. Create `app/retrieval/bm25_store.py`:
Copy `experiments/retrieval_experiments/sparse.py` → `app/retrieval/bm25_store.py`.
Remove the experiment-specific imports; wire it to `app/config.py`.

### 4c. Create `app/retrieval/hybrid_retriever.py`:
Copy `experiments/retrieval_experiments/hybrid.py` → `app/retrieval/hybrid_retriever.py`.
Update imports to point to the production modules.

### 4d. Update `app/retrieval/retriever.py`:
```python
# Replace the existing retrieve() function body:
def retrieve(query: str, top_k: Optional[int] = None) -> List[Document]:
    k = top_k or settings.retrieval_top_k
    hybrid = HybridRetriever(
        chroma_store=get_vector_store(),
        bm25_retriever=get_bm25_store(),   # new singleton
        dense_top_k=k * 3,
        bm25_top_k=k * 3,
        rrf_k=60,
        final_top_k=k,
    )
    return hybrid.retrieve(query)
```

### 4e. BM25 index persistence:
BM25 index must be rebuilt from the current ChromaDB chunks on startup.
Add a `get_bm25_store()` function that:
1. Fetches all documents from ChromaDB via `vector_store.get()`.
2. Builds `BM25Retriever(documents)`.
3. Caches with `@lru_cache`.

Note: BM25 index is in-memory only (no persistence). Rebuilding on startup
adds ~1-2s for a corpus of this size.

### 4f. Cross-encoder reranking (optional, if reranked config wins):
Add `sentence-transformers` is already in `requirements.txt`.
Wrap `CrossEncoderReranker` from `experiments/retrieval_experiments/reranking.py`
into a `get_reranker()` singleton, and call it at the end of `retrieve()`.

---

## Step 5 — Verify integration

Run the existing test suite to confirm nothing broke:
```bash
pytest -v
```

Then run a quick end-to-end smoke test:
```bash
# Ingest and query
python scripts/ingest.py --verbose
uvicorn app.api.main:app --port 8000 &
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the return policy?", "session_id": "test"}'
```

Finally, re-run the eval harness to confirm the metric improvement holds
in the integrated production system:
```bash
python experiments/retrieval_experiments/run_retrieval_ablation.py \
    --config experiments/config/retrieval.yaml \
    --phases retrieval
```

---

## What NOT to change

- `app/api/routes.py`, `app/api/models.py`, `app/session.py` — no changes needed for any experiment winner.
- `ui/streamlit_app.py` — unaffected unless the SSE event format changes (it does not).
- `docker-compose.yml`, `Dockerfile` — only update if the new embedding model requires different compute requirements.
- `app/ingestion/pipeline.py` — idempotent ingestion works regardless of chunking config.

---

## Configuration matrix for common winning scenarios

| If winner is... | Files to edit |
|---|---|
| New embedding model | `.env` → `EMBEDDING_MODEL`, re-ingest |
| New chunk size | `.env` → `CHUNK_SIZE`, `CHUNK_OVERLAP`, re-ingest |
| Few-shot prompt | `app/generation/prompt.py` |
| Hybrid retrieval | `requirements.txt`, new `bm25_store.py`, `hybrid_retriever.py`, update `retriever.py` |
| Different `top_k` | `.env` → `RETRIEVAL_TOP_K` |
| Different threshold | `.env` → `SIMILARITY_THRESHOLD` |
