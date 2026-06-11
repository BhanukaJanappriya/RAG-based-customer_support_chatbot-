"""Chunking strategy implementations for the retrieval ablation.

Strategies tested:
- fixed: RecursiveCharacterTextSplitter with fixed size and overlap
- recursive: Same, but with hierarchical separator list (production config)
- semantic: Split at embedding-similarity breakpoints (langchain SemanticChunker)
- sentence_window: Small sentence-level chunks, expanded at query time

Note on semantic chunking: SemanticChunker calls the embedding model for
every potential split point — expensive on CPU. Expect ~5-10x longer
indexing time vs. fixed-size splitting. Only worthwhile if it substantially
improves recall on multi-sentence concepts.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

ChunkStrategy = Literal["fixed", "recursive", "semantic", "sentence_window"]


def chunk_fixed(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Fixed-size character splitting without hierarchy.

    Uses only the empty-string separator so chunks are strictly
    ``chunk_size`` characters (no preference for paragraph/sentence breaks).
    Overlap is absolute in characters.

    Args:
        documents: Raw loaded documents.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        Chunked documents with ``chunk_index`` metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[""],  # force character-level split
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    _add_chunk_index(chunks)
    logger.info(f"Fixed chunking: {len(documents)} docs → {len(chunks)} chunks "
                f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def chunk_recursive(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Recursive character splitting with semantic separator hierarchy.

    Mirrors the production chunker: tries paragraph → sentence → word →
    character splits in order. This is the production baseline.

    Args:
        documents: Raw loaded documents.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap in characters.

    Returns:
        Chunked documents with ``chunk_index`` metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    _add_chunk_index(chunks)
    logger.info(f"Recursive chunking: {len(documents)} docs → {len(chunks)} chunks "
                f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def chunk_semantic(
    documents: list[Document],
    embedding_model,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
) -> list[Document]:
    """Semantic chunking using embedding-similarity breakpoints.

    Splits at points where consecutive sentence embeddings have low cosine
    similarity (i.e., a topic shift). Produces variable-length chunks that
    respect semantic boundaries.

    Requires ``langchain-experimental`` to be installed:
    ``pip install langchain-experimental``

    Args:
        documents: Raw loaded documents.
        embedding_model: A LangChain embeddings instance (used to detect splits).
        breakpoint_threshold_type: "percentile", "standard_deviation", or "interquartile".
        breakpoint_threshold_amount: Threshold value; higher = fewer splits = larger chunks.

    Returns:
        Chunked documents with ``chunk_index`` metadata.

    Raises:
        ImportError: If ``langchain-experimental`` is not installed.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        raise ImportError(
            "Semantic chunking requires langchain-experimental. "
            "Install with: pip install langchain-experimental"
        )

    chunker = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    all_chunks: list[Document] = []
    for doc in documents:
        doc_chunks = chunker.create_documents(
            [doc.page_content],
            metadatas=[doc.metadata],
        )
        all_chunks.extend(doc_chunks)

    _add_chunk_index(all_chunks)
    sizes = [len(c.page_content) for c in all_chunks]
    logger.info(
        f"Semantic chunking: {len(documents)} docs → {len(all_chunks)} chunks "
        f"(avg_size={sum(sizes)/len(sizes):.0f}, threshold={breakpoint_threshold_type}/{breakpoint_threshold_amount})"
    )
    return all_chunks


def chunk_sentence_window(
    documents: list[Document],
    window_size: int = 3,
    sentence_splitter_sep: str = ". ",
) -> tuple[list[Document], dict[str, list[int]]]:
    """Sentence-window chunking: index small chunks, expand at query time.

    Each chunk is a single sentence. The ``window_size`` controls how many
    surrounding sentences to add at query time (done in retrieval, not here).
    This module produces the indexed chunks and the window mapping.

    Args:
        documents: Raw loaded documents.
        window_size: Number of sentences before+after to include at query time.
        sentence_splitter_sep: Separator used to split into sentences.

    Returns:
        Tuple of (sentence_chunks, chunk_id_to_context_window_ids):
        - sentence_chunks: One Document per sentence.
        - window_map: chunk_index → [context_chunk_indices] (the window).
    """
    sentence_chunks: list[Document] = []
    chunk_idx = 0

    for doc in documents:
        sentences = [s.strip() for s in doc.page_content.split(sentence_splitter_sep) if s.strip()]
        doc_chunk_indices = []

        for sent in sentences:
            chunk = Document(
                page_content=sent,
                metadata={**doc.metadata, "chunk_index": chunk_idx},
            )
            sentence_chunks.append(chunk)
            doc_chunk_indices.append(chunk_idx)
            chunk_idx += 1

    # Build window map
    n = len(sentence_chunks)
    window_map: dict[str, list[int]] = {}
    for i in range(n):
        window = list(range(
            max(0, i - window_size),
            min(n, i + window_size + 1),
        ))
        cid = f"chunk_{i}"
        window_map[cid] = window

    logger.info(
        f"Sentence-window chunking: {len(documents)} docs → {len(sentence_chunks)} "
        f"sentence chunks (window_size={window_size})"
    )
    return sentence_chunks, window_map


def expand_with_window(
    retrieved_chunks: list[Document],
    all_chunks: list[Document],
    window_map: dict[str, list[int]],
) -> list[Document]:
    """Expand retrieved sentence chunks to their window context.

    Called at query time: after retrieving the top-k sentence chunks,
    replace each with the surrounding window of sentences concatenated.

    Args:
        retrieved_chunks: Small sentence chunks returned by the retriever.
        all_chunks: The full indexed sentence chunk list.
        window_map: Mapping from chunk_id → window chunk indices.

    Returns:
        Expanded chunks with window context merged into page_content.
    """
    expanded = []
    seen_windows: set[tuple] = set()

    for chunk in retrieved_chunks:
        idx = chunk.metadata.get("chunk_index", 0)
        cid = f"chunk_{idx}"
        window_indices = tuple(window_map.get(cid, [idx]))

        if window_indices in seen_windows:
            continue
        seen_windows.add(window_indices)

        window_text = " ".join(
            all_chunks[i].page_content
            for i in window_indices
            if i < len(all_chunks)
        )
        expanded.append(
            Document(
                page_content=window_text,
                metadata={**chunk.metadata, "window_indices": list(window_indices)},
            )
        )

    return expanded


def _add_chunk_index(chunks: list[Document]) -> None:
    """In-place: add sequential chunk_index metadata and default page=1."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata.setdefault("page", 1)


def get_chunker(strategy: str, **kwargs):
    """Factory function: return a chunk function bound to kwargs.

    Args:
        strategy: One of "fixed", "recursive", "semantic", "sentence_window".
        **kwargs: Strategy-specific parameters.

    Returns:
        A callable(documents: list[Document]) → list[Document].

    Raises:
        ValueError: If strategy is unknown.
    """
    if strategy == "fixed":
        def fn(docs):
            return chunk_fixed(docs, kwargs["chunk_size"], kwargs["chunk_overlap"])
    elif strategy == "recursive":
        def fn(docs):
            return chunk_recursive(docs, kwargs["chunk_size"], kwargs["chunk_overlap"])
    elif strategy == "semantic":
        embedding_model = kwargs.pop("embedding_model")
        def fn(docs):
            return chunk_semantic(docs, embedding_model, **kwargs)
    elif strategy == "sentence_window":
        def fn(docs):
            chunks, window_map = chunk_sentence_window(docs, kwargs.get("window_size", 3))
            return chunks, window_map
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
    return fn
