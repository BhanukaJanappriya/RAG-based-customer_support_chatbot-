"""LangChain LCEL chain for RAG generation with streaming support."""

import logging
from typing import AsyncGenerator, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app.config import settings
from app.generation.prompt import build_prompt
from app.retrieval.retriever import retrieve

logger = logging.getLogger(__name__)


def format_context(documents: List[Document]) -> str:
    """Render retrieved chunks into a single cited context block.

    Each chunk is prefixed with its source citation so the LLM can echo
    it back in the response (e.g. "[Source: handbook.pdf, p.3]").

    Args:
        documents: Documents returned by the retriever.

    Returns:
        Multi-section string, or a fallback message if the list is empty.
    """
    if not documents:
        return "No relevant context found in the knowledge base."

    sections = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        sections.append(f"[Source: {source}, p.{page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


def _get_llm() -> ChatOllama:
    """Instantiate a ChatOllama client with project settings."""
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
    )


async def astream_response(
    query: str,
    chat_history: List[BaseMessage],
) -> AsyncGenerator[dict, None]:
    """Async generator that streams events for a single query.

    Yields dictionaries with a ``type`` discriminator:

    - ``{"type": "token",   "content": "<partial text>"}``
    - ``{"type": "sources", "content": [<source dicts>]}``
    - ``{"type": "done",    "content": ""}``

    The API layer serialises these as Server-Sent Events; the UI layer
    reads them to progressively render the response and source panel.

    Args:
        query: User question.
        chat_history: Prior conversation turns for this session.

    Yields:
        Event dictionaries as described above.
    """
    yield {"type": "thinking", "content": "Searching knowledge base…"}
    documents = retrieve(query)
    context = format_context(documents)

    yield {"type": "thinking", "content": "Generating response…"}
    chain = build_prompt() | _get_llm() | StrOutputParser()

    async for token in chain.astream(
        {"question": query, "context": context, "chat_history": chat_history}
    ):
        yield {"type": "token", "content": token}

    source_payload = [
        {
            "content": (
                doc.page_content[:300] + "…"
                if len(doc.page_content) > 300
                else doc.page_content
            ),
            "source": doc.metadata.get("source", "unknown"),
            "page": str(doc.metadata.get("page", "?")),
            "chunk_index": int(doc.metadata.get("chunk_index", 0)),
        }
        for doc in documents
    ]
    yield {"type": "sources", "content": source_payload}
    yield {"type": "done", "content": ""}


def generate_response_sync(
    query: str,
    chat_history: List[BaseMessage],
) -> Tuple[str, List[Document]]:
    """Blocking (non-streaming) generation for scripts and tests.

    Args:
        query: User question.
        chat_history: Prior conversation turns.

    Returns:
        Tuple of (response text, retrieved Documents).
    """
    documents = retrieve(query)
    context = format_context(documents)

    chain = build_prompt() | _get_llm() | StrOutputParser()
    response = chain.invoke(
        {"question": query, "context": context, "chat_history": chat_history}
    )
    return response, documents
