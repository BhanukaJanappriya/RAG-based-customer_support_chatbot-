"""FastAPI route definitions."""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.models import ChatRequest, HealthResponse
from app.config import settings
from app.generation.chain import astream_response
from app.retrieval.vector_store import get_collection_count
from app.session import add_to_history, clear_history, get_history

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    """Return API health status plus basic system info.

    Does not probe Ollama so it stays fast and always responds,
    even if the LLM is still loading.
    """
    return HealthResponse(
        status="ok",
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        document_count=get_collection_count(),
    )


@router.post("/chat", summary="Stream a RAG-grounded response")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Accept a user query and stream back an SSE response.

    Each SSE ``data:`` frame carries a JSON object with one of these shapes:

    - ``{"type": "token",   "content": "<partial text>"}``
    - ``{"type": "sources", "content": [<source objects>]}``
    - ``{"type": "done",    "content": ""}``
    - ``{"type": "error",   "content": "<message>"}``

    Session history is updated after the full response is assembled.
    """
    chat_history = get_history(request.session_id)

    async def event_stream() -> AsyncGenerator[str, None]:
        full_response = ""
        try:
            async for event in astream_response(request.query, chat_history):
                if event["type"] == "token":
                    full_response += event["content"]
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.error(f"Streaming error for session {request.session_id}: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        finally:
            if full_response:
                add_to_history(request.session_id, request.query, full_response)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Prevents nginx from buffering the stream
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/chat/{session_id}", summary="Clear conversation history")
async def clear_session(session_id: str) -> dict:
    """Wipe conversation history for the given session ID."""
    clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}
