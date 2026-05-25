"""Pydantic models for API request and response payloads."""

from typing import List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload sent by the client to the POST /chat endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: str = Field(
        default="default",
        description="Unique conversation identifier; use a UUID per browser tab",
    )


class SourceChunk(BaseModel):
    """A single retrieved document chunk returned alongside the LLM answer."""

    content: str = Field(..., description="Truncated chunk text (≤300 chars)")
    source: str = Field(..., description="Source filename (e.g. handbook.pdf)")
    page: str = Field(..., description="Page number within the source file")
    chunk_index: int = Field(default=0, description="Position of the chunk in the corpus")


class ChatResponse(BaseModel):
    """Full (non-streaming) chat response — used in tests and the sync fallback."""

    response: str
    sources: List[SourceChunk]
    session_id: str


class HealthResponse(BaseModel):
    """Liveness check response."""

    status: str
    llm_model: str
    embedding_model: str
    document_count: int
