"""Embedding model wrapper (singleton) using sentence-transformers."""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache the HuggingFace embedding model.

    The model is downloaded on first call and cached in ``~/.cache/huggingface``.
    Subsequent calls return the same in-memory instance (lru_cache ensures this).

    Returns:
        A configured ``HuggingFaceEmbeddings`` instance ready for encoding.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        # Normalise to unit vectors so cosine similarity == dot product
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model ready")
    return model
