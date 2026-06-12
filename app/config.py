"""Centralised application settings loaded from environment / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama / LLM ---
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:1b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # --- ChromaDB ---
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "customer_support_docs"

    # --- Analytics ---
    analytics_db_path: str = "./data/analytics.db"

    # --- Ingestion ---
    raw_data_dir: str = "./data/raw"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Retrieval ---
    retrieval_top_k: int = 4
    similarity_threshold: float = 0.3

    # --- FastAPI ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # --- Streamlit ---
    streamlit_api_url: str = "http://localhost:8000"

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def raw_data_path(self) -> Path:
        return Path(self.raw_data_dir)

    @property
    def analytics_path(self) -> Path:
        return Path(self.analytics_db_path)


settings = Settings()
