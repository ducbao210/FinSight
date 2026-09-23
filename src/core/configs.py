"""FinSight core configuration using pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vector database. Chroma persists its SQLite and HNSW files locally, so
    # ingestion does not depend on a remote HTTP request completing.
    chroma_persist_dir: Path = PROJECT_ROOT / "data/index/chroma"
    chroma_collection: str = "finsight_chunks"
    chroma_upsert_retries: int = 3

    # LLM API
    groq_api_key: str = ""

    # Model name
    model_name: str = "openai/gpt-oss-120b"
    # HF TOKEN
    hf_token: str = ""

    # Embedding model
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # Reranker model. Enable with USE_RERANKER=true when the extra latency is acceptable.
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = False

    # Retrieval
    top_k_dense: int = 30
    top_k_bm25: int = 30
    top_k_hybrid: int = 12

    # CRAG
    max_retries: int = 2
    max_hops: int = 2

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"

    def model_post_init(self, __context: object) -> None:
        """Resolve configured paths against the repository root, not shell cwd."""
        if not self.chroma_persist_dir.is_absolute():
            self.chroma_persist_dir = PROJECT_ROOT / self.chroma_persist_dir
        if not self.log_dir.is_absolute():
            self.log_dir = PROJECT_ROOT / self.log_dir

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def normalize_groq_api_key(cls, value: str | None) -> str:
        """Accept keys copied from dotenv files without surrounding whitespace/quotes."""
        if value is None:
            return ""
        return str(value).strip().strip('"').strip("'").strip()

    @field_validator("chroma_collection", mode="before")
    @classmethod
    def default_collection(cls, value: str | None) -> str:
        return value or "finsight_chunks"

    @property
    def groq_api_key_or_raise(self) -> str:
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment or .env file")
        return self.groq_api_key


settings = Settings()
