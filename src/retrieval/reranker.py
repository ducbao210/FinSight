from __future__ import annotations

from threading import Lock

from sentence_transformers import CrossEncoder

from src.core.logging import get_logger
from src.core.configs import settings
from src.schemas import RetrievedChunk

logger = get_logger(__name__)

_reranker: CrossEncoder | None = None
_reranker_lock = Lock()


def get_reranker() -> CrossEncoder:
    """Load the local cross-encoder once per worker process."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                logger.info("Loading reranker model: %s", settings.reranker_model)
                _reranker = CrossEncoder(
                    settings.reranker_model,
                    token=settings.hf_token or None,
                )
    return _reranker


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Rerank a bounded candidate list with one cached local cross-encoder."""
    if not chunks:
        return []
    model = get_reranker()
    pairs = [(query, chunk.chunk.text) for chunk in chunks]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [chunk for chunk, _ in ranked[:top_k]]
