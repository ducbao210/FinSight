"""FastAPI application for FinSight."""

from __future__ import annotations

import pickle
import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.core.configs import settings
from src.core.logging import get_logger, setup_logging
from src.graph.workflow import CRAGPipeline
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import get_collection, get_embedding_model
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import get_reranker
from src.schemas import QueryRequest, QueryResponse

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data" / "index" / "bm25.pkl"


def _load_bm25() -> BM25Retriever:
    if not ARTIFACT.exists():
        raise FileNotFoundError(
            f"BM25 artifact not found: {ARTIFACT}. Run scripts/ingest.py first."
        )
    with ARTIFACT.open("rb") as handle:
        return pickle.load(handle)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("backend")
    logger = get_logger(__name__)
    app.state.startup_error = ""
    try:
        collection = get_collection()
        bm25 = _load_bm25()
        await asyncio.to_thread(get_embedding_model)
        if settings.use_reranker:
            await asyncio.to_thread(get_reranker)
        app.state.collection = collection
        app.state.bm25 = bm25
        app.state.pipeline = CRAGPipeline(collection, bm25, settings.model_name)
        logger.info(
            "Index loaded: chroma=%d chunks, bm25=%d chunks, dir=%s",
            collection.count(),
            len(bm25.chunks),
            settings.chroma_persist_dir,
        )
    except Exception as exc:
        app.state.startup_error = str(exc)
        logger.exception("Startup failed: %s", exc)
    yield


app = FastAPI(title="FinSight API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    """Report index counts, not just process liveness.

    An empty or mismatched index is the most common cause of blanket
    abstentions, so it must be visible without reading the logs.
    """
    error = getattr(app.state, "startup_error", "")
    collection = getattr(app.state, "collection", None)
    bm25 = getattr(app.state, "bm25", None)
    chroma_count = collection.count() if collection is not None else 0
    bm25_count = len(bm25.chunks) if bm25 is not None else 0
    problems: list[str] = []
    if error:
        problems.append(error)
    if not chroma_count:
        problems.append("Chroma collection is empty")
    if not bm25_count:
        problems.append("BM25 artifact is empty")
    if chroma_count and bm25_count and chroma_count != bm25_count:
        problems.append(
            f"Index mismatch: chroma={chroma_count} vs bm25={bm25_count}; re-run ingest"
        )
    payload = {
        "status": "error" if problems else "ok",
        "detail": "; ".join(problems),
        "chroma_chunks": chroma_count,
        "bm25_chunks": bm25_count,
        "chroma_dir": str(settings.chroma_persist_dir),
        "collection": settings.chroma_collection,
        "embedding_model": settings.embedding_model,
        "use_reranker": settings.use_reranker,
        "top_k_hybrid": settings.top_k_hybrid,
    }
    if problems:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/debug/retrieve")
def debug_retrieve(
    q: str,
    top_k: int = 10,
    company: str | None = None,
    fiscal_year: int | None = None,
) -> dict[str, object]:
    """Run retrieval only, bypassing the LLM graph.

    This separates "retrieval found nothing" from "the grader rejected
    everything" - the two failures look identical in the answer UI.
    """
    if not hasattr(app.state, "pipeline"):
        raise HTTPException(
            status_code=503,
            detail=getattr(app.state, "startup_error", "Pipeline is not ready"),
        )
    results = hybrid_search(
        app.state.collection,
        app.state.bm25,
        q,
        top_k=top_k,
        company_filter=company,
        fiscal_year_filter=fiscal_year,
    )
    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "rank": rank,
                "score": item.score,
                "chunk_id": item.chunk.metadata.chunk_id,
                "company": item.chunk.metadata.company,
                "doc_id": item.chunk.metadata.doc_id,
                "fiscal_year": item.chunk.metadata.fiscal_year,
                "page": item.chunk.metadata.page,
                "section": item.chunk.metadata.section,
                "content_type": item.chunk.metadata.content_type,
                "preview": item.chunk.text[:400],
            }
            for rank, item in enumerate(results, 1)
        ],
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    request_id = uuid.uuid4().hex[:12]
    if not hasattr(app.state, "pipeline"):
        raise HTTPException(
            status_code=503,
            detail=getattr(app.state, "startup_error", "Pipeline is not ready"),
        )
    return await app.state.pipeline.answer(request, request_id=request_id)
