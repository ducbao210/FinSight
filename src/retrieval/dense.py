"""Dense retrieval using sentence-transformers embeddings and local ChromaDB."""

from __future__ import annotations

from threading import Lock
from typing import Any, cast

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from src.core.configs import settings
from src.core.logging import get_logger
from src.schemas import Chunk, ChunkMetadata, RetrievedChunk

logger = get_logger(__name__)

_embedding_model: SentenceTransformer | None = None
_embedding_model_lock = Lock()


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                logger.info("Loading embedding model: %s", settings.embedding_model)
                _embedding_model = SentenceTransformer(
                    settings.embedding_model, token=settings.hf_token or None
                )
    return _embedding_model


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return cast(list[float], embedding.tolist())


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [cast(list[float], emb.tolist()) for emb in embeddings]


def get_chroma_client() -> chromadb.ClientAPI:
    """Open the on-disk Chroma database without starting a server."""
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_persist_dir))


def collection_exists(client: chromadb.ClientAPI) -> bool:
    try:
        client.get_collection(settings.chroma_collection)
        return True
    except NotFoundError:
        return False
    except Exception:
        logger.exception("Unable to inspect Chroma collection")
        raise


def get_collection(client: chromadb.ClientAPI | None = None) -> Any:
    """Return the existing collection, failing clearly when ingestion was not run."""
    client = client or get_chroma_client()
    try:
        return client.get_collection(settings.chroma_collection)
    except NotFoundError as exc:
        raise FileNotFoundError(
            "Chroma collection is missing. Run scripts/ingest.py first."
        ) from exc


def create_collection(client: chromadb.ClientAPI | None = None) -> Any:
    """Create or reopen the collection configured for cosine-distance search."""
    client = client or get_chroma_client()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        configuration={"hnsw": {"space": "cosine"}},
    )
    logger.info("Opened local Chroma collection '%s'", settings.chroma_collection)
    return collection


def reset_collection(client: chromadb.ClientAPI | None = None) -> Any:
    """Delete and recreate only this application's collection."""
    client = client or get_chroma_client()
    if collection_exists(client):
        client.delete_collection(settings.chroma_collection)
        logger.info("Deleted Chroma collection '%s'", settings.chroma_collection)
    return create_collection(client)


def dense_search(
    collection: Any,
    query: str,
    top_k: int | None = None,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
) -> list[RetrievedChunk]:
    if top_k is None:
        top_k = settings.top_k_dense

    clauses: list[dict[str, object]] = []
    if company_filter:
        clauses.append({"company": company_filter})
    if fiscal_year_filter is not None:
        clauses.append({"fiscal_year": fiscal_year_filter})
    where = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)

    query_embedding = embed_text(query)

    def run_query(query_where: dict | None) -> dict:
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=query_where,
            include=["metadatas", "documents", "distances"],
        )

    results = run_query(where)
    first_batch = (results.get("documents") or [[]])[0] or []
    if not first_batch and company_filter and fiscal_year_filter is not None:
        company_only = {"company": company_filter}
        logger.warning(
            "Dense year filter returned no chunks; retrying with company only "
            "(company=%r, fiscal_year=%r)",
            company_filter,
            fiscal_year_filter,
        )
        results = run_query(company_only)
        first_batch = (results.get("documents") or [[]])[0] or []
    if not first_batch and where is not None and not company_filter:
        logger.warning(
            "Dense strict filters returned no chunks; retrying without filters "
            "(company=%r, fiscal_year=%r)",
            company_filter,
            fiscal_year_filter,
        )
        results = run_query(None)
    metadatas = (results.get("metadatas") or [[]])[0] or []
    documents = (results.get("documents") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    retrieved: list[RetrievedChunk] = []
    for metadata, text, distance in zip(metadatas, documents, distances):
        payload = metadata or {}
        chunk = Chunk(
            metadata=ChunkMetadata(
                chunk_id=str(payload.get("chunk_id", "")),
                doc_id=str(payload.get("doc_id", "")),
                company=str(payload.get("company", "")),
                ticker=str(payload.get("ticker", "")),
                doc_type=str(payload.get("doc_type", "")),
                fiscal_year=payload.get("fiscal_year"),
                page=payload.get("page"),
                section=str(payload.get("section", "")),
                source_url=str(payload.get("source_url", "")),
                content_type=str(payload.get("content_type", "narrative")),
            ),
            text=text or "",
        )
        # Chroma returns a cosine distance (lower is better); keep the public
        # retriever convention where a higher score is better.
        retrieved.append(
            RetrievedChunk(chunk=chunk, score=1.0 - float(distance), source="dense")
        )
    logger.info(
        "Dense search: query=%r query_len=%d filters=%s n_results=%d returned=%d",
        query,
        len(query),
        where,
        top_k,
        len(retrieved),
    )
    return retrieved
