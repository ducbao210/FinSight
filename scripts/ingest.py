"""Parse FinanceBench PDFs and index chunks in local ChromaDB and BM25."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import pickle
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.configs import settings
from src.core.logging import get_logger, setup_logging
from src.ingestion.chunker import create_chunks_from_pdf
from src.ingestion.metadata import (
    load_document_info_from_jsonl,
    resolve_pdf_path,
)
from src.ingestion.parser import PARSER_VERSION, parse_pdf
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import (
    collection_exists,
    create_collection,
    embed_batch,
    get_chroma_client,
    get_collection,
    reset_collection,
)
from src.schemas import Chunk, DocumentInfo, IngestionReport

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs",
        type=Path,
        default=ROOT / "data/raw/financebench_document_information.jsonl",
    )
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/raw/pdfs")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "data/index")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", default=True)
    mode.add_argument(
        "--force",
        action="store_true",
        help="Clear cached state and rebuild the local Chroma collection",
    )
    return parser.parse_args()


def _metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Return Chroma-compatible scalar metadata (it does not accept nulls)."""
    return {
        key: value
        for key, value in chunk.metadata.model_dump().items()
        if value is not None
    }


def _cache_path(cache_dir: Path, doc_id: str) -> Path:
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{digest}.pkl"


def _parse_document(
    task: tuple[DocumentInfo, str, int, int],
) -> tuple[str, list[Chunk] | None, str | None]:
    doc, pdf_path, chunk_size, chunk_overlap = task
    try:
        pages = parse_pdf(Path(pdf_path))
        return (
            doc.doc_id,
            create_chunks_from_pdf(pages, doc, chunk_size, chunk_overlap),
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return doc.doc_id, None, str(exc)


def _connect_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY, pdf_path TEXT NOT NULL, pdf_mtime_ns INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL, chunk_overlap INTEGER NOT NULL, cache_path TEXT NOT NULL,
            status TEXT NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '',
            parser_version TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS batches (
            batch_key TEXT PRIMARY KEY, batch_index INTEGER NOT NULL, start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL, status TEXT NOT NULL, updated_at REAL NOT NULL
        );
        """)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
    if "parser_version" not in columns:
        connection.execute(
            "ALTER TABLE documents ADD COLUMN parser_version TEXT NOT NULL DEFAULT ''"
        )
    connection.commit()
    return connection


def _reset_state(connection: sqlite3.Connection, cache_dir: Path) -> None:
    connection.execute("DELETE FROM documents")
    connection.execute("DELETE FROM batches")
    connection.commit()
    for cache_file in cache_dir.glob("*.pkl"):
        cache_file.unlink()


def _load_cached_chunks(
    connection: sqlite3.Connection,
    doc: DocumentInfo,
    pdf_path: Path,
    args: argparse.Namespace,
) -> list[Chunk] | None:
    row = connection.execute(
        "SELECT pdf_path, pdf_mtime_ns, chunk_size, chunk_overlap, cache_path, status, "
        "parser_version FROM documents WHERE doc_id = ?",
        (doc.doc_id,),
    ).fetchone()
    if not row or row[5] != "complete":
        return None
    if (
        row[:4]
        != (
            str(pdf_path),
            pdf_path.stat().st_mtime_ns,
            args.chunk_size,
            args.chunk_overlap,
        )
        or row[6] != PARSER_VERSION
    ):
        return None
    try:
        with Path(row[4]).open("rb") as handle:
            chunks = pickle.load(handle)
        return (
            chunks
            if isinstance(chunks, list)
            and all(isinstance(item, Chunk) for item in chunks)
            else None
        )
    except (OSError, EOFError, pickle.PickleError, TypeError):
        return None


def _store_document(
    connection: sqlite3.Connection,
    cache_dir: Path,
    doc: DocumentInfo,
    pdf_path: Path,
    chunks: list[Chunk],
    args: argparse.Namespace,
) -> None:
    cache_path = _cache_path(cache_dir, doc.doc_id)
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(cache_path)
    connection.execute(
        "INSERT INTO documents (doc_id, pdf_path, pdf_mtime_ns, chunk_size, chunk_overlap, cache_path, "
        "status, chunk_count, error, parser_version, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'complete', ?, '', ?, ?) "
        "ON CONFLICT(doc_id) DO UPDATE SET pdf_path=excluded.pdf_path, pdf_mtime_ns=excluded.pdf_mtime_ns, "
        "chunk_size=excluded.chunk_size, chunk_overlap=excluded.chunk_overlap, cache_path=excluded.cache_path, "
        "status='complete', chunk_count=excluded.chunk_count, error='', parser_version=excluded.parser_version, "
        "updated_at=excluded.updated_at",
        (
            doc.doc_id,
            str(pdf_path),
            pdf_path.stat().st_mtime_ns,
            args.chunk_size,
            args.chunk_overlap,
            str(cache_path),
            len(chunks),
            PARSER_VERSION,
            time.time(),
        ),
    )
    connection.commit()


def _store_error(
    connection: sqlite3.Connection,
    cache_dir: Path,
    doc: DocumentInfo,
    pdf_path: Path | None,
    args: argparse.Namespace,
    error: str,
) -> None:
    connection.execute(
        "INSERT INTO documents (doc_id, pdf_path, pdf_mtime_ns, chunk_size, chunk_overlap, cache_path, "
        "status, chunk_count, error, parser_version, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'error', 0, ?, ?, ?) "
        "ON CONFLICT(doc_id) DO UPDATE SET status='error', error=excluded.error, "
        "parser_version=excluded.parser_version, updated_at=excluded.updated_at",
        (
            doc.doc_id,
            str(pdf_path or ""),
            pdf_path.stat().st_mtime_ns if pdf_path and pdf_path.exists() else 0,
            args.chunk_size,
            args.chunk_overlap,
            str(_cache_path(cache_dir, doc.doc_id)),
            error,
            PARSER_VERSION,
            time.time(),
        ),
    )
    connection.commit()


def _prepare_chunks(
    docs: list[DocumentInfo], args: argparse.Namespace, connection: sqlite3.Connection
) -> tuple[list[Chunk], int, list[str]]:
    cache_dir = args.artifact_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    pending: list[tuple[DocumentInfo, Path]] = []
    parsed_by_doc: dict[str, list[Chunk]] = {}
    errors: list[str] = []
    processed = 0
    for doc in docs:
        pdf_path = resolve_pdf_path(args.pdf_dir, doc)
        if pdf_path is None:
            error = f"{doc.doc_id}: PDF not found"
            errors.append(error)
            _store_error(connection, cache_dir, doc, None, args, error)
            continue
        cached = _load_cached_chunks(connection, doc, pdf_path, args)
        if cached is None:
            pending.append((doc, pdf_path))
        else:
            chunks.extend(cached)
            processed += 1

    tasks = [
        (doc, str(pdf_path), args.chunk_size, args.chunk_overlap)
        for doc, pdf_path in pending
    ]
    if tasks:
        logger.info("Parsing %d documents with %d workers", len(tasks), args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_parse_document, task): item
                for task, item in zip(tasks, pending)
            }
            for future in as_completed(futures):
                doc, pdf_path = futures[future]
                doc_id, parsed, error = future.result()
                if error or parsed is None:
                    message = f"{doc_id}: {error or 'unknown parse error'}"
                    errors.append(message)
                    _store_error(connection, cache_dir, doc, pdf_path, args, message)
                else:
                    _store_document(connection, cache_dir, doc, pdf_path, parsed, args)
                    parsed_by_doc[doc.doc_id] = parsed
                    processed += 1
    for doc, _ in pending:
        if doc.doc_id in parsed_by_doc:
            chunks.extend(parsed_by_doc[doc.doc_id])
    with (args.artifact_dir / "chunks.pkl").open("wb") as handle:
        pickle.dump(chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return chunks, processed, errors


def _retry_vector_store(operation, description: str):
    for attempt in range(1, settings.chroma_upsert_retries + 1):
        try:
            return operation()
        except Exception as exc:
            if isinstance(exc, ValueError) or type(exc).__name__ in {
                "DuplicateIDError",
                "InvalidArgumentError",
            }:
                raise
            if attempt == settings.chroma_upsert_retries:
                raise
            delay = min(2 ** (attempt - 1), 10)
            logger.warning(
                "%s failed (attempt %d/%d); retrying in %ds",
                description,
                attempt,
                settings.chroma_upsert_retries,
                delay,
            )
            time.sleep(delay)


def _batch_key(batch: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in batch:
        digest.update(chunk.metadata.chunk_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest(args: argparse.Namespace, docs: list[DocumentInfo]) -> dict[str, object]:
    return {
        "parser_version": PARSER_VERSION,
        "embedding_model": settings.embedding_model,
        "collection": settings.chroma_collection,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents": [doc.doc_id for doc in docs],
    }


def _manifest_path(artifact_dir: Path) -> Path:
    return artifact_dir / "ingest_manifest.json"


def _manifest_fingerprint(manifest: dict[str, object]) -> str:
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _index_chroma(
    chunks: list[Chunk], args: argparse.Namespace, connection: sqlite3.Connection
) -> None:
    client = get_chroma_client()
    exists = collection_exists(client)
    if args.force:
        collection = reset_collection(client)
        exists = False
    elif exists:
        collection = get_collection(client)
    else:
        collection = create_collection(client)

    # Batch checkpoints are only valid while their backing collection exists.
    if not exists:
        connection.execute("DELETE FROM batches")
        connection.commit()

    for batch_index, start in enumerate(range(0, len(chunks), args.batch_size)):
        batch = chunks[start : start + args.batch_size]
        key = _batch_key(batch)
        existing = connection.execute(
            "SELECT status FROM batches WHERE batch_key = ?", (key,)
        ).fetchone()
        if existing and existing[0] == "complete":
            logger.info("Skipping completed Chroma batch %d", batch_index)
            continue
        ids = [chunk.metadata.chunk_id for chunk in batch]
        duplicates = sorted(
            chunk_id for chunk_id, count in Counter(ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                "Duplicate chunk IDs before Chroma upsert "
                f"in batch {batch_index}: {duplicates}"
            )
        vectors = embed_batch([chunk.embedding_text for chunk in batch])
        _retry_vector_store(
            lambda: collection.upsert(
                ids=[chunk.metadata.chunk_id for chunk in batch],
                embeddings=vectors,
                documents=[chunk.text for chunk in batch],
                metadatas=[_metadata(chunk) for chunk in batch],
            ),
            f"Chroma upsert for batch {batch_index}",
        )
        connection.execute(
            "INSERT OR REPLACE INTO batches VALUES (?, ?, ?, ?, 'complete', ?)",
            (key, batch_index, start, start + len(batch), time.time()),
        )
        connection.commit()


def ingest(args: argparse.Namespace) -> IngestionReport:
    started = time.perf_counter()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.artifact_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    connection = _connect_state(args.artifact_dir / "ingest_state.sqlite3")
    try:
        if args.force:
            _reset_state(connection, cache_dir)
        docs = load_document_info_from_jsonl(args.docs)
        if args.limit:
            docs = docs[: args.limit]
        manifest = _manifest(args, docs)
        manifest_path = _manifest_path(args.artifact_dir)
        fingerprint = _manifest_fingerprint(manifest)
        if not args.force and manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != fingerprint:
                logger.warning(
                    "Ingestion configuration changed; clearing old checkpoints and index"
                )
                args.force = True
                _reset_state(connection, cache_dir)
        chunks, processed, errors = _prepare_chunks(docs, args, connection)
        bm25 = BM25Retriever()
        bm25.index(chunks)
        with (args.artifact_dir / "bm25.pkl").open("wb") as handle:
            pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _index_chroma(chunks, args, connection)
        manifest_path.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "manifest": manifest},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return IngestionReport(
            documents_processed=processed,
            chunks_created=len(chunks),
            errors=errors,
            duration_seconds=time.perf_counter() - started,
        )
    finally:
        connection.close()


def main() -> int:
    setup_logging("ingest")
    report = ingest(parse_args())
    print(report.model_dump_json(indent=2))

    if report.errors:
        print(f"Ingestion completed with {len(report.errors)} error(s).")
    else:
        print("Ingestion completed successfully.")

    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
