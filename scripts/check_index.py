"""Validate that the local Chroma and BM25 indexes are ready for serving."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.configs import settings


def main() -> int:
    bm25_path = ROOT / "data" / "index" / "bm25.pkl"
    chroma_dir = settings.chroma_persist_dir

    if not bm25_path.is_file():
        print(f"Index missing: {bm25_path}")
        return 1
    if not chroma_dir.is_dir():
        print(f"Chroma directory missing: {chroma_dir}")
        return 1

    try:
        with bm25_path.open("rb") as handle:
            bm25 = pickle.load(handle)
        bm25_chunks = len(bm25.chunks)

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(settings.chroma_collection)
        chroma_chunks = collection.count()
    except Exception as exc:
        print(f"Index validation failed: {type(exc).__name__}: {exc}")
        return 1

    if not bm25_chunks or not chroma_chunks:
        print(f"Index is empty: chroma={chroma_chunks}, bm25={bm25_chunks}")
        return 1
    if chroma_chunks != bm25_chunks:
        print(f"Index count mismatch: chroma={chroma_chunks}, bm25={bm25_chunks}")
        return 1

    print(
        f"Index is ready: collection={settings.chroma_collection}, "
        f"chroma={chroma_chunks}, bm25={bm25_chunks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
