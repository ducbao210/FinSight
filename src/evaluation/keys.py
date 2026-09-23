"""Canonical keys used to compare retrieved chunks against FinanceBench gold evidence.

FinanceBench identifies evidence by ``doc_name`` + ``evidence_page_num``.
Chunks carry ``doc_id`` (== doc_name when ingested with metadata) plus
company/ticker/fiscal_year/doc_type/page, so we can always build a comparable
key, even for chunks ingested before ``doc_id`` existed.
"""

from __future__ import annotations

import re

from src.schemas import ChunkMetadata

# FinanceBench evidence pages are 0-indexed in the JSONL, while PDF parsers
# usually number pages from 1. Gold page + PAGE_OFFSET == parser page.
DEFAULT_PAGE_OFFSET = 1


def normalize_doc_name(value: str) -> str:
    """Lowercase alphanumeric form of a document name, for robust matching."""
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def doc_key_from_metadata(meta: ChunkMetadata) -> str:
    """Document-level key for a retrieved chunk."""
    if meta.doc_id:
        return normalize_doc_name(meta.doc_id)
    # Legacy fallback: rebuild the FinanceBench naming convention.
    name = meta.ticker or meta.company
    parts = [name, str(meta.fiscal_year or ""), meta.doc_type]
    return normalize_doc_name("_".join(p for p in parts if p))


def page_key(doc_key: str, page: int | None) -> str:
    """Page-level key. Chunks without a page number never match gold pages."""
    return f"{doc_key}#p{page}" if page is not None else f"{doc_key}#p?"


def page_key_from_metadata(meta: ChunkMetadata) -> str:
    return page_key(doc_key_from_metadata(meta), meta.page)


def gold_doc_key(doc_name: str) -> str:
    return normalize_doc_name(doc_name)


def gold_page_key(doc_name: str, page_num: int | None, page_offset: int = DEFAULT_PAGE_OFFSET) -> str:
    page = None if page_num is None else page_num + page_offset
    return page_key(normalize_doc_name(doc_name), page)
