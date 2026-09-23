from __future__ import annotations

import re
from typing import Any

from src.core.configs import settings
from src.core.logging import get_logger
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import dense_search
from src.retrieval.reranker import rerank
from src.schemas import RetrievedChunk

logger = get_logger(__name__)


def reciprocal_rank_fusion_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


EXPANSIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("income statement", "statements of operations", "statement of operations"),
        "Consolidated Statements of Operations",
    ),
    (
        ("balance sheet", "statement of financial position"),
        "Consolidated Balance Sheets",
    ),
    (
        ("cash flow", "cashflow", "statement of cash flows", "operating cash"),
        "Consolidated Statements of Cash Flows net cash provided by operating activities",
    ),
    (
        ("free cash flow", "free cashflow", "fcf"),
        "net cash provided by operating activities less purchases of property and "
        "equipment capital expenditures",
    ),
    (
        ("conversion", "converted"),
        "net income net cash provided by operating activities",
    ),
    (
        ("pp&e", "p.p.&e.", "property plant and equipment"),
        "property, plant and equipment net PP&E",
    ),
    (
        ("capex", "capital expenditure", "capital expenditures"),
        "capital expenditures purchases of property and equipment",
    ),
    (
        ("margin", "profitability"),
        "revenue cost of revenue gross profit operating income net income",
    ),
    (
        ("debt", "leverage", "liabilities"),
        "total liabilities long-term debt current portion of long-term debt",
    ),
    (
        ("dividend", "buyback", "repurchase"),
        "dividends paid repurchases of common stock financing activities",
    ),
)

# Words that describe the question rather than the filing. Removing them turns a
# verbose decomposed sub-question into a short lexical probe that actually
# matches a bare statement table.
_STOPWORDS = {
    "what",
    "was",
    "were",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "as",
    "its",
    "it",
    "and",
    "or",
    "to",
    "on",
    "at",
    "by",
    "with",
    "does",
    "did",
    "do",
    "have",
    "has",
    "had",
    "please",
    "provide",
    "value",
    "values",
    "amount",
    "report",
    "reported",
    "reports",
    "disclosed",
    "disclose",
    "according",
    "per",
    "form",
    "filing",
    "filings",
    "10-k",
    "10k",
    "10-q",
    "10q",
    "annual",
    "fiscal",
    "year",
    "years",
    "fy",
    "company",
    "also",
    "expressed",
    "using",
    "based",
    "answer",
    "question",
    "respond",
    "state",
    "show",
    "shown",
    "figure",
    "figures",
    "usd",
    "million",
    "millions",
    "billion",
    "billions",
    "percentage",
    "percent",
    "rate",
    "ratio",
}


def _keyword_query(query: str) -> str:
    """Strip question scaffolding, keep finance nouns and numbers."""
    tokens = re.findall(r"[A-Za-z0-9&\.\-']+", query)
    kept = [
        token
        for token in tokens
        if token.casefold() not in _STOPWORDS and len(token) > 1
    ]
    return " ".join(kept)


def _query_variants(query: str) -> list[str]:
    """Return a small deterministic set of finance-specific query variants."""
    normalized = query.casefold()
    variants = [query]
    keyword_only = _keyword_query(query)
    if keyword_only and keyword_only.casefold() != normalized:
        variants.append(keyword_only)
    for needles, expansion in EXPANSIONS:
        if any(needle in normalized for needle in needles):
            variants.append(f"{query} {expansion}")
            # A short, purely lexical probe. Long sub-questions drift towards
            # verbose note pages; the statement line items live in short tables.
            variants.append(expansion)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for variant in variants:
        key = variant.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    # Each variant runs both dense and BM25 retrieval. More than three variants
    # adds latency faster than it adds recall on this 222k-chunk index.
    return unique[:3]


def _diversify_by_page(
    ranked: list[tuple[RetrievedChunk, float]], top_k: int, max_per_page: int = 3
) -> list[tuple[RetrievedChunk, float]]:
    """Select one best chunk per page first, then allow more chunks per page.

    A financial statement page normally yields one table chunk plus a narrative
    chunk. Keeping both is desirable, so a table is admitted immediately even if
    the page already contributed a narrative chunk.
    """
    selected: list[tuple[RetrievedChunk, float]] = []
    deferred: list[tuple[RetrievedChunk, float]] = []
    page_counts: dict[str, int] = {}
    page_types: dict[str, set[str]] = {}
    for item in ranked:
        meta = item[0].chunk.metadata
        page = f"{meta.doc_id}#p{meta.page}"
        seen_types = page_types.setdefault(page, set())
        if page_counts.get(page, 0) == 0 or (
            meta.content_type == "table" and "table" not in seen_types
        ):
            selected.append(item)
            page_counts[page] = page_counts.get(page, 0) + 1
            seen_types.add(meta.content_type)
        else:
            deferred.append(item)
        if len(selected) >= top_k:
            return selected
    for item in deferred:
        meta = item[0].chunk.metadata
        page = f"{meta.doc_id}#p{meta.page}"
        if page_counts.get(page, 0) >= max_per_page:
            continue
        selected.append(item)
        page_counts[page] = page_counts.get(page, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def _fuse_rrf_by_variant(
    scores: dict[str, tuple[RetrievedChunk, float]],
    result_sets: list[list[RetrievedChunk]],
) -> None:
    """Fuse rankings with rank reset for every independent query variant.

    The old implementation concatenated all variant results first, then assigned
    ranks 1..N. This made rank 1 from variant 2 look like rank N+1. Here the
    enumerate call is intentionally inside the per-variant loop.
    """
    for variant_results in result_sets:
        seen: set[str] = set()
        for rank, result in enumerate(variant_results, start=1):
            cid = result.chunk.metadata.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            existing = scores.get(cid)
            current_score = existing[1] if existing else 0.0
            chunk = existing[0] if existing else result
            scores[cid] = (
                chunk,
                current_score + reciprocal_rank_fusion_score(rank),
            )


_PAGE_INDEX_CACHE: dict[int, dict[tuple[str, int], list[Any]]] = {}


def _page_index(bm25: BM25Retriever) -> dict[tuple[str, int], list[Any]]:
    """Map (doc_id, page) to chunks once, instead of scanning 200k+ chunks."""
    key = id(bm25)
    cached = _PAGE_INDEX_CACHE.get(key)
    if cached is None:
        cached = {}
        for chunk in bm25.chunks:
            meta = chunk.metadata
            if meta.page is None:
                continue
            cached.setdefault((meta.doc_id, meta.page), []).append(chunk)
        _PAGE_INDEX_CACHE[key] = cached
        logger.info("Built page index over %d pages", len(cached))
    return cached


def _preferred_sections(normalized_query: str) -> set[str]:
    sections: set[str] = set()
    if re.search(
        r"income statement|statements? of operations|revenue|gross margin|"
        r"operating income|earnings per share|net income",
        normalized_query,
    ):
        sections.add("Income Statement")
    if re.search(
        r"balance sheet|total assets|total liabilities|inventory|equity|"
        r"working capital",
        normalized_query,
    ):
        sections.add("Balance Sheet")
    if re.search(
        r"cash ?flow|operating activities|capital expenditure|capex|fcf|"
        r"dividend|repurchase",
        normalized_query,
    ):
        sections.add("Cash Flow")
    return sections


def hybrid_search(
    collection: Any,
    bm25: BM25Retriever,
    query: str,
    top_k: int | None = None,
    company_filter: str | None = None,
    fiscal_year_filter: int | None = None,
    evidence_goal: str | None = None,
) -> list[RetrievedChunk]:
    if top_k is None:
        top_k = settings.top_k_hybrid

    explicit_filters = company_filter is not None or fiscal_year_filter is not None
    if company_filter is None or fiscal_year_filter is None:
        inferred_company, inferred_year = bm25.infer_filters(query)
        company_filter = company_filter or inferred_company
        fiscal_year_filter = (
            fiscal_year_filter if fiscal_year_filter is not None else inferred_year
        )
        logger.debug(
            "Inferred retrieval filters: company=%s fiscal_year=%s",
            company_filter,
            fiscal_year_filter,
        )

    query_variants = _query_variants(query)
    dense_sets: list[list[RetrievedChunk]] = []
    bm25_sets: list[list[RetrievedChunk]] = []
    candidate_k = max(settings.top_k_dense, top_k * 5)
    lexical_k = max(settings.top_k_bm25, top_k * 5)
    for variant in query_variants:
        dense_sets.append(
            dense_search(
                collection,
                variant,
                top_k=candidate_k,
                company_filter=company_filter,
                fiscal_year_filter=fiscal_year_filter,
            )
        )
        bm25_sets.append(
            bm25.search(
                variant,
                top_k=lexical_k,
                company_filter=company_filter,
                fiscal_year_filter=fiscal_year_filter,
            )
        )

    # Inferred filters are only ranking aids, not hard user constraints. If a
    # legacy index uses a different company spelling (for example ``Adobe Inc``
    # versus ``Adobe``), retry without inferred filters before declaring that no
    # evidence exists. Explicit API filters remain honored by the first pass.
    if not any(dense_sets) and not any(bm25_sets) and not explicit_filters:
        logger.warning(
            "Filtered retrieval returned no chunks; retrying without inferred filters"
        )
        dense_sets = [
            dense_search(collection, variant, top_k=candidate_k)
            for variant in query_variants
        ]
        bm25_sets = [
            bm25.search(variant, top_k=lexical_k) for variant in query_variants
        ]

    scores: dict[str, tuple[RetrievedChunk, float]] = {}
    _fuse_rrf_by_variant(scores, dense_sets)
    _fuse_rrf_by_variant(scores, bm25_sets)

    normalized_query = query.casefold()
    preferred_sections = _preferred_sections(normalized_query)
    prefers_tables = evidence_goal in (None, "numeric_table") and bool(
        re.search(r"\d|amount|total|margin|cash|revenue|income|ratio", normalized_query)
    )

    def ranking_value(item: tuple[RetrievedChunk, float]) -> float:
        chunk, score = item
        meta = chunk.chunk.metadata
        bonus = 0.0
        if preferred_sections and meta.section in preferred_sections:
            bonus += 0.01
        if prefers_tables and meta.content_type == "table":
            bonus += 0.004
        return score + bonus

    # Context expansion. Section metadata is unreliable on many filings, so the
    # neighbourhood is defined by a page window around each strong anchor: the
    # statement table and its continuation almost always sit on adjacent pages.
    preliminary = sorted(scores.values(), key=ranking_value, reverse=True)[:top_k]
    page_index = _page_index(bm25)
    for anchor, anchor_score in preliminary:
        anchor_meta = anchor.chunk.metadata
        if anchor_meta.page is None:
            continue
        for offset in (0, -1, 1):
            for candidate in page_index.get(
                (anchor_meta.doc_id, anchor_meta.page + offset), []
            ):
                penalty = 1e-4 if offset == 0 else 5e-4
                scores.setdefault(
                    candidate.metadata.chunk_id,
                    (
                        RetrievedChunk(
                            chunk=candidate,
                            score=anchor_score - penalty,
                            source="hybrid",
                        ),
                        anchor_score - penalty,
                    ),
                )

    ranked = sorted(scores.values(), key=ranking_value, reverse=True)

    # RRF is a cheap candidate generator. The local cross-encoder then scores
    # only the best candidates before page diversification and final top-k.
    # This is local model inference and does not consume LLM API quota.
    if settings.use_reranker and ranked:
        rerank_k = min(len(ranked), max(top_k * 5, 50))
        candidate_items = ranked[:rerank_k]
        reranked_chunks = rerank(
            query,
            [chunk for chunk, _ in candidate_items],
            top_k=rerank_k,
        )
        original_scores = {
            chunk.chunk.metadata.chunk_id: score for chunk, score in candidate_items
        }
        reranked_ids = {chunk.chunk.metadata.chunk_id for chunk in reranked_chunks}
        ranked = [
            (chunk, original_scores.get(chunk.chunk.metadata.chunk_id, 0.0))
            for chunk in reranked_chunks
        ] + [
            item
            for item in ranked[rerank_k:]
            if item[0].chunk.metadata.chunk_id not in reranked_ids
        ]

    fused = _diversify_by_page(ranked, top_k)
    results: list[RetrievedChunk] = []
    for chunk, score in fused:
        chunk.source = "hybrid"
        chunk.score = score
        results.append(chunk)

    logger.info(
        "Hybrid search: variants=%d dense=%d bm25=%d candidates=%d fused=%d "
        "filters=(company=%s, fiscal_year=%s) top=%s",
        len(query_variants),
        sum(len(items) for items in dense_sets),
        sum(len(items) for items in bm25_sets),
        len(scores),
        len(results),
        company_filter,
        fiscal_year_filter,
        [
            f"{item.chunk.metadata.doc_id}#p{item.chunk.metadata.page}"
            f":{item.chunk.metadata.content_type}"
            for item in results[:3]
        ],
    )
    return results
