"""Citation metrics.

citation_coverage : share of answer claims (sentences carrying a figure, or all
                    sentences when the answer has no numbers) that are backed by
                    at least one resolvable citation marker.
citation_validity : share of cited chunk_ids that really point at the gold
                    evidence (same document, and same page when the gold
                    evidence gives a page number).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.schemas import Citation, CitationMetrics

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[$])|\n+")
_NUMBER = re.compile(r"\d")
# Accepts [1], [1,2], [c3], [ticker_2018_10K_p012_t000]
_MARKER = re.compile(r"\[([^\[\]]{1,120})\]")


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def _resolve_marker(token: str, citations: Sequence[Citation]) -> Citation | None:
    """Map one marker token to a citation, by 1-based index or by chunk_id."""
    token = token.strip().lstrip("^#cC ").strip()
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < len(citations):
            return citations[index]
        return None
    for citation in citations:
        if citation.chunk_id and citation.chunk_id.lower() == token.lower():
            return citation
    return None


def _markers(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _MARKER.finditer(text or ""):
        tokens.extend(part for part in re.split(r"[;,]", match.group(1)) if part.strip())
    return tokens


def compute_citation_coverage(answer: str, citations: Sequence[Citation]) -> float:
    """Fraction of claim sentences that carry at least one resolvable citation."""
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    numeric = [s for s in sentences if _NUMBER.search(s)]
    claims = numeric or sentences

    covered = 0
    for sentence in claims:
        if any(_resolve_marker(token, citations) for token in _markers(sentence)):
            covered += 1
    return covered / len(claims)


def compute_citation_validity(
    answer: str,
    citations: Sequence[Citation],
    retrieved_chunk_ids: Sequence[str],
    gold_doc_keys: Sequence[str],
    gold_page_keys: Sequence[str],
    chunk_key_index: dict[str, str] | None = None,
) -> float:
    """Fraction of cited chunk_ids that exist and land on gold evidence.

    `chunk_key_index` maps a retrieved chunk_id to its "<doc_key>#p<page>" key;
    it is the reliable way to resolve a citation, because chunk ids are built
    from the ticker while FinanceBench gold uses doc_name.
    """
    from src.evaluation.keys import normalize_doc_name, page_key

    chunk_key_index = chunk_key_index or {}

    cited: list[Citation] = []
    for token in _markers(answer):
        citation = _resolve_marker(token, citations)
        if citation and citation not in cited:
            cited.append(citation)
    if not cited:
        return 0.0

    known_ids = {cid for cid in retrieved_chunk_ids if cid}
    gold_docs = set(gold_doc_keys)
    gold_pages = set(gold_page_keys)

    valid = 0
    for citation in cited:
        if citation.chunk_id and known_ids and citation.chunk_id not in known_ids:
            continue

        resolved = chunk_key_index.get(citation.chunk_id, "")
        if resolved:
            doc_key, _, page_part = resolved.partition("#p")
            page = None if page_part in ("", "?") else int(page_part)
        else:
            doc_key = _doc_key_from_chunk_id(citation.chunk_id) or normalize_doc_name(
                citation.filing
            )
            page = citation.page

        if gold_docs and doc_key and doc_key not in gold_docs:
            continue
        if gold_pages and page_key(doc_key, page) not in gold_pages:
            continue
        valid += 1
    return valid / len(cited)


def _doc_key_from_chunk_id(chunk_id: str) -> str:
    """chunk ids look like `mmm_2018_10K_p012_t000`; strip the page/part suffix."""
    from src.evaluation.keys import normalize_doc_name

    if not chunk_id:
        return ""
    stem = re.split(r"_p\d{1,4}(?:_|$)", chunk_id)[0]
    return normalize_doc_name(stem)


def aggregate_citation_metrics(
    coverages: Sequence[float], validities: Sequence[float]
) -> CitationMetrics:
    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return CitationMetrics(
        citation_coverage=mean(coverages),
        citation_validity=mean(validities),
    )
