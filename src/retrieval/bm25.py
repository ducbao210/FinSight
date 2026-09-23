from __future__ import annotations
import re

from rank_bm25 import BM25Okapi
from src.core.configs import settings
from src.core.logging import get_logger
from src.schemas import Chunk, RetrievedChunk

logger = get_logger(__name__)


class BM25Retriever:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def index(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(chunk.embedding_text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 indexed %d chunks", len(chunks))

    @property
    def chunks(self) -> list[Chunk]:
        """Return indexed chunks for deterministic local context expansion."""
        return self._chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Preserve finance-specific forms such as PP&E, 10-K and FY2018,
        # while also indexing ordinary words and numeric values.
        normalized = text.casefold().replace("−", "-").replace("–", "-")
        return re.findall(r"[a-z0-9]+(?:[.&/-][a-z0-9]+)*", normalized)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        company_filter: str | None = None,
        fiscal_year_filter: int | None = None,
    ) -> list[RetrievedChunk]:
        if self._bm25 is None:
            logger.warning("BM25 not indexed yet")
            return []
        if top_k is None:
            top_k = settings.top_k_bm25

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        scored = [(idx, score) for idx, score in enumerate(scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        def collect(
            company_constraint: str | None,
            year_constraint: int | None,
        ) -> list[RetrievedChunk]:
            found: list[RetrievedChunk] = []
            for idx, score in scored:
                chunk = self._chunks[idx]
                if (
                    company_constraint
                    and chunk.metadata.company.casefold()
                    != company_constraint.casefold()
                ):
                    continue
                if (
                    year_constraint is not None
                    and chunk.metadata.fiscal_year != year_constraint
                ):
                    continue
                found.append(
                    RetrievedChunk(chunk=chunk, score=float(score), source="bm25")
                )
                if len(found) >= top_k:
                    break
            return found

        retrieved = collect(company_filter, fiscal_year_filter)
        if not retrieved and company_filter and fiscal_year_filter is not None:
            logger.warning(
                "BM25 year filter returned no chunks; retrying with company only "
                "(company=%r, fiscal_year=%r)",
                company_filter,
                fiscal_year_filter,
            )
            retrieved = collect(company_filter, None)
        if not retrieved and not company_filter and fiscal_year_filter is not None:
            logger.warning(
                "BM25 strict filters returned no chunks; retrying without filters "
                "(company=%r, fiscal_year=%r)",
                company_filter,
                fiscal_year_filter,
            )
            retrieved = collect(None, None)
        return retrieved

    def infer_filters(self, query: str) -> tuple[str | None, int | None]:
        """Infer safe metadata filters when the API caller did not provide them.

        FinanceBench questions normally name the company and fiscal year. Applying
        those constraints before ranking prevents high-frequency terms such as
        ``restructuring`` from surfacing another company's filing first.
        """
        normalized_query = query.casefold()
        companies = {
            chunk.metadata.company
            for chunk in self._chunks
            if chunk.metadata.company.strip()
        }
        company_matches = [
            company for company in companies if company.casefold() in normalized_query
        ]
        if not company_matches:
            query_tokens = set(re.findall(r"[a-z0-9]+", normalized_query))
            generic = {
                "the", "inc", "incorporated", "corp", "corporation", "co",
                "company", "limited", "ltd", "plc", "holdings", "group",
            }
            scored_matches = []
            for company in companies:
                company_tokens = set(re.findall(r"[a-z0-9]+", company.casefold())) - generic
                overlap = company_tokens & query_tokens
                if overlap:
                    scored_matches.append((len(overlap), len(company_tokens), company))
            if scored_matches:
                company_matches = [max(scored_matches)[2]]
        company_filter = max(company_matches, key=len) if company_matches else None

        year_matches = sorted(
            {
                int(value)
                for value in re.findall(r"\b(?:fy\s*)?(20\d{2})\b", normalized_query)
            }
        )
        fiscal_year_filter = year_matches[0] if len(year_matches) == 1 else None
        return company_filter, fiscal_year_filter
