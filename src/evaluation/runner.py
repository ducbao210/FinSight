"""Evaluation runner: executes the pipeline over FinanceBench and scores it."""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Sequence
from typing import Protocol

from src.core.logging import get_logger
from src.evaluation.citation_metrics import (
    aggregate_citation_metrics,
    compute_citation_coverage,
    compute_citation_validity,
)
from src.evaluation.dataset import (
    filter_questions,
    gold_contexts,
    gold_keys,
    load_questions,
    question_key,
    sample_questions,
)
from src.evaluation.keys import doc_key_from_metadata, page_key_from_metadata
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.schemas import (
    Citation,
    EvalConfig,
    EvalReport,
    EvalSample,
    FinanceBenchQuestion,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

logger = get_logger(__name__)


class Answerer(Protocol):
    """Anything that can answer a question (the CRAG pipeline, or a baseline)."""

    async def answer(self, request: QueryRequest) -> QueryResponse: ...


class Retriever(Protocol):
    """Retrieval-only entry point, used when generation scoring is skipped."""

    def __call__(self, question: str, top_k: int) -> list[RetrievedChunk]: ...


def _retrieved_keys(
    chunks: Sequence[RetrievedChunk],
) -> tuple[list[str], list[str], list[str], list[str], dict[str, str]]:
    doc_keys: list[str] = []
    page_keys: list[str] = []
    chunk_ids: list[str] = []
    contexts: list[str] = []
    index: dict[str, str] = {}
    for item in chunks:
        meta = item.chunk.metadata
        doc_keys.append(doc_key_from_metadata(meta))
        page_keys.append(page_key_from_metadata(meta))
        chunk_ids.append(meta.chunk_id)
        contexts.append(item.chunk.text)
        if meta.chunk_id:
            index[meta.chunk_id] = page_key_from_metadata(meta)
    return doc_keys, page_keys, chunk_ids, contexts, index


def _citation_keys(
    citations: Sequence[Citation],
) -> tuple[list[str], list[str], list[str], list[str], dict[str, str]]:
    """Build evaluation keys from the exact citations used by the answerer."""
    doc_keys: list[str] = []
    page_keys: list[str] = []
    chunk_ids: list[str] = []
    contexts: list[str] = []
    index: dict[str, str] = {}
    for citation in citations:
        doc_key = doc_key_from_metadata(citation_to_metadata(citation))
        page_key = (
            f"{doc_key}#p{citation.page}"
            if citation.page is not None
            else f"{doc_key}#p?"
        )
        doc_keys.append(doc_key)
        page_keys.append(page_key)
        chunk_ids.append(citation.chunk_id)
        contexts.append(citation.text_snippet)
        if citation.chunk_id:
            index[citation.chunk_id] = page_key
    return doc_keys, page_keys, chunk_ids, contexts, index


def citation_to_metadata(citation: Citation):
    """Create the minimal metadata needed by the canonical key helper."""
    from src.schemas import ChunkMetadata

    return ChunkMetadata(
        chunk_id=citation.chunk_id,
        doc_id=citation.filing,
        company=citation.company,
        page=citation.page,
        section=citation.section,
        source_url=citation.source_url,
    )


def _sample_from_question(
    question: FinanceBenchQuestion, config: EvalConfig, index: int
) -> EvalSample:
    docs, pages = gold_keys(question, config.page_offset)
    return EvalSample(
        question_id=question_key(question, index),
        question=question.question,
        gold_answer=question.answer,
        gold_doc_keys=docs,
        gold_page_keys=pages,
    )


async def _run_one(
    question: FinanceBenchQuestion,
    config: EvalConfig,
    answerer: Answerer | None,
    retriever: Retriever | None,
    index: int,
) -> EvalSample:
    sample = _sample_from_question(question, config, index)
    start = time.perf_counter()

    try:
        if config.skip_generation or answerer is None:
            if retriever is None:
                raise ValueError("A retriever is required when generation is skipped")
            chunks = await asyncio.to_thread(retriever, question.question, config.top_k)
            docs, pages, ids, contexts, index = _retrieved_keys(chunks)
            sample.retrieved_doc_keys = docs
            sample.retrieved_page_keys = pages
            sample.retrieved_chunk_ids = ids
            sample.contexts = contexts
            sample.chunk_key_index = index
            sample.status = "answer" if chunks else "abstain"
        else:
            response = await answerer.answer(
                QueryRequest(question=question.question, include_trace=True)
            )
            sample.predicted_answer = response.answer
            sample.status = response.status
            sample.citations = response.citations
            sample.retry_count = response.trace.retry_count if response.trace else 0
            # Use the exact retrieval result produced during answering; do not retrieve twice.
            if response.retrieved_chunks:
                docs, pages, ids, contexts, index = _retrieved_keys(
                    response.retrieved_chunks
                )
                contexts = response.contexts or contexts
            else:
                docs, pages, ids, contexts, index = _citation_keys(response.citations)
                contexts = response.contexts or contexts
            sample.retrieved_doc_keys = docs
            sample.retrieved_page_keys = pages
            sample.retrieved_chunk_ids = ids
            sample.contexts = [context for context in contexts if context]
            sample.chunk_key_index = index
    except Exception as exc:
        logger.error("Question %s failed: %s", sample.question_id, exc)
        sample.status = "error"
        sample.error = str(exc)

    sample.latency_ms = (time.perf_counter() - start) * 1000

    if sample.status == "answer":
        sample.citation_coverage = compute_citation_coverage(
            sample.predicted_answer, sample.citations
        )
        sample.citation_validity = compute_citation_validity(
            sample.predicted_answer,
            sample.citations,
            sample.retrieved_chunk_ids,
            sample.gold_doc_keys,
            sample.gold_page_keys,
            sample.chunk_key_index,
        )
    return sample


async def run_evaluation(
    config: EvalConfig,
    answerer: Answerer | None = None,
    retriever: Retriever | None = None,
) -> EvalReport:
    """Run the pipeline over the selected questions and build the report."""
    questions = filter_questions(
        load_questions(config.questions_path),
        company=config.company,
        question_type=config.question_type,
    )
    questions = sample_questions(questions, config.limit, config.seed, config.shuffle)
    if not questions:
        raise ValueError("No questions selected - check your filters")

    logger.info("Evaluating %d questions (%s)", len(questions), config.config_name)
    semaphore = asyncio.Semaphore(max(1, config.concurrency))

    async def guarded(index: int, question: FinanceBenchQuestion) -> EvalSample:
        async with semaphore:
            return await _run_one(question, config, answerer, retriever, index)

    samples = await asyncio.gather(*(guarded(i, q) for i, q in enumerate(questions)))

    report = EvalReport(
        config_name=config.config_name,
        num_questions=len(samples),
        k_values=list(config.k_values),
        samples=list(samples),
    )
    if config.run_retrieval_metrics:
        report.retrieval = compute_retrieval_metrics(samples, config.k_values)

    if not config.skip_generation and config.run_generation_metrics:
        from src.evaluation.generation_metrics import compute_generation_metrics

        try:
            report.generation = compute_generation_metrics(
                samples,
                references={
                    question_key(q, i): q.answer for i, q in enumerate(questions)
                },
                reference_contexts={
                    question_key(q, i): gold_contexts(q)
                    for i, q in enumerate(questions)
                },
            )
        except Exception as exc:
            logger.error("Ragas scoring failed: %s", exc)

    if config.run_citation_metrics:
        answered = [s for s in samples if s.status == "answer"]
        report.citation = aggregate_citation_metrics(
            [s.citation_coverage for s in answered],
            [s.citation_validity for s in answered],
        )
    answered = [s for s in samples if s.status == "answer"]
    report.median_latency_ms = (
        statistics.median([s.latency_ms for s in answered]) if answered else 0.0
    )
    report.median_retry_count = (
        statistics.median([s.retry_count for s in answered]) if answered else 0.0
    )
    report.abstain_rate = sum(s.status == "abstain" for s in samples) / len(samples)
    report.error_rate = sum(s.status == "error" for s in samples) / len(samples)
    return report
