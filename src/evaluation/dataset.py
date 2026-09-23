"""Loading and filtering of the FinanceBench evaluation set."""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.core.logging import get_logger
from src.evaluation.keys import DEFAULT_PAGE_OFFSET, gold_doc_key, gold_page_key
from src.ingestion.metadata import build_document_index, load_document_info_from_jsonl
from src.schemas import DocumentInfo, FinanceBenchQuestion

logger = get_logger(__name__)


def load_questions(jsonl_path: Path) -> list[FinanceBenchQuestion]:
    """Read financebench_open_source.jsonl into typed questions."""
    questions: list[FinanceBenchQuestion] = []
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Question file not found: {jsonl_path}")

    with jsonl_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(FinanceBenchQuestion.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.warning("Skipping invalid question at line %d: %s", line_no, exc)

    logger.info("Loaded %d FinanceBench questions from %s", len(questions), jsonl_path)
    return questions


def question_key(question: FinanceBenchQuestion, index: int) -> str:
    """Return the stable key shared by Pass 1, Pass 2, and Ragas."""
    return question.financebench_id or f"q{index:04d}"


def load_documents(jsonl_path: Path) -> dict[str, DocumentInfo]:
    """Read financebench_document_information.jsonl, indexed by doc_name."""
    return build_document_index(load_document_info_from_jsonl(jsonl_path))


def filter_questions(
    questions: list[FinanceBenchQuestion],
    company: str | None = None,
    question_type: str | None = None,
    doc_names: set[str] | None = None,
    require_evidence: bool = True,
) -> list[FinanceBenchQuestion]:
    """Keep only the questions we can actually grade."""
    selected: list[FinanceBenchQuestion] = []
    for question in questions:
        if company and question.company.lower() != company.lower():
            continue
        if question_type and question.question_type != question_type:
            continue
        if require_evidence and not question.gold_doc_names:
            continue
        if doc_names is not None and not set(question.gold_doc_names) & doc_names:
            continue
        selected.append(question)
    return selected


def sample_questions(
    questions: list[FinanceBenchQuestion],
    limit: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
) -> list[FinanceBenchQuestion]:
    """Take a deterministic subset, useful for cheap smoke runs."""
    items = list(questions)
    if shuffle:
        random.Random(seed).shuffle(items)
    if limit is not None and limit > 0:
        items = items[:limit]
    return items


def gold_keys(
    question: FinanceBenchQuestion,
    page_offset: int = DEFAULT_PAGE_OFFSET,
) -> tuple[list[str], list[str]]:
    """Return (document-level keys, page-level keys) of the gold evidence."""
    docs: list[str] = []
    pages: list[str] = []
    for evidence in question.evidence:
        name = evidence.doc_name or question.doc_name
        if not name:
            continue
        key = gold_doc_key(name)
        if key not in docs:
            docs.append(key)
        if evidence.evidence_page_num is not None:
            page = gold_page_key(name, evidence.evidence_page_num, page_offset)
            if page not in pages:
                pages.append(page)
    if not docs and question.doc_name:
        docs.append(gold_doc_key(question.doc_name))
    return docs, pages


def gold_contexts(question: FinanceBenchQuestion) -> list[str]:
    """Gold evidence text, used as the reference context for Ragas context_recall."""
    return [e.evidence_text for e in question.evidence if e.evidence_text.strip()]
