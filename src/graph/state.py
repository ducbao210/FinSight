"""LangGraph state definition for CRAG pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class FinSightState(TypedDict):
    """State for the CRAG reasoning graph."""

    question: str
    question_type: str
    question_reasoning: list[str]
    requires_calculation: bool
    requires_multi_year_comparison: bool
    sub_questions: list[str]
    evidence_goals: list[str]
    current_hop: int
    retrieved_chunks: list[dict]
    accepted_evidence: list[dict]
    evidence_scores: list[dict]
    calculations: list[dict]
    draft_answer: str
    validation_issues: list[str]
    citations: list[dict]
    rewritten_queries: list[str]
    retry_count: int
    hop_accepted: int
    retrieval_log: list[dict]
    final_status: str
    company_filter: str | None
    fiscal_year_filter: int | None
    client: Any
    bm25: Any
    model_name: str
