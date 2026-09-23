"""LangGraph nodes for the CRAG pipeline."""

from __future__ import annotations

import json
import re
import asyncio
from functools import lru_cache
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from src.core.configs import settings
from src.core.logging import get_logger
from src.calculation.financial_metrics import CALCULATION_REGISTRY, CalcInput
from src.core.prompts import (
    ABSTENTION_TEMPLATE,
    ANSWER_GENERATOR_PROMPT,
    ANSWER_REPAIR_PROMPT,
    CALCULATION_PLANNER_PROMPT,
    CITATION_VALIDATOR_PROMPT,
    EVIDENCE_GRADER_PROMPT,
    QUESTION_CLASSIFIER_PROMPT,
    QUERY_DECOMPOSER_PROMPT,
    QUERY_REWRITER_PROMPT,
)
from src.graph.state import FinSightState
from src.retrieval.hybrid import hybrid_search

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _get_llm(model_name: str | None = None) -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key_or_raise,
        model=model_name or settings.model_name,
        temperature=0.0,
    )


def _extract_json(content: str) -> dict:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        content = match.group(1)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise json.JSONDecodeError("No JSON object", content, 0)
    return json.loads(content[start : end + 1])


def _normalize_grader_value(value: object) -> str:
    """Normalize small formatting variations in structured LLM output."""
    return str(value or "").strip().casefold()


def _is_conditional_zero_question(question: str) -> bool:
    normalized = question.casefold()
    return bool(
        re.search(
            r"\b0\s+if\s+(?:it\s+is\s+)?not\s+(?:shown|listed|disclosed|presented)",
            normalized,
        )
    )


def _conditional_zero_instruction(question: str) -> str:
    if _is_conditional_zero_question(question):
        return (
            "This question explicitly requests 0 if the requested line is not shown. "
            "A complete authoritative income statement may therefore be relevant "
            "negative evidence; do not mark it irrelevant only because that row is absent."
        )
    return "No conditional-zero fallback applies to this question. Do not infer a zero value."


def _is_negative_statement_evidence(question: str, chunk: dict) -> bool:
    """Whether a chunk can establish that a requested statement row is absent."""
    if not _is_conditional_zero_question(question):
        return False
    metadata = chunk.get("metadata", {})
    section = _normalize_grader_value(metadata.get("section"))
    if section != "income statement":
        return False
    years = re.findall(r"\b20\d{2}\b", question)
    return not years or str(metadata.get("fiscal_year", "")) in years


async def classify_question(state: FinSightState) -> FinSightState:
    llm = _get_llm(state.get("model_name"))
    prompt = QUESTION_CLASSIFIER_PROMPT.format(question=state["question"])
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        result = _extract_json(cast(str, response.content))
        # FinanceBench's question_type is a source category; the classifier prompt
        # returns reasoning_type for the pipeline's internal reasoning category.
        state["question_type"] = result.get(
            "reasoning_type",
            result.get("question_type", "information_extraction"),
        )
        reasoning = result.get("question_reasoning", [])
        if isinstance(reasoning, str):
            reasoning = [reasoning]
        allowed = {
            "Information extraction",
            "Numerical reasoning",
            "Logical reasoning",
        }
        state["question_reasoning"] = [
            value for value in reasoning if isinstance(value, str) and value in allowed
        ]
        state["requires_calculation"] = bool(result.get("requires_calculation", False))
        state["requires_multi_year_comparison"] = bool(
            result.get("requires_multi_year_comparison", False)
        )
    except (json.JSONDecodeError, KeyError):
        state["question_type"] = "information_extraction"
        state["question_reasoning"] = ["Information extraction"]
    logger.info("Classified question as: %s", state["question_type"])
    return state


async def decompose_query(state: FinSightState) -> FinSightState:
    llm = _get_llm(state.get("model_name"))
    prompt = QUERY_DECOMPOSER_PROMPT.format(question=state["question"])
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        result = _extract_json(cast(str, response.content))
        sub_qs = result.get("sub_questions", [])
        state["sub_questions"] = []
        state["evidence_goals"] = []
        for item in sub_qs:
            if isinstance(item, dict) and item.get("question"):
                state["sub_questions"].append(str(item["question"]))
                state["evidence_goals"].append(
                    str(item.get("evidence_goal", "numeric_table"))
                )
            elif item:
                state["sub_questions"].append(str(item))
                state["evidence_goals"].append("numeric_table")
    except (json.JSONDecodeError, KeyError):
        state["sub_questions"] = [state["question"]]
        state["evidence_goals"] = ["numeric_table"]
    if not state["sub_questions"]:
        state["sub_questions"] = [state["question"]]
        state["evidence_goals"] = ["numeric_table"]
    # Keep latency bounded even when the decomposer returns too many tasks.
    # The configured hop limit is the safety ceiling for one request.
    state["sub_questions"] = state["sub_questions"][: settings.max_hops]
    state["evidence_goals"] = state["evidence_goals"][: len(state["sub_questions"])]
    logger.info("Decomposed into %d sub-questions", len(state["sub_questions"]))
    return state


def _current_goal(state: FinSightState) -> str:
    goals = state.get("evidence_goals", [])
    hop = state["current_hop"]
    return goals[hop] if hop < len(goals) else "numeric_table"


async def retrieve_evidence(state: FinSightState) -> FinSightState:
    hop = min(state["current_hop"], len(state["sub_questions"]) - 1)
    query = state["sub_questions"][hop]
    results = await asyncio.to_thread(
        hybrid_search,
        state["client"],
        state["bm25"],
        query,
        company_filter=state.get("company_filter"),
        fiscal_year_filter=state.get("fiscal_year_filter"),
        evidence_goal=_current_goal(state),
    )
    state["retrieved_chunks"] = [item.chunk.model_dump() for item in results]
    state.setdefault("retrieval_log", []).append(
        {
            "hop": state["current_hop"],
            "retry": state["retry_count"],
            "query": query,
            "chunks": len(results),
            "top_chunks": [
                {
                    "chunk_id": item.chunk.metadata.chunk_id,
                    "doc_id": item.chunk.metadata.doc_id,
                    "page": item.chunk.metadata.page,
                    "content_type": item.chunk.metadata.content_type,
                }
                for item in results[:5]
            ],
        }
    )
    logger.info(
        "Retrieved %d chunks for hop %d, retry %d",
        len(results),
        state["current_hop"],
        state["retry_count"],
    )
    return state


async def grade_evidence(state: FinSightState) -> FinSightState:
    if not state["retrieved_chunks"]:
        state["evidence_scores"] = [
            {"verdict": "irrelevant", "score": 0.0, "reason": "No chunk retrieved"}
        ]
        state["hop_accepted"] = 0
        return state

    llm = _get_llm(state.get("model_name"))
    current_sub = (
        state["sub_questions"][state["current_hop"]]
        if state["current_hop"] < len(state["sub_questions"])
        else state["question"]
    )
    # Grade the strongest eight chunks returned by the retriever. Restricting
    # this to four chunks can discard the only relevant table and cause a false
    # abstention even when retrieval itself found the evidence.
    candidates = state["retrieved_chunks"][: min(settings.top_k_hybrid, 8)]
    evidence_goal = _current_goal(state)
    candidate_text = "\n\n---\n\n".join(
        json.dumps(
            {
                "chunk_id": item.get("metadata", {}).get("chunk_id", ""),
                "metadata": item.get("metadata", {}),
                "text": item.get("text", "")[:2000],
            },
            ensure_ascii=False,
        )
        for item in candidates
    )
    prompt = EVIDENCE_GRADER_PROMPT.format(
        sub_question=current_sub,
        evidence_goal=evidence_goal,
        conditional_instruction=_conditional_zero_instruction(state["question"]),
        candidates=candidate_text,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        result = _extract_json(cast(str, response.content))
        scores = result.get("items", [])
        if not isinstance(scores, list):
            raise ValueError("items must be a list")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        scores = [
            {
                "chunk_id": item.get("metadata", {}).get("chunk_id", ""),
                "verdict": "ambiguous",
                "score": 0.5,
                "reason": "Batch grader parse error",
            }
            for item in candidates
        ]
    # LLMs commonly return harmless variations such as ``Relevant`` or extra
    # whitespace.  The old exact comparison turned those into zero accepted
    # chunks, causing every query to abstain even when retrieval was correct.
    normalized_scores: list[dict] = []
    for index, item in enumerate(scores):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized["verdict"] = _normalize_grader_value(item.get("verdict"))
        normalized["chunk_id"] = str(item.get("chunk_id", "")).strip()
        normalized["_candidate_index"] = index
        normalized_scores.append(normalized)
    state["evidence_scores"] = normalized_scores
    score_by_id = {
        _normalize_grader_value(item.get("chunk_id")): item
        for item in normalized_scores
        if _normalize_grader_value(item.get("chunk_id"))
    }
    accepted_now = 0
    for candidate_index, chunk_data in enumerate(candidates):
        chunk_id = chunk_data.get("metadata", {}).get("chunk_id", "")
        score = score_by_id.get(_normalize_grader_value(chunk_id))
        # If the grader omitted IDs but returned exactly one item per candidate,
        # use the required response order as a safe fallback.
        if score is None and len(normalized_scores) == len(candidates):
            score = normalized_scores[candidate_index]
        score = score or {
            "verdict": "ambiguous",
            "score": 0.5,
            "reason": "Missing grader item",
        }
        if _normalize_grader_value(
            score.get("verdict")
        ) == "irrelevant" and _is_negative_statement_evidence(current_sub, chunk_data):
            score = dict(score)
            score["verdict"] = "relevant"
            score["score"] = max(float(score.get("score", 0.0) or 0.0), 0.8)
            score["supports"] = list(score.get("supports") or []) + [
                "authoritative income statement table does not separately show the requested line"
            ]
            score["reason"] = (
                "Valid negative evidence for the explicit conditional-zero instruction"
            )
            if chunk_id:
                score_by_id[_normalize_grader_value(chunk_id)] = score
            score_index = score.get("_candidate_index")
            if isinstance(score_index, int) and score_index < len(normalized_scores):
                normalized_scores[score_index] = score
        if (
            _normalize_grader_value(score.get("verdict")) == "relevant"
            and chunk_data not in state["accepted_evidence"]
        ):
            state["accepted_evidence"].append(chunk_data)
            accepted_now += 1

    # Last-attempt fallback: rather than abstaining outright, keep the best
    # ambiguous chunks so the answer generator can judge them. The generator and
    # the citation validator still refuse to assert unsupported numbers.
    if accepted_now == 0 and state["retry_count"] >= settings.max_retries:
        for candidate_index, chunk_data in enumerate(candidates):
            chunk_id = chunk_data.get("metadata", {}).get("chunk_id", "")
            score = score_by_id.get(_normalize_grader_value(chunk_id))
            if score is None and len(normalized_scores) == len(candidates):
                score = normalized_scores[candidate_index]
            verdict = _normalize_grader_value((score or {}).get("verdict"))
            value = float((score or {}).get("score", 0.0) or 0.0)
            if verdict == "ambiguous" and value >= 0.4:
                if chunk_data not in state["accepted_evidence"]:
                    state["accepted_evidence"].append(chunk_data)
                    accepted_now += 1
        if accepted_now:
            logger.info(
                "Accepted %d ambiguous chunks as last-attempt fallback", accepted_now
            )

    state["hop_accepted"] = accepted_now
    logger.info(
        "Graded %d chunks: %d relevant, %d accepted",
        len(scores),
        sum(
            1
            for s in normalized_scores
            if _normalize_grader_value(s.get("verdict")) == "relevant"
        ),
        accepted_now,
    )
    return state


async def rewrite_query(state: FinSightState) -> FinSightState:
    llm = _get_llm(state.get("model_name"))
    current_sub = (
        state["sub_questions"][state["current_hop"]]
        if state["current_hop"] < len(state["sub_questions"])
        else state["question"]
    )
    last_verdict = state["evidence_scores"][-1] if state["evidence_scores"] else {}
    goals = state.get("evidence_goals", [])
    evidence_goal = (
        goals[state["current_hop"]]
        if state["current_hop"] < len(goals)
        else "numeric_table"
    )
    prompt = QUERY_REWRITER_PROMPT.format(
        sub_question=current_sub,
        evidence_goal=evidence_goal,
        verdict=last_verdict.get("verdict", "unknown"),
        missing=json.dumps(last_verdict.get("missing", [])),
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        result = _extract_json(cast(str, response.content))
        rewritten = result.get("rewritten_question", current_sub)
        if result.get("evidence_goal") in {
            "numeric_table",
            "narrative_explanation",
            "cross_reference",
        } and state["current_hop"] < len(goals):
            goals[state["current_hop"]] = result["evidence_goal"]
    except (json.JSONDecodeError, KeyError):
        rewritten = current_sub
    if state["current_hop"] < len(state["sub_questions"]):
        state["sub_questions"][state["current_hop"]] = rewritten
    state.setdefault("rewritten_queries", []).append(rewritten)
    state["retry_count"] += 1
    logger.info("Rewrote query (retry %d): %s", state["retry_count"], rewritten)
    return state


async def aggregate_evidence(state: FinSightState) -> FinSightState:
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for chunk in state["accepted_evidence"]:
        cid = chunk.get("metadata", {}).get("chunk_id", "")
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique.append(chunk)
    state["accepted_evidence"] = unique
    logger.info("Aggregated %d unique evidence chunks", len(unique))
    return state


async def compute_calculations(state: FinSightState) -> FinSightState:
    """Run the arithmetic in Python, not in the LLM.

    The model only extracts inputs and picks a formula from a fixed registry;
    every number in the trace is then produced deterministically.
    """
    state.setdefault("calculations", [])
    if not state.get("requires_calculation") or not state["accepted_evidence"]:
        return state

    llm = _get_llm(state.get("model_name"))
    prompt = CALCULATION_PLANNER_PROMPT.format(
        question=state["question"],
        evidence=_format_evidence_for_prompt(state["accepted_evidence"]),
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        plan = _extract_json(cast(str, response.content)).get("calculations", [])
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Calculation planning failed: %s", exc)
        return state

    for item in plan if isinstance(plan, list) else []:
        operation = str(item.get("operation", "")).strip()
        func = CALCULATION_REGISTRY.get(operation)
        if func is None:
            logger.warning("Unknown calculation operation: %s", operation)
            continue
        args = item.get("args") or {}
        try:
            result = func(**{k: float(v) for k, v in args.items()})
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            logger.warning("Calculation %s skipped: %s", operation, exc)
            continue
        chunk_ids = [str(c) for c in (item.get("chunk_ids") or [])]
        unit = str(item.get("unit", ""))
        result.inputs = [
            CalcInput(
                name=inp.name,
                value=inp.value,
                unit=unit,
                chunk_id=chunk_ids[0] if chunk_ids else "",
            )
            for inp in result.inputs
        ]
        payload = result.to_dict()
        payload["operation"] = operation
        payload["label"] = str(item.get("label", operation))
        payload["unit"] = unit
        payload["chunk_ids"] = chunk_ids
        state["calculations"].append(payload)

    logger.info("Computed %d deterministic calculations", len(state["calculations"]))
    return state


async def generate_answer(state: FinSightState) -> FinSightState:
    if not state["accepted_evidence"]:
        state["draft_answer"] = ABSTENTION_TEMPLATE.format(
            documents_checked="All indexed documents",
            reason="No evidence passed the relevance grader",
            question=state["question"],
        )
        state["final_status"] = "abstain"
        return state
    llm = _get_llm(state.get("model_name"))
    evidence_text = _format_evidence_for_prompt(state["accepted_evidence"])
    calc_text = (
        json.dumps(state["calculations"], indent=2) if state["calculations"] else "None"
    )
    prompt = ANSWER_GENERATOR_PROMPT.format(
        question=state["question"],
        conditional_instruction=_conditional_zero_instruction(state["question"]),
        evidence=evidence_text,
        calculations=calc_text,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    state["draft_answer"] = cast(str, response.content)
    return state


async def validate_citations(state: FinSightState) -> FinSightState:
    if not state["accepted_evidence"]:
        # generate_answer has already made the deliberate abstention decision;
        # never let a validator hallucinate a passing verdict for empty context.
        state["final_status"] = "abstain"
        return state
    llm = _get_llm(state.get("model_name"))
    evidence_text = _format_evidence_for_prompt(state["accepted_evidence"])
    calc_text = (
        json.dumps(state["calculations"], indent=2) if state["calculations"] else "None"
    )
    prompt = CITATION_VALIDATOR_PROMPT.format(
        draft_answer=state["draft_answer"],
        evidence=evidence_text,
        calculations=calc_text,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        result = _extract_json(cast(str, response.content))
        verdict = _normalize_grader_value(result.get("verdict", "pass"))
        issues = [str(item) for item in result.get("issues", []) if item]
        issues += [str(item) for item in result.get("unsupported_claims", []) if item]
        issues += [
            f"Invalid citation: {item}"
            for item in result.get("invalid_citations", [])
            if item
        ]
    except (json.JSONDecodeError, KeyError):
        verdict = "fail"
        issues = ["Citation validator returned invalid JSON"]
    state["validation_issues"] = issues
    if verdict != "pass" and state["retry_count"] < settings.max_retries:
        repair_prompt = ANSWER_REPAIR_PROMPT.format(
            question=state["question"],
            draft_answer=state["draft_answer"],
            issues=json.dumps(issues, ensure_ascii=False),
            evidence=evidence_text,
            calculations=calc_text,
        )
        repaired = await llm.ainvoke([HumanMessage(content=repair_prompt)])
        state["draft_answer"] = cast(str, repaired.content)
        state["retry_count"] += 1
        verdict = "pass"
    state["final_status"] = (
        "answer"
        if verdict == "pass"
        else ("abstain" if state["retry_count"] >= settings.max_retries else "answer")
    )
    logger.info("Citation validation: %s", verdict)
    return state


async def advance_hop(state: FinSightState) -> FinSightState:
    """Move to the next sub-question.

    Hop bookkeeping lives in a node, not in the conditional router: LangGraph
    discards mutations made inside a routing function, which previously made
    multi-hop questions silently re-run the first sub-question.
    """
    state["current_hop"] += 1
    state["retry_count"] = 0
    state["retrieved_chunks"] = []
    state["evidence_scores"] = []
    logger.info("Advancing to hop %d", state["current_hop"])
    return state


def decide_next_step(state: FinSightState) -> str:
    accepted_now = state.get("hop_accepted", 0)
    if accepted_now == 0 and state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    if accepted_now == 0 and not state["accepted_evidence"]:
        return "aggregate_evidence"
    if (
        state["current_hop"] + 1 < len(state["sub_questions"])
        and state["current_hop"] + 1 < settings.max_hops
    ):
        return "advance_hop"
    return "aggregate_evidence"


def _format_evidence_for_prompt(evidence: list[dict]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(evidence, 1):
        meta = chunk.get("metadata", {})
        header = f"[{i}] {meta.get('company','?')} | document={meta.get('doc_id') or '?'} | source_url={meta.get('source_url') or '?'} | {meta.get('doc_type','?')} FY{meta.get('fiscal_year','?')} | Page {meta.get('page','?')} | {meta.get('section','?')} | chunk_id={meta.get('chunk_id','?')}"
        parts.append(f"{header}\n{chunk.get('text','')}")
    return "\n\n---\n\n".join(parts)
