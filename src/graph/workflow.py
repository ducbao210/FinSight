from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.core.logging import get_logger
from src.graph.nodes import (
    advance_hop,
    aggregate_evidence,
    compute_calculations,
    classify_question,
    decide_next_step,
    decompose_query,
    generate_answer,
    grade_evidence,
    retrieve_evidence,
    rewrite_query,
    validate_citations,
)
from src.graph.state import FinSightState
from src.retrieval.bm25 import BM25Retriever
from src.schemas import (
    Citation,
    Chunk,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    TraceInfo,
)

logger = get_logger(__name__)


def build_crag_graph() -> StateGraph:
    workflow = StateGraph(FinSightState)
    workflow.add_node("classify_question", classify_question)
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("retrieve_evidence", retrieve_evidence)
    workflow.add_node("grade_evidence", grade_evidence)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("advance_hop", advance_hop)
    workflow.add_node("aggregate_evidence", aggregate_evidence)
    workflow.add_node("compute_calculations", compute_calculations)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("validate_citations", validate_citations)
    workflow.set_entry_point("classify_question")
    workflow.add_edge("classify_question", "decompose_query")
    workflow.add_edge("decompose_query", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "grade_evidence")
    workflow.add_conditional_edges(
        "grade_evidence",
        decide_next_step,
        {
            "rewrite_query": "rewrite_query",
            "advance_hop": "advance_hop",
            "aggregate_evidence": "aggregate_evidence",
            "generate_answer": "generate_answer",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve_evidence")
    workflow.add_edge("advance_hop", "retrieve_evidence")
    workflow.add_edge("aggregate_evidence", "compute_calculations")
    workflow.add_edge("compute_calculations", "generate_answer")
    workflow.add_edge("generate_answer", "validate_citations")
    workflow.add_edge("validate_citations", END)
    return workflow.compile()


class CRAGPipeline:
    """Corrective RAG pipeline using LangGraph for multi-hop reasoning."""

    def __init__(
        self,
        client: object,
        bm25: BM25Retriever,
        model_name: str = "llama-3.1-8b-instant",
    ) -> None:
        self._client = client
        self._bm25 = bm25
        self._model_name = model_name
        self._graph = build_crag_graph()

    @property
    def name(self) -> str:
        return "CRAG"

    async def answer(
        self, request: QueryRequest, request_id: str = "unknown"
    ) -> QueryResponse:
        initial_state: FinSightState = {
            "question": request.question,
            "question_type": "",
            "question_reasoning": [],
            "requires_calculation": False,
            "requires_multi_year_comparison": False,
            "sub_questions": [],
            "evidence_goals": [],
            "current_hop": 0,
            "retrieved_chunks": [],
            "accepted_evidence": [],
            "evidence_scores": [],
            "calculations": [],
            "draft_answer": "",
            "validation_issues": [],
            "citations": [],
            "rewritten_queries": [],
            "retry_count": 0,
            "hop_accepted": 0,
            "retrieval_log": [],
            "final_status": "",
            "company_filter": request.company,
            "fiscal_year_filter": request.fiscal_year,
            "client": self._client,
            "bm25": self._bm25,
            "model_name": self._model_name,
        }
        try:
            result = await self._graph.ainvoke(initial_state)
        except Exception as exc:
            logger.exception("CRAG graph execution failed (request_id=%s)", request_id)
            if (
                type(exc).__name__ == "AuthenticationError"
                or getattr(exc, "status_code", None) == 401
            ):
                message = (
                    "LLM authentication failed: GROQ_API_KEY is invalid or expired. "
                    "Update the key in .env and restart the backend."
                )
            else:
                message = (
                    "Unable to generate an answer at this time. "
                    f"Please retry later. Request ID: {request_id}"
                )
            return QueryResponse(
                answer=message,
                status="error",
                citations=[],
                calculations=[],
            )

        citations = self._build_citations(result.get("accepted_evidence", []))
        trace = None
        if request.include_trace:
            trace = TraceInfo(
                question_type=result.get("question_type", ""),
                question_reasoning=result.get("question_reasoning", []),
                requires_calculation=result.get("requires_calculation", False),
                requires_multi_year_comparison=result.get(
                    "requires_multi_year_comparison", False
                ),
                sub_questions=result.get("sub_questions", []),
                retrieval_attempts=result.get("retry_count", 0) + 1,
                grader_verdicts=result.get("evidence_scores", []),
                validation_issues=result.get("validation_issues", []),
                rewritten_queries=result.get("rewritten_queries", []),
                retry_count=result.get("retry_count", 0),
                retrieval_log=result.get("retrieval_log", []),
            )
        return QueryResponse(
            answer=result.get("draft_answer", "No answer generated"),
            status=result.get("final_status", "error"),
            retrieved_chunks=[
                RetrievedChunk(chunk=Chunk.model_validate(item), score=0.0, source="hybrid")
                for item in result.get("retrieved_chunks", [])
            ],
            contexts=[item.get("text", "") for item in result.get("accepted_evidence", [])],
            citations=citations,
            calculations=result.get("calculations", []),
            trace=trace,
        )

    def _build_citations(self, evidence: list[dict]) -> list[Citation]:
        citations: list[Citation] = []
        for chunk in evidence[:5]:
            meta = chunk.get("metadata", {})
            citations.append(
                Citation(
                    company=meta.get("company", "Unknown"),
                    filing=meta.get("doc_id")
                    or f"{meta.get('fiscal_year','?')} {meta.get('doc_type','?')}",
                    source_url=meta.get("source_url", ""),
                    page=meta.get("page"),
                    chunk_id=meta.get("chunk_id", ""),
                    section=meta.get("section", ""),
                    text_snippet=chunk.get("text", "")[:200],
                )
            )
        return citations
