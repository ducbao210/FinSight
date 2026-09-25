"""Pydantic schemas for FinSight."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Document / Chunk


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk."""

    chunk_id: str
    # FinanceBench doc_name (e.g. "3M_2018_10K"); empty for legacy chunks.
    doc_id: str = ""
    company: str
    ticker: str = ""
    doc_type: str = ""
    fiscal_year: int | None = None
    page: int | None = None
    section: str = ""
    source_url: str = ""
    content_type: Literal[
        "table", "narrative", "footnote", "risk_factor", "unknown"
    ] = "narrative"


class Chunk(BaseModel):
    """A text chunk with metadata."""

    metadata: ChunkMetadata
    text: str
    # Short provenance line ("Adobe | 10K FY2022 | page 57 | Cash Flow | ...").
    # It is prepended for embedding and lexical indexing only, so that a bare
    # numeric table still matches a query naming the company, year or statement.
    # Citations and the answer prompt keep using ``text`` unchanged.
    context_header: str = ""

    @property
    def embedding_text(self) -> str:
        """Text used for dense embedding and BM25 indexing."""
        header = (self.context_header or "").strip()
        return f"{header}\n{self.text}" if header else self.text


class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval with a score."""

    chunk: Chunk
    score: float
    source: Literal["dense", "bm25", "hybrid"] = "dense"


# API


class QueryRequest(BaseModel):
    """Request to the /query endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The financial question to answer",
    )
    company: str | None = Field(None, description="Optional company filter")
    fiscal_year: int | None = Field(
        None, ge=1900, le=2100, description="Optional fiscal year filter"
    )
    include_trace: bool = Field(False, description="Include full trace in response")

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Question must not be blank")
        return value

    @field_validator("company")
    @classmethod
    def normalize_company(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        return value or None


class Citation(BaseModel):
    """A citation to a specific chunk."""

    company: str
    filing: str
    source_url: str = ""
    page: int | None = None
    chunk_id: str
    section: str = ""
    text_snippet: str = ""


class CalculationTrace(BaseModel):
    """Trace of a deterministic calculation."""

    formula: str
    inputs: list[dict]
    result: float
    rounding: str = "one decimal place"


class TraceInfo(BaseModel):
    """Full trace of the reasoning process."""

    question_type: str = ""
    question_reasoning: list[str] = []
    requires_calculation: bool = False
    requires_multi_year_comparison: bool = False
    sub_questions: list[str] = []
    retrieval_attempts: int = 0
    grader_verdicts: list[dict] = []
    validation_issues: list[str] = []
    rewritten_queries: list[str] = []
    retry_count: int = 0
    retrieval_log: list[dict] = []


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""

    answer: str
    model_name: str = ""
    status: Literal["answer", "abstain", "error"] = "answer"
    # Keep the exact retrieval output available to evaluators. The evaluation
    # runner uses this to score chunks produced during answering without a
    # second retrieval request.
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    # Exact texts supplied to the answer model; snippets remain UI-only.
    contexts: list[str] = Field(default_factory=list)
    citations: list[Citation] = []
    calculations: list[CalculationTrace] = []
    trace: TraceInfo | None = None


# Evaluation


class FinanceBenchEvidence(BaseModel):
    """One gold evidence span from financebench_open_source.jsonl."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    doc_name: str = ""
    evidence_page_num: int | None = None
    evidence_text: str = ""

    @field_validator("doc_name", "evidence_text", mode="before")
    @classmethod
    def coerce_null_text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("evidence_page_num", mode="before")
    @classmethod
    def coerce_page(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


class FinanceBenchQuestion(BaseModel):
    """One FinanceBench question row."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    financebench_id: str = Field(
        "", validation_alias=AliasChoices("financebench_id", "question_id")
    )
    company: str = ""
    doc_name: str = ""
    question_type: str = ""
    question: str
    answer: str = ""
    justification: str = ""
    evidence: list[FinanceBenchEvidence] = []

    @field_validator(
        "financebench_id",
        "company",
        "doc_name",
        "question_type",
        "answer",
        "justification",
        mode="before",
    )
    @classmethod
    def coerce_null_text(cls, value: Any) -> str:
        """FinanceBench leaves several text fields as null."""
        return "" if value is None else str(value)

    @property
    def gold_doc_names(self) -> list[str]:
        names = [e.doc_name for e in self.evidence if e.doc_name]
        return names or ([self.doc_name] if self.doc_name else [])


class RetrievalMetrics(BaseModel):
    """IR metrics, computed with ranx at document level and page level."""

    doc_level: dict[str, float] = {}
    page_level: dict[str, float] = {}
    num_scored_questions: int = 0


class GenerationMetrics(BaseModel):
    """Ragas generation metrics (None when generation scoring is skipped)."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    answer_correctness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class CitationMetrics(BaseModel):
    """Citation quality metrics."""

    citation_coverage: float = 0.0
    citation_validity: float = 0.0


class EvalSample(BaseModel):
    """Per-question evaluation record."""

    question_id: str
    question: str
    gold_answer: str = ""
    predicted_answer: str = ""
    status: Literal["answer", "abstain", "error"] = "answer"
    gold_doc_keys: list[str] = []
    gold_page_keys: list[str] = []
    retrieved_doc_keys: list[str] = []
    retrieved_page_keys: list[str] = []
    retrieved_chunk_ids: list[str] = []
    # chunk_id -> "<doc_key>#p<page>", used to validate citations.
    chunk_key_index: dict[str, str] = {}
    contexts: list[str] = []
    citations: list[Citation] = []
    citation_coverage: float = 0.0
    citation_validity: float = 0.0
    generation: GenerationMetrics = Field(default_factory=lambda: GenerationMetrics())
    latency_ms: float = 0.0
    retry_count: int = 0
    error: str = ""


class EvalConfig(BaseModel):
    """Validated configuration for one evaluation run."""

    questions_path: Path
    config_name: str = "crag-hybrid"
    limit: int | None = Field(default=None, ge=1)
    shuffle: bool = False
    seed: int = 42
    company: str | None = None
    question_type: str | None = None
    top_k: int = Field(default=10, ge=1)
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    page_offset: int = 1
    concurrency: int = Field(default=4, ge=1)
    skip_generation: bool = False
    run_generation_metrics: bool = True
    run_retrieval_metrics: bool = True
    run_citation_metrics: bool = True


class EvalReport(BaseModel):
    """Aggregate evaluation report."""

    config_name: str = "default"
    dataset: str = "financebench"
    num_questions: int = 0
    k_values: list[int] = [1, 3, 5, 10]
    retrieval: RetrievalMetrics = Field(default_factory=lambda: RetrievalMetrics())
    generation: GenerationMetrics = Field(default_factory=lambda: GenerationMetrics())
    citation: CitationMetrics = Field(default_factory=lambda: CitationMetrics())
    median_latency_ms: float = 0.0
    median_retry_count: float = 0.0
    abstain_rate: float = 0.0
    error_rate: float = 0.0
    samples: list[EvalSample] = []
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# Ingestion


class DocumentInfo(BaseModel):
    """Canonical document metadata, compatible with FinSight and FinanceBench JSONL."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # Internal name is doc_id; FinanceBench calls this doc_name.
    doc_id: str = Field(
        "",
        validation_alias=AliasChoices("doc_id", "doc_name"),
    )
    company: str = "Unknown"
    ticker: str = ""
    # FinanceBench uses doc_type/document_type; keep the canonical internal name.
    doc_type: str = Field(
        "10K",
        validation_alias=AliasChoices("doc_type", "document_type"),
    )
    # Internal name is fiscal_year; FinanceBench calls this doc_period.
    fiscal_year: int | None = Field(
        None,
        validation_alias=AliasChoices("fiscal_year", "doc_period", "year"),
    )
    source_path: str = ""
    source_url: str = Field(
        "",
        validation_alias=AliasChoices("source_url", "doc_link", "url"),
    )
    num_pages: int = 0

    @field_validator("doc_type", mode="before")
    @classmethod
    def normalize_doc_type(cls, value: Any) -> str:
        """Normalize metadata spelling without conflating annual-report filenames."""
        if value is None:
            return "10K"
        value = str(value).strip().upper().replace("-", "_")
        # FinanceBench metadata uses 10k_annualreport; its PDF stem uses annualreport.
        if value == "10K_ANNUALREPORT":
            return "10K_ANNUAL"
        return value

    @model_validator(mode="after")
    def default_ticker(self) -> "DocumentInfo":
        """Derive a usable ticker when FinanceBench metadata omits it."""
        if not self.ticker.strip():
            self.ticker = self.company.upper()
        return self


class IngestionReport(BaseModel):
    """Report from an ingestion run."""

    documents_processed: int
    chunks_created: int
    errors: list[str] = []
    duration_seconds: float = 0.0
