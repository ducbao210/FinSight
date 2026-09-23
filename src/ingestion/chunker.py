from __future__ import annotations

import hashlib
import re
from typing import Literal, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.logging import get_logger
from src.schemas import Chunk, ChunkMetadata, DocumentInfo

logger = get_logger(__name__)
ContentType = Literal["table", "narrative", "footnote", "risk_factor", "unknown"]


SECTION_PATTERNS = [
    (
        re.compile(
            r"^\s*(?:PART\s+I\s*[,.:\-]?\s*)?ITEM\s+1\s*[,.:\-]?\s+BUSINESS\s*[.:-]?\s*$",
            re.I,
        ),
        "Business",
    ),
    (
        re.compile(
            r"^\s*(?:PART\s+I\s*[,.:\-]?\s*)?ITEM\s+1A\s*[,.:\-]?\s+RISK\s+FACTORS?\s*[.:-]?\s*$",
            re.I,
        ),
        "Risk Factors",
    ),
    (
        re.compile(
            r"^\s*(?:PART\s+II\s*[,.:\-]?\s*)?ITEM\s+7A?\s*[,.:\-]?\s+MANAGEMENT[’'`]?S\s+DISCUSSION(?:\s+AND)?\s+ANALYSIS.*\s*[.:-]?\s*$",
            re.I,
        ),
        "Management's Discussion and Analysis",
    ),
    (
        re.compile(
            r"^\s*(?:PART\s+II\s*[,.:\-]?\s*)?ITEM\s+8\s*[,.:\-]?\s+FINANCIAL\s+STATEMENTS?(?:\s+AND\s+SUPPLEMENT(?:ARY|AL)\s+DATA)?\s*[.:-]?\s*$",
            re.I,
        ),
        "Financial Statements",
    ),
    (
        re.compile(
            r"^\s*(?:CONSOLIDATED\s+)?STATEMENTS?\s+OF\s+(?:OPERATIONS|INCOME|EARNINGS|LOSS)(?:\s+AND\s+COMPREHENSIVE\s+(?:INCOME|LOSS))?.*$",
            re.I,
        ),
        "Income Statement",
    ),
    (
        re.compile(
            r"^\s*(?:CONSOLIDATED\s+)?BALANCE\s+SHEETS?(?:\s+(?:AT|AS\s+OF|FOR)\s+.*)?\s*$",
            re.I,
        ),
        "Balance Sheet",
    ),
    (
        re.compile(
            r"^\s*(?:CONSOLIDATED\s+)?STATEMENTS?\s+OF\s+CASH\s+FLOWS?.*$", re.I
        ),
        "Cash Flow",
    ),
    (
        re.compile(
            r"^\s*NOTES?\s+TO\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS?\s*[.:-]?\s*$",
            re.I,
        ),
        "Footnotes",
    ),
    (re.compile(r"^\s*RISK\s+FACTORS?\s*[.:-]?\s*$", re.I), "Risk Factors"),
]


def normalize_line(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    # PyMuPDF Layout emits semantic headings as Markdown. Detection patterns
    # intentionally continue to operate on the plain heading text.
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text)
    # Headings are frequently emitted as bold text, or as the first cell of a
    # markdown table row. Both used to defeat the anchored patterns below and
    # were the main reason 89% of indexed chunks had section "Unknown".
    text = text.strip().strip("|").strip()
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(
        r"(?i)\b(factor|statement|financial|consolidated|management)\s+s\b",
        r"\1s",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def detect_heading(line: str, page_num: int) -> str | None:
    line = normalize_line(line)
    if not line or len(line) > 180:
        return None
    # Ignore table-of-contents entries: dot leaders + printed page number.
    if page_num <= 18 and re.search(r"(?:\.{2,}|\s)\d{1,4}\s*$", line):
        return None
    if re.search(r"\bfor\s+obligor\s+group\b", line, re.I):
        return None
    for pattern, section in SECTION_PATTERNS:
        if pattern.match(line):
            return section
    return None


# Content fallbacks: a statement table often carries no heading on its own page
# (the title sits on the previous page or inside an image). These row labels are
# specific enough to identify the statement from the table body itself.
CONTENT_SECTION_PATTERNS = [
    (
        re.compile(
            r"(?i)cash\s+flows?\s+from\s+operating\s+activities"
            r"|net\s+cash\s+(?:provided\s+by|used\s+(?:in|for))\s+operating\s+activities"
        ),
        "Cash Flow",
    ),
    (
        re.compile(
            r"(?i)total\s+(?:current\s+)?liabilities\s+and\s+(?:stockholders|shareholders)"
            r"|total\s+current\s+assets"
        ),
        "Balance Sheet",
    ),
    (
        re.compile(
            r"(?i)(?:total\s+)?(?:net\s+)?revenues?\b[\s\S]{0,400}?"
            r"(?:cost\s+of\s+(?:revenue|sales|goods)|gross\s+(?:profit|margin))"
            r"|operating\s+income\b[\s\S]{0,200}?(?:net\s+income|income\s+before)"
            r"|(?:basic\s+and\s+)?diluted\s+(?:net\s+)?(?:income|earnings)\s+per\s+share"
        ),
        "Income Statement",
    ),
]


def infer_section_from_text(text: str) -> str | None:
    """Identify a financial statement from distinctive row labels in the body."""
    head = text[:1500]
    for pattern, section in CONTENT_SECTION_PATTERNS:
        if pattern.search(head):
            return section
    return None


def build_context_header(metadata: ChunkMetadata, heading: str = "") -> str:
    """Provenance line prepended to a chunk for embedding and BM25 only."""
    parts = [metadata.company or "Unknown company"]
    if metadata.ticker:
        parts.append(metadata.ticker)
    period = " ".join(
        value
        for value in (
            metadata.doc_type or "filing",
            f"FY{metadata.fiscal_year}" if metadata.fiscal_year else "",
        )
        if value
    )
    parts.append(period)
    if metadata.page is not None:
        parts.append(f"page {metadata.page}")
    if metadata.section and metadata.section != "Unknown":
        parts.append(metadata.section)
    cleaned_heading = normalize_line(heading or "")
    if cleaned_heading and cleaned_heading.casefold() != (
        metadata.section or ""
    ).casefold():
        parts.append(cleaned_heading)
    return " | ".join(part for part in parts if part)


def assign_sections_to_pages(pages: Sequence[dict]) -> list[dict]:
    """Propagate the most recent section heading to subsequent pages."""
    current = "Unknown"
    current_heading = ""
    result = []
    for page in pages:
        item = dict(page)
        page_num = int(item.get("page_num", len(result)))
        heading = None
        # Headings also appear inside extracted tables (statement titles are
        # often the first row), so scan those lines as well.
        lines = list((item.get("text", "") or "").splitlines())
        for table in item.get("tables", []) or []:
            lines.extend((table or "").splitlines()[:3])
        for line in lines:
            section = detect_heading(line, page_num)
            if section:
                current, heading = section, normalize_line(line)
        if heading:
            current_heading = heading
        item["section"] = current
        item["section_heading"] = heading or current_heading
        result.append(item)
    return result


def detect_content_type(text: str, section: str, is_table: bool = False) -> ContentType:
    """Use parser table metadata first; use only conservative text fallbacks."""
    text = text.strip()
    if not text:
        return "unknown"
    if is_table or sum(line.count("|") >= 2 for line in text.splitlines()) >= 3:
        return "table"
    if section == "Risk Factors":
        return "risk_factor"
    if section == "Footnotes" or re.search(
        r"\b(?:note|footnote)\s+\d+[A-Za-z]?\b", text, re.I
    ):
        return "footnote"
    return "narrative"


def _split_text(
    text: str,
    metadata: ChunkMetadata,
    splitter: RecursiveCharacterTextSplitter,
    heading: str = "",
) -> list[Chunk]:
    documents = splitter.split_documents([Document(page_content=text.strip())])
    texts = [doc.page_content.strip() for doc in documents if doc.page_content.strip()]
    if len(texts) > 1 and len(texts[-1]) < 300:
        texts[-2] = f"{texts[-2]}\n\n{texts[-1]}"
        texts.pop()

    chunks = []
    for index, value in enumerate(texts):
        meta = metadata.model_copy(deep=True)
        meta.chunk_id = f"{metadata.chunk_id}_c{index:03d}"
        if meta.section in ("", "Unknown"):
            meta.section = infer_section_from_text(value) or meta.section
        chunks.append(
            Chunk(
                metadata=meta,
                text=value,
                context_header=build_context_header(meta, heading),
            )
        )
    return chunks


def _document_key(doc_info: DocumentInfo) -> str:
    """Return a stable short key that scopes chunk IDs to one document."""
    raw = doc_info.doc_id or (
        f"{doc_info.ticker}_{doc_info.fiscal_year}_{doc_info.doc_type}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def chunk_page(
    page: dict, doc_info: DocumentInfo, splitter: RecursiveCharacterTextSplitter
) -> list[Chunk]:
    """Chunk one parser output page. PDF parsing is deliberately not done here."""
    page_num = int(page["page_num"])
    section = page.get("section", "Unknown")
    base_id = f"{_document_key(doc_info)}_p{page_num:03d}"
    base = ChunkMetadata(
        chunk_id=base_id,
        doc_id=doc_info.doc_id,
        company=doc_info.company,
        ticker=doc_info.ticker,
        doc_type=doc_info.doc_type,
        fiscal_year=doc_info.fiscal_year,
        page=page_num,
        section=section,
        source_url=doc_info.source_url,
        content_type="narrative",
    )
    result: list[Chunk] = []
    heading = str(page.get("section_heading", "") or "")

    for table_index, table in enumerate(page.get("tables", []) or []):
        if table and table.strip():
            meta = base.model_copy(deep=True)
            meta.chunk_id = f"{base_id}_t{table_index:03d}"
            meta.content_type = "table"
            # A bare numbers table is unretrievable without this: identify the
            # statement from its own row labels when the page carries no heading.
            if meta.section in ("", "Unknown"):
                meta.section = infer_section_from_text(table) or meta.section
            result.append(
                Chunk(
                    metadata=meta,
                    text=table.strip(),
                    context_header=build_context_header(meta, heading),
                )
            )

    text = page.get("text", "") or ""
    if text.strip():
        meta = base.model_copy(deep=True)
        meta.chunk_id = f"{base_id}_n{len(result):03d}"
        meta.content_type = detect_content_type(text, section)
        result.extend(_split_text(text, meta, splitter, heading))
    return result


def create_chunks_from_pdf(
    pages: list[dict],
    doc_info: DocumentInfo,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Convert parser pages to chunks; this is the only public chunking entry point."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )
    pages_with_sections = assign_sections_to_pages(pages)
    chunks = [
        chunk
        for page in pages_with_sections
        for chunk in chunk_page(page, doc_info, splitter)
    ]
    logger.info(
        "Created %d chunks from %s (%d pages)", len(chunks), doc_info.doc_id, len(pages)
    )
    return chunks
