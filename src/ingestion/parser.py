from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from src.core.logging import get_logger
from src.schemas import DocumentInfo

logger = get_logger(__name__)
PARSER_VERSION = "pymupdf-layout-markdown-v2-contextual-headers"


def parse_pdf(file_path: Path) -> list[dict]:
    """Parse a PDF with Layout-aware Markdown, falling back to native PyMuPDF."""
    try:
        pages = _parse_pdf_with_layout(file_path)
    except Exception as exc:
        # Layout installs a hook on PyMuPDF pages. Turn it off before using
        # the native fallback, otherwise table detection would retry the same
        # failed model on every page.
        pymupdf4llm.use_layout(False)
        logger.warning(
            "PyMuPDF Layout failed for %s; using native PyMuPDF fallback: %s",
            file_path.name,
            exc,
        )
        pages = _parse_pdf_native(file_path)

    logger.info("Parsed %s: %d pages with content", file_path.name, len(pages))
    return pages


def _parse_pdf_with_layout(file_path: Path) -> list[dict]:
    """Keep Markdown reading order and tables emitted by PyMuPDF Layout."""
    pymupdf4llm.use_layout(True)
    page_chunks = pymupdf4llm.to_markdown(
        str(file_path),
        page_chunks=True,
        header=False,
        footer=False,
        use_ocr=False,
        write_images=False,
    )
    if not isinstance(page_chunks, list):
        raise TypeError("PyMuPDF4LLM must return one Markdown record per page")

    pages: list[dict] = []
    for item in page_chunks:
        text = str(item.get("text", ""))
        tables, narrative = _separate_layout_tables(text, item.get("page_boxes", []))
        if narrative or tables:
            metadata = item.get("metadata", {})
            pages.append(
                {
                    "page_num": int(metadata.get("page_number", len(pages) + 1)),
                    "text": narrative,
                    "tables": tables,
                }
            )
    return pages


def _separate_layout_tables(
    markdown: str, page_boxes: object
) -> tuple[list[str], str]:
    """Split Layout's table boxes from narrative Markdown without duplication."""
    if not isinstance(page_boxes, list):
        return [], markdown.strip()

    tables: list[str] = []
    narrative_parts: list[str] = []
    cursor = 0
    for box in page_boxes:
        if not isinstance(box, dict):
            continue
        pos = box.get("pos")
        if not (
            isinstance(pos, (tuple, list))
            and len(pos) == 2
            and all(isinstance(value, int) for value in pos)
        ):
            continue
        start, stop = pos
        start = max(cursor, start)
        stop = max(start, stop)
        narrative_parts.append(markdown[cursor:start])
        content = markdown[start:stop]
        if box.get("class") == "table":
            if content.strip():
                tables.append(content.strip())
        else:
            narrative_parts.append(content)
        cursor = stop
    narrative_parts.append(markdown[cursor:])
    return tables, "".join(narrative_parts).strip()


def _parse_pdf_native(file_path: Path) -> list[dict]:
    """Retain a usable parser when the optional Layout runtime is unavailable."""
    pages: list[dict] = []
    try:
        document = pymupdf.open(str(file_path))
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", file_path, exc)
        return pages

    try:
        for page_num, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()
            tables = _extract_tables_raw(page)
            if text or tables:
                pages.append(
                    {
                        "page_num": page_num,
                        "text": text,
                        "tables": tables,
                    }
                )
    finally:
        document.close()

    return pages


def _extract_tables_raw(page: pymupdf.Page) -> list[str]:
    """Extract tables as pipe-delimited text for table-preserving chunks."""
    tables: list[str] = []
    try:
        found = page.find_tables()
        for table in found.tables:
            cells = table.extract()
            if cells:
                tables.append(
                    "\n".join(
                        " | ".join(str(cell or "") for cell in row) for row in cells
                    )
                )
    except Exception as exc:
        logger.debug("Table extraction failed on page %s: %s", page.number + 1, exc)
    return tables


def extract_document_info(file_path: Path) -> DocumentInfo | None:
    """Build basic document metadata from the filename and first page."""
    name = file_path.stem
    parts = name.replace(" ", "_").split("_")
    ticker = parts[0].upper() if parts else "UNKNOWN"
    fiscal_year = 2023
    if len(parts) > 1:
        match = re.search(r"\d{4}", parts[1])
        if match:
            fiscal_year = int(match.group())
    doc_type = parts[2].replace("-", "_") if len(parts) > 2 else "10-K"
    company = ticker

    try:
        document = pymupdf.open(str(file_path))
        try:
            first_page = (
                document[0].get_text("text", sort=True)[:2000] if document else ""
            )
        finally:
            document.close()
        match = re.search(
            r"\b([A-Z][A-Za-z0-9&.,'\- ]+?\s+(?:Inc\.|Corp\.|Corporation|Company|Ltd\.|plc))\b",
            first_page,
        )
        if match:
            company = match.group(1).strip()
    except Exception as exc:
        logger.debug("Could not infer company from %s: %s", file_path, exc)

    return DocumentInfo(
        doc_id=name,
        company=company,
        ticker=ticker,
        doc_type=doc_type,
        fiscal_year=fiscal_year,
        source_path=str(file_path),
        source_url="",
    )


def normalize_number(raw_text: str) -> tuple[str, float | None, str]:
    """Normalize a disclosed financial number without changing its source text."""
    original = raw_text
    text = raw_text.strip()
    if text in {"—", "-", "–", "N/A", ""}:
        return original, None, ""

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    units: list[str] = []
    if re.search(r"\$|\bUSD\b", text, re.I):
        units.append("USD")
    text = re.sub(r"(?:,|\$|\bUSD\b)", "", text, flags=re.I)

    for word, unit in (
        ("billion", "billions"),
        ("million", "millions"),
        ("thousand", "thousands"),
    ):
        if re.search(rf"\b{word}s?\b", text, re.I):
            text = re.sub(rf"\s*\b{word}s?\b", "", text, flags=re.I)
            units.append(unit)
    if "%" in text or re.search(r"\bpercent\b", text, re.I):
        text = re.sub(r"%|\s*\bpercent\b", "", text, flags=re.I)
        units.append("percentage")

    try:
        value = float(text.strip())
        return original, (-value if negative else value), " ".join(units)
    except ValueError:
        return original, None, ""
