from __future__ import annotations

import json
import re
from pathlib import Path

from src.core.logging import get_logger
from src.schemas import DocumentInfo

logger = get_logger(__name__)


def load_document_info_from_jsonl(jsonl_path: Path) -> list[DocumentInfo]:
    """Load internal DocumentInfo objects from FinanceBench-compatible JSONL."""
    docs: list[DocumentInfo] = []
    if not jsonl_path.exists():
        logger.warning("Document info JSONL not found: %s", jsonl_path)
        return docs

    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                document = DocumentInfo.model_validate(row)
                if any(existing.doc_id == document.doc_id for existing in docs):
                    logger.warning(
                        "Skipping duplicate document metadata at line %d: %s",
                        line_no,
                        document.doc_id,
                    )
                    continue
                docs.append(document)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping invalid document metadata at line %d: %s", line_no, exc
                )

    logger.info("Loaded %d document info entries from %s", len(docs), jsonl_path)
    return docs


def build_document_index(docs: list[DocumentInfo]) -> dict[str, DocumentInfo]:
    return {doc.doc_id: doc for doc in docs}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _stem_tokens(stem: str) -> set[str]:
    """Split a PDF stem on filename separators and normalize each token."""
    return {token for part in re.split(r"[_-]+", stem) if (token := _norm(part))}


def _type_aliases(doc_type: str) -> set[str]:
    """Return filename spellings accepted for a canonical document type."""
    value = _norm(doc_type)
    aliases = {value}
    if value in {"10kannual", "10kannualreport"}:
        # FinanceBench metadata says 10k_annualreport, while PDF stems say annualreport.
        aliases.add("annualreport")
    return aliases


def _name_matches(name: str, stem_tokens: set[str]) -> bool:
    """Match a company/ticker as a complete filename token, not a substring."""
    compact_name = _norm(name)
    if compact_name and compact_name in stem_tokens:
        return True

    name_tokens = {
        token
        for part in re.split(r"[^A-Za-z0-9]+", str(name))
        if (token := _norm(part)) and len(token) > 1
    }
    return bool(name_tokens) and name_tokens.issubset(stem_tokens)


def resolve_pdf_path(pdf_dir: Path, doc_info: DocumentInfo) -> Path | None:
    """Resolve a FinanceBench document to a local PDF path."""
    if not pdf_dir.exists():
        return None

    # 1. Exact FinanceBench convention: doc_name == PDF stem.
    exact = pdf_dir / f"{doc_info.doc_id}.pdf"
    if exact.is_file():
        return exact

    files = sorted(pdf_dir.glob("*.pdf"))
    by_norm = {_norm(file.stem): file for file in files}

    # 2. Normalized doc_id: handles case and separator differences.
    if hit := by_norm.get(_norm(doc_info.doc_id)):
        return hit

    # 3. Reconstruct the conventional filename from metadata.
    year = _norm(str(doc_info.fiscal_year)) if doc_info.fiscal_year is not None else ""
    type_aliases = _type_aliases(doc_info.doc_type)
    names = (doc_info.ticker, doc_info.company)

    for name in names:
        name_norm = _norm(name)
        for type_name in type_aliases:
            if year and (hit := by_norm.get(_norm(f"{name_norm}_{year}_{type_name}"))):
                return hit

    # 4. Boundary/token fallback. This avoids AES matching ADVANCED-like names.
    candidates: list[Path] = []
    for file in files:
        stem_tokens = _stem_tokens(file.stem)
        if (
            year
            and year in stem_tokens
            and stem_tokens.intersection(type_aliases)
            and any(_name_matches(name, stem_tokens) for name in names if name)
        ):
            candidates.append(file)

    if candidates:
        # Stable fallback for multiple same-year/same-type documents.
        return candidates[0]

    return None
