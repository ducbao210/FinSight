"""Console, checkpoint, and file reporting for evaluation runs."""
from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
from src.core.logging import get_logger
from src.schemas import EvalReport
logger = get_logger(__name__)

def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"

def print_report(report: EvalReport) -> None:
    line = "-" * 62
    print(line)
    print(f"FinanceBench evaluation | config={report.config_name} | n={report.num_questions}")
    print(line)
    print("Retrieval - document level")
    for name in sorted(report.retrieval.doc_level):
        print(f"  {name:<24} {report.retrieval.doc_level[name]:.3f}")
    print("Retrieval - page level")
    for name in sorted(report.retrieval.page_level):
        print(f"  {name:<24} {report.retrieval.page_level[name]:.3f}")
    print("Generation (Ragas)")
    for name in ("faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"):
        print(f"  {name:<24} {_fmt(getattr(report.generation, name))}")
    print("Citations")
    print(f"  {'citation_coverage':<24} {report.citation.citation_coverage:.3f}")
    print(f"  {'citation_validity':<24} {report.citation.citation_validity:.3f}")
    print("Operational")
    print(f"  {'median_latency_ms':<24} {report.median_latency_ms:.0f}")
    print(f"  {'median_retry_count':<24} {report.median_retry_count:.1f}")
    print(f"  {'abstain_rate':<24} {report.abstain_rate:.3f}")
    print(f"  {'error_rate':<24} {report.error_rate:.3f}")
    print(line)

def save_intermediate(report: EvalReport, path: Path) -> Path:
    """Save a complete report checkpoint, normally after answer generation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Intermediate report written to %s", path)
    return path

def load_intermediate(path: Path) -> EvalReport:
    """Load and validate a report checkpoint produced by save_intermediate."""
    return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))

def save_report(report: EvalReport, output_dir: Path) -> tuple[Path, Path]:
    """Write the full JSON report plus a per-question CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = report.config_name.replace(" ", "-")
    json_path = output_dir / f"eval_{slug}_{stamp}.json"
    save_intermediate(report, json_path)
    csv_path = output_dir / f"eval_{slug}_{stamp}_samples.csv"
    columns = ["question_id", "question", "status", "gold_answer", "predicted_answer", "citation_coverage", "citation_validity", "faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall", "latency_ms", "retry_count", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sample in report.samples:
            writer.writerow({
                "question_id": sample.question_id, "question": sample.question, "status": sample.status,
                "gold_answer": sample.gold_answer, "predicted_answer": sample.predicted_answer,
                "citation_coverage": round(sample.citation_coverage, 4), "citation_validity": round(sample.citation_validity, 4),
                "faithfulness": sample.generation.faithfulness, "answer_relevancy": sample.generation.answer_relevancy,
                "answer_correctness": sample.generation.answer_correctness, "context_precision": sample.generation.context_precision,
                "context_recall": sample.generation.context_recall, "latency_ms": round(sample.latency_ms, 1),
                "retry_count": sample.retry_count, "error": sample.error,
            })
    logger.info("Report written to %s", json_path)
    return json_path, csv_path

__all__ = ["load_intermediate", "print_report", "save_intermediate", "save_report"]
