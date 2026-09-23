"""Retrieval metrics computed with `ranx` (precision, recall, MRR, nDCG, hit rate).

We score the same run twice:
  * document level  -> did we retrieve the right filing?
  * page level      -> did we retrieve the right page of that filing?
    (`page_recall@k` is simply `recall@k` on the page-level qrels)
"""

from __future__ import annotations

from collections.abc import Sequence

from ranx import Qrels, Run, evaluate

from src.core.logging import get_logger
from src.schemas import EvalSample, RetrievalMetrics

logger = get_logger(__name__)

DEFAULT_K_VALUES = (1, 3, 5, 10)


def metric_names(k_values: Sequence[int] = DEFAULT_K_VALUES) -> list[str]:
    names: list[str] = ["mrr"]
    for k in k_values:
        names.extend([f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}"])
    return names


def _build(
    samples: Sequence[EvalSample],
    gold_attr: str,
    run_attr: str,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}

    for sample in samples:
        gold: list[str] = getattr(sample, gold_attr)
        retrieved: list[str] = getattr(sample, run_attr)
        if not gold:
            continue
        qrels[sample.question_id] = {key: 1 for key in gold}
        scores: dict[str, float] = {}
        for rank, key in enumerate(retrieved, start=1):
            # Several chunks map to the same doc/page: keep the best rank.
            scores[key] = max(scores.get(key, 0.0), 1.0 / rank)
        # ranx requires a non-empty run entry per query.
        run[sample.question_id] = scores or {"__empty__": 0.0}

    return qrels, run


def _evaluate(
    samples: Sequence[EvalSample],
    gold_attr: str,
    run_attr: str,
    k_values: Sequence[int],
) -> dict[str, float]:
    qrels_dict, run_dict = _build(samples, gold_attr, run_attr)
    if not qrels_dict:
        return {}
    scores = evaluate(
        Qrels(qrels_dict),
        Run(run_dict),
        metrics=metric_names(k_values),
        make_comparable=True,
    )
    return {name: float(value) for name, value in dict(scores).items()}


def compute_retrieval_metrics(
    samples: Sequence[EvalSample],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalMetrics:
    """Document-level and page-level IR metrics for a list of eval samples."""
    doc_level = _evaluate(samples, "gold_doc_keys", "retrieved_doc_keys", k_values)
    page_level = _evaluate(samples, "gold_page_keys", "retrieved_page_keys", k_values)

    # Convenience aliases so `page_recall@k` shows up under its usual name.
    for k in k_values:
        if f"recall@{k}" in page_level:
            page_level[f"page_recall@{k}"] = page_level[f"recall@{k}"]

    scored = sum(1 for s in samples if s.gold_doc_keys)
    logger.info("Scored retrieval on %d questions", scored)
    return RetrievalMetrics(
        doc_level=doc_level,
        page_level=page_level,
        num_scored_questions=scored,
    )
