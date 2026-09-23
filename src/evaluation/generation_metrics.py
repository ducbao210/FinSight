"""Generation metrics via Ragas.

The caller supplies a LangChain-compatible local judge. Pass 2 loads that model
after Pass 1 has written its checkpoint; it does not regenerate answers. Ragas
is used for dataset orchestration and metric computation, while the supplied
local model performs the actual judging. Local inference is intentionally
serialized because the model occupies the GPU; the LangChain adapter may still
expose an async entrypoint for Ragas compatibility.
"""

from __future__ import annotations

import os
import pandas as pd
from tqdm import tqdm
from collections.abc import Sequence

from src.core.configs import settings
from src.core.logging import get_logger
from src.schemas import EvalSample, GenerationMetrics

logger = get_logger(__name__)

METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
)

# Ragas renames result columns between releases, so each metric lists the
# columns it may appear under, most specific first.
COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "faithfulness": ("faithfulness", "faithfulness_with_hhem"),
    "answer_relevancy": ("answer_relevancy", "response_relevancy"),
    "answer_correctness": ("answer_correctness",),
    "context_precision": (
        "llm_context_precision_with_reference",
        "context_precision",
        "context_precision_with_reference",
    ),
    "context_recall": ("context_recall", "llm_context_recall"),
}


# --------------------------------------------------------------------------- #
# Judge construction
# --------------------------------------------------------------------------- #
def build_groq_judge(model: str | None = None):
    """Groq judge, or None when no API key is configured."""
    key = settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=key,
        model=model or settings.model_name or "llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=3,
    )


def build_embeddings(embedding_model: str | None = None):
    """Build embeddings without competing with the local GGUF judge for VRAM.

    Set ``RAGAS_EMBEDDING_DEVICE=cuda`` explicitly when there is spare VRAM.
    CPU is the safe default for the two-pass runner because the judge already
    occupies most of the GPU and simultaneous llama/torch allocation can end
    in a native allocator crash rather than a catchable Python exception.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    device = os.environ.get("RAGAS_EMBEDDING_DEVICE", "cpu").strip().lower()
    if device not in {"cpu", "cuda"}:
        logger.warning("Unsupported RAGAS_EMBEDDING_DEVICE=%s; using cpu", device)
        device = "cpu"

    name = embedding_model or settings.embedding_model or "BAAI/bge-base-en-v1.5"
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _wrap(llm, embeddings):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [v for v in values if v is not None and v == v]  # drop None / NaN
    return sum(clean) / len(clean) if clean else None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _build_dataset(scorable, references, reference_contexts):
    from ragas import EvaluationDataset, SingleTurnSample

    return EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s.question,
                response=s.predicted_answer,
                retrieved_contexts=list(s.contexts),
                reference=references.get(s.question_id) or s.gold_answer or "",
                reference_contexts=reference_contexts.get(s.question_id) or None,
                metadata={"question_id": s.question_id},
            )
            for s in scorable
        ]
    )


def _build_metrics(llm, embeddings):
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    try:
        from ragas.metrics import AnswerCorrectness

        metrics.append(AnswerCorrectness(llm=llm, embeddings=embeddings))
    except (ImportError, TypeError):
        logger.warning("AnswerCorrectness is unavailable in this Ragas version")
    return metrics


def _score_single_sample_sync(
    single_dataset, metrics, llm, embeddings, max_workers: int, timeout: int
):
    """Đánh giá 1 sample duy nhất một cách đồng bộ."""
    from ragas import evaluate
    from ragas.run_config import RunConfig

    run_cfg = RunConfig(
        timeout=timeout,
        max_workers=max(1, max_workers),
        max_retries=1,
    )

    res = evaluate(
        dataset=single_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_cfg,
        raise_exceptions=False,
    )
    return res.to_pandas()


def _score(dataset, metrics, llm, embeddings, max_workers: int, timeout: int):
    """Feed trực tiếp từng câu của Pass 1 vào local judge thay vì batch async."""
    from ragas import EvaluationDataset

    total_samples = len(dataset)
    all_frames = []

    logger.info("Starting sequential evaluation for %d samples...", total_samples)

    for i in tqdm(range(total_samples), desc="Evaluating Pass 2 (Sync)"):
        sample_dataset = EvaluationDataset(samples=[dataset[i]])
        try:
            sample_frame = _score_single_sample_sync(
                sample_dataset,
                metrics,
                llm,
                embeddings,
                max_workers=max_workers,
                timeout=timeout,
            )
            metadata = getattr(dataset[i], "metadata", {}) or {}
            sample_frame["question_id"] = metadata.get(
                "question_id", dataset[i].user_input
            )
            all_frames.append(sample_frame)
        except Exception as e:
            logger.warning("Error evaluating sample %d: %s", i, e)
            all_frames.append(
                pd.DataFrame(
                    {
                        "question_id": [
                            (getattr(dataset[i], "metadata", {}) or {}).get(
                                "question_id", dataset[i].user_input
                            )
                        ]
                    }
                )
            )

    if all_frames:
        frame = pd.concat(all_frames, ignore_index=True)
    else:
        frame = pd.DataFrame({"question_id": []})

    return frame


def _aggregate(frame, scorable) -> dict[str, float | None]:
    aggregate: dict[str, float | None] = {}
    columns = list(frame.columns)
    for key in METRIC_KEYS:
        column = next(
            (c for c in COLUMN_CANDIDATES[key] if c in columns),
            "",
        )
        if not column:
            aggregate[key] = None
            continue
        score_by_id: dict[str, float | None] = {}
        for row in frame.itertuples(index=False):
            question_id = getattr(row, "question_id", "")
            raw = getattr(row, column, None)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = None
            score_by_id[str(question_id)] = value if value == value else None
        values: list[float | None] = []
        for sample in scorable:
            value = score_by_id.get(sample.question_id)
            setattr(sample.generation, key, value)
            values.append(value)
        aggregate[key] = _mean(values)
    return aggregate


def compute_generation_metrics(
    samples: Sequence[EvalSample],
    references: dict[str, str] | None = None,
    reference_contexts: dict[str, list[str]] | None = None,
    judge=None,
    judge_label: str = "local",
    embedding_model: str | None = None,
    max_workers: int | None = None,
    timeout: int = 1800,
    groq_fallback: bool = False,
    groq_model: str | None = None,
    return_backend: bool = False,
):
    """Score answered samples with Ragas and the supplied local judge.

    ``samples`` is mutated in place: each sample's ``.generation`` receives its
    own per-question scores.

    The local ``judge`` is tried first. Groq is used only when the caller
    explicitly enables ``groq_fallback``; evaluation does not use an API by
    default. A local judge should normally be called with ``max_workers=1``.

    Returns ``GenerationMetrics``; with ``return_backend=True`` returns
    ``(GenerationMetrics, backend_name)``.
    """
    references = references or {}
    reference_contexts = reference_contexts or {}

    scorable = [
        s
        for s in samples
        if s.status == "answer" and s.predicted_answer.strip() and s.contexts
    ]
    if not scorable:
        logger.warning("No answerable samples to score with Ragas")
        empty = GenerationMetrics()
        return (empty, "none") if return_backend else empty

    dataset = _build_dataset(scorable, references, reference_contexts)
    embeddings_model = build_embeddings(embedding_model)

    attempts: list[tuple[str, object, int]] = []
    if judge is not None:
        label = judge_label or "local"
        # An in-process local judge owns the model directly and must serialize
        # requests. A local OpenAI-compatible server ("local-api") can manage
        # concurrency itself, so only that backend keeps the caller's worker
        # count.
        workers = (
            1 if label in {"local", "local_llm_judge"} else max(1, max_workers or 1)
        )
        attempts.append((label, judge, workers))
    if groq_fallback:
        groq = build_groq_judge(groq_model)
        if groq is not None:
            attempts.append(("groq", groq, max_workers or 4))
    elif judge is None:
        raise RuntimeError(
            "No local Ragas judge available; evaluation will not call an API."
        )

    aggregate: dict[str, float | None] = {k: None for k in METRIC_KEYS}
    used = "none"
    for name, llm_obj, workers in attempts:
        logger.info(
            "Running Ragas on %d samples with the %s judge", len(scorable), name
        )
        wrapped_llm, wrapped_emb = _wrap(llm_obj, embeddings_model)
        metrics = _build_metrics(wrapped_llm, wrapped_emb)
        try:
            frame = _score(dataset, metrics, wrapped_llm, wrapped_emb, workers, timeout)
            aggregate = _aggregate(frame, scorable)
        except Exception as exc:
            logger.error("Ragas failed on the %s judge: %s", name, exc)
            continue
        used = name
        if any(v is not None for v in aggregate.values()):
            break
        logger.warning("The %s judge produced no usable scores", name)

    if used == "none" or all(v is None for v in aggregate.values()):
        logger.error("Ragas produced no scores with any judge")

    result = GenerationMetrics(**aggregate)
    return (result, used) if return_backend else result
