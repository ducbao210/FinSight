"""Two-pass GPT-OSS-20B NF4 FinanceBench evaluation for Kaggle GPUs.

Pass 1 loads the pre-quantized NF4 model, retrieves evidence, generates answers,
and writes an EvalReport checkpoint. Pass 2 releases the answer model, loads the
same model fresh as a LangChain judge, and scores the checkpoint with Ragas.
GPT-OSS uses the Harmony chat template, so generation preserves its channel
markers and extracts only the final channel.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.configs import settings
from src.core.logging import get_logger, setup_logging
from src.evaluation import EvalConfig, run_evaluation
from src.evaluation.dataset import (
    filter_questions,
    load_questions,
    question_key,
    sample_questions,
)
from src.evaluation.citation_metrics import aggregate_citation_metrics
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.generation_metrics import compute_generation_metrics
from src.evaluation.reporting import (
    load_intermediate,
    print_report,
    save_intermediate,
    save_report,
)
from src.schemas import Citation, QueryRequest, QueryResponse, TraceInfo

logger = get_logger("run_eval")
DEFAULT_QUESTIONS = ROOT / "data/raw/financebench_open_source.jsonl"
DEFAULT_ARTIFACT_DIR = ROOT / "data/index"
DEFAULT_OUTPUT = ROOT / "results"
DEFAULT_MODEL = "models/gpt-oss-20b-Q4_K_M.gguf"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-pass local GPT-OSS + Ragas FinanceBench evaluation"
    )
    p.add_argument("--phase", choices=("all", "generate", "judge"), default="all")
    p.add_argument(
        "--backend",
        choices=("transformers", "gguf"),
        default="gguf",
        help="Local inference backend. GGUF uses llama-cpp-python and avoids MXFP4 kernels.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="GGUF file path when --backend gguf; HF model id otherwise.",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Defaults to --model; loaded fresh in Pass 2.",
    )
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--samples-path",
        type=Path,
        default=None,
        help="Pass 1 JSON checkpoint for --phase judge.",
    )
    p.add_argument("--config-name", default="gpt-oss-20b-gguf-two-pass")
    p.add_argument(
        "--pipeline",
        choices=("plain", "crag"),
        default="plain",
        help="Answer pipeline to evaluate. crag uses the configured CRAG/Groq pipeline.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--company", default=None)
    p.add_argument("--question-type", default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--k", default="1,3,5,10")
    p.add_argument("--page-offset", type=int, default=1)
    p.add_argument("--max-context-chars", type=int, default=1600)
    p.add_argument("--max-contexts", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=1536)
    p.add_argument("--judge-max-new-tokens", type=int, default=2048)
    p.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="medium",
        help="GPT-OSS Harmony reasoning effort.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n-ctx", type=int, default=8192, help="GGUF context size")
    p.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="GGUF layers to offload; -1 means all possible layers",
    )
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--no-double-quant", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--run-metrics", default="retrieval,citation,ragas")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--ragas-workers", type=int, default=1)
    p.add_argument("--ragas-timeout", type=int, default=1800)
    p.add_argument("--no-save", action="store_true")
    return p.parse_args(argv)


def describe_gpus() -> tuple[int, list[str]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0, []
        return torch.cuda.device_count(), [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except ImportError:
        return 0, []


def load_local_model(model_name: str, args: argparse.Namespace):
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    count, names = describe_gpus()
    if not count:
        raise RuntimeError(
            "No CUDA device is visible; the local GPT-OSS backend needs a GPU."
        )
    logger.info("Visible GPU(s): %d -> %s", count, ", ".join(names))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=args.trust_remote_code
    )
    has_native_quantization = (
        getattr(model_config, "quantization_config", None) is not None
    )
    prequantized = has_native_quantization or any(
        x in model_name.lower() for x in ("4bit", "bnb", "gptq", "awq", "nf4", "fp4")
    )
    if prequantized and args.no_4bit:
        raise ValueError("The selected model is pre-quantized; remove --no-4bit.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if has_native_quantization:
        kwargs["dtype"] = torch.float16
    elif prequantized:
        kwargs["dtype"] = "auto"
    else:
        kwargs["dtype"] = dtype
        if not args.no_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=not args.no_double_quant,
            )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


class GGUFTokenizer:
    """Minimal Harmony renderer for llama.cpp GGUF inference."""

    chat_template = "gpt-oss-harmony"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="medium",
        **_,
    ):
        del tokenize
        system = next(
            (str(m["content"]) for m in messages if m["role"] == "system"), ""
        )
        rendered = (
            "<|start|>system<|message|>"
            f"{system}\n\nReasoning: {reasoning_effort}\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message."
            "<|end|>"
        )
        for message in messages:
            role = message["role"]
            if role == "system":
                continue
            rendered += f"<|start|>{role}<|message|>{message['content']}<|end|>"
        if add_generation_prompt:
            rendered += "<|start|>assistant<|channel|>analysis<|message|>"
        return rendered


class GGUFGenerator:
    """Callable adapter matching the Hugging Face pipeline result shape."""

    def __init__(self, model, max_new_tokens: int, temperature: float):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = GGUFTokenizer()

    def __call__(self, prompt: str, **_):
        result = self.model(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=1.0,
            top_k=0,
            # <|end|> closes analysis; stopping there prevents the final channel.
            stop=["<|return|>"],
            echo=False,
        )
        text = result["choices"][0].get("text", "")
        return [{"generated_text": text}]


def load_gguf_model(model_path: str, args: argparse.Namespace, max_new_tokens: int):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "GGUF backend requires llama-cpp-python. Install a CUDA-enabled build "
            "or use --backend transformers."
        ) from exc

    model = Llama(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=512,
        verbose=False,
    )
    generator = GGUFGenerator(model, max_new_tokens, args.temperature)
    return generator.tokenizer, model, generator


def load_eval_backend(model_path: str, args: argparse.Namespace, max_new_tokens: int):
    if args.backend == "gguf":
        return load_gguf_model(model_path, args, max_new_tokens)
    tokenizer, model = load_local_model(model_path, args)
    return (
        tokenizer,
        model,
        build_pipeline(tokenizer, model, max_new_tokens, args.temperature),
    )


def build_pipeline(tokenizer, model, max_new_tokens: int, temperature: float):
    from transformers import pipeline

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
    gc = gen.generation_config
    gc.max_new_tokens = max_new_tokens
    gc.pad_token_id = tokenizer.pad_token_id
    if temperature <= 0:
        gc.do_sample = False
    else:
        gc.do_sample = True
        gc.temperature = temperature
        gc.top_p = 0.9
    return gen


def render_chat_prompt(tokenizer, messages, reasoning_effort: str) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return "\n\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "reasoning_effort": reasoning_effort,
    }
    template_text = str(chat_template)
    if "enable_thinking" in template_text:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def extract_generated_text(result: object) -> str:
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        return str(result.get("generated_text", ""))
    return str(result or "")


def extract_final_text(text: str) -> str:
    text = str(text or "")
    final_marker = "<|channel|>final<|message|>"
    analysis_marker = "<|channel|>analysis<|message|>"
    if final_marker in text:
        text = text.rsplit(final_marker, 1)[-1]
    elif analysis_marker in text:
        return ""
    text = text.split("<|return|>", 1)[0]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|(?:im_end|end|return)\|>", "", text)
    return text.strip()


def clean_judge_output(content: object) -> str:
    text = extract_final_text(str(content or ""))
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL
    ).strip()
    for start, marker in ((i, text[i]) for i in range(len(text)) if text[i] in "{["):
        closing = "}" if marker == "{" else "]"
        depth, quoted, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\" and quoted:
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif not quoted and char == marker:
                depth += 1
            elif not quoted and char == closing:
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        value = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    return json.dumps(value, ensure_ascii=False)
    return text


def as_langchain_llm(generator, reasoning_effort: str = "low"):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    class LocalJudgeModel(BaseChatModel):
        _generator = PrivateAttr()

        def __init__(self, generator):
            super().__init__()
            self._generator = generator

        @property
        def _llm_type(self) -> str:
            return "sync_local_gguf"

        def _prompt(self, messages: list[BaseMessage]) -> str:
            mapped = []
            for message in messages:
                role = "user" if message.type == "human" else message.type
                if role == "ai":
                    role = "assistant"
                mapped.append({"role": role, "content": str(message.content)})
            return render_chat_prompt(self._generator.tokenizer, mapped, "low")

        def _generate(
            self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
        ) -> ChatResult:
            prompt = self._prompt(messages)
            suffix = (
                "\n\nIMPORTANT: Return only the requested JSON data instance. "
                "Do not repeat or describe the JSON Schema. Do not output keys "
                "such as properties, required, title, or type unless they are "
                "explicitly requested as answer fields."
            )
            result = self._generator(prompt + suffix, skip_special_tokens=False)
            text = extract_generated_text(result)
            cleaned = clean_judge_output(text)
            try:
                value = json.loads(cleaned)
            except (TypeError, json.JSONDecodeError):
                value = None
            if (
                isinstance(value, dict)
                and "properties" in value
                and "required" in value
                and not any(
                    key not in {"properties", "required", "title", "type"}
                    for key in value
                )
            ):
                retry_prompt = (
                    prompt
                    + "\n\nThe previous response was a JSON Schema, not the answer. "
                    "Now produce the actual JSON object required by the schema, "
                    "with concrete values. Output JSON only."
                )
                retry = self._generator(retry_prompt, skip_special_tokens=False)
                cleaned = clean_judge_output(extract_generated_text(retry))
            message = AIMessage(content=cleaned)
            return ChatResult(generations=[ChatGeneration(message=message)])

        async def _agenerate(
            self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
        ) -> ChatResult:
            return self._generate(messages, stop, run_manager, **kwargs)

    return LocalJudgeModel(generator)


class Answerer:
    def __init__(self, retriever, args, tokenizer=None, generator=None):
        self.retriever, self.args, self.tokenizer, self.generator = (
            retriever,
            args,
            tokenizer,
            generator,
        )
        self._gpu_lock = asyncio.Lock()

    def _prompt(self, question: str, contexts: list[str]) -> str:
        evidence = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(contexts))
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer financial questions using only the evidence provided. "
                    "Cite evidence as [1], [2], etc. If evidence is insufficient, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nEvidence:\n{evidence}",
            },
        ]
        return render_chat_prompt(self.tokenizer, messages, self.args.reasoning_effort)

    def _select_contexts(self, chunks):
        max_chars = max(
            1,
            min(
                self.args.max_context_chars,
                (self.args.n_ctx - self.args.max_new_tokens - 512) * 4,
            ),
        )
        selected = []
        used_chars = 0
        for chunk in chunks[: self.args.max_contexts]:
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            text = chunk.chunk.text[: min(self.args.max_context_chars, remaining)]
            if not text:
                break
            selected.append((chunk, text))
            used_chars += len(text)
        return selected

    def _generate(self, prompt: str) -> str:
        result = self.generator(prompt, skip_special_tokens=False)
        return extract_final_text(extract_generated_text(result))

    async def answer(self, request: QueryRequest) -> QueryResponse:
        chunks = await asyncio.to_thread(
            self.retriever, request.question, self.args.top_k
        )
        selected_with_text = self._select_contexts(chunks)
        selected = [item[0] for item in selected_with_text]
        contexts = [item[1] for item in selected_with_text]
        citations = [
            Citation(
                company=c.chunk.metadata.company,
                filing=c.chunk.metadata.doc_id,
                page=c.chunk.metadata.page,
                chunk_id=c.chunk.metadata.chunk_id,
                section=c.chunk.metadata.section,
                source_url=c.chunk.metadata.source_url,
                text_snippet=c.chunk.text[:200],
            )
            for c in selected
        ]
        async with self._gpu_lock:
            answer = await asyncio.to_thread(
                self._generate, self._prompt(request.question, contexts)
            )
        return QueryResponse(
            answer=answer,
            status="abstain" if not answer else "answer",
            contexts=contexts,
            citations=citations,
            trace=TraceInfo(),
            retrieved_chunks=chunks,
        )


def make_retriever(args):
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import get_collection
    from src.retrieval.hybrid import hybrid_search

    path = args.artifact_dir / "bm25.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 artifact not found: {path}. Run scripts/ingest.py first."
        )
    with path.open("rb") as handle:
        bm25: BM25Retriever = pickle.load(handle)
    return lambda q, k: hybrid_search(get_collection(), bm25, q, top_k=k)


def make_crag_answerer(args):
    from src.graph.workflow import CRAGPipeline
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import get_collection

    path = args.artifact_dir / "bm25.pkl"
    with path.open("rb") as handle:
        bm25: BM25Retriever = pickle.load(handle)
    return CRAGPipeline(get_collection(), bm25, settings.model_name)


def selected_questions(args):
    questions = filter_questions(
        load_questions(args.questions),
        company=args.company,
        question_type=args.question_type,
    )
    questions = sample_questions(questions, args.limit, args.seed, args.shuffle)
    if not questions:
        raise ValueError("No questions selected - check your filters")
    return questions


def release_model(*objects) -> None:
    for obj in objects:
        close = getattr(obj, "close", None)
        if callable(close):
            close()
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def checkpoint_path(args) -> Path:
    if args.samples_path is not None:
        return args.samples_path
    return args.output_dir / f"{args.config_name}_pass1.json"


def require_checkpoint(path: Path, output_dir: Path) -> Path:
    """Validate the Pass 1 artifact and provide an actionable error."""
    if path.is_file():
        return path
    candidates = sorted(output_dir.glob("*pass1*.json"))
    hint = ""
    if candidates:
        hint = " Available checkpoints: " + ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Pass 1 checkpoint not found: {path}. Run --phase generate first or "
        f"pass the exact file with --samples-path.{hint}"
    )


def pass1_generate(args, k_values) -> tuple[Path, dict[str, Any]]:
    retriever = make_retriever(args)
    tokenizer = model = generator = None
    if args.pipeline == "crag":
        answerer = make_crag_answerer(args)
    else:
        tokenizer, model, generator = load_eval_backend(
            args.model, args, args.max_new_tokens
        )
        answerer = Answerer(retriever, args, tokenizer=tokenizer, generator=generator)
    config = EvalConfig(
        questions_path=args.questions,
        config_name=args.config_name,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        company=args.company,
        question_type=args.question_type,
        top_k=args.top_k,
        k_values=k_values,
        page_offset=args.page_offset,
        concurrency=args.concurrency,
        run_generation_metrics=False,
        run_retrieval_metrics="retrieval" in args.run_metrics,
        run_citation_metrics="citation" in args.run_metrics,
    )
    report = asyncio.run(run_evaluation(config, answerer=answerer, retriever=retriever))
    path = checkpoint_path(args)
    save_intermediate(report, path)
    n_gpu, gpu_names = describe_gpus()
    meta = {
        "phase": "generate",
        "model": args.model,
        "backend": args.backend,
        "quantization": "GGUF" if args.backend == "gguf" else "checkpoint-defined",
        "checkpoint": str(path),
        "gpu_count": n_gpu,
        "gpus": gpu_names,
    }
    meta_path = path.with_name(path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    release_model(generator, model, tokenizer, answerer, retriever)
    return path, meta


def pass2_judge(args, checkpoint: Path) -> tuple[Any, str]:
    """Judge an existing Pass 1 checkpoint; never regenerates answers."""
    report = load_intermediate(checkpoint)
    model_name = args.judge_model or args.model
    tokenizer, model, generator = load_eval_backend(
        model_name, args, args.judge_max_new_tokens
    )
    judge = as_langchain_llm(generator, args.reasoning_effort)
    questions = selected_questions(args)
    references = {question_key(q, i): q.answer for i, q in enumerate(questions)}
    reference_contexts = {
        question_key(q, i): [e.evidence_text for e in q.evidence if e.evidence_text]
        for i, q in enumerate(questions)
    }
    selected_ids = {question_key(q, i) for i, q in enumerate(questions)}
    samples = [s for s in report.samples if s.question_id in selected_ids]
    if not samples:
        raise ValueError(
            "The selected Pass 2 questions do not exist in the Pass 1 checkpoint. "
            "Use the same --questions/--company/--question-type filters, or omit "
            "--limit while locating the checkpoint."
        )

    try:
        metrics, backend = compute_generation_metrics(
            samples,
            references=references,
            reference_contexts=reference_contexts,
            judge=judge,
            judge_label="local",
            embedding_model=args.embedding_model,
            max_workers=1,
            timeout=args.ragas_timeout,
            groq_fallback=False,
            return_backend=True,
        )
        report.generation = metrics
        report.samples = samples
        report.num_questions = len(samples)
        if "retrieval" in {
            x.strip().lower() for x in args.run_metrics.split(",") if x.strip()
        }:
            report.retrieval = compute_retrieval_metrics(samples, report.k_values)
        if "citation" in {
            x.strip().lower() for x in args.run_metrics.split(",") if x.strip()
        }:
            answered = [sample for sample in samples if sample.status == "answer"]
            report.citation = aggregate_citation_metrics(
                [sample.citation_coverage for sample in answered],
                [sample.citation_validity for sample in answered],
            )
        report.abstain_rate = sum(
            sample.status == "abstain" for sample in samples
        ) / len(samples)
        report.error_rate = sum(sample.status == "error" for sample in samples) / len(
            samples
        )
    finally:
        release_model(generator, model, tokenizer, judge)
    return report, backend


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging("eval_two_pass")
    metrics_requested = {
        x.strip().lower() for x in args.run_metrics.split(",") if x.strip()
    }
    if metrics_requested - {"retrieval", "citation", "ragas"}:
        raise SystemExit(
            f"Unsupported metrics: {sorted(metrics_requested - {'retrieval', 'citation', 'ragas'})}"
        )
    args.run_metrics = ",".join(sorted(metrics_requested))
    k_values = tuple(int(x) for x in args.k.split(",") if x.strip()) or (1, 3, 5, 10)
    checkpoint = checkpoint_path(args)
    meta: dict[str, Any] = {}
    if args.phase in ("all", "generate"):
        checkpoint, meta = pass1_generate(args, k_values)
        logger.info("Pass 1 complete: %s", checkpoint)
        if args.phase == "generate":
            print(f"Pass 1 checkpoint: {checkpoint}")
            return 0
    if args.phase == "judge":
        checkpoint = require_checkpoint(checkpoint, args.output_dir)
    report = load_intermediate(checkpoint)
    backend = "skipped"
    if "ragas" in metrics_requested:
        report, backend = pass2_judge(args, checkpoint)
    print_report(report)
    if not args.no_save:
        json_path, csv_path = save_report(report, args.output_dir)
        meta_path = args.output_dir / "two_pass_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.update(
            {
                "phase": args.phase,
                "ragas_backend": backend,
                "checkpoint": str(checkpoint),
                "judge_model": args.judge_model or args.model,
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved: {json_path}\nSaved: {csv_path}\nSaved: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
