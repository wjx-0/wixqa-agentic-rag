from __future__ import annotations

from contextlib import nullcontext
import math
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.agentic.context_budget import ContextBudgetConfig
from src.agentic.evidence_context import EvidenceContextError, build_initial_evidence_context
from src.agentic.evidence_loop import (
    EvidenceCompletionLoopResult,
    EvidenceLoopConfig,
    EvidenceLoopError,
    run_evidence_completion_loop,
)
from src.evaluation.chunk_eval import (
    classify_chunk_case,
    compute_chunk_retrieval_metrics,
    select_chunk_ks,
    summarize_chunk_metrics,
)
from src.evaluation.eval_utils import CASE_INVALID, markdown_table
from src.evaluation.run_chunk_bm25_eval import load_kb_chunks, metric_rows
from src.evaluation.run_evidence_context_eval import (
    DEFAULT_RERANK_RUN_DIR,
    load_rerank_trace_lookup,
    required_config_value,
    validate_qid_sets,
)
from src.evaluation.run_rerank_eval import (
    RerankEvalError,
    build_chunk_lookup,
    candidate_row_to_example,
    load_hybrid_candidates,
    load_required_json,
)
from src.llm.evidence_checker import (
    CHECKER_MODE_COMPACT,
    CHECKER_MODE_TRACEABLE,
    DEFAULT_CHECKER_MAX_TOKENS,
    DEFAULT_CHECKER_TEMPERATURE,
    DEFAULT_CHECKER_TIMEOUT,
    DEFAULT_CONTEXT_PREVIEW_CHARS,
    DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
    VALID_CHECKER_MODES,
    CompactEvidenceChecker,
    OpenAICompatibleChatClient,
    RetryingEvidenceChecker,
    TraceableEvidenceChecker,
)
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_INSTRUCTION_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL_NAME,
    CrossEncoderReranker,
    CrossEncoderRerankerError,
)
from src.rerankers.dashscope_reranker import (
    DEFAULT_DASHSCOPE_RERANK_MODEL,
    DEFAULT_DASHSCOPE_RERANK_TIMEOUT,
    DEFAULT_DASHSCOPE_RERANK_URL,
    DashScopeReranker,
    DashScopeRerankerError,
)
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME
from src.retrievers.dense_worker_client import VALID_WORKER_MODES
from src.retrievers.hybrid_retriever import HybridRetriever, HybridRetrieverError
from src.retrievers.rrf import DEFAULT_BM25_WEIGHT, DEFAULT_DENSE_WEIGHT
from src.utils.env_utils import (
    dashscope_rerank_model_from_env,
    dashscope_rerank_url_from_env,
    deepseek_base_url_from_env,
)
from src.utils.io_utils import ensure_dir, model_to_dict, write_json, write_jsonl
from src.utils.metrics import average
from src.utils.text_utils import compact_text


DEFAULT_OUTPUT_DIR = "outputs/evidence_loop"
DEFAULT_INDEX_DIR = "indexes/faiss_bge_m3"
DEFAULT_BRANCH_TOP_K_CHUNKS = 50
DEFAULT_SECOND_HOP_TOP_K_CHUNKS = 20
DEFAULT_RRF_K = 60
DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS = 16384
DEFAULT_MAX_RAW_CHUNKS_PER_CHECKER_CALL = 30
RERANKER_PROVIDER_LOCAL = "local"
RERANKER_PROVIDER_DASHSCOPE = "dashscope"
VALID_RERANKER_PROVIDERS = {RERANKER_PROVIDER_LOCAL, RERANKER_PROVIDER_DASHSCOPE}


class EvidenceLoopEvalError(RuntimeError):
    pass


def run_evidence_loop_eval(
    *,
    rerank_run_dir: str | Path = DEFAULT_RERANK_RUN_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    dense_model_name: str = DEFAULT_DENSE_MODEL_NAME,
    local_files_only: bool = True,
    dense_worker_mode: str = "full",
    dense_query_batch_size: int = 16,
    device: str | None = None,
    branch_top_k_chunks: int = DEFAULT_BRANCH_TOP_K_CHUNKS,
    second_hop_top_k_chunks: int = DEFAULT_SECOND_HOP_TOP_K_CHUNKS,
    rrf_k: int = DEFAULT_RRF_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    reranker_provider: str = RERANKER_PROVIDER_LOCAL,
    reranker_model_name: str = DEFAULT_RERANK_MODEL_NAME,
    reranker_local_files_only: bool = True,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    reranker_max_length: int = DEFAULT_MAX_LENGTH,
    reranker_instruction_name: str = DEFAULT_INSTRUCTION_NAME,
    reranker_instruction: str = DEFAULT_RERANK_INSTRUCTION,
    dashscope_rerank_url: str | None = None,
    dashscope_rerank_api_key: str | None = None,
    dashscope_rerank_model: str | None = None,
    dashscope_rerank_timeout: float = DEFAULT_DASHSCOPE_RERANK_TIMEOUT,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float = DEFAULT_CHECKER_TEMPERATURE,
    llm_max_tokens: int = DEFAULT_CHECKER_MAX_TOKENS,
    llm_timeout: float = DEFAULT_CHECKER_TIMEOUT,
    checker_mode: str = CHECKER_MODE_TRACEABLE,
    checker_retry_attempts: int = 1,
    context_preview_chars: int = DEFAULT_CONTEXT_PREVIEW_CHARS,
    model_context_window_tokens: int = DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
    max_rounds: int = 4,
    min_retrieval_rounds: int = 0,
    max_queries_per_round: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
    max_raw_chunks_per_checker_call: int = DEFAULT_MAX_RAW_CHUNKS_PER_CHECKER_CALL,
    max_new_raw_chunks_per_round: int = 5,
    max_new_chunks_per_article: int = 1,
    max_visible_chunks_per_article: int = 2,
    limit: int | None = None,
    qids: list[str] | None = None,
    source_case_prefixes: list[str] | None = None,
    checker: Any | None = None,
    checker_client: Any | None = None,
    retriever: Any | None = None,
    reranker: Any | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    validate_args(
        branch_top_k_chunks=branch_top_k_chunks,
        second_hop_top_k_chunks=second_hop_top_k_chunks,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        reranker_provider=reranker_provider,
        dense_query_batch_size=dense_query_batch_size,
        reranker_batch_size=reranker_batch_size,
        reranker_max_length=reranker_max_length,
        dashscope_rerank_timeout=dashscope_rerank_timeout,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_timeout=llm_timeout,
        checker_mode=checker_mode,
        checker_retry_attempts=checker_retry_attempts,
        context_preview_chars=context_preview_chars,
        model_context_window_tokens=model_context_window_tokens,
        max_rounds=max_rounds,
        min_retrieval_rounds=min_retrieval_rounds,
        max_queries_per_round=max_queries_per_round,
        max_raw_chunks_per_checker_call=max_raw_chunks_per_checker_call,
        max_new_raw_chunks_per_round=max_new_raw_chunks_per_round,
        max_new_chunks_per_article=max_new_chunks_per_article,
        max_visible_chunks_per_article=max_visible_chunks_per_article,
        limit=limit,
    )
    console = console or Console()
    rerank_run_dir = Path(rerank_run_dir)
    index_dir = Path(index_dir)

    try:
        rerank_run_config = load_required_json(rerank_run_dir / "run_config.json")
        rerank_metrics = load_required_json(rerank_run_dir / "metrics.json")
        source_hybrid_run_dir = Path(required_config_value(rerank_run_config, "source_hybrid_run_dir"))
        chunks_path = Path(required_config_value(rerank_run_config, "chunks_path"))
        candidate_top_k_chunks = int(
            required_config_value(rerank_run_config, "candidate_top_k_chunks")
        )
        chunks = load_kb_chunks(chunks_path)
        chunk_lookup = build_chunk_lookup(chunks)
        candidate_rows = load_hybrid_candidates(
            source_hybrid_run_dir / "candidates.jsonl",
            chunk_lookup=chunk_lookup,
            expected_top_k_chunks=candidate_top_k_chunks,
            expected_records=int(rerank_metrics["records"]),
            expected_dataset_name=rerank_metrics["dataset_name"],
        )
        rerank_traces = load_rerank_trace_lookup(rerank_run_dir / "rerank_traces.jsonl")
        validate_qid_sets(candidate_rows, rerank_traces)
        candidate_rows = filter_candidate_rows(
            candidate_rows,
            rerank_traces=rerank_traces,
            qids=qids,
            source_case_prefixes=source_case_prefixes,
            limit=limit,
        )
    except (OSError, ValueError, RerankEvalError) as exc:
        raise EvidenceLoopEvalError(str(exc)) from exc

    llm_config = resolve_llm_config(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        checker=checker,
        checker_client=checker_client,
    )
    if checker is None:
        checker_client = checker_client or OpenAICompatibleChatClient(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            model=llm_config["model"],
            timeout=llm_timeout,
        )
        checker_cls = (
            CompactEvidenceChecker
            if checker_mode == CHECKER_MODE_COMPACT
            else TraceableEvidenceChecker
        )
        checker = checker_cls(
            client=checker_client,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            max_next_queries=max_queries_per_round,
            context_preview_chars=context_preview_chars,
        )
        if checker_retry_attempts > 1:
            checker = RetryingEvidenceChecker(checker, max_attempts=checker_retry_attempts)
    reranker_config = build_loop_reranker(
        reranker=reranker,
        provider=reranker_provider,
        model_name=reranker_model_name,
        local_files_only=reranker_local_files_only,
        device=device,
        instruction=reranker_instruction,
        max_length=reranker_max_length,
        dashscope_url=dashscope_rerank_url,
        dashscope_api_key=dashscope_rerank_api_key,
        dashscope_model=dashscope_rerank_model,
        dashscope_timeout=dashscope_rerank_timeout,
    )
    reranker = reranker_config["reranker"]
    effective_reranker_model_name = reranker_config["model_name"]

    loop_config = EvidenceLoopConfig(
        max_rounds=max_rounds,
        min_retrieval_rounds=min_retrieval_rounds,
        max_queries_per_round=max_queries_per_round,
        per_query_retrieve_top_k_chunks=second_hop_top_k_chunks,
        max_raw_chunks_per_checker_call=max_raw_chunks_per_checker_call,
        max_new_raw_chunks_per_round=max_new_raw_chunks_per_round,
        max_new_chunks_per_article=max_new_chunks_per_article,
        max_visible_chunks_per_article=max_visible_chunks_per_article,
        rerank_batch_size=reranker_batch_size,
        budget=ContextBudgetConfig(model_context_window_tokens=model_context_window_tokens),
    )

    retriever_context = (
        nullcontext(retriever)
        if retriever is not None
        else HybridRetriever(
            chunks=chunks,
            index_dir=index_dir,
            model_name=dense_model_name,
            local_files_only=local_files_only,
            device=device,
            dense_worker_mode=dense_worker_mode,
        )
    )
    try:
        with retriever_context as base_retriever:
            loop_retriever = build_loop_retriever_adapter(
                base_retriever,
                branch_top_k_chunks=branch_top_k_chunks,
                rrf_k=rrf_k,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                dense_query_batch_size=dense_query_batch_size,
            )
            traces = []
            prompt_manifests = []
            usage_snapshots = []
            compact_boundaries = []
            for candidate_row in tqdm(candidate_rows, desc="EvidenceContext 8B loop eval"):
                qid = candidate_row["qid"]
                try:
                    initial_context = build_initial_evidence_context(
                        candidate_row=candidate_row,
                        rerank_trace=rerank_traces[qid],
                        chunk_lookup=chunk_lookup,
                    )
                    loop_result = run_evidence_completion_loop(
                        context=initial_context,
                        checker=checker,
                        retriever=loop_retriever,
                        reranker=reranker,
                        chunk_lookup=chunk_lookup,
                        config=loop_config,
                    )
                except (EvidenceContextError, EvidenceLoopError, HybridRetrieverError) as exc:
                    raise EvidenceLoopEvalError(f"Evidence loop failed for qid={qid}: {exc}") from exc
                trace = build_loop_trace(
                    candidate_row=candidate_row,
                    rerank_trace=rerank_traces[qid],
                    loop_result=loop_result,
                    active_top_k_chunks=max_raw_chunks_per_checker_call,
                )
                traces.append(trace)
                prompt_manifests.extend(model_to_dict(row) for row in loop_result.prompt_manifests)
                usage_snapshots.extend(model_to_dict(row) for row in loop_result.usage_snapshots)
                compact_boundaries.extend(model_to_dict(row) for row in loop_result.compact_boundaries)
    except HybridRetrieverError as exc:
        raise EvidenceLoopEvalError(str(exc)) from exc

    summary = summarize_loop_traces(
        traces,
        rerank_metrics=rerank_metrics,
        candidate_rows=candidate_rows,
        active_top_k_chunks=max_raw_chunks_per_checker_call,
        checker_mode=checker_mode,
    )
    run_dir = ensure_dir(
        Path(output_dir)
        / source_hybrid_run_dir.name
        / rerank_run_dir.name
        / build_run_name(
            llm_model=llm_config["model"],
            reranker_model_name=effective_reranker_model_name,
            second_hop_top_k_chunks=second_hop_top_k_chunks,
            model_context_window_tokens=model_context_window_tokens,
            max_raw_chunks_per_checker_call=max_raw_chunks_per_checker_call,
            min_retrieval_rounds=min_retrieval_rounds,
            checker_mode=checker_mode,
        )
    )
    write_outputs(
        run_dir,
        summary=summary,
        run_config={
            "source_hybrid_run": source_hybrid_run_dir.name,
            "source_hybrid_run_dir": str(source_hybrid_run_dir),
            "source_rerank_run": rerank_run_dir.name,
            "source_rerank_run_dir": str(rerank_run_dir),
            "chunks_path": str(chunks_path),
            "index_dir": str(index_dir),
            "dense_model_name": dense_model_name,
            "local_files_only": local_files_only,
            "dense_worker_mode": dense_worker_mode,
            "dense_query_batch_size": dense_query_batch_size,
            "device": device,
            "branch_top_k_chunks": branch_top_k_chunks,
            "second_hop_top_k_chunks": second_hop_top_k_chunks,
            "rrf_k": rrf_k,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "reranker_provider": reranker_config["provider"],
            "reranker_model_name": reranker_model_name,
            "effective_reranker_model_name": effective_reranker_model_name,
            "reranker_local_files_only": reranker_local_files_only,
            "reranker_batch_size": reranker_batch_size,
            "reranker_max_length": reranker_max_length,
            "reranker_instruction_name": reranker_instruction_name,
            "reranker_instruction": reranker_instruction,
            "dashscope_rerank_url": reranker_config["dashscope_url"],
            "dashscope_rerank_api_key_provided": reranker_config["dashscope_api_key_provided"],
            "dashscope_rerank_model": reranker_config["dashscope_model"],
            "dashscope_rerank_timeout": dashscope_rerank_timeout,
            "llm_base_url": llm_config["base_url"],
            "llm_api_key_provided": bool(llm_config["api_key"]),
            "llm_model": llm_config["model"],
            "llm_temperature": llm_temperature,
            "llm_max_tokens": llm_max_tokens,
            "llm_timeout": llm_timeout,
            "checker_mode": checker_mode,
            "checker_retry_attempts": checker_retry_attempts,
            "context_preview_chars": context_preview_chars,
            "model_context_window_tokens": model_context_window_tokens,
            "max_rounds": max_rounds,
            "min_retrieval_rounds": min_retrieval_rounds,
            "max_queries_per_round": max_queries_per_round,
            "max_raw_chunks_per_checker_call": max_raw_chunks_per_checker_call,
            "max_new_raw_chunks_per_round": max_new_raw_chunks_per_round,
            "max_new_chunks_per_article": max_new_chunks_per_article,
            "max_visible_chunks_per_article": max_visible_chunks_per_article,
            "limit": limit,
            "qids": qids or [],
            "source_case_prefixes": source_case_prefixes or [],
            "context_budget": model_to_dict(loop_config.budget),
        },
        traces=traces,
        prompt_manifests=prompt_manifests,
        usage_snapshots=usage_snapshots,
        compact_boundaries=compact_boundaries,
    )
    print_summary(console, summary, run_dir)
    return summary


def validate_args(
    *,
    branch_top_k_chunks: int,
    second_hop_top_k_chunks: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
    reranker_provider: str,
    dense_query_batch_size: int,
    reranker_batch_size: int,
    reranker_max_length: int,
    dashscope_rerank_timeout: float,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_timeout: float,
    checker_mode: str,
    checker_retry_attempts: int,
    context_preview_chars: int,
    model_context_window_tokens: int,
    max_rounds: int,
    min_retrieval_rounds: int,
    max_queries_per_round: int,
    max_raw_chunks_per_checker_call: int,
    max_new_raw_chunks_per_round: int,
    max_new_chunks_per_article: int,
    max_visible_chunks_per_article: int,
    limit: int | None,
) -> None:
    positive_values = {
        "branch_top_k_chunks": branch_top_k_chunks,
        "second_hop_top_k_chunks": second_hop_top_k_chunks,
        "rrf_k": rrf_k,
        "dense_query_batch_size": dense_query_batch_size,
        "reranker_batch_size": reranker_batch_size,
        "reranker_max_length": reranker_max_length,
        "llm_max_tokens": llm_max_tokens,
        "context_preview_chars": context_preview_chars,
        "model_context_window_tokens": model_context_window_tokens,
        "max_queries_per_round": max_queries_per_round,
        "max_raw_chunks_per_checker_call": max_raw_chunks_per_checker_call,
        "max_new_raw_chunks_per_round": max_new_raw_chunks_per_round,
        "max_new_chunks_per_article": max_new_chunks_per_article,
        "max_visible_chunks_per_article": max_visible_chunks_per_article,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise EvidenceLoopEvalError(f"{name} must be positive.")
    if max_rounds < 0:
        raise EvidenceLoopEvalError("max_rounds must be non-negative.")
    if min_retrieval_rounds < 0:
        raise EvidenceLoopEvalError("min_retrieval_rounds must be non-negative.")
    if min_retrieval_rounds > max_rounds:
        raise EvidenceLoopEvalError("min_retrieval_rounds cannot exceed max_rounds.")
    if reranker_provider not in VALID_RERANKER_PROVIDERS:
        raise EvidenceLoopEvalError(
            f"reranker_provider must be one of {sorted(VALID_RERANKER_PROVIDERS)}."
        )
    if checker_mode not in VALID_CHECKER_MODES:
        raise EvidenceLoopEvalError(f"checker_mode must be one of {sorted(VALID_CHECKER_MODES)}.")
    if checker_retry_attempts <= 0:
        raise EvidenceLoopEvalError("checker_retry_attempts must be positive.")
    if limit is not None and limit <= 0:
        raise EvidenceLoopEvalError("limit must be positive when provided.")
    if second_hop_top_k_chunks > branch_top_k_chunks:
        raise EvidenceLoopEvalError("second_hop_top_k_chunks cannot exceed branch_top_k_chunks.")
    if max_queries_per_round > DEFAULT_TRACEABLE_MAX_NEXT_QUERIES:
        raise EvidenceLoopEvalError("Route B defaults to at most 2 next queries per round.")
    if max_new_raw_chunks_per_round > max_raw_chunks_per_checker_call:
        raise EvidenceLoopEvalError("max_new_raw_chunks_per_round cannot exceed raw chunk window.")
    for name, value in (
        ("bm25_weight", bm25_weight),
        ("dense_weight", dense_weight),
        ("dashscope_rerank_timeout", dashscope_rerank_timeout),
        ("llm_temperature", llm_temperature),
        ("llm_timeout", llm_timeout),
    ):
        if not math.isfinite(value) or value < 0:
            raise EvidenceLoopEvalError(f"{name} must be a non-negative finite number.")
    if bm25_weight == 0 or dense_weight == 0 or llm_timeout == 0 or dashscope_rerank_timeout == 0:
        raise EvidenceLoopEvalError(
            "bm25_weight, dense_weight, llm_timeout, and dashscope_rerank_timeout must be positive."
        )


def build_loop_reranker(
    *,
    reranker: Any | None,
    provider: str,
    model_name: str,
    local_files_only: bool,
    device: str | None,
    instruction: str,
    max_length: int,
    dashscope_url: str | None,
    dashscope_api_key: str | None,
    dashscope_model: str | None,
    dashscope_timeout: float,
) -> dict[str, Any]:
    if reranker is not None:
        model = compact_text(model_name) or compact_text(getattr(reranker, "model_name", "")) or "custom_reranker"
        return {
            "reranker": reranker,
            "provider": "custom",
            "model_name": model,
            "dashscope_url": "",
            "dashscope_model": "",
            "dashscope_api_key_provided": False,
        }

    if provider == RERANKER_PROVIDER_LOCAL:
        try:
            return {
                "reranker": CrossEncoderReranker(
                    model_name=model_name,
                    local_files_only=local_files_only,
                    device=device,
                    instruction=instruction,
                    max_length=max_length,
                ),
                "provider": RERANKER_PROVIDER_LOCAL,
                "model_name": model_name,
                "dashscope_url": "",
                "dashscope_model": "",
                "dashscope_api_key_provided": False,
            }
        except CrossEncoderRerankerError as exc:
            raise EvidenceLoopEvalError(str(exc)) from exc

    if provider == RERANKER_PROVIDER_DASHSCOPE:
        config = resolve_dashscope_reranker_config(
            url=dashscope_url,
            api_key=dashscope_api_key,
            model=dashscope_model,
        )
        try:
            reranker = DashScopeReranker(
                url=config["url"],
                api_key=config["api_key"],
                model=config["model"],
                instruction=instruction,
                timeout=dashscope_timeout,
            )
        except DashScopeRerankerError as exc:
            raise EvidenceLoopEvalError(str(exc)) from exc
        return {
            "reranker": reranker,
            "provider": RERANKER_PROVIDER_DASHSCOPE,
            "model_name": f"dashscope/{config['model']}",
            "dashscope_url": config["url"],
            "dashscope_model": config["model"],
            "dashscope_api_key_provided": bool(config["api_key"]),
        }

    raise EvidenceLoopEvalError(f"Unsupported reranker_provider: {provider}.")


def resolve_dashscope_reranker_config(
    *,
    url: str | None,
    api_key: str | None,
    model: str | None,
) -> dict[str, str]:
    resolved_url = first_text(url, os.environ.get("DASHSCOPE_RERANK_URL"), dashscope_rerank_url_from_env())
    resolved_api_key = first_text(
        api_key,
        os.environ.get("DASHSCOPE_RERANK_API_KEY"),
        os.environ.get("DASHSCOPE_API_KEY"),
    )
    resolved_model = first_text(
        model,
        os.environ.get("DASHSCOPE_RERANK_MODEL"),
        dashscope_rerank_model_from_env(),
        DEFAULT_DASHSCOPE_RERANK_MODEL,
    )
    if not resolved_api_key:
        raise EvidenceLoopEvalError(
            "DashScope reranker api key is required. Set DASHSCOPE_RERANK_API_KEY "
            "or DASHSCOPE_API_KEY, or pass --dashscope_rerank_api_key."
        )
    if not resolved_url:
        resolved_url = DEFAULT_DASHSCOPE_RERANK_URL
    return {"url": resolved_url, "api_key": resolved_api_key, "model": resolved_model}


def filter_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    rerank_traces: dict[str, dict[str, Any]] | None = None,
    qids: list[str] | None,
    source_case_prefixes: list[str] | None = None,
    limit: int | None,
) -> list[dict[str, Any]]:
    output = rows
    if qids:
        qid_set = set(qids)
        output = [row for row in output if row["qid"] in qid_set]
        missing = sorted(qid_set - {row["qid"] for row in output})
        if missing:
            raise EvidenceLoopEvalError(f"Requested qids not found: {missing[:5]}.")
    if source_case_prefixes:
        prefixes = tuple(prefix.strip() for prefix in source_case_prefixes if prefix.strip())
        if prefixes:
            if rerank_traces is None:
                raise EvidenceLoopEvalError("source_case_prefixes requires rerank traces.")
            output = [
                row
                for row in output
                if str(rerank_traces[row["qid"]].get("case_type") or "").startswith(prefixes)
            ]
    if limit is not None:
        output = output[:limit]
    if not output:
        raise EvidenceLoopEvalError("No candidate rows selected.")
    return output


def resolve_llm_config(
    *,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
    checker: Any | None,
    checker_client: Any | None,
) -> dict[str, str]:
    model_from_checker = compact_text(getattr(checker, "model", "")) if checker else ""
    model_from_client = compact_text(getattr(checker_client, "model", "")) if checker_client else ""
    base_url = first_text(
        llm_base_url,
        os.environ.get("LLM_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        deepseek_base_url_from_env(),
    )
    api_key = first_text(
        llm_api_key,
        os.environ.get("LLM_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("DEEPSEEK_API_KEY"),
    )
    model = first_text(
        llm_model,
        os.environ.get("LLM_MODEL"),
        os.environ.get("OPENAI_MODEL"),
        os.environ.get("DEEPSEEK_MODEL"),
        model_from_checker,
        model_from_client,
    )
    if checker is None and checker_client is None:
        if not base_url:
            raise EvidenceLoopEvalError(
                "llm_base_url is required. Pass --llm_base_url or set LLM_BASE_URL."
            )
        if not model:
            raise EvidenceLoopEvalError("llm_model is required. Pass --llm_model or set LLM_MODEL.")
    if not model:
        model = "custom_checker"
    return {"base_url": base_url, "api_key": api_key, "model": model}


def first_text(*values: Any) -> str:
    for value in values:
        text = compact_text(value)
        if text:
            return text
    return ""


class HybridLoopRetrieverAdapter:
    def __init__(
        self,
        retriever: Any,
        *,
        branch_top_k_chunks: int,
        rrf_k: int,
        bm25_weight: float,
        dense_weight: float,
        dense_query_batch_size: int,
    ) -> None:
        self.retriever = retriever
        self.branch_top_k_chunks = branch_top_k_chunks
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.dense_query_batch_size = dense_query_batch_size

    def search(self, query_text: str, *, top_k_chunks: int) -> list[dict[str, Any]]:
        batches = self.search_batch([query_text], top_k_chunks=top_k_chunks)
        return batches[0] if batches else []

    def search_batch(self, queries: list[str], *, top_k_chunks: int) -> list[list[dict[str, Any]]]:
        batches = self.retriever.search_batch(
            queries,
            branch_top_k_chunks=self.branch_top_k_chunks,
            fused_top_k_chunks=top_k_chunks,
            rrf_k=self.rrf_k,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            dense_query_batch_size=self.dense_query_batch_size,
        )
        if len(batches) != len(queries):
            raise HybridRetrieverError(
                "Hybrid retriever returned an unexpected batch count: "
                f"results={len(batches)}, queries={len(queries)}."
            )
        return [list(batch["hybrid"][:top_k_chunks]) for batch in batches]


def build_loop_retriever_adapter(
    retriever: Any,
    *,
    branch_top_k_chunks: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
    dense_query_batch_size: int,
) -> Any:
    if hasattr(retriever, "search"):
        return retriever
    return HybridLoopRetrieverAdapter(
        retriever,
        branch_top_k_chunks=branch_top_k_chunks,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        dense_query_batch_size=dense_query_batch_size,
    )


def build_loop_trace(
    *,
    candidate_row: dict[str, Any],
    rerank_trace: dict[str, Any],
    loop_result: EvidenceCompletionLoopResult,
    active_top_k_chunks: int,
) -> dict[str, Any]:
    context = loop_result.context
    example = candidate_row_to_example(candidate_row)
    final_results = [
        {
            "rank": rank,
            "chunk_id": item.chunk_id,
            "article_id": item.article_id,
            "title": item.title,
            "score": item.rerank_score if item.rerank_score is not None else item.score,
            "text_preview": item.text_preview,
        }
        for rank, item in enumerate(context.active_items[:active_top_k_chunks], start=1)
    ]
    ks = select_chunk_ks(active_top_k_chunks)
    metrics = compute_chunk_retrieval_metrics(example.article_ids, final_results, ks)
    final_article_ids = [row["article_id"] for row in final_results]
    baseline_top10_article_ids = [row["article_id"] for row in rerank_trace["top10_reranked_chunks"]]
    source_candidate_article_ids = [
        row["article_id"]
        for row in candidate_row.get("hybrid_candidates", [])
        if row.get("article_id")
    ]
    source_case_type = rerank_trace.get("case_type")
    checker_outputs = loop_result.checker_outputs
    invalid_provenance_chunk_ids = sorted(
        {
            chunk_id
            for output in checker_outputs
            for chunk_id in output.get("invalid_provenance_chunk_ids", [])
        }
    )
    prompt_manifest_rows = [model_to_dict(row) for row in loop_result.prompt_manifests]
    usage_snapshot_rows = [model_to_dict(row) for row in loop_result.usage_snapshots]
    compact_boundary_rows = [model_to_dict(row) for row in loop_result.compact_boundaries]
    gap_query_rows = [model_to_dict(row) for row in context.gap_query_provenance]
    second_hop_retrieved_article_ids = unique_texts(
        article_id
        for row in gap_query_rows
        for article_id in row.get("retrieved_article_ids", [])
    )
    second_hop_candidate_article_ids = unique_texts(
        article_id
        for row in gap_query_rows
        for article_id in row.get("candidate_article_ids", [])
    )
    second_hop_selected_article_ids = unique_texts(
        article_id
        for row in gap_query_rows
        for article_id in row.get("selected_article_ids", [])
    )
    baseline_missing_article_ids = missing_article_ids(example.article_ids, baseline_top10_article_ids)
    final_missing_article_ids = missing_article_ids(example.article_ids, final_article_ids)
    trace = {
        "qid": context.qid,
        "dataset_name": context.dataset_name,
        "question": context.question,
        "answer": context.answer,
        "gold_article_ids": example.article_ids,
        "num_gold_articles": example.num_gold_articles,
        "is_multi_article": example.is_multi_article,
        "source_case_type": source_case_type,
        "baseline_top10_chunk_ids": [row["chunk_id"] for row in rerank_trace["top10_reranked_chunks"]],
        "baseline_top10_article_ids": baseline_top10_article_ids,
        "baseline_missing_article_ids": baseline_missing_article_ids,
        "source_candidate_article_ids": unique_texts(source_candidate_article_ids),
        "source_candidate_full_article_hit": is_full_article_hit(example.article_ids, source_candidate_article_ids),
        "final_active_chunk_ids": [row["chunk_id"] for row in final_results],
        "final_active_article_ids": final_article_ids,
        "final_missing_article_ids": final_missing_article_ids,
        "final_context_top_k_chunks": active_top_k_chunks,
        "baseline_context_full_article_hit": is_full_article_hit(
            example.article_ids,
            baseline_top10_article_ids,
        ),
        "final_context_full_article_hit": is_full_article_hit(example.article_ids, final_article_ids),
        "final_case_type": classify_chunk_case(example.article_ids, final_article_ids, active_top_k_chunks),
        "second_hop_retrieved_article_ids": second_hop_retrieved_article_ids,
        "second_hop_candidate_article_ids": second_hop_candidate_article_ids,
        "second_hop_selected_article_ids": second_hop_selected_article_ids,
        "second_hop_retrieved_missing_article_ids": [
            article_id for article_id in baseline_missing_article_ids if article_id in second_hop_retrieved_article_ids
        ],
        "second_hop_selected_missing_article_ids": [
            article_id for article_id in baseline_missing_article_ids if article_id in second_hop_selected_article_ids
        ],
        "candidate_items_count": len(context.candidate_items),
        "rounds_completed": loop_result.rounds_completed,
        "completed": loop_result.completed,
        "checker_call_count": len(checker_outputs),
        "minimum_retrieval_forced_count": sum(
            bool(output.get("minimum_retrieval_forced")) for output in checker_outputs
        ),
        "retrieval_rounds": loop_result.retrieval_rounds,
        "second_hop_query_count": loop_result.second_hop_query_count,
        "prompt_manifests": prompt_manifest_rows,
        "context_usage_snapshots": usage_snapshot_rows,
        "compact_boundaries": compact_boundary_rows,
        "checker_outputs": checker_outputs,
        "query_history": context.query_history,
        "gap_query_provenance": gap_query_rows,
        "llm_seen_chunk_ids": sorted(
            {
                chunk_id
                for manifest in prompt_manifest_rows
                for chunk_id in manifest["input_chunk_ids"]
            }
        ),
        "provenance_valid": not invalid_provenance_chunk_ids
        and all(output.get("provenance_valid", True) for output in checker_outputs),
        "invalid_provenance_chunk_ids": invalid_provenance_chunk_ids,
        "max_usage_ratio": max(
            (row["usage_ratio"] for row in usage_snapshot_rows),
            default=0.0,
        ),
        "max_budget_status": max_budget_status(usage_snapshot_rows),
    }
    trace["case_type"] = trace["final_case_type"]
    trace.update({key: value for key, value in metrics.items() if key != "is_valid"})
    trace["failure_stage"] = classify_failure_stage(trace)
    return trace


def max_budget_status(snapshots: list[dict[str, Any]]) -> str:
    severity = {
        "healthy": 0,
        "watch": 1,
        "soft_auto_compact": 2,
        "hard_auto_compact": 3,
        "emergency_compact": 4,
    }
    if not snapshots:
        return "healthy"
    return max((row["budget_status"] for row in snapshots), key=lambda status: severity[status])


def summarize_loop_traces(
    traces: list[dict[str, Any]],
    *,
    rerank_metrics: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    active_top_k_chunks: int,
    checker_mode: str = CHECKER_MODE_TRACEABLE,
) -> dict[str, Any]:
    valid_rows = [trace for trace in traces if trace["final_case_type"] != CASE_INVALID]
    invalid_rows = [trace for trace in traces if trace["final_case_type"] == CASE_INVALID]
    ks = select_chunk_ks(active_top_k_chunks)
    summary = summarize_chunk_metrics(
        dataset_name=rerank_metrics["dataset_name"],
        records=len(candidate_rows),
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        ks=ks,
        top_k_chunks=active_top_k_chunks,
    )
    baseline_full = float(rerank_metrics.get("chunk_full_article_hit@10", 0.0))
    final_top10_full = float(summary.get("chunk_full_article_hit@10", 0.0))
    final_context_full = float(summary.get(f"chunk_full_article_hit@{active_top_k_chunks}", final_top10_full))
    sample_baseline_full = average(
        is_full_article_hit(row["gold_article_ids"], row["baseline_top10_article_ids"])
        for row in traces
    )
    loop_name = (
        "Compact EvidenceContext Loop"
        if checker_mode == CHECKER_MODE_COMPACT
        else "Traceable EvidenceContext Loop"
    )
    summary.update(
        {
            "retriever_type": loop_name,
            "checker_mode": checker_mode,
            "retrieval_unit": "chunk",
            "source_rerank_chunk_full_article_hit@10": baseline_full,
            "sample_source_rerank_chunk_full_article_hit@10": sample_baseline_full,
            "final_context_top_k_chunks": active_top_k_chunks,
            "final_chunk_full_article_hit@10": final_top10_full,
            "final_context_chunk_full_article_hit": final_context_full,
            "delta_chunk_full_article_hit@10": final_top10_full - baseline_full,
            "sample_delta_chunk_full_article_hit@10": final_top10_full - sample_baseline_full,
            "delta_context_chunk_full_article_hit": final_context_full - baseline_full,
            "sample_delta_context_chunk_full_article_hit": final_context_full - sample_baseline_full,
            "completed_count": sum(row["completed"] for row in traces),
            "minimum_retrieval_forced_count": sum(row["minimum_retrieval_forced_count"] for row in traces),
            "provenance_invalid_count": sum(not row["provenance_valid"] for row in traces),
            "avg_checker_calls": average(row["checker_call_count"] for row in traces),
            "avg_retrieval_rounds": average(row["retrieval_rounds"] for row in traces),
            "avg_second_hop_queries": average(row["second_hop_query_count"] for row in traces),
            "avg_candidate_items": average(row["candidate_items_count"] for row in traces),
            "avg_max_usage_ratio": average(row["max_usage_ratio"] for row in traces),
            "max_usage_ratio": max((row["max_usage_ratio"] for row in traces), default=0.0),
            "soft_or_worse_budget_count": sum(
                row["max_budget_status"] != "healthy" and row["max_budget_status"] != "watch"
                for row in traces
            ),
            "source_case_B_count": sum(str(row.get("source_case_type", "")).startswith("B_") for row in traces),
            "source_case_C_count": sum(str(row.get("source_case_type", "")).startswith("C_") for row in traces),
            "source_B_or_C_final_full_count": sum(
                str(row.get("source_case_type", "")).startswith(("B_", "C_"))
                and row["final_context_full_article_hit"]
                for row in traces
            ),
            "failure_stage_counts": count_by_key(traces, "failure_stage"),
        }
    )
    return summary


def is_full_article_hit(gold_article_ids: list[str], retrieved_article_ids: list[str]) -> bool:
    gold = set(gold_article_ids)
    return bool(gold) and gold.issubset(set(retrieved_article_ids))


def missing_article_ids(gold_article_ids: list[str], retrieved_article_ids: list[str]) -> list[str]:
    retrieved = set(retrieved_article_ids)
    return [article_id for article_id in gold_article_ids if article_id not in retrieved]


def classify_failure_stage(trace: dict[str, Any]) -> str:
    if trace["final_context_full_article_hit"]:
        return "rescued"
    if trace["baseline_context_full_article_hit"]:
        return "already_full"
    if trace["checker_call_count"] <= 1 and trace["second_hop_query_count"] == 0:
        return "checker_stopped_without_retrieval"
    if trace["second_hop_query_count"] == 0:
        return "checker_no_query"
    if not trace["second_hop_retrieved_missing_article_ids"]:
        return "retrieval_missing_gold"
    if not trace["second_hop_selected_missing_article_ids"]:
        return "reranker_or_diversity_missed_gold"
    if not trace["completed"]:
        return "max_rounds_still_insufficient"
    return "packing_or_metric_missed_gold"


def write_outputs(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    run_config: dict[str, Any],
    traces: list[dict[str, Any]],
    prompt_manifests: list[dict[str, Any]],
    usage_snapshots: list[dict[str, Any]],
    compact_boundaries: list[dict[str, Any]],
) -> None:
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "metrics.json", summary)
    (run_dir / "metrics.md").write_text(render_metrics_markdown(summary), encoding="utf-8")
    write_jsonl(run_dir / "evidence_loop_traces.jsonl", traces)
    write_jsonl(run_dir / "prompt_manifests.jsonl", prompt_manifests)
    write_jsonl(run_dir / "context_usage_snapshots.jsonl", usage_snapshots)
    write_jsonl(run_dir / "compact_boundaries.jsonl", compact_boundaries)
    write_jsonl(
        run_dir / "cases_final_context_not_full.jsonl",
        [row for row in traces if not row["final_context_full_article_hit"]],
    )
    write_jsonl(
        run_dir / "cases_final_context_full.jsonl",
        [row for row in traces if row["final_context_full_article_hit"]],
    )
    write_jsonl(
        run_dir / "cases_invalid_provenance.jsonl",
        [row for row in traces if not row["provenance_valid"]],
    )
    for failure_stage in sorted({row.get("failure_stage", "") for row in traces if row.get("failure_stage")}):
        write_jsonl(
            run_dir / f"cases_{safe_name(failure_stage)}.jsonl",
            [row for row in traces if row.get("failure_stage") == failure_stage],
        )


def render_metrics_markdown(summary: dict[str, Any]) -> str:
    top_k_chunks = int(summary["final_context_top_k_chunks"])
    title = compact_text(summary.get("retriever_type")) or "EvidenceContext Loop"
    lines = [f"# {title}", "", "## 基本信息", ""]
    lines.extend(
        markdown_table(
            ["指标", "数值"],
            [
                ["records", summary["records"]],
                ["valid_records", summary["valid_records"]],
                ["source_rerank_full@10_global", f"{summary['source_rerank_chunk_full_article_hit@10']:.4f}"],
                ["source_rerank_full@10_sample", f"{summary['sample_source_rerank_chunk_full_article_hit@10']:.4f}"],
                ["final_full@10", f"{summary['final_chunk_full_article_hit@10']:.4f}"],
                [f"final_context_full@{top_k_chunks}", f"{summary['final_context_chunk_full_article_hit']:.4f}"],
                ["delta_full@10_vs_global", f"{summary['delta_chunk_full_article_hit@10']:.4f}"],
                ["delta_full@10_sample", f"{summary['sample_delta_chunk_full_article_hit@10']:.4f}"],
                [f"delta_context@{top_k_chunks}_vs_global", f"{summary['delta_context_chunk_full_article_hit']:.4f}"],
                [f"delta_context@{top_k_chunks}_sample", f"{summary['sample_delta_context_chunk_full_article_hit']:.4f}"],
                ["completed_count", summary["completed_count"]],
                ["minimum_retrieval_forced_count", summary["minimum_retrieval_forced_count"]],
                ["provenance_invalid_count", summary["provenance_invalid_count"]],
                ["avg_checker_calls", f"{summary['avg_checker_calls']:.2f}"],
                ["avg_retrieval_rounds", f"{summary['avg_retrieval_rounds']:.2f}"],
                ["avg_second_hop_queries", f"{summary['avg_second_hop_queries']:.2f}"],
                ["avg_candidate_items", f"{summary['avg_candidate_items']:.2f}"],
                ["avg_max_usage_ratio", f"{summary['avg_max_usage_ratio']:.4f}"],
                ["max_usage_ratio", f"{summary['max_usage_ratio']:.4f}"],
                ["soft_or_worse_budget_count", summary["soft_or_worse_budget_count"]],
                ["source_B_or_C_final_full_count", summary["source_B_or_C_final_full_count"]],
            ],
        )
    )
    if summary.get("failure_stage_counts"):
        lines.extend(["", "## Failure Stage Counts", ""])
        lines.extend(
            markdown_table(
                ["failure_stage", "count"],
                [[key, value] for key, value in sorted(summary["failure_stage_counts"].items())],
            )
        )
    lines.extend(["", "## Final Context 指标", ""])
    lines.extend(
        markdown_table(
            [
                "top_k_chunks",
                "chunk_hit",
                "chunk_full_article_hit",
                "chunk_article_recall",
                "chunk_gold_rate",
                "unique_articles",
                "duplicate_article_ratio",
            ],
            metric_rows({**summary, "top_k_chunks": top_k_chunks}),
        )
    )
    return "\n".join(lines)


def print_summary(console: Console, summary: dict[str, Any], run_dir: Path) -> None:
    console.print(
        "[bold green]EvidenceContext 8B loop complete[/bold green] "
        f"records={summary['records']} "
        f"source_full@10_sample={summary['sample_source_rerank_chunk_full_article_hit@10']:.4f} "
        f"final_context_full@{summary['final_context_top_k_chunks']}="
        f"{summary['final_context_chunk_full_article_hit']:.4f} "
        f"context_sample_delta={summary['sample_delta_context_chunk_full_article_hit']:.4f} "
        f"run_dir={run_dir}"
    )


def build_run_name(
    *,
    llm_model: str,
    reranker_model_name: str,
    second_hop_top_k_chunks: int,
    model_context_window_tokens: int,
    max_raw_chunks_per_checker_call: int,
    min_retrieval_rounds: int = 0,
    checker_mode: str = CHECKER_MODE_TRACEABLE,
) -> str:
    name = (
        f"loop_{safe_name(llm_model)}_h{second_hop_top_k_chunks}_"
        f"{safe_name(reranker_model_name)}_w{max_raw_chunks_per_checker_call}_"
        f"ctx{model_context_window_tokens // 1024}k"
    )
    if min_retrieval_rounds:
        name += f"_rmin{min_retrieval_rounds}"
    if checker_mode != CHECKER_MODE_TRACEABLE:
        name += f"_{safe_name(checker_mode)}"
    return name


def safe_name(value: str) -> str:
    output = compact_text(value).lower()
    output = output.replace("/", "-").replace(".", "p")
    output = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in output)
    while "--" in output:
        output = output.replace("--", "-")
    return output.strip("-") or "model"


def count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def unique_texts(values: Any) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
