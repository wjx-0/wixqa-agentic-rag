from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.evaluation.chunk_eval import (
    classify_chunk_case,
    result_article_ids,
)
from src.evaluation.eval_utils import BM25EvalError, markdown_table
from src.evaluation.run_chunk_bm25_eval import load_kb_chunks
from src.evaluation.run_rerank_eval import (
    RerankEvalError,
    build_chunk_lookup,
    candidate_row_to_example,
    load_hybrid_candidates,
)
from src.evaluation.run_rule_second_hop_eval import (
    DEFAULT_BASELINE_RERANK_RUN_DIR,
    DEFAULT_BRANCH_TOP_K_CHUNKS,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_FIRST_HOP_HYBRID_RUN_DIR,
    DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
    DEFAULT_INDEX_DIR,
    DEFAULT_RRF_K,
    DEFAULT_SECOND_HOP_TOP_K_CHUNKS,
    DEFAULT_TOP100_CONTROL_RERANK_RUN_DIR,
    RuleSecondHopEvalError,
    average,
    count_true,
    load_control_hybrid_run_dir,
    load_json_object,
    load_rerank_trace_lookup,
    run_second_hop_retrieval,
    validate_candidate_cutoff,
    validate_fair_top100_control,
)
from src.llm.evidence_checker import (
    DEFAULT_CHECKER_MAX_TOKENS,
    DEFAULT_CHECKER_TEMPERATURE,
    DEFAULT_CHECKER_TIMEOUT,
    DEFAULT_CONTEXT_PREVIEW_CHARS,
    DEFAULT_MAX_NEXT_QUERIES,
    EvidenceCheckerError,
    OpenAICompatibleChatClient,
    build_evidence_checker_messages,
    parse_checker_response,
)
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME
from src.retrievers.rrf import DEFAULT_BM25_WEIGHT
from src.retrievers.rule_second_hop import RuleSecondHopError, merge_chunk_candidates
from src.utils.io_utils import ensure_dir, write_json, write_jsonl
from src.utils.text_utils import compact_text


DEFAULT_OUTPUT_DIR = "outputs/llm_evidence_checker"
DEFAULT_RULE_SECOND_HOP_RUN_DIR = (
    "outputs/rule_second_hop/"
    "hybrid_rrf_b100_f50_k60_bw1_dw2_wixqa_expertwritten/"
    "qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024/"
    "title_expand_s3_h20"
)


class LLMEvidenceCheckerEvalError(RuntimeError):
    pass


def run_llm_evidence_checker_eval(
    *,
    first_hop_hybrid_run_dir: str | Path = DEFAULT_FIRST_HOP_HYBRID_RUN_DIR,
    baseline_rerank_run_dir: str | Path = DEFAULT_BASELINE_RERANK_RUN_DIR,
    top100_control_rerank_run_dir: str | Path = DEFAULT_TOP100_CONTROL_RERANK_RUN_DIR,
    rule_second_hop_run_dir: str | Path = DEFAULT_RULE_SECOND_HOP_RUN_DIR,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dense_model_name: str = DEFAULT_DENSE_MODEL_NAME,
    local_files_only: bool = True,
    dense_worker_mode: str = "full",
    dense_query_batch_size: int = 16,
    device: str | None = None,
    branch_top_k_chunks: int = DEFAULT_BRANCH_TOP_K_CHUNKS,
    second_hop_top_k_chunks: int = DEFAULT_SECOND_HOP_TOP_K_CHUNKS,
    max_next_queries: int = DEFAULT_MAX_NEXT_QUERIES,
    rrf_k: int = DEFAULT_RRF_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float = DEFAULT_CHECKER_TEMPERATURE,
    llm_max_tokens: int = DEFAULT_CHECKER_MAX_TOKENS,
    llm_timeout: float = DEFAULT_CHECKER_TIMEOUT,
    llm_concurrency: int = 1,
    context_preview_chars: int = DEFAULT_CONTEXT_PREVIEW_CHARS,
    checker_client: Any | None = None,
    retriever: Any | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    validate_args(
        branch_top_k_chunks=branch_top_k_chunks,
        second_hop_top_k_chunks=second_hop_top_k_chunks,
        max_next_queries=max_next_queries,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        dense_query_batch_size=dense_query_batch_size,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_timeout=llm_timeout,
        llm_concurrency=llm_concurrency,
        context_preview_chars=context_preview_chars,
    )
    console = console or Console()
    first_hop_hybrid_run_dir = Path(first_hop_hybrid_run_dir)
    baseline_rerank_run_dir = Path(baseline_rerank_run_dir)
    top100_control_rerank_run_dir = Path(top100_control_rerank_run_dir)
    rule_second_hop_run_dir = Path(rule_second_hop_run_dir)
    chunks_path = Path(chunks_path)

    try:
        first_hop_metrics = load_json_object(first_hop_hybrid_run_dir / "metrics.json")
        validate_candidate_cutoff(
            first_hop_metrics,
            DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
            artifact_name="First-hop Hybrid metrics",
        )
        chunks = load_kb_chunks(chunks_path)
        chunk_lookup = build_chunk_lookup(chunks)
        candidate_rows = load_hybrid_candidates(
            first_hop_hybrid_run_dir / "candidates.jsonl",
            chunk_lookup=chunk_lookup,
            expected_top_k_chunks=DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
            expected_records=int(first_hop_metrics["records"]),
            expected_dataset_name=first_hop_metrics["dataset_name"],
        )
        baseline_metrics = load_json_object(baseline_rerank_run_dir / "metrics.json")
        validate_candidate_cutoff(
            baseline_metrics,
            DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
            artifact_name="Baseline reranker metrics",
        )
        baseline_traces = load_rerank_trace_lookup(
            baseline_rerank_run_dir / "rerank_traces.jsonl"
        )
        control_metrics = load_json_object(top100_control_rerank_run_dir / "metrics.json")
        validate_candidate_cutoff(
            control_metrics,
            100,
            artifact_name="Top100 control reranker metrics",
        )
        control_config = load_json_object(top100_control_rerank_run_dir / "run_config.json")
        control_hybrid_run_dir = load_control_hybrid_run_dir(control_config)
        control_hybrid_metrics = load_json_object(control_hybrid_run_dir / "metrics.json")
        validate_fair_top100_control(
            first_hop_metrics=first_hop_metrics,
            control_hybrid_metrics=control_hybrid_metrics,
        )
        rule_second_hop_metrics = load_json_object(rule_second_hop_run_dir / "metrics.json")
        validate_candidate_baseline_qids(candidate_rows, baseline_traces)
    except (
        BM25EvalError,
        OSError,
        ValueError,
        RerankEvalError,
        RuleSecondHopEvalError,
    ) as exc:
        raise LLMEvidenceCheckerEvalError(str(exc)) from exc

    llm_config = resolve_llm_config(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
        checker_client=checker_client,
    )
    checker_client = checker_client or OpenAICompatibleChatClient(
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
        model=llm_config["model"],
        timeout=llm_timeout,
    )

    checker_traces = run_checker_batch(
        candidate_rows,
        baseline_traces,
        checker_client=checker_client,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        max_next_queries=max_next_queries,
        context_preview_chars=context_preview_chars,
        llm_concurrency=llm_concurrency,
    )
    query_plans: dict[str, list[dict[str, Any]]] = {
        trace["qid"]: trace["second_hop_queries"] for trace in checker_traces
    }

    retrieval_batches = run_second_hop_retrieval(
        query_plans,
        chunks=chunks,
        index_dir=index_dir,
        dense_model_name=dense_model_name,
        local_files_only=local_files_only,
        device=device,
        dense_worker_mode=dense_worker_mode,
        branch_top_k_chunks=branch_top_k_chunks,
        second_hop_top_k_chunks=second_hop_top_k_chunks,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        dense_query_batch_size=dense_query_batch_size,
        retriever=retriever,
    )

    merged_candidate_upper_bound = (
        DEFAULT_FIRST_HOP_TOP_K_CHUNKS + max_next_queries * second_hop_top_k_chunks
    )
    traces = []
    checker_traces_by_qid = {row["qid"]: row for row in checker_traces}
    for candidate_row in tqdm(candidate_rows, desc="LLM checker pool eval"):
        qid = candidate_row["qid"]
        checker_trace = checker_traces_by_qid[qid]
        try:
            merged_candidates = merge_chunk_candidates(
                candidate_row["hybrid_candidates"],
                retrieval_batches[qid],
            )
        except RuleSecondHopError as exc:
            raise LLMEvidenceCheckerEvalError(f"Candidate merge failed for qid={qid}: {exc}") from exc
        if len(merged_candidates) > merged_candidate_upper_bound:
            raise LLMEvidenceCheckerEvalError(
                f"Merged candidate pool exceeds {merged_candidate_upper_bound} chunks for qid={qid}."
            )
        traces.append(
            build_pool_eval_trace(
                candidate_row=candidate_row,
                baseline_trace=baseline_traces[qid],
                checker_trace=checker_trace,
                second_hop_batches=retrieval_batches[qid],
                merged_candidates=merged_candidates,
            )
        )

    summary = summarize_checker_traces(
        traces,
        baseline_metrics=baseline_metrics,
        control_metrics=control_metrics,
        rule_second_hop_metrics=rule_second_hop_metrics,
        merged_candidate_upper_bound=merged_candidate_upper_bound,
    )
    run_dir = ensure_dir(
        Path(output_dir)
        / first_hop_hybrid_run_dir.name
        / baseline_rerank_run_dir.name
        / build_run_name(
            model_name=llm_config["model"],
            max_next_queries=max_next_queries,
            second_hop_top_k_chunks=second_hop_top_k_chunks,
        )
    )
    write_outputs(
        run_dir,
        traces=traces,
        summary=summary,
        run_config={
            "first_hop_hybrid_run_dir": str(first_hop_hybrid_run_dir),
            "baseline_rerank_run_dir": str(baseline_rerank_run_dir),
            "top100_control_rerank_run_dir": str(top100_control_rerank_run_dir),
            "rule_second_hop_run_dir": str(rule_second_hop_run_dir),
            "chunks_path": str(chunks_path),
            "index_dir": str(index_dir),
            "dense_model_name": dense_model_name,
            "local_files_only": local_files_only,
            "dense_worker_mode": dense_worker_mode,
            "dense_query_batch_size": dense_query_batch_size,
            "device": device,
            "branch_top_k_chunks": branch_top_k_chunks,
            "second_hop_top_k_chunks": second_hop_top_k_chunks,
            "max_next_queries": max_next_queries,
            "rrf_k": rrf_k,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "llm_base_url": llm_config["base_url"],
            "llm_api_key_provided": bool(llm_config["api_key"]),
            "llm_model": llm_config["model"],
            "llm_temperature": llm_temperature,
            "llm_max_tokens": llm_max_tokens,
            "llm_timeout": llm_timeout,
            "llm_concurrency": llm_concurrency,
            "context_preview_chars": context_preview_chars,
            "eval_scope": "pool_level_only_no_final_rerank",
            "merged_candidate_upper_bound": merged_candidate_upper_bound,
        },
    )
    print_summary(console, summary, run_dir)
    return summary


def validate_args(
    *,
    branch_top_k_chunks: int,
    second_hop_top_k_chunks: int,
    max_next_queries: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
    dense_query_batch_size: int,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_timeout: float,
    llm_concurrency: int,
    context_preview_chars: int,
) -> None:
    positive_values = {
        "branch_top_k_chunks": branch_top_k_chunks,
        "second_hop_top_k_chunks": second_hop_top_k_chunks,
        "max_next_queries": max_next_queries,
        "rrf_k": rrf_k,
        "dense_query_batch_size": dense_query_batch_size,
        "llm_max_tokens": llm_max_tokens,
        "llm_concurrency": llm_concurrency,
        "context_preview_chars": context_preview_chars,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise LLMEvidenceCheckerEvalError(f"{name} must be a positive integer.")
    if second_hop_top_k_chunks > branch_top_k_chunks:
        raise LLMEvidenceCheckerEvalError(
            "second_hop_top_k_chunks must be smaller than or equal to branch_top_k_chunks."
        )
    for name, value in (
        ("bm25_weight", bm25_weight),
        ("dense_weight", dense_weight),
        ("llm_temperature", llm_temperature),
        ("llm_timeout", llm_timeout),
    ):
        if not math.isfinite(value) or value < 0:
            raise LLMEvidenceCheckerEvalError(f"{name} must be a non-negative number.")
    if bm25_weight == 0 or dense_weight == 0 or llm_timeout == 0:
        raise LLMEvidenceCheckerEvalError("bm25_weight, dense_weight, and llm_timeout must be positive.")


def validate_candidate_baseline_qids(
    candidate_rows: list[dict[str, Any]],
    baseline_traces: dict[str, dict[str, Any]],
) -> None:
    candidate_qids = {row["qid"] for row in candidate_rows}
    baseline_qids = set(baseline_traces)
    if candidate_qids != baseline_qids:
        raise LLMEvidenceCheckerEvalError(
            "Qid set mismatch between first-hop candidates and baseline reranker traces: "
            f"candidates={len(candidate_qids)}, traces={len(baseline_qids)}."
        )


def resolve_llm_config(
    *,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
    llm_timeout: float,
    checker_client: Any | None,
) -> dict[str, str]:
    model_from_client = compact_text(getattr(checker_client, "model", "")) if checker_client else ""
    base_url = first_text(llm_base_url, os.environ.get("LLM_BASE_URL"), os.environ.get("OPENAI_BASE_URL"))
    api_key = first_text(llm_api_key, os.environ.get("LLM_API_KEY"), os.environ.get("OPENAI_API_KEY"))
    model = first_text(llm_model, os.environ.get("LLM_MODEL"), os.environ.get("OPENAI_MODEL"), model_from_client)
    if checker_client is None:
        if not base_url:
            raise LLMEvidenceCheckerEvalError(
                "llm_base_url is required. Pass --llm_base_url or set LLM_BASE_URL."
            )
        if not model:
            raise LLMEvidenceCheckerEvalError(
                "llm_model is required. Pass --llm_model or set LLM_MODEL."
            )
    if checker_client is not None and not model:
        model = "custom_llm"
    if llm_timeout <= 0:
        raise LLMEvidenceCheckerEvalError("llm_timeout must be positive.")
    return {"base_url": base_url, "api_key": api_key, "model": model}


def first_text(*values: Any) -> str:
    for value in values:
        text = compact_text(value)
        if text:
            return text
    return ""


def run_checker_for_row(
    *,
    candidate_row: dict[str, Any],
    baseline_trace: dict[str, Any],
    checker_client: Any,
    llm_temperature: float,
    llm_max_tokens: int,
    max_next_queries: int,
    context_preview_chars: int,
) -> dict[str, Any]:
    qid = candidate_row["qid"]
    top10_chunks = baseline_trace["top10_reranked_chunks"]
    try:
        messages = build_evidence_checker_messages(
            candidate_row["question"],
            top10_chunks,
            max_context_chars=context_preview_chars,
        )
        raw_text = checker_client.complete(
            messages,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )
    except EvidenceCheckerError as exc:
        raise LLMEvidenceCheckerEvalError(f"LLM checker call failed for qid={qid}: {exc}") from exc

    checker_error = None
    try:
        checker_result = parse_checker_response(raw_text, max_next_queries=max_next_queries)
        checker_valid = True
    except EvidenceCheckerError as exc:
        checker_result = empty_checker_result()
        checker_valid = False
        checker_error = str(exc)

    should_retrieve = (
        checker_valid
        and checker_result["sufficient"] is False
        and bool(checker_result["blocking_missing_evidence"])
    )
    next_queries = checker_result["next_queries"] if should_retrieve else []
    second_hop_queries = [
        {"query_id": f"gap_query_{index}", "query_text": query}
        for index, query in enumerate(next_queries, start=1)
    ]
    return {
        "qid": qid,
        "checker_raw_text": raw_text,
        "checker_result": checker_result,
        "checker_valid": checker_valid,
        "checker_error": checker_error,
        "checker_sufficient": checker_result["sufficient"] if checker_valid else None,
        "known_facts": checker_result["known_facts"],
        "missing_evidence": checker_result["missing_evidence"],
        "blocking_missing_evidence": checker_result["blocking_missing_evidence"],
        "nice_to_have_missing_evidence": checker_result["nice_to_have_missing_evidence"],
        "next_queries": next_queries,
        "second_hop_queries": second_hop_queries,
        "retrieval_triggered": bool(second_hop_queries),
        "llm_calls": 1,
    }


def run_checker_batch(
    candidate_rows: list[dict[str, Any]],
    baseline_traces: dict[str, dict[str, Any]],
    *,
    checker_client: Any,
    llm_temperature: float,
    llm_max_tokens: int,
    max_next_queries: int,
    context_preview_chars: int,
    llm_concurrency: int,
) -> list[dict[str, Any]]:
    if llm_concurrency <= 1:
        traces = []
        for candidate_row in tqdm(candidate_rows, desc="LLM evidence checker"):
            qid = candidate_row["qid"]
            traces.append(
                run_checker_for_row(
                    candidate_row=candidate_row,
                    baseline_trace=baseline_traces[qid],
                    checker_client=checker_client,
                    llm_temperature=llm_temperature,
                    llm_max_tokens=llm_max_tokens,
                    max_next_queries=max_next_queries,
                    context_preview_chars=context_preview_chars,
                )
            )
        return traces

    traces: list[dict[str, Any] | None] = [None] * len(candidate_rows)
    with ThreadPoolExecutor(max_workers=llm_concurrency) as executor:
        futures = {
            executor.submit(
                run_checker_for_row,
                candidate_row=candidate_row,
                baseline_trace=baseline_traces[candidate_row["qid"]],
                checker_client=checker_client,
                llm_temperature=llm_temperature,
                llm_max_tokens=llm_max_tokens,
                max_next_queries=max_next_queries,
                context_preview_chars=context_preview_chars,
            ): index
            for index, candidate_row in enumerate(candidate_rows)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM evidence checker"):
            traces[futures[future]] = future.result()

    return [trace for trace in traces if trace is not None]


def empty_checker_result() -> dict[str, Any]:
    return {
        "sufficient": None,
        "known_facts": [],
        "missing_evidence": [],
        "blocking_missing_evidence": [],
        "nice_to_have_missing_evidence": [],
        "next_queries": [],
        "reason": "",
    }


def build_pool_eval_trace(
    *,
    candidate_row: dict[str, Any],
    baseline_trace: dict[str, Any],
    checker_trace: dict[str, Any],
    second_hop_batches: list[dict[str, Any]],
    merged_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    example = candidate_row_to_example(candidate_row)
    first_hop_candidates = candidate_row["hybrid_candidates"]
    baseline_top10 = baseline_trace["top10_reranked_chunks"]
    diagnostics = build_pool_diagnostics(
        gold_article_ids=example.article_ids,
        first_hop_candidates=first_hop_candidates,
        baseline_top10=baseline_top10,
        merged_candidates=merged_candidates,
        is_multi_article=example.is_multi_article,
    )
    first_hop_chunk_ids = {row["chunk_id"] for row in first_hop_candidates}
    first_hop_article_ids = set(result_article_ids(first_hop_candidates))
    new_candidates = [
        row for row in merged_candidates if row["chunk_id"] not in first_hop_chunk_ids
    ]
    return {
        "qid": example.qid,
        "dataset_name": example.dataset_name,
        "question": example.question,
        "answer": example.answer,
        "gold_article_ids": example.article_ids,
        "num_gold_articles": example.num_gold_articles,
        "is_multi_article": example.is_multi_article,
        "baseline_top10_chunk_ids": [row["chunk_id"] for row in baseline_top10],
        "baseline_top10_article_ids": result_article_ids(baseline_top10),
        "first_hop_top50_article_ids": result_article_ids(first_hop_candidates),
        "merged_pool_article_ids": result_article_ids(merged_candidates),
        "source_case_type": classify_chunk_case(
            example.article_ids,
            result_article_ids(first_hop_candidates),
            DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
        ),
        **checker_trace,
        "second_hop_results": second_hop_batches,
        "merged_candidate_count": len(merged_candidates),
        "new_chunk_ids": [row["chunk_id"] for row in new_candidates],
        "new_article_ids": sorted(
            set(result_article_ids(new_candidates)) - first_hop_article_ids
        ),
        "retrieval_rounds": 1 + int(bool(second_hop_batches)),
        **diagnostics,
    }


def build_pool_diagnostics(
    *,
    gold_article_ids: list[str],
    first_hop_candidates: list[dict[str, Any]],
    baseline_top10: list[dict[str, Any]],
    merged_candidates: list[dict[str, Any]],
    is_multi_article: bool,
) -> dict[str, Any]:
    gold_set = {article_id for article_id in gold_article_ids if article_id}
    first_pool = set(result_article_ids(first_hop_candidates))
    merged_pool = set(result_article_ids(merged_candidates))
    baseline_top10_ids = set(result_article_ids(baseline_top10))
    source_c = bool(gold_set) and not gold_set.issubset(first_pool)
    source_a = bool(gold_set) and gold_set.issubset(baseline_top10_ids)
    return {
        "source_C": source_c,
        "source_A": source_a,
        "LLM_C_pool_rescued": source_c and gold_set.issubset(merged_pool),
        "merged_pool_full_article_hit": int(bool(gold_set) and gold_set.issubset(merged_pool)),
        "merged_pool_article_recall": len(gold_set & merged_pool) / len(gold_set) if gold_set else 0.0,
        "pool_new_gold_article_ids": sorted((gold_set & merged_pool) - (gold_set & first_pool)),
        "pool_still_missing_gold_article_ids": sorted(gold_set - merged_pool),
        "is_multi_article": is_multi_article,
    }


def summarize_checker_traces(
    traces: list[dict[str, Any]],
    *,
    baseline_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    rule_second_hop_metrics: dict[str, Any],
    merged_candidate_upper_bound: int,
) -> dict[str, Any]:
    source_c_rows = [row for row in traces if row["source_C"]]
    source_a_rows = [row for row in traces if row["source_A"]]
    checker_valid_rows = [row for row in traces if row["checker_valid"]]
    checker_insufficient_rows = [
        row for row in checker_valid_rows if row["checker_sufficient"] is False
    ]
    source_c_checker_insufficient_rows = [
        row for row in source_c_rows
        if row["checker_valid"] and row["checker_sufficient"] is False
    ]
    source_a_checker_sufficient_rows = [
        row for row in source_a_rows
        if row["checker_valid"] and row["checker_sufficient"] is True
    ]
    source_a_checker_insufficient_rows = [
        row for row in source_a_rows
        if row["checker_valid"] and row["checker_sufficient"] is False
    ]
    source_a_retrieval_rows = [
        row for row in source_a_rows if row.get("retrieval_triggered")
    ]
    summary = {
        "dataset_name": baseline_metrics["dataset_name"],
        "records": len(traces),
        "eval_scope": "pool_level_only_no_final_rerank",
        "retriever_type": "LLM Evidence Sufficiency Checker + Gap Query Pool Eval",
        "first_hop_candidate_top_k_chunks": DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
        "merged_candidate_upper_bound": merged_candidate_upper_bound,
        "source_baseline_metrics": baseline_metrics,
        "top100_control_metrics": control_metrics,
        "phase7_rule_second_hop_metrics": rule_second_hop_metrics,
        "checker_valid_count": len(checker_valid_rows),
        "invalid_json_count": len(traces) - len(checker_valid_rows),
        "checker_sufficient_count": sum(
            row["checker_sufficient"] is True for row in checker_valid_rows
        ),
        "checker_insufficient_count": len(checker_insufficient_rows),
        "source_A_count": len(source_a_rows),
        "source_C_count": len(source_c_rows),
        "source_A_sufficient_rate": (
            len(source_a_checker_sufficient_rows) / len(source_a_rows) if source_a_rows else 0.0
        ),
        "source_A_insufficient_count": len(source_a_checker_insufficient_rows),
        "source_A_unnecessary_retrieval_count": len(source_a_retrieval_rows),
        "avg_queries_for_source_A": average(
            len(row["second_hop_queries"]) for row in source_a_rows
        ),
        "source_C_insufficient_rate": (
            len(source_c_checker_insufficient_rows) / len(source_c_rows) if source_c_rows else 0.0
        ),
        "source_C_checker_sufficient_count": sum(
            row["checker_valid"] and row["checker_sufficient"] is True
            for row in source_c_rows
        ),
        "LLM_C_pool_rescued_count": count_true(traces, "LLM_C_pool_rescued"),
        "multi_LLM_C_pool_rescued_count": count_true(
            traces,
            "LLM_C_pool_rescued",
            multi_only=True,
        ),
        "pool_new_gold_articles_count": sum(
            len(row["pool_new_gold_article_ids"]) for row in traces
        ),
        "multi_pool_new_gold_articles_count": sum(
            len(row["pool_new_gold_article_ids"])
            for row in traces
            if row["is_multi_article"]
        ),
        "merged_pool_full_article_hit_rate": average(
            row["merged_pool_full_article_hit"] for row in traces
        ),
        "multi_merged_pool_full_article_hit_rate": average(
            row["merged_pool_full_article_hit"]
            for row in traces
            if row["is_multi_article"]
        ),
        "avg_llm_calls": average(row["llm_calls"] for row in traces),
        "avg_generated_queries": average(len(row["next_queries"]) for row in traces),
        "avg_second_hop_queries": average(len(row["second_hop_queries"]) for row in traces),
        "avg_merged_candidates": average(row["merged_candidate_count"] for row in traces),
        "avg_retrieval_rounds": average(row["retrieval_rounds"] for row in traces),
    }
    return summary


def write_outputs(
    run_dir: Path,
    *,
    traces: list[dict[str, Any]],
    summary: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "metrics.json", summary)
    (run_dir / "metrics.md").write_text(render_metrics_markdown(summary), encoding="utf-8")
    (run_dir / "comparison.md").write_text(
        render_comparison_markdown(summary),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "checker_traces.jsonl", traces)
    write_jsonl(
        run_dir / "cases_invalid_checker_json.jsonl",
        [row for row in traces if not row["checker_valid"]],
    )
    write_jsonl(
        run_dir / "cases_checker_insufficient.jsonl",
        [row for row in traces if row["checker_valid"] and row["checker_sufficient"] is False],
    )
    write_jsonl(
        run_dir / "cases_C_pool_rescued_by_llm_checker.jsonl",
        [row for row in traces if row["LLM_C_pool_rescued"]],
    )
    write_jsonl(
        run_dir / "multi_cases_C_pool_rescued_by_llm_checker.jsonl",
        [row for row in traces if row["LLM_C_pool_rescued"] and row["is_multi_article"]],
    )
    write_jsonl(
        run_dir / "cases_source_C_checker_sufficient_miss.jsonl",
        [
            row for row in traces
            if row["source_C"] and row["checker_valid"] and row["checker_sufficient"] is True
        ],
    )


def render_metrics_markdown(summary: dict[str, Any]) -> str:
    rows = [
        ["records", summary["records"]],
        ["eval_scope", summary["eval_scope"]],
        ["checker_valid_count", summary["checker_valid_count"]],
        ["invalid_json_count", summary["invalid_json_count"]],
        ["checker_sufficient_count", summary["checker_sufficient_count"]],
        ["checker_insufficient_count", summary["checker_insufficient_count"]],
        ["source_A_sufficient_rate", f"{summary['source_A_sufficient_rate']:.4f}"],
        ["source_A_insufficient_count", summary["source_A_insufficient_count"]],
        [
            "source_A_unnecessary_retrieval_count",
            summary["source_A_unnecessary_retrieval_count"],
        ],
        ["avg_queries_for_source_A", f"{summary['avg_queries_for_source_A']:.4f}"],
        ["source_C_insufficient_rate", f"{summary['source_C_insufficient_rate']:.4f}"],
        ["source_C_checker_sufficient_count", summary["source_C_checker_sufficient_count"]],
        ["LLM_C_pool_rescued_count", summary["LLM_C_pool_rescued_count"]],
        ["multi_LLM_C_pool_rescued_count", summary["multi_LLM_C_pool_rescued_count"]],
        ["pool_new_gold_articles_count", summary["pool_new_gold_articles_count"]],
        ["multi_pool_new_gold_articles_count", summary["multi_pool_new_gold_articles_count"]],
        ["merged_pool_full_article_hit_rate", f"{summary['merged_pool_full_article_hit_rate']:.4f}"],
        [
            "multi_merged_pool_full_article_hit_rate",
            f"{summary['multi_merged_pool_full_article_hit_rate']:.4f}",
        ],
        ["avg_llm_calls", f"{summary['avg_llm_calls']:.4f}"],
        ["avg_generated_queries", f"{summary['avg_generated_queries']:.4f}"],
        ["avg_second_hop_queries", f"{summary['avg_second_hop_queries']:.4f}"],
        ["avg_merged_candidates", f"{summary['avg_merged_candidates']:.4f}"],
        ["avg_retrieval_rounds", f"{summary['avg_retrieval_rounds']:.4f}"],
    ]
    lines = ["# LLM Evidence Sufficiency Checker Pool-level Eval", ""]
    lines.extend(markdown_table(["metric", "value"], rows))
    return "\n".join(lines)


def render_comparison_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["source_baseline_metrics"]
    control = summary["top100_control_metrics"]
    rule = summary["phase7_rule_second_hop_metrics"]
    rows = [
        [
            "Top50 Hybrid + Qwen3 baseline",
            "final top10 rerank",
            metric_value(baseline, "chunk_full_article_hit@10"),
            metric_value(baseline, "multi_chunk_full_article_hit@10"),
            "-",
            "-",
            "-",
        ],
        [
            "Top100 Hybrid + Qwen3 control",
            "final top10 rerank",
            metric_value(control, "chunk_full_article_hit@10"),
            metric_value(control, "multi_chunk_full_article_hit@10"),
            "-",
            "-",
            "-",
        ],
        [
            "Phase 7 rule title-expanded second-hop",
            "pool + final top10 diagnostics",
            metric_value(rule, "chunk_full_article_hit@10"),
            metric_value(rule, "multi_chunk_full_article_hit@10"),
            str(rule.get("C_pool_rescued_count", "-")),
            str(rule.get("multi_C_pool_rescued_count", "-")),
            str(rule.get("C_top10_rescued_count", "-")),
        ],
        [
            "Phase 8 LLM checker gap-query pool eval",
            "pool-level eval only",
            "N/A",
            "N/A",
            str(summary["LLM_C_pool_rescued_count"]),
            str(summary["multi_LLM_C_pool_rescued_count"]),
            "N/A",
        ],
    ]
    lines = [
        "# Phase 8 LLM Checker Gap-query Pool Evaluation",
        "",
        "Phase 8 is pool-level evaluation only.",
        "It does not rerank the merged pool and does not measure final top10 context quality.",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "method",
                "scope",
                "final full@10",
                "final multi_full@10",
                "C_pool_rescued",
                "multi_C_pool_rescued",
                "C_top10_rescued",
            ],
            rows,
        )
    )
    return "\n".join(lines)


def metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_run_name(
    *,
    model_name: str,
    max_next_queries: int,
    second_hop_top_k_chunks: int,
) -> str:
    return f"{model_slug(model_name)}_s{max_next_queries}_h{second_hop_top_k_chunks}_pool_eval"


def model_slug(model_name: str) -> str:
    normalized = compact_text(model_name).lower()
    if "qwen3" in normalized and "8b" in normalized:
        return "qwen3_8b"
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "custom_llm"


def print_summary(console: Console, summary: dict[str, Any], run_dir: Path) -> None:
    console.print()
    console.print("[bold green]LLM Evidence Checker Pool Eval Finished[/bold green]")
    console.print()
    console.print(f"checker_valid_count: {summary['checker_valid_count']}")
    console.print(f"LLM_C_pool_rescued_count: {summary['LLM_C_pool_rescued_count']}")
    console.print(
        f"multi_LLM_C_pool_rescued_count: {summary['multi_LLM_C_pool_rescued_count']}"
    )
    console.print(f"source_C_insufficient_rate: {summary['source_C_insufficient_rate']:.4f}")
    console.print(f"avg_generated_queries: {summary['avg_generated_queries']:.4f}")
    console.print()
    console.print(f"Artifacts: {run_dir}")
