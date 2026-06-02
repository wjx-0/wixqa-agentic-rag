from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.evaluation.chunk_eval import (
    build_chunk_trace,
    classify_chunk_case,
    compute_chunk_retrieval_metrics,
    result_article_ids,
    select_chunk_ks,
    summarize_chunk_metrics,
)
from src.evaluation.eval_utils import BM25EvalError, CASE_INVALID, markdown_table
from src.evaluation.run_chunk_bm25_eval import load_kb_chunks
from src.evaluation.run_rerank_eval import (
    RerankEvalError,
    build_chunk_lookup,
    candidate_row_to_example,
    load_hybrid_candidates,
)
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_INSTRUCTION_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL_NAME,
    CrossEncoderReranker,
    CrossEncoderRerankerError,
)
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME
from src.retrievers.hybrid_retriever import HybridRetriever, HybridRetrieverError
from src.retrievers.rrf import DEFAULT_BM25_WEIGHT
from src.retrievers.rule_second_hop import (
    DEFAULT_MAX_SEED_ARTICLES,
    RuleSecondHopError,
    build_title_expansion_queries,
    merge_chunk_candidates,
)
from src.utils.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


DEFAULT_FIRST_HOP_HYBRID_RUN_DIR = (
    "outputs/hybrid_rrf_baseline/"
    "hybrid_rrf_b100_f50_k60_bw1_dw2_wixqa_expertwritten"
)
DEFAULT_BASELINE_RERANK_RUN_DIR = (
    "outputs/rerank_baseline/"
    "hybrid_rrf_b100_f50_k60_bw1_dw2_wixqa_expertwritten/"
    "qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024"
)
DEFAULT_TOP100_CONTROL_RERANK_RUN_DIR = (
    "outputs/rerank_baseline/"
    "hybrid_rrf_b100_f100_k60_bw1_dw2_wixqa_expertwritten/"
    "qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024"
)
DEFAULT_CHUNKS_PATH = "data/processed/wix_kb_chunks.jsonl"
DEFAULT_INDEX_DIR = "indexes/faiss_bge_m3"
DEFAULT_OUTPUT_DIR = "outputs/rule_second_hop"
DEFAULT_FIRST_HOP_TOP_K_CHUNKS = 50
DEFAULT_BRANCH_TOP_K_CHUNKS = 100
DEFAULT_SECOND_HOP_TOP_K_CHUNKS = 20
DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 2.0
DEFAULT_RERANK_BATCH_SIZE = 32


class RuleSecondHopEvalError(RuntimeError):
    pass


def run_rule_second_hop_eval(
    *,
    first_hop_hybrid_run_dir: str | Path = DEFAULT_FIRST_HOP_HYBRID_RUN_DIR,
    baseline_rerank_run_dir: str | Path = DEFAULT_BASELINE_RERANK_RUN_DIR,
    top100_control_rerank_run_dir: str | Path | None = None,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dense_model_name: str = DEFAULT_DENSE_MODEL_NAME,
    reranker_model_name: str = DEFAULT_RERANK_MODEL_NAME,
    local_files_only: bool = True,
    dense_worker_mode: str = "full",
    dense_query_batch_size: int = 16,
    device: str | None = None,
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
    instruction_name: str = DEFAULT_INSTRUCTION_NAME,
    max_seed_articles: int = DEFAULT_MAX_SEED_ARTICLES,
    branch_top_k_chunks: int = DEFAULT_BRANCH_TOP_K_CHUNKS,
    second_hop_top_k_chunks: int = DEFAULT_SECOND_HOP_TOP_K_CHUNKS,
    rrf_k: int = DEFAULT_RRF_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    console: Console | None = None,
    retriever: Any | None = None,
    reranker: Any | None = None,
) -> dict[str, Any]:
    validate_args(
        max_seed_articles=max_seed_articles,
        branch_top_k_chunks=branch_top_k_chunks,
        second_hop_top_k_chunks=second_hop_top_k_chunks,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        dense_query_batch_size=dense_query_batch_size,
        rerank_batch_size=rerank_batch_size,
        max_length=max_length,
    )
    console = console or Console()
    first_hop_hybrid_run_dir = Path(first_hop_hybrid_run_dir)
    baseline_rerank_run_dir = Path(baseline_rerank_run_dir)
    top100_control_rerank_run_dir = optional_artifact_path(top100_control_rerank_run_dir)
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
        control_metrics, control_traces, control_hybrid_run_dir = load_optional_top100_control(
            top100_control_rerank_run_dir,
            first_hop_metrics=first_hop_metrics,
        )
        validate_qid_sets(candidate_rows, baseline_traces, control_traces)
        reranker = reranker or CrossEncoderReranker(
            model_name=reranker_model_name,
            local_files_only=local_files_only,
            device=device,
            instruction=instruction,
            max_length=max_length,
        )
    except (
        BM25EvalError,
        OSError,
        ValueError,
        RerankEvalError,
        CrossEncoderRerankerError,
    ) as exc:
        raise RuleSecondHopEvalError(str(exc)) from exc

    query_plans = build_query_plans(
        candidate_rows,
        baseline_traces,
        max_seed_articles=max_seed_articles,
    )
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
        DEFAULT_FIRST_HOP_TOP_K_CHUNKS + max_seed_articles * second_hop_top_k_chunks
    )
    ks = sorted(set(select_chunk_ks(merged_candidate_upper_bound) + [merged_candidate_upper_bound]))
    traces = []
    for candidate_row in tqdm(candidate_rows, desc="Rule second-hop rerank"):
        qid = candidate_row["qid"]
        baseline_trace = baseline_traces[qid]
        second_hop_batches = retrieval_batches[qid]
        try:
            merged_candidates = merge_chunk_candidates(
                candidate_row["hybrid_candidates"],
                second_hop_batches,
            )
            if len(merged_candidates) > merged_candidate_upper_bound:
                raise RuleSecondHopEvalError(
                    f"Merged candidate pool exceeds {merged_candidate_upper_bound} chunks for qid={qid}."
                )
            final_results = reranker.rerank(
                candidate_row["question"],
                merged_candidates,
                chunk_lookup,
                batch_size=rerank_batch_size,
            )
        except (RuleSecondHopError, CrossEncoderRerankerError) as exc:
            raise RuleSecondHopEvalError(f"Second-hop reranking failed for qid={qid}: {exc}") from exc
        traces.append(
            build_second_hop_trace(
                candidate_row=candidate_row,
                baseline_trace=baseline_trace,
                second_hop_batches=second_hop_batches,
                merged_candidates=merged_candidates,
                final_results=final_results,
                ks=ks,
                merged_candidate_upper_bound=merged_candidate_upper_bound,
            )
        )

    summary = summarize_second_hop_traces(
        traces,
        ks=ks,
        merged_candidate_upper_bound=merged_candidate_upper_bound,
        baseline_metrics=baseline_metrics,
        control_metrics=control_metrics,
    )
    run_dir = ensure_dir(
        Path(output_dir)
        / first_hop_hybrid_run_dir.name
        / baseline_rerank_run_dir.name
        / build_run_name(
            max_seed_articles=max_seed_articles,
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
            "top100_control_rerank_run_dir": (
                str(top100_control_rerank_run_dir)
                if top100_control_rerank_run_dir is not None
                else None
            ),
            "top100_control_hybrid_run_dir": (
                str(control_hybrid_run_dir) if control_hybrid_run_dir is not None else None
            ),
            "chunks_path": str(chunks_path),
            "index_dir": str(index_dir),
            "dense_model_name": dense_model_name,
            "reranker_model_name": reranker_model_name,
            "local_files_only": local_files_only,
            "dense_worker_mode": dense_worker_mode,
            "dense_query_batch_size": dense_query_batch_size,
            "device": device,
            "rerank_batch_size": rerank_batch_size,
            "max_length": max_length,
            "instruction_name": instruction_name,
            "instruction": instruction,
            "max_seed_articles": max_seed_articles,
            "branch_top_k_chunks": branch_top_k_chunks,
            "second_hop_top_k_chunks": second_hop_top_k_chunks,
            "rrf_k": rrf_k,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "merged_candidate_upper_bound": merged_candidate_upper_bound,
        },
    )
    print_summary(console, summary, run_dir)
    return summary


def validate_args(
    *,
    max_seed_articles: int,
    branch_top_k_chunks: int,
    second_hop_top_k_chunks: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
    dense_query_batch_size: int,
    rerank_batch_size: int,
    max_length: int,
) -> None:
    positive_values = {
        "max_seed_articles": max_seed_articles,
        "branch_top_k_chunks": branch_top_k_chunks,
        "second_hop_top_k_chunks": second_hop_top_k_chunks,
        "rrf_k": rrf_k,
        "dense_query_batch_size": dense_query_batch_size,
        "rerank_batch_size": rerank_batch_size,
        "max_length": max_length,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise RuleSecondHopEvalError(f"{name} must be a positive integer.")
    if second_hop_top_k_chunks > branch_top_k_chunks:
        raise RuleSecondHopEvalError(
            "second_hop_top_k_chunks must be smaller than or equal to branch_top_k_chunks."
        )
    for name, value in (("bm25_weight", bm25_weight), ("dense_weight", dense_weight)):
        if not math.isfinite(value) or value <= 0:
            raise RuleSecondHopEvalError(f"{name} must be a positive number.")


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuleSecondHopEvalError(f"Missing required artifact: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise RuleSecondHopEvalError(f"Expected a JSON object in {path}.")
    return value


def optional_artifact_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"none", "null", "skip"}:
        return None
    return Path(text)


def validate_candidate_cutoff(
    metrics: dict[str, Any],
    expected: int,
    *,
    artifact_name: str,
) -> None:
    actual = int(metrics.get("candidate_top_k_chunks") or metrics.get("fused_top_k_chunks") or 0)
    if actual != expected:
        raise RuleSecondHopEvalError(f"{artifact_name} must use {expected} chunks, got {actual}.")


def load_control_hybrid_run_dir(control_config: dict[str, Any]) -> Path:
    path = control_config.get("source_hybrid_run_dir")
    if not isinstance(path, str) or not path.strip():
        raise RuleSecondHopEvalError(
            "Top100 control reranker run_config.json is missing source_hybrid_run_dir."
        )
    return Path(path)


def validate_fair_top100_control(
    *,
    first_hop_metrics: dict[str, Any],
    control_hybrid_metrics: dict[str, Any],
) -> None:
    validate_candidate_cutoff(
        control_hybrid_metrics,
        100,
        artifact_name="Top100 control Hybrid metrics",
    )
    for key in (
        "dataset_name",
        "branch_top_k_chunks",
        "rrf_k",
        "bm25_weight",
        "dense_weight",
    ):
        if control_hybrid_metrics.get(key) != first_hop_metrics.get(key):
            raise RuleSecondHopEvalError(
                "Top100 control Hybrid metrics must match first-hop configuration except "
                f"for fused_top_k_chunks; mismatch at {key}: "
                f"first_hop={first_hop_metrics.get(key)!r}, "
                f"control={control_hybrid_metrics.get(key)!r}."
            )


def load_optional_top100_control(
    top100_control_rerank_run_dir: Path | None,
    *,
    first_hop_metrics: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]] | None, Path | None]:
    if top100_control_rerank_run_dir is None:
        return None, None, None
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
    control_traces = load_rerank_trace_lookup(
        top100_control_rerank_run_dir / "rerank_traces.jsonl"
    )
    return control_metrics, control_traces, control_hybrid_run_dir


def load_rerank_trace_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise RuleSecondHopEvalError(f"Missing required artifact: {path}")
    lookup = {}
    for row in read_jsonl(path):
        qid = row.get("qid")
        if not qid:
            raise RuleSecondHopEvalError(f"Rerank trace in {path} is missing qid.")
        if qid in lookup:
            raise RuleSecondHopEvalError(f"Duplicate qid in {path}: {qid}.")
        top10 = row.get("top10_reranked_chunks")
        if not isinstance(top10, list):
            raise RuleSecondHopEvalError(f"Rerank trace qid={qid} is missing top10_reranked_chunks.")
        lookup[qid] = row
    if not lookup:
        raise RuleSecondHopEvalError(f"No rerank traces found in {path}.")
    return lookup


def validate_qid_sets(
    candidate_rows: list[dict[str, Any]],
    baseline_traces: dict[str, dict[str, Any]],
    control_traces: dict[str, dict[str, Any]] | None,
) -> None:
    candidate_qids = {row["qid"] for row in candidate_rows}
    qid_sets = [("baseline reranker traces", set(baseline_traces))]
    if control_traces is not None:
        qid_sets.append(("top100 control reranker traces", set(control_traces)))
    for label, qids in qid_sets:
        if qids != candidate_qids:
            raise RuleSecondHopEvalError(
                f"Qid set mismatch between first-hop candidates and {label}: "
                f"candidates={len(candidate_qids)}, traces={len(qids)}."
            )


def build_query_plans(
    candidate_rows: list[dict[str, Any]],
    baseline_traces: dict[str, dict[str, Any]],
    *,
    max_seed_articles: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        row["qid"]: build_title_expansion_queries(
            row["question"],
            baseline_traces[row["qid"]]["top10_reranked_chunks"],
            max_seed_articles=max_seed_articles,
        )
        for row in candidate_rows
    }


def run_second_hop_retrieval(
    query_plans: dict[str, list[dict[str, Any]]],
    *,
    chunks: list[Any],
    index_dir: str | Path,
    dense_model_name: str,
    local_files_only: bool,
    device: str | None,
    dense_worker_mode: str,
    branch_top_k_chunks: int,
    second_hop_top_k_chunks: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
    dense_query_batch_size: int,
    retriever: Any | None,
) -> dict[str, list[dict[str, Any]]]:
    flat_queries = [
        (qid, query)
        for qid, queries in query_plans.items()
        for query in queries
    ]
    output = {qid: [] for qid in query_plans}
    if not flat_queries:
        return output
    context = (
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
        with context as active_retriever:
            batches = active_retriever.search_batch(
                [query["query_text"] for _, query in flat_queries],
                branch_top_k_chunks=branch_top_k_chunks,
                fused_top_k_chunks=second_hop_top_k_chunks,
                rrf_k=rrf_k,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                dense_query_batch_size=dense_query_batch_size,
            )
    except HybridRetrieverError as exc:
        raise RuleSecondHopEvalError(str(exc)) from exc
    if len(batches) != len(flat_queries):
        raise RuleSecondHopEvalError(
            "Second-hop retrieval result count does not match query count: "
            f"results={len(batches)}, queries={len(flat_queries)}."
        )
    for (qid, query), batch in zip(flat_queries, batches):
        output[qid].append(
            {
                **query,
                "results": batch["hybrid"],
            }
        )
    return output


def build_second_hop_trace(
    *,
    candidate_row: dict[str, Any],
    baseline_trace: dict[str, Any],
    second_hop_batches: list[dict[str, Any]],
    merged_candidates: list[dict[str, Any]],
    final_results: list[dict[str, Any]],
    ks: list[int],
    merged_candidate_upper_bound: int,
) -> dict[str, Any]:
    example = candidate_row_to_example(candidate_row)
    metrics = compute_chunk_retrieval_metrics(example.article_ids, final_results, ks)
    trace = build_chunk_trace(
        example,
        final_results,
        metrics,
        {},
        top_k_chunks=merged_candidate_upper_bound,
    )
    diagnostics = build_rescue_diagnostics(
        gold_article_ids=example.article_ids,
        first_hop_candidates=candidate_row["hybrid_candidates"],
        baseline_top10=baseline_trace["top10_reranked_chunks"],
        merged_candidates=merged_candidates,
        final_top10=final_results[:10],
        is_multi_article=example.is_multi_article,
    )
    first_hop_chunk_ids = {row["chunk_id"] for row in candidate_row["hybrid_candidates"]}
    first_hop_article_ids = set(result_article_ids(candidate_row["hybrid_candidates"]))
    new_candidates = [
        row for row in merged_candidates if row["chunk_id"] not in first_hop_chunk_ids
    ]
    trace.update(
        {
            "source_case_type": classify_chunk_case(
                example.article_ids,
                result_article_ids(candidate_row["hybrid_candidates"]),
                DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
            ),
            "seed_articles": [
                {
                    "article_id": batch["seed_article_id"],
                    "title": batch["seed_title"],
                }
                for batch in second_hop_batches
            ],
            "second_hop_queries": [
                {
                    key: batch[key]
                    for key in ("query_id", "query_text", "seed_article_id", "seed_title")
                }
                for batch in second_hop_batches
            ],
            "second_hop_results": second_hop_batches,
            "merged_candidate_count": len(merged_candidates),
            "new_chunk_ids": [row["chunk_id"] for row in new_candidates],
            "new_article_ids": sorted(
                set(result_article_ids(new_candidates)) - first_hop_article_ids
            ),
            "final_top10_chunks": final_results[:10],
            "retrieval_rounds": 1 + int(bool(second_hop_batches)),
            **diagnostics,
        }
    )
    return trace


def build_rescue_diagnostics(
    *,
    gold_article_ids: list[str],
    first_hop_candidates: list[dict[str, Any]],
    baseline_top10: list[dict[str, Any]],
    merged_candidates: list[dict[str, Any]],
    final_top10: list[dict[str, Any]],
    is_multi_article: bool,
) -> dict[str, Any]:
    gold_set = {article_id for article_id in gold_article_ids if article_id}
    first_pool = set(result_article_ids(first_hop_candidates))
    merged_pool = set(result_article_ids(merged_candidates))
    baseline_top10_ids = set(result_article_ids(baseline_top10))
    final_top10_ids = set(result_article_ids(final_top10))
    source_c = bool(gold_set) and not gold_set.issubset(first_pool)
    source_a = bool(gold_set) and gold_set.issubset(baseline_top10_ids)
    c_pool_rescued = source_c and gold_set.issubset(merged_pool)
    c_top10_rescued = source_c and gold_set.issubset(final_top10_ids)
    if c_top10_rescued and not c_pool_rescued:
        raise RuleSecondHopEvalError("C_top10_rescued requires C_pool_rescued.")
    return {
        "source_C": source_c,
        "source_A": source_a,
        "C_pool_rescued": c_pool_rescued,
        "C_top10_rescued": c_top10_rescued,
        "A_dropped": source_a and not gold_set.issubset(final_top10_ids),
        "is_multi_article": is_multi_article,
        "merged_pool_full_article_hit": int(bool(gold_set) and gold_set.issubset(merged_pool)),
        "pool_new_gold_article_ids": sorted((gold_set & merged_pool) - (gold_set & first_pool)),
        "pool_still_missing_gold_article_ids": sorted(gold_set - merged_pool),
        "top10_new_gold_article_ids": sorted(
            (gold_set & final_top10_ids) - (gold_set & baseline_top10_ids)
        ),
        "top10_still_missing_gold_article_ids": sorted(gold_set - final_top10_ids),
    }


def summarize_second_hop_traces(
    traces: list[dict[str, Any]],
    *,
    ks: list[int],
    merged_candidate_upper_bound: int,
    baseline_metrics: dict[str, Any],
    control_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    valid_rows = [trace for trace in traces if trace["case_type"] != CASE_INVALID]
    summary = summarize_chunk_metrics(
        dataset_name=baseline_metrics["dataset_name"],
        records=len(traces),
        valid_rows=valid_rows,
        invalid_rows=[trace for trace in traces if trace["case_type"] == CASE_INVALID],
        ks=ks,
        top_k_chunks=merged_candidate_upper_bound,
    )
    summary.update(
        {
            "retriever_type": "Rule-based Second-hop Retrieval",
            "first_hop_candidate_top_k_chunks": DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
            "merged_candidate_upper_bound": merged_candidate_upper_bound,
            "source_baseline_metrics": baseline_metrics,
            "top100_control_metrics": control_metrics,
            "C_pool_rescued_count": count_true(valid_rows, "C_pool_rescued"),
            "multi_C_pool_rescued_count": count_true(
                valid_rows,
                "C_pool_rescued",
                multi_only=True,
            ),
            "C_top10_rescued_count": count_true(valid_rows, "C_top10_rescued"),
            "multi_C_top10_rescued_count": count_true(
                valid_rows,
                "C_top10_rescued",
                multi_only=True,
            ),
            "A_dropped_count": count_true(valid_rows, "A_dropped"),
            "avg_second_hop_queries": average(len(row["second_hop_queries"]) for row in traces),
            "avg_merged_candidates": average(row["merged_candidate_count"] for row in traces),
            "avg_retrieval_rounds": average(row["retrieval_rounds"] for row in traces),
        }
    )
    return summary


def count_true(
    rows: list[dict[str, Any]],
    key: str,
    *,
    multi_only: bool = False,
) -> int:
    return sum(
        bool(row.get(key)) and (not multi_only or bool(row.get("is_multi_article")))
        for row in rows
    )


def average(values: Any) -> float:
    items = list(values)
    return sum(float(value) for value in items) / len(items) if items else 0.0


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
    write_jsonl(run_dir / "second_hop_traces.jsonl", traces)
    write_jsonl(
        run_dir / "cases_C_pool_rescued_by_second_hop.jsonl",
        [row for row in traces if row["C_pool_rescued"]],
    )
    write_jsonl(
        run_dir / "cases_C_top10_rescued_by_second_hop.jsonl",
        [row for row in traces if row["C_top10_rescued"]],
    )
    write_jsonl(
        run_dir / "multi_cases_C_top10_rescued_by_second_hop.jsonl",
        [row for row in traces if row["C_top10_rescued"] and row["is_multi_article"]],
    )
    write_jsonl(
        run_dir / "cases_A_dropped_by_second_hop.jsonl",
        [row for row in traces if row["A_dropped"]],
    )


def render_metrics_markdown(summary: dict[str, Any]) -> str:
    rows = [
        ["records", summary["records"]],
        ["chunk_full_article_hit@10", f"{summary['chunk_full_article_hit@10']:.4f}"],
        ["chunk_article_recall@10", f"{summary['chunk_article_recall@10']:.4f}"],
        [
            "multi_chunk_full_article_hit@10",
            f"{summary['multi_chunk_full_article_hit@10']:.4f}",
        ],
        ["multi_chunk_article_recall@10", f"{summary['multi_chunk_article_recall@10']:.4f}"],
        ["C_pool_rescued_count", summary["C_pool_rescued_count"]],
        ["multi_C_pool_rescued_count", summary["multi_C_pool_rescued_count"]],
        ["C_top10_rescued_count", summary["C_top10_rescued_count"]],
        ["multi_C_top10_rescued_count", summary["multi_C_top10_rescued_count"]],
        ["A_dropped_count", summary["A_dropped_count"]],
        ["avg_second_hop_queries", f"{summary['avg_second_hop_queries']:.4f}"],
        ["avg_merged_candidates", f"{summary['avg_merged_candidates']:.4f}"],
        ["avg_retrieval_rounds", f"{summary['avg_retrieval_rounds']:.4f}"],
    ]
    lines = ["# Rule-based Second-hop Retrieval", ""]
    lines.extend(markdown_table(["metric", "value"], rows))
    return "\n".join(lines)


def render_comparison_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["source_baseline_metrics"]
    control = summary["top100_control_metrics"]
    rows = [
        comparison_row("Top50 Hybrid + Qwen3 baseline", baseline),
        comparison_row("Top100 Hybrid + Qwen3 control", control),
        comparison_row("Top50 + title-expanded second-hop + Qwen3", summary),
    ]
    lines = ["# Rule-based Second-hop Retrieval Comparison", ""]
    lines.extend(
        markdown_table(
            [
                "method",
                "full@10",
                "recall@10",
                "multi_full@10",
                "multi_recall@10",
            ],
            rows,
        )
    )
    return "\n".join(lines)


def comparison_row(label: str, metrics: dict[str, Any] | None) -> list[Any]:
    if metrics is None:
        return [label, "N/A", "N/A", "N/A", "N/A"]
    return [
        label,
        f"{float(metrics['chunk_full_article_hit@10']):.4f}",
        f"{float(metrics['chunk_article_recall@10']):.4f}",
        f"{float(metrics['multi_chunk_full_article_hit@10']):.4f}",
        f"{float(metrics['multi_chunk_article_recall@10']):.4f}",
    ]


def build_run_name(*, max_seed_articles: int, second_hop_top_k_chunks: int) -> str:
    return f"title_expand_s{max_seed_articles}_h{second_hop_top_k_chunks}"


def print_summary(console: Console, summary: dict[str, Any], run_dir: Path) -> None:
    console.print()
    console.print("[bold green]Rule-based Second-hop Retrieval Finished[/bold green]")
    console.print()
    console.print(f"chunk_full_article_hit@10: {summary['chunk_full_article_hit@10']:.4f}")
    console.print(
        "multi_chunk_full_article_hit@10: "
        f"{summary['multi_chunk_full_article_hit@10']:.4f}"
    )
    console.print(f"C_pool_rescued_count: {summary['C_pool_rescued_count']}")
    console.print(f"C_top10_rescued_count: {summary['C_top10_rescued_count']}")
    console.print(f"A_dropped_count: {summary['A_dropped_count']}")
    console.print()
    console.print(f"Results saved to {run_dir}")
