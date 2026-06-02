from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.evaluation.chunk_eval import (
    build_chunk_trace,
    compute_chunk_retrieval_metrics,
    result_article_ids,
    select_chunk_ks,
    summarize_chunk_metrics,
)
from src.evaluation.eval_utils import BM25EvalError, CASE_INVALID, markdown_table
from src.evaluation.run_chunk_bm25_eval import load_kb_chunks
from src.evaluation.run_llm_evidence_checker_eval import DEFAULT_RULE_SECOND_HOP_RUN_DIR
from src.evaluation.run_rerank_eval import (
    RerankEvalError,
    build_chunk_lookup,
    candidate_row_to_example,
    load_hybrid_candidates,
)
from src.evaluation.run_rule_second_hop_eval import (
    DEFAULT_BASELINE_RERANK_RUN_DIR,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_FIRST_HOP_HYBRID_RUN_DIR,
    DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_TOP100_CONTROL_RERANK_RUN_DIR,
    RuleSecondHopEvalError,
    average,
    count_true,
    load_control_hybrid_run_dir,
    load_json_object,
    load_rerank_trace_lookup,
    validate_candidate_cutoff,
    validate_fair_top100_control,
)
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_INSTRUCTION_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL_NAME,
    CrossEncoderReranker,
    CrossEncoderRerankerError,
)
from src.retrievers.rule_second_hop import RuleSecondHopError, merge_chunk_candidates
from src.utils.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.utils.text_utils import compact_text


DEFAULT_LLM_CHECKER_RUN_DIR = (
    "outputs/llm_evidence_checker/"
    "hybrid_rrf_b100_f50_k60_bw1_dw2_wixqa_expertwritten/"
    "qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024/"
    "qwen3_8b_s3_h20_pool_eval"
)
DEFAULT_OUTPUT_DIR = "outputs/agentic_rag"
DEFAULT_AGENTIC_RUN_NAME = "gap_aware_rerank_merged_pool_a0p6_b0p4"
GAP_AWARE_ALPHA = 0.6
GAP_AWARE_BETA = 0.4
GAP_QUERY_MAPPING = "best_second_hop_rank"
GAP_AWARE_APPLIES_TO = "pure_second_hop_only"
GAP_AWARE_EVAL_SCOPE = (
    "phase8_merged_pool_offline_gap_aware_rerank_no_llm_no_retrieval"
)


class AgenticRerankEvalError(RuntimeError):
    pass


def run_agentic_rerank_eval(
    *,
    first_hop_hybrid_run_dir: str | Path = DEFAULT_FIRST_HOP_HYBRID_RUN_DIR,
    baseline_rerank_run_dir: str | Path = DEFAULT_BASELINE_RERANK_RUN_DIR,
    top100_control_rerank_run_dir: str | Path = DEFAULT_TOP100_CONTROL_RERANK_RUN_DIR,
    rule_second_hop_run_dir: str | Path = DEFAULT_RULE_SECOND_HOP_RUN_DIR,
    llm_checker_run_dir: str | Path = DEFAULT_LLM_CHECKER_RUN_DIR,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    reranker_model_name: str = DEFAULT_RERANK_MODEL_NAME,
    local_files_only: bool = True,
    device: str | None = None,
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
    instruction_name: str = DEFAULT_INSTRUCTION_NAME,
    dense_worker_mode: str | None = None,
    console: Console | None = None,
    reranker: Any | None = None,
) -> dict[str, Any]:
    validate_args(
        rerank_batch_size=rerank_batch_size,
        max_length=max_length,
        instruction=instruction,
        instruction_name=instruction_name,
    )
    console = console or Console()
    first_hop_hybrid_run_dir = Path(first_hop_hybrid_run_dir)
    baseline_rerank_run_dir = Path(baseline_rerank_run_dir)
    top100_control_rerank_run_dir = Path(top100_control_rerank_run_dir)
    rule_second_hop_run_dir = Path(rule_second_hop_run_dir)
    llm_checker_run_dir = Path(llm_checker_run_dir)
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
        control_traces = load_rerank_trace_lookup(
            top100_control_rerank_run_dir / "rerank_traces.jsonl"
        )
        phase7_metrics = load_json_object(rule_second_hop_run_dir / "metrics.json")
        phase8_metrics = load_json_object(llm_checker_run_dir / "metrics.json")
        checker_traces = load_checker_trace_lookup(llm_checker_run_dir / "checker_traces.jsonl")
        validate_qid_sets(candidate_rows, baseline_traces, control_traces, checker_traces)
        merged_candidate_upper_bound = int(
            phase8_metrics.get("merged_candidate_upper_bound") or 0
        )
        if merged_candidate_upper_bound < 20:
            raise AgenticRerankEvalError(
                "Phase 8 metrics.json is missing a usable merged_candidate_upper_bound."
            )
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
        RuleSecondHopEvalError,
        CrossEncoderRerankerError,
    ) as exc:
        raise AgenticRerankEvalError(str(exc)) from exc

    ks = sorted(set(select_chunk_ks(merged_candidate_upper_bound) + [merged_candidate_upper_bound]))
    traces = []
    for candidate_row in tqdm(candidate_rows, desc="Agentic merged-pool rerank"):
        qid = candidate_row["qid"]
        checker_trace = checker_traces[qid]
        try:
            second_hop_batches = checker_trace["second_hop_results"]
            merged_candidates = merge_chunk_candidates(
                candidate_row["hybrid_candidates"],
                second_hop_batches,
            )
            if len(merged_candidates) > merged_candidate_upper_bound:
                raise AgenticRerankEvalError(
                    f"Merged candidate pool exceeds {merged_candidate_upper_bound} chunks for qid={qid}."
                )
            if not merged_candidates:
                raise AgenticRerankEvalError(f"Merged candidate pool is empty for qid={qid}.")
            final_results = rerank_gap_aware_candidates(
                question=candidate_row["question"],
                checker_trace=checker_trace,
                candidates=merged_candidates,
                reranker=reranker,
                chunk_lookup=chunk_lookup,
                batch_size=rerank_batch_size,
            )
            if len(final_results) != len(merged_candidates):
                raise AgenticRerankEvalError(
                    "Reranker result count does not match merged candidate count for "
                    f"qid={qid}: results={len(final_results)}, "
                    f"candidates={len(merged_candidates)}."
                )
        except (KeyError, TypeError, RuleSecondHopError, CrossEncoderRerankerError) as exc:
            raise AgenticRerankEvalError(f"Agentic rerank failed for qid={qid}: {exc}") from exc
        traces.append(
            build_agentic_rerank_trace(
                candidate_row=candidate_row,
                baseline_trace=baseline_traces[qid],
                checker_trace=checker_trace,
                second_hop_batches=second_hop_batches,
                merged_candidates=merged_candidates,
                final_results=final_results,
                ks=ks,
                merged_candidate_upper_bound=merged_candidate_upper_bound,
            )
        )

    summary = summarize_agentic_rerank_traces(
        traces,
        ks=ks,
        merged_candidate_upper_bound=merged_candidate_upper_bound,
        baseline_metrics=baseline_metrics,
        control_metrics=control_metrics,
        phase7_metrics=phase7_metrics,
        phase8_metrics=phase8_metrics,
    )
    run_dir = ensure_dir(
        Path(output_dir)
        / first_hop_hybrid_run_dir.name
        / baseline_rerank_run_dir.name
        / llm_checker_run_dir.name
        / DEFAULT_AGENTIC_RUN_NAME
    )
    write_outputs(
        run_dir,
        traces=traces,
        summary=summary,
        run_config={
            "first_hop_hybrid_run_dir": str(first_hop_hybrid_run_dir),
            "baseline_rerank_run_dir": str(baseline_rerank_run_dir),
            "top100_control_rerank_run_dir": str(top100_control_rerank_run_dir),
            "top100_control_hybrid_run_dir": str(control_hybrid_run_dir),
            "rule_second_hop_run_dir": str(rule_second_hop_run_dir),
            "llm_checker_run_dir": str(llm_checker_run_dir),
            "chunks_path": str(chunks_path),
            "reranker_model_name": reranker_model_name,
            "local_files_only": local_files_only,
            "device": device,
            "rerank_batch_size": rerank_batch_size,
            "max_length": max_length,
            "instruction_name": instruction_name,
            "instruction": instruction,
            "dense_worker_mode": dense_worker_mode,
            "eval_scope": GAP_AWARE_EVAL_SCOPE,
            "rerank_scoring_mode": "gap_aware",
            "gap_aware_alpha": GAP_AWARE_ALPHA,
            "gap_aware_beta": GAP_AWARE_BETA,
            "gap_query_mapping": GAP_QUERY_MAPPING,
            "gap_aware_applies_to": GAP_AWARE_APPLIES_TO,
            "merged_candidate_upper_bound": merged_candidate_upper_bound,
            "top10_primary_context_budget": 10,
            "top20_diagnostic_context_budget": 20,
        },
    )
    print_summary(console, summary, run_dir)
    return summary


def validate_args(
    *,
    rerank_batch_size: int,
    max_length: int,
    instruction: str,
    instruction_name: str,
) -> None:
    if rerank_batch_size <= 0:
        raise AgenticRerankEvalError("rerank_batch_size must be a positive integer.")
    if max_length <= 0:
        raise AgenticRerankEvalError("max_length must be a positive integer.")
    if not instruction.strip():
        raise AgenticRerankEvalError("instruction must not be empty.")
    if not instruction_name.strip():
        raise AgenticRerankEvalError("instruction_name must not be empty.")


def rerank_gap_aware_candidates(
    *,
    question: str,
    checker_trace: dict[str, Any],
    candidates: list[dict[str, Any]],
    reranker: Any,
    chunk_lookup: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    original_results = reranker.rerank(
        question,
        candidates,
        chunk_lookup,
        batch_size=batch_size,
    )
    if len(original_results) != len(candidates):
        raise AgenticRerankEvalError(
            "Original-query reranker result count does not match candidate count: "
            f"results={len(original_results)}, candidates={len(candidates)}."
        )

    original_by_chunk_id = index_rerank_results(original_results, label="original rerank")
    gap_query_lookup = build_gap_query_lookup(checker_trace)
    selected_gap_queries: dict[str, tuple[str, str]] = {}
    gap_candidates_by_query_id: dict[str, list[dict[str, Any]]] = {}

    for candidate in candidates:
        query_id = select_gap_query_id_for_chunk(candidate)
        if not query_id:
            continue
        query_text = gap_query_lookup.get(query_id)
        if not query_text:
            raise AgenticRerankEvalError(
                "Pure second-hop chunk references an unknown gap query_id: "
                f"chunk_id={candidate.get('chunk_id')}, query_id={query_id}."
            )
        chunk_id = compact_text(candidate.get("chunk_id"))
        selected_gap_queries[chunk_id] = (query_id, query_text)
        gap_candidates_by_query_id.setdefault(query_id, []).append(candidate)

    gap_scores_by_chunk_id: dict[str, float] = {}
    for query_id, query_candidates in gap_candidates_by_query_id.items():
        query_text = gap_query_lookup[query_id]
        gap_results = reranker.rerank(
            query_text,
            query_candidates,
            chunk_lookup,
            batch_size=batch_size,
        )
        if len(gap_results) != len(query_candidates):
            raise AgenticRerankEvalError(
                "Gap-query reranker result count does not match candidate count: "
                f"query_id={query_id}, results={len(gap_results)}, "
                f"candidates={len(query_candidates)}."
            )
        for result in gap_results:
            chunk_id = compact_text(result.get("chunk_id"))
            gap_scores_by_chunk_id[chunk_id] = float(result["rerank_score"])

    scored_results = []
    for candidate in candidates:
        chunk_id = compact_text(candidate.get("chunk_id"))
        original_result = original_by_chunk_id[chunk_id]
        score_original = float(original_result["rerank_score"])
        selected_query = selected_gap_queries.get(chunk_id)
        if selected_query is None:
            score_gap = None
            gap_query_id = None
            gap_query_text = None
            gap_aware_score = score_original
        else:
            gap_query_id, gap_query_text = selected_query
            score_gap = gap_scores_by_chunk_id.get(chunk_id)
            if score_gap is None:
                raise AgenticRerankEvalError(
                    "Missing gap-query score for pure second-hop chunk: "
                    f"chunk_id={chunk_id}, query_id={gap_query_id}."
                )
            gap_aware_score = GAP_AWARE_ALPHA * score_original + GAP_AWARE_BETA * score_gap

        scored_results.append(
            {
                **original_result,
                "score_original": score_original,
                "score_gap": score_gap,
                "gap_query_id": gap_query_id,
                "gap_query_text": gap_query_text,
                "gap_aware_score": gap_aware_score,
                "gap_aware_alpha": GAP_AWARE_ALPHA,
                "gap_aware_beta": GAP_AWARE_BETA,
                "rerank_score": gap_aware_score,
                "score": gap_aware_score,
            }
        )

    ranked = sorted(
        scored_results,
        key=lambda result: (
            -float(result["gap_aware_score"]),
            int(result.get("merged_rank") or result.get("hybrid_rank") or result.get("rank") or 0),
            str(result["chunk_id"]),
        ),
    )
    return [{**result, "rank": rank} for rank, result in enumerate(ranked, start=1)]


def index_rerank_results(results: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for result in results:
        chunk_id = compact_text(result.get("chunk_id"))
        if not chunk_id:
            raise AgenticRerankEvalError(f"{label} result is missing chunk_id.")
        if chunk_id in indexed:
            raise AgenticRerankEvalError(f"Duplicate chunk_id in {label}: {chunk_id}.")
        indexed[chunk_id] = result
    return indexed


def build_gap_query_lookup(checker_trace: dict[str, Any]) -> dict[str, str]:
    lookup = {}
    queries = checker_trace.get("second_hop_queries") or []
    if not isinstance(queries, list):
        raise AgenticRerankEvalError("Checker trace second_hop_queries must be a list.")
    for query in queries:
        if not isinstance(query, dict):
            raise AgenticRerankEvalError("Checker trace second_hop_queries must contain objects.")
        query_id = compact_text(query.get("query_id"))
        query_text = compact_text(query.get("query_text"))
        if not query_id or not query_text:
            raise AgenticRerankEvalError(
                "Checker trace second_hop_queries entries require query_id and query_text."
            )
        if query_id in lookup:
            raise AgenticRerankEvalError(f"Duplicate second-hop query_id: {query_id}.")
        lookup[query_id] = query_text
    return lookup


def select_gap_query_id_for_chunk(candidate: dict[str, Any]) -> str | None:
    if candidate.get("first_hop_rank") is not None:
        return None

    query_ids = [
        query_id
        for query_id in (compact_text(value) for value in candidate.get("second_hop_query_ids") or [])
        if query_id
    ]
    if not query_ids:
        raise AgenticRerankEvalError(
            f"Pure second-hop chunk is missing second_hop_query_ids: {candidate.get('chunk_id')}."
        )
    if len(query_ids) == 1:
        return query_ids[0]

    query_order = {query_id: index for index, query_id in enumerate(query_ids)}
    ranked_query_ids = []
    for item in candidate.get("second_hop_ranks") or []:
        if not isinstance(item, dict):
            raise AgenticRerankEvalError(
                f"second_hop_ranks entries must be objects: {candidate.get('chunk_id')}."
            )
        query_id = compact_text(item.get("query_id"))
        if query_id not in query_order:
            continue
        try:
            rank = int(item.get("rank"))
        except (TypeError, ValueError) as exc:
            raise AgenticRerankEvalError(
                f"Invalid second-hop rank for chunk_id={candidate.get('chunk_id')}, "
                f"query_id={query_id}."
            ) from exc
        ranked_query_ids.append((rank, query_order[query_id], query_id))

    if not ranked_query_ids:
        raise AgenticRerankEvalError(
            f"Pure second-hop chunk has multiple query_ids but no usable second_hop_ranks: "
            f"{candidate.get('chunk_id')}."
        )
    return min(ranked_query_ids)[2]


def load_checker_trace_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise AgenticRerankEvalError(f"Missing required artifact: {path}")
    lookup = {}
    for row in read_jsonl(path):
        qid = row.get("qid")
        if not qid:
            raise AgenticRerankEvalError(f"Checker trace in {path} is missing qid.")
        if qid in lookup:
            raise AgenticRerankEvalError(f"Duplicate qid in {path}: {qid}.")
        if not isinstance(row.get("second_hop_results"), list):
            raise AgenticRerankEvalError(
                f"Checker trace qid={qid} is missing second_hop_results."
            )
        lookup[qid] = row
    if not lookup:
        raise AgenticRerankEvalError(f"No checker traces found in {path}.")
    return lookup


def validate_qid_sets(
    candidate_rows: list[dict[str, Any]],
    baseline_traces: dict[str, dict[str, Any]],
    control_traces: dict[str, dict[str, Any]],
    checker_traces: dict[str, dict[str, Any]],
) -> None:
    candidate_qids = {row["qid"] for row in candidate_rows}
    for label, qids in (
        ("baseline reranker traces", set(baseline_traces)),
        ("top100 control reranker traces", set(control_traces)),
        ("Phase 8 checker traces", set(checker_traces)),
    ):
        if qids != candidate_qids:
            raise AgenticRerankEvalError(
                f"Qid set mismatch between first-hop candidates and {label}: "
                f"candidates={len(candidate_qids)}, traces={len(qids)}."
            )


def build_agentic_rerank_trace(
    *,
    candidate_row: dict[str, Any],
    baseline_trace: dict[str, Any],
    checker_trace: dict[str, Any],
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
    diagnostics = build_agentic_rerank_diagnostics(
        gold_article_ids=example.article_ids,
        first_hop_candidates=candidate_row["hybrid_candidates"],
        baseline_top10=baseline_trace["top10_reranked_chunks"],
        merged_candidates=merged_candidates,
        final_top10=final_results[:10],
        final_top20=final_results[:20],
        is_multi_article=example.is_multi_article,
    )
    first_hop_chunk_ids = {row["chunk_id"] for row in candidate_row["hybrid_candidates"]}
    first_hop_article_ids = set(result_article_ids(candidate_row["hybrid_candidates"]))
    new_candidates = [
        row for row in merged_candidates if row["chunk_id"] not in first_hop_chunk_ids
    ]
    trace.update(
        {
            "eval_scope": GAP_AWARE_EVAL_SCOPE,
            "baseline_top10_chunks": baseline_trace["top10_reranked_chunks"],
            "baseline_top10_article_ids": result_article_ids(
                baseline_trace["top10_reranked_chunks"]
            ),
            "phase8_checker_valid": checker_trace.get("checker_valid"),
            "phase8_checker_sufficient": checker_trace.get("checker_sufficient"),
            "checker_valid": checker_trace.get("checker_valid"),
            "checker_sufficient": checker_trace.get("checker_sufficient"),
            "llm_calls": int(checker_trace.get("llm_calls") or 0),
            "next_queries": checker_trace.get("next_queries") or [],
            "second_hop_queries": checker_trace.get("second_hop_queries") or [],
            "second_hop_results": second_hop_batches,
            "merged_candidate_count": len(merged_candidates),
            "rerank_candidate_count": len(final_results),
            "new_chunk_ids": [row["chunk_id"] for row in new_candidates],
            "new_article_ids": sorted(
                set(result_article_ids(new_candidates)) - first_hop_article_ids
            ),
            "final_top10_chunks": final_results[:10],
            "final_top20_chunks": final_results[:20],
            "final_top10_article_ids": result_article_ids(final_results[:10]),
            "final_top20_article_ids": result_article_ids(final_results[:20]),
            "retrieval_rounds": int(checker_trace.get("retrieval_rounds") or 1),
            **diagnostics,
        }
    )
    return trace


def build_agentic_rerank_diagnostics(
    *,
    gold_article_ids: list[str],
    first_hop_candidates: list[dict[str, Any]],
    baseline_top10: list[dict[str, Any]],
    merged_candidates: list[dict[str, Any]],
    final_top10: list[dict[str, Any]],
    final_top20: list[dict[str, Any]],
    is_multi_article: bool,
) -> dict[str, Any]:
    gold_set = {article_id for article_id in gold_article_ids if article_id}
    first_pool = set(result_article_ids(first_hop_candidates))
    merged_pool = set(result_article_ids(merged_candidates))
    baseline_top10_ids = set(result_article_ids(baseline_top10))
    final_top10_ids = set(result_article_ids(final_top10))
    final_top20_ids = set(result_article_ids(final_top20))
    source_c = bool(gold_set) and not gold_set.issubset(first_pool)
    source_a = bool(gold_set) and gold_set.issubset(baseline_top10_ids)
    c_pool_rescued = source_c and gold_set.issubset(merged_pool)
    c_top10_rescued = source_c and gold_set.issubset(final_top10_ids)
    c_top20_rescued = source_c and gold_set.issubset(final_top20_ids)
    if c_top10_rescued and not c_top20_rescued:
        raise AgenticRerankEvalError("LLM_C_top10_rescued requires LLM_C_top20_rescued.")
    if c_top20_rescued and not c_pool_rescued:
        raise AgenticRerankEvalError("LLM_C_top20_rescued requires LLM_C_pool_rescued.")
    return {
        "source_C": source_c,
        "source_A": source_a,
        "LLM_C_pool_rescued": c_pool_rescued,
        "LLM_C_top10_rescued": c_top10_rescued,
        "LLM_C_top20_rescued": c_top20_rescued,
        "A_dropped@10": source_a and not gold_set.issubset(final_top10_ids),
        "A_dropped@20": source_a and not gold_set.issubset(final_top20_ids),
        "pool_rescued_but_not_top10": c_pool_rescued and not c_top10_rescued,
        "pool_rescued_but_top20_only": (
            c_pool_rescued and c_top20_rescued and not c_top10_rescued
        ),
        "is_multi_article": is_multi_article,
        "merged_pool_full_article_hit": int(bool(gold_set) and gold_set.issubset(merged_pool)),
        "pool_new_gold_article_ids": sorted((gold_set & merged_pool) - (gold_set & first_pool)),
        "pool_still_missing_gold_article_ids": sorted(gold_set - merged_pool),
        "top10_new_gold_article_ids": sorted(
            (gold_set & final_top10_ids) - (gold_set & baseline_top10_ids)
        ),
        "top10_still_missing_gold_article_ids": sorted(gold_set - final_top10_ids),
        "top20_new_gold_article_ids": sorted(
            (gold_set & final_top20_ids) - (gold_set & baseline_top10_ids)
        ),
        "top20_still_missing_gold_article_ids": sorted(gold_set - final_top20_ids),
    }


def summarize_agentic_rerank_traces(
    traces: list[dict[str, Any]],
    *,
    ks: list[int],
    merged_candidate_upper_bound: int,
    baseline_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    phase7_metrics: dict[str, Any],
    phase8_metrics: dict[str, Any],
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
    multi_rows = [row for row in valid_rows if row["is_multi_article"]]
    summary.update(
        {
            "eval_scope": GAP_AWARE_EVAL_SCOPE,
            "retriever_type": "LLM Gap-query Merged Pool + Qwen3 Gap-aware Rerank",
            "first_hop_candidate_top_k_chunks": DEFAULT_FIRST_HOP_TOP_K_CHUNKS,
            "merged_candidate_upper_bound": merged_candidate_upper_bound,
            "source_baseline_metrics": baseline_metrics,
            "top100_control_metrics": control_metrics,
            "phase7_rule_second_hop_metrics": phase7_metrics,
            "phase8_llm_checker_pool_metrics": phase8_metrics,
            "multi_chunk_full_article_hit@20": average(
                row.get("chunk_full_article_hit@20", 0) for row in multi_rows
            ),
            "multi_chunk_article_recall@20": average(
                row.get("chunk_article_recall@20", 0.0) for row in multi_rows
            ),
            "merged_pool_full_article_hit_rate": average(
                row["merged_pool_full_article_hit"] for row in valid_rows
            ),
            "multi_merged_pool_full_article_hit_rate": average(
                row["merged_pool_full_article_hit"] for row in multi_rows
            ),
            "LLM_C_pool_rescued_count": count_true(valid_rows, "LLM_C_pool_rescued"),
            "multi_LLM_C_pool_rescued_count": count_true(
                valid_rows,
                "LLM_C_pool_rescued",
                multi_only=True,
            ),
            "LLM_C_top10_rescued_count": count_true(valid_rows, "LLM_C_top10_rescued"),
            "multi_LLM_C_top10_rescued_count": count_true(
                valid_rows,
                "LLM_C_top10_rescued",
                multi_only=True,
            ),
            "LLM_C_top20_rescued_count": count_true(valid_rows, "LLM_C_top20_rescued"),
            "multi_LLM_C_top20_rescued_count": count_true(
                valid_rows,
                "LLM_C_top20_rescued",
                multi_only=True,
            ),
            "pool_rescued_but_not_top10_count": count_true(
                valid_rows,
                "pool_rescued_but_not_top10",
            ),
            "pool_rescued_but_top20_only_count": count_true(
                valid_rows,
                "pool_rescued_but_top20_only",
            ),
            "A_dropped@10_count": count_true(valid_rows, "A_dropped@10"),
            "A_dropped@20_count": count_true(valid_rows, "A_dropped@20"),
            "checker_valid_count": count_true(traces, "checker_valid"),
            "invalid_json_count": sum(not row.get("checker_valid") for row in traces),
            "avg_llm_calls": average(row["llm_calls"] for row in traces),
            "avg_second_hop_queries": average(len(row["second_hop_queries"]) for row in traces),
            "avg_merged_candidates": average(row["merged_candidate_count"] for row in traces),
            "avg_rerank_candidates": average(row["rerank_candidate_count"] for row in traces),
            "avg_retrieval_rounds": average(row["retrieval_rounds"] for row in traces),
        }
    )
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
    write_jsonl(run_dir / "agentic_rerank_traces.jsonl", traces)
    write_jsonl(
        run_dir / "cases_LLM_C_top10_rescued_by_rerank.jsonl",
        [row for row in traces if row["LLM_C_top10_rescued"]],
    )
    write_jsonl(
        run_dir / "cases_LLM_C_top20_rescued_by_rerank.jsonl",
        [row for row in traces if row["LLM_C_top20_rescued"]],
    )
    write_jsonl(
        run_dir / "multi_cases_LLM_C_top10_rescued_by_rerank.jsonl",
        [row for row in traces if row["LLM_C_top10_rescued"] and row["is_multi_article"]],
    )
    write_jsonl(
        run_dir / "multi_cases_LLM_C_top20_rescued_by_rerank.jsonl",
        [row for row in traces if row["LLM_C_top20_rescued"] and row["is_multi_article"]],
    )
    write_jsonl(
        run_dir / "cases_LLM_C_pool_rescued_but_not_top10.jsonl",
        [row for row in traces if row["pool_rescued_but_not_top10"]],
    )
    write_jsonl(
        run_dir / "cases_LLM_C_pool_rescued_but_top20_only.jsonl",
        [row for row in traces if row["pool_rescued_but_top20_only"]],
    )
    write_jsonl(
        run_dir / "cases_A_dropped_by_agentic_rerank.jsonl",
        [row for row in traces if row["A_dropped@10"] or row["A_dropped@20"]],
    )


def render_metrics_markdown(summary: dict[str, Any]) -> str:
    rows = [
        ["records", summary["records"]],
        ["eval_scope", summary["eval_scope"]],
        ["chunk_full_article_hit@10", f"{summary['chunk_full_article_hit@10']:.4f}"],
        ["chunk_article_recall@10", f"{summary['chunk_article_recall@10']:.4f}"],
        [
            "multi_chunk_full_article_hit@10",
            f"{summary['multi_chunk_full_article_hit@10']:.4f}",
        ],
        ["multi_chunk_article_recall@10", f"{summary['multi_chunk_article_recall@10']:.4f}"],
        ["chunk_full_article_hit@20", f"{summary['chunk_full_article_hit@20']:.4f}"],
        ["chunk_article_recall@20", f"{summary['chunk_article_recall@20']:.4f}"],
        [
            "multi_chunk_full_article_hit@20",
            f"{summary['multi_chunk_full_article_hit@20']:.4f}",
        ],
        ["multi_chunk_article_recall@20", f"{summary['multi_chunk_article_recall@20']:.4f}"],
        ["LLM_C_pool_rescued_count", summary["LLM_C_pool_rescued_count"]],
        ["multi_LLM_C_pool_rescued_count", summary["multi_LLM_C_pool_rescued_count"]],
        ["LLM_C_top10_rescued_count", summary["LLM_C_top10_rescued_count"]],
        ["multi_LLM_C_top10_rescued_count", summary["multi_LLM_C_top10_rescued_count"]],
        ["LLM_C_top20_rescued_count", summary["LLM_C_top20_rescued_count"]],
        ["multi_LLM_C_top20_rescued_count", summary["multi_LLM_C_top20_rescued_count"]],
        ["pool_rescued_but_not_top10_count", summary["pool_rescued_but_not_top10_count"]],
        ["pool_rescued_but_top20_only_count", summary["pool_rescued_but_top20_only_count"]],
        ["A_dropped@10_count", summary["A_dropped@10_count"]],
        ["A_dropped@20_count", summary["A_dropped@20_count"]],
        ["checker_valid_count", summary["checker_valid_count"]],
        ["invalid_json_count", summary["invalid_json_count"]],
        ["avg_llm_calls", f"{summary['avg_llm_calls']:.4f}"],
        ["avg_second_hop_queries", f"{summary['avg_second_hop_queries']:.4f}"],
        ["avg_merged_candidates", f"{summary['avg_merged_candidates']:.4f}"],
        ["avg_rerank_candidates", f"{summary['avg_rerank_candidates']:.4f}"],
    ]
    lines = ["# Phase 9 LLM Gap-query Merged Pool Gap-aware Rerank", ""]
    lines.extend(markdown_table(["metric", "value"], rows))
    return "\n".join(lines)


def render_comparison_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["source_baseline_metrics"]
    control = summary["top100_control_metrics"]
    rule = summary["phase7_rule_second_hop_metrics"]
    phase8 = summary["phase8_llm_checker_pool_metrics"]
    rows = [
        [
            "Top50 Hybrid + Qwen3 baseline",
            "final top10 rerank",
            metric_value(baseline, "chunk_full_article_hit@10"),
            metric_value(baseline, "multi_chunk_full_article_hit@10"),
            metric_value(baseline, "chunk_full_article_hit@20"),
            metric_value(baseline, "multi_chunk_full_article_hit@20"),
            "-",
            "-",
            "-",
            "-",
        ],
        [
            "Top100 Hybrid + Qwen3 control",
            "final top10 rerank",
            metric_value(control, "chunk_full_article_hit@10"),
            metric_value(control, "multi_chunk_full_article_hit@10"),
            metric_value(control, "chunk_full_article_hit@20"),
            metric_value(control, "multi_chunk_full_article_hit@20"),
            "-",
            "-",
            "-",
            "-",
        ],
        [
            "Phase 7 rule title-expanded second-hop + Qwen3",
            "pool + final top10 diagnostics",
            metric_value(rule, "chunk_full_article_hit@10"),
            metric_value(rule, "multi_chunk_full_article_hit@10"),
            metric_value(rule, "chunk_full_article_hit@20"),
            metric_value(rule, "multi_chunk_full_article_hit@20"),
            str(rule.get("C_pool_rescued_count", "-")),
            str(rule.get("multi_C_pool_rescued_count", "-")),
            str(rule.get("C_top10_rescued_count", "-")),
            "-",
        ],
        [
            "Phase 8 LLM checker gap-query pool eval",
            "pool-level eval only",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            str(phase8.get("LLM_C_pool_rescued_count", "-")),
            str(phase8.get("multi_LLM_C_pool_rescued_count", "-")),
            "N/A",
            "N/A",
        ],
        [
            "Phase 9 LLM gap-query merged-pool + Qwen3 gap-aware rerank",
            "final top10 primary, top20 diagnostic",
            metric_value(summary, "chunk_full_article_hit@10"),
            metric_value(summary, "multi_chunk_full_article_hit@10"),
            metric_value(summary, "chunk_full_article_hit@20"),
            metric_value(summary, "multi_chunk_full_article_hit@20"),
            str(summary["LLM_C_pool_rescued_count"]),
            str(summary["multi_LLM_C_pool_rescued_count"]),
            str(summary["LLM_C_top10_rescued_count"]),
            str(summary["LLM_C_top20_rescued_count"]),
        ],
    ]
    lines = [
        "# Phase 9 LLM Gap-query Merged Pool Rerank Comparison",
        "",
        "Phase 8 measures pool-level rescue only.",
        "Phase 9 measures final context quality after reranking the Phase 8 merged pool.",
        "Top10 is the primary context budget; top20 is diagnostic only.",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "method",
                "scope",
                "full@10",
                "multi_full@10",
                "full@20",
                "multi_full@20",
                "C_pool_rescued",
                "multi_C_pool_rescued",
                "C_top10_rescued",
                "C_top20_rescued",
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


def print_summary(console: Console, summary: dict[str, Any], run_dir: Path) -> None:
    console.print()
    console.print("[bold green]Phase 9 Agentic Rerank Finished[/bold green]")
    console.print()
    console.print(f"chunk_full_article_hit@10: {summary['chunk_full_article_hit@10']:.4f}")
    console.print(
        f"multi_chunk_full_article_hit@10: {summary['multi_chunk_full_article_hit@10']:.4f}"
    )
    console.print(f"chunk_full_article_hit@20: {summary['chunk_full_article_hit@20']:.4f}")
    console.print(
        f"multi_chunk_full_article_hit@20: {summary['multi_chunk_full_article_hit@20']:.4f}"
    )
    console.print(f"LLM_C_pool_rescued_count: {summary['LLM_C_pool_rescued_count']}")
    console.print(f"LLM_C_top10_rescued_count: {summary['LLM_C_top10_rescued_count']}")
    console.print(f"LLM_C_top20_rescued_count: {summary['LLM_C_top20_rescued_count']}")
    console.print(f"A_dropped@10_count: {summary['A_dropped@10_count']}")
    console.print()
    console.print(f"Results saved to {run_dir}")
