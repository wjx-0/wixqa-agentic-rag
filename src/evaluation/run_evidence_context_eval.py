from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.agentic.context_budget import ContextBudgetConfig, build_context_usage_snapshot
from src.agentic.evidence_context import (
    DEFAULT_ACTIVE_TOP_K_CHUNKS,
    DEFAULT_INITIAL_PHASE,
    EvidenceContext,
    EvidenceContextError,
    build_initial_evidence_context,
)
from src.evaluation.chunk_eval import (
    CASE_A_CHUNKS,
    chunk_case_labels,
    compute_chunk_retrieval_metrics,
    select_chunk_ks,
    summarize_chunk_metrics,
)
from src.evaluation.eval_utils import CASE_INVALID, markdown_table
from src.evaluation.run_chunk_bm25_eval import load_kb_chunks, metric_rows
from src.evaluation.run_rerank_eval import (
    RerankEvalError,
    build_chunk_lookup,
    candidate_row_to_example,
    load_hybrid_candidates,
    load_required_json,
)
from src.utils.io_utils import ensure_dir, model_to_dict, read_jsonl, write_json, write_jsonl
from src.utils.metrics import average


DEFAULT_RERANK_RUN_DIR = (
    "outputs/rerank_baseline/"
    "qwen3-reranker-4b_inst-wixqa_help_center_v1_ml1024"
)
DEFAULT_OUTPUT_DIR = "outputs/evidence_context"


class EvidenceContextEvalError(RuntimeError):
    pass


def run_evidence_context_eval(
    *,
    rerank_run_dir: str | Path = DEFAULT_RERANK_RUN_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    active_top_k_chunks: int = DEFAULT_ACTIVE_TOP_K_CHUNKS,
    model_context_window_tokens: int = ContextBudgetConfig().model_context_window_tokens,
    console: Console | None = None,
) -> dict[str, Any]:
    validate_args(
        active_top_k_chunks=active_top_k_chunks,
        model_context_window_tokens=model_context_window_tokens,
    )
    console = console or Console()
    rerank_run_dir = Path(rerank_run_dir)
    output_dir = Path(output_dir)

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
    except (OSError, ValueError, RerankEvalError) as exc:
        raise EvidenceContextEvalError(str(exc)) from exc

    budget_config = ContextBudgetConfig(
        model_context_window_tokens=model_context_window_tokens
    )
    contexts: list[EvidenceContext] = []
    prompt_manifests = []
    usage_snapshots = []
    metric_traces = []
    valid_metric_traces = []
    invalid_metric_traces = []
    cases_b_context_not_full = []
    cases_c_context_not_full = []
    multi_cases_context_not_full = []

    for candidate_row in tqdm(candidate_rows, desc="EvidenceContext initial eval"):
        qid = candidate_row["qid"]
        rerank_trace = rerank_traces[qid]
        try:
            context = build_initial_evidence_context(
                candidate_row=candidate_row,
                rerank_trace=rerank_trace,
                chunk_lookup=chunk_lookup,
                active_top_k_chunks=active_top_k_chunks,
            )
        except EvidenceContextError as exc:
            raise EvidenceContextEvalError(str(exc)) from exc

        manifest = context.prompt_manifests[0]
        snapshot = build_context_usage_snapshot(
            llm_call_id=manifest.llm_call_id,
            phase=manifest.phase,
            qid=qid,
            input_texts=[context.question, *[item.text_preview for item in context.active_items]],
            input_chunk_ids=manifest.input_chunk_ids,
            config=budget_config,
        )
        manifest.prompt_token_count = snapshot.projected_input_tokens

        trace = build_context_metric_trace(context)
        contexts.append(context)
        prompt_manifests.append(manifest)
        usage_snapshots.append(snapshot)
        metric_traces.append(trace)
        if trace["case_type"] == CASE_INVALID:
            invalid_metric_traces.append(trace)
        else:
            valid_metric_traces.append(trace)

        source_case_type = context.eval_info.get("source_case_type")
        if source_case_type and source_case_type.startswith("B_"):
            cases_b_context_not_full.append(trace)
        if source_case_type and source_case_type.startswith("C_"):
            cases_c_context_not_full.append(trace)
        if context.eval_info.get("is_multi_article") and trace["case_type"] != CASE_A_CHUNKS:
            multi_cases_context_not_full.append(trace)

    ks = select_chunk_ks(active_top_k_chunks)
    summary = summarize_chunk_metrics(
        dataset_name=rerank_metrics["dataset_name"],
        records=len(metric_traces),
        valid_rows=valid_metric_traces,
        invalid_rows=invalid_metric_traces,
        ks=ks,
        top_k_chunks=active_top_k_chunks,
    )
    summary.update(
        {
            "retriever_type": "EvidenceContext initial context",
            "retrieval_unit": "chunk",
            "source_hybrid_run": source_hybrid_run_dir.name,
            "source_hybrid_run_dir": str(source_hybrid_run_dir),
            "source_rerank_run": rerank_run_dir.name,
            "source_rerank_run_dir": str(rerank_run_dir),
            "chunks_path": str(chunks_path),
            "candidate_top_k_chunks": candidate_top_k_chunks,
            "active_top_k_chunks": active_top_k_chunks,
            "prompt_manifest_count": len(prompt_manifests),
            "context_usage_snapshot_count": len(usage_snapshots),
            "case_B_context_not_full_count": len(cases_b_context_not_full),
            "case_C_context_not_full_count": len(cases_c_context_not_full),
            "multi_cases_context_not_full_count": len(multi_cases_context_not_full),
            "avg_candidate_items": average(len(context.candidate_items) for context in contexts),
            "avg_active_items": average(len(context.active_items) for context in contexts),
            "avg_projected_input_tokens": average(
                snapshot.projected_input_tokens for snapshot in usage_snapshots
            ),
            "avg_usage_ratio": average(snapshot.usage_ratio for snapshot in usage_snapshots),
            "max_usage_ratio": max(
                (snapshot.usage_ratio for snapshot in usage_snapshots),
                default=0.0,
            ),
        }
    )

    run_dir = ensure_dir(
        output_dir / source_hybrid_run_dir.name / rerank_run_dir.name / DEFAULT_INITIAL_PHASE
    )
    write_outputs(
        run_dir,
        summary=summary,
        run_config={
            "phase": DEFAULT_INITIAL_PHASE,
            "source_hybrid_run": source_hybrid_run_dir.name,
            "source_hybrid_run_dir": str(source_hybrid_run_dir),
            "source_rerank_run": rerank_run_dir.name,
            "source_rerank_run_dir": str(rerank_run_dir),
            "chunks_path": str(chunks_path),
            "candidate_top_k_chunks": candidate_top_k_chunks,
            "active_top_k_chunks": active_top_k_chunks,
            "context_budget": model_to_dict(budget_config),
        },
        contexts=contexts,
        prompt_manifests=prompt_manifests,
        usage_snapshots=usage_snapshots,
        cases_b_context_not_full=cases_b_context_not_full,
        cases_c_context_not_full=cases_c_context_not_full,
        multi_cases_context_not_full=multi_cases_context_not_full,
    )
    print_summary(console, summary, run_dir)
    return summary


def validate_args(*, active_top_k_chunks: int, model_context_window_tokens: int) -> None:
    if active_top_k_chunks <= 0:
        raise EvidenceContextEvalError("--active_top_k_chunks must be positive.")
    if active_top_k_chunks > 10:
        raise EvidenceContextEvalError(
            "--active_top_k_chunks must be <= 10 for the initial checker context."
        )
    if model_context_window_tokens <= 0:
        raise EvidenceContextEvalError("--model_context_window_tokens must be positive.")


def required_config_value(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if value in (None, ""):
        raise EvidenceContextEvalError(f"Reranker run_config.json is missing {key!r}.")
    return value


def load_rerank_trace_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise EvidenceContextEvalError(f"Missing rerank trace artifact: {path}.")
    traces: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = row.get("qid")
        if not qid:
            raise EvidenceContextEvalError(f"Rerank trace in {path} is missing qid.")
        if qid in traces:
            raise EvidenceContextEvalError(f"Duplicate qid in {path}: {qid}.")
        if not isinstance(row.get("top10_reranked_chunks"), list):
            raise EvidenceContextEvalError(
                f"Rerank trace qid={qid} is missing top10_reranked_chunks."
            )
        traces[qid] = row
    return traces


def validate_qid_sets(
    candidate_rows: list[dict[str, Any]],
    rerank_traces: dict[str, dict[str, Any]],
) -> None:
    candidate_qids = {row["qid"] for row in candidate_rows}
    trace_qids = set(rerank_traces)
    if candidate_qids != trace_qids:
        missing = sorted(candidate_qids - trace_qids)[:5]
        extra = sorted(trace_qids - candidate_qids)[:5]
        raise EvidenceContextEvalError(
            f"Hybrid candidates and rerank traces disagree on qids. "
            f"missing_traces={missing}, extra_traces={extra}."
        )


def build_context_metric_trace(context: EvidenceContext) -> dict[str, Any]:
    example = candidate_row_to_example(
        {
            "qid": context.qid,
            "dataset_name": context.dataset_name,
            "question": context.question,
            "answer": context.answer,
            "gold_article_ids": context.eval_info.get("gold_article_ids") or [],
            "num_gold_articles": context.eval_info.get("num_gold_articles") or 0,
            "is_multi_article": context.eval_info.get("is_multi_article") or False,
        }
    )
    active_results = [evidence_item_to_result(item) for item in context.active_items]
    metrics = compute_chunk_retrieval_metrics(
        example.article_ids,
        active_results,
        select_chunk_ks(len(active_results)),
    )
    top_chunk_article_ids = [item.article_id for item in context.active_items]
    gold_set = {article_id for article_id in example.article_ids if article_id}
    trace = {
        "qid": context.qid,
        "dataset_name": context.dataset_name,
        "question": context.question,
        "answer": context.answer,
        "gold_article_ids": example.article_ids,
        "num_gold_articles": example.num_gold_articles,
        "is_multi_article": example.is_multi_article,
        "source_case_type": context.eval_info.get("source_case_type"),
        "candidate_chunk_ids": [item.chunk_id for item in context.candidate_items],
        "active_chunk_ids": [item.chunk_id for item in context.active_items],
        "active_article_ids": top_chunk_article_ids,
        "active_top10_chunk_ids": [item.chunk_id for item in context.active_items[:10]],
        "active_top10_article_ids": top_chunk_article_ids[:10],
        "missing_articles_in_active_context": sorted(gold_set - set(top_chunk_article_ids)),
        "candidate_items_count": len(context.candidate_items),
        "active_items_count": len(context.active_items),
        "prompt_manifest_llm_call_id": context.prompt_manifests[0].llm_call_id,
        "prompt_manifest_input_chunk_ids": context.prompt_manifests[0].input_chunk_ids,
    }
    trace.update({key: value for key, value in metrics.items() if key != "is_valid"})
    trace["case_type"] = classify_initial_context_case(example.article_ids, top_chunk_article_ids)
    return trace


def classify_initial_context_case(
    gold_article_ids: list[str],
    active_article_ids: list[str],
) -> str:
    gold_set = {article_id for article_id in gold_article_ids if article_id}
    if not gold_set:
        return CASE_INVALID
    if gold_set.issubset(set(active_article_ids[:10])):
        return CASE_A_CHUNKS
    return chunk_case_labels(10)[2]


def evidence_item_to_result(item: Any) -> dict[str, Any]:
    return {
        "rank": item.rank,
        "chunk_id": item.chunk_id,
        "article_id": item.article_id,
        "title": item.title,
        "score": item.score,
        "text_preview": item.text_preview,
    }


def write_outputs(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    run_config: dict[str, Any],
    contexts: list[EvidenceContext],
    prompt_manifests: list[Any],
    usage_snapshots: list[Any],
    cases_b_context_not_full: list[dict[str, Any]],
    cases_c_context_not_full: list[dict[str, Any]],
    multi_cases_context_not_full: list[dict[str, Any]],
) -> None:
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "metrics.json", summary)
    (run_dir / "metrics.md").write_text(render_metrics_markdown(summary), encoding="utf-8")
    write_jsonl(run_dir / "evidence_context_traces.jsonl", contexts)
    write_jsonl(run_dir / "prompt_manifests.jsonl", prompt_manifests)
    write_jsonl(run_dir / "context_usage_snapshots.jsonl", usage_snapshots)
    write_jsonl(run_dir / "cases_B_context_not_full.jsonl", cases_b_context_not_full)
    write_jsonl(run_dir / "cases_C_context_not_full.jsonl", cases_c_context_not_full)
    write_jsonl(run_dir / "multi_cases_context_not_full.jsonl", multi_cases_context_not_full)


def render_metrics_markdown(summary: dict[str, Any]) -> str:
    top_k_chunks = summary["active_top_k_chunks"]
    lines = [
        "# EvidenceContext Initial Context",
        "",
        "## 基本信息",
        "",
    ]
    lines.extend(
        markdown_table(
            ["指标", "数值"],
            [
                ["source_hybrid_run", summary["source_hybrid_run"]],
                ["source_rerank_run", summary["source_rerank_run"]],
                ["candidate_top_k_chunks", summary["candidate_top_k_chunks"]],
                ["active_top_k_chunks", top_k_chunks],
                ["records", summary["records"]],
                ["valid_records", summary["valid_records"]],
                ["prompt_manifest_count", summary["prompt_manifest_count"]],
                ["context_usage_snapshot_count", summary["context_usage_snapshot_count"]],
                ["avg_candidate_items", f"{summary['avg_candidate_items']:.2f}"],
                ["avg_active_items", f"{summary['avg_active_items']:.2f}"],
                ["avg_projected_input_tokens", f"{summary['avg_projected_input_tokens']:.2f}"],
                ["avg_usage_ratio", f"{summary['avg_usage_ratio']:.4f}"],
                ["max_usage_ratio", f"{summary['max_usage_ratio']:.4f}"],
            ],
        )
    )
    lines.extend(["", "## 初始上下文指标", ""])
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
            metric_rows(summary),
        )
    )
    lines.extend(["", *render_context_group_metrics(summary, top_k_chunks)])
    lines.extend(["", "## 需要后续补证据的样本", ""])
    lines.extend(
        markdown_table(
            ["case_file", "数量"],
            [
                ["cases_B_context_not_full.jsonl", summary["case_B_context_not_full_count"]],
                ["cases_C_context_not_full.jsonl", summary["case_C_context_not_full_count"]],
                ["multi_cases_context_not_full.jsonl", summary["multi_cases_context_not_full_count"]],
            ],
        )
    )
    return "\n".join(lines)


def render_context_group_metrics(summary: dict[str, Any], top_k_chunks: int) -> list[str]:
    headers = [
        "分组",
        "样本数",
        "chunk_full_article_hit@10",
        "chunk_article_recall@10",
    ]
    if top_k_chunks != 10:
        headers.extend(
            [
                f"chunk_full_article_hit@{top_k_chunks}",
                f"chunk_article_recall@{top_k_chunks}",
            ]
        )

    rows = []
    for group in ("single", "multi"):
        row = [
            group,
            summary[f"{group}_article_records"],
            f"{summary[f'{group}_chunk_full_article_hit@10']:.4f}",
            f"{summary[f'{group}_chunk_article_recall@10']:.4f}",
        ]
        if top_k_chunks != 10:
            row.extend(
                [
                    f"{summary[f'{group}_chunk_full_article_hit@{top_k_chunks}']:.4f}",
                    f"{summary[f'{group}_chunk_article_recall@{top_k_chunks}']:.4f}",
                ]
            )
        rows.append(row)

    return [
        "## 单文章 vs 多文章",
        "",
        *markdown_table(headers, rows),
    ]


def print_summary(console: Console, summary: dict[str, Any], run_dir: Path) -> None:
    console.print(
        "[bold green]EvidenceContext initial eval complete[/bold green] "
        f"records={summary['records']} "
        f"full@10={summary['chunk_full_article_hit@10']:.4f} "
        f"run_dir={run_dir}"
    )
