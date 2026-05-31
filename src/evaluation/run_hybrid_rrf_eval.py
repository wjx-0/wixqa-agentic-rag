from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rich.console import Console

from src.evaluation.chunk_eval import (
    build_chunk_trace,
    chunk_case_labels,
    compute_chunk_retrieval_metrics,
    result_article_ids,
    select_chunk_ks,
    summarize_chunk_metrics,
)
from src.evaluation.eval_utils import (
    BM25EvalError,
    CASE_INVALID,
    VALID_QA_DATASETS,
    load_kb_articles,
    load_qa_examples,
    markdown_table,
)
from src.evaluation.run_chunk_bm25_eval import (
    load_kb_chunks,
    metric_rows,
    render_failed_examples,
    render_group_metrics,
)
from src.evaluation.run_dense_faiss_eval import DenseFaissEvalError, load_index_config
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME
from src.retrievers.hybrid_retriever import HybridRetriever, HybridRetrieverError
from src.retrievers.rrf import DEFAULT_BM25_WEIGHT, DEFAULT_DENSE_WEIGHT
from src.utils.io_utils import ensure_dir, write_json, write_jsonl


class HybridRRFEvalError(RuntimeError):
    pass


def run_hybrid_rrf_eval(
    *,
    processed_dir: str | Path,
    chunks_path: str | Path,
    index_dir: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    model_name: str = DEFAULT_DENSE_MODEL_NAME,
    local_files_only: bool = True,
    branch_top_k_chunks: int = 50,
    fused_top_k_chunks: int = 50,
    rrf_k: int = 60,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    dense_worker_mode: str = "full",
    dense_query_batch_size: int = 16,
    device: str | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    validate_args(
        dataset_name=dataset_name,
        branch_top_k_chunks=branch_top_k_chunks,
        fused_top_k_chunks=fused_top_k_chunks,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        dense_query_batch_size=dense_query_batch_size,
    )
    console = console or Console()
    processed_dir = Path(processed_dir)
    index_dir = Path(index_dir)
    output_dir = ensure_dir(output_dir)

    try:
        index_config = load_index_config(index_dir)
        kb_articles = load_kb_articles(processed_dir)
        kb_chunks = load_kb_chunks(Path(chunks_path))
        qa_examples = load_qa_examples(processed_dir, dataset_name)
        with HybridRetriever(
            chunks=kb_chunks,
            index_dir=index_dir,
            model_name=model_name,
            local_files_only=local_files_only,
            device=device,
            dense_worker_mode=dense_worker_mode,
        ) as retriever:
            result_batches = retriever.search_batch(
                [example.question for example in qa_examples],
                branch_top_k_chunks=branch_top_k_chunks,
                fused_top_k_chunks=fused_top_k_chunks,
                rrf_k=rrf_k,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                dense_query_batch_size=dense_query_batch_size,
            )
    except (BM25EvalError, DenseFaissEvalError, HybridRetrieverError) as exc:
        raise HybridRRFEvalError(str(exc)) from exc

    article_lookup = {article.article_id: article for article in kb_articles}
    ks = select_chunk_ks(fused_top_k_chunks)
    case_a, case_b, case_c = chunk_case_labels(fused_top_k_chunks)
    case_rows: dict[str, list[dict[str, Any]]] = {case_a: [], case_b: [], case_c: []}
    method_rows: dict[str, list[dict[str, Any]]] = {"bm25": [], "dense": [], "hybrid": []}
    traces = []
    complementarity_rows = []

    for example, results in zip(qa_examples, result_batches):
        method_metrics = {
            method: compute_chunk_retrieval_metrics(example.article_ids, chunk_results, ks)
            for method, chunk_results in results.items()
        }
        for method, chunk_results in results.items():
            method_rows[method].append(
                build_evaluation_row(
                    example=example,
                    chunk_results=chunk_results,
                    metrics=method_metrics[method],
                    top_k_chunks=fused_top_k_chunks,
                )
            )

        trace = build_chunk_trace(
            example,
            results["hybrid"],
            method_metrics["hybrid"],
            article_lookup,
            top_k_chunks=fused_top_k_chunks,
        )
        trace.update(
            {
                "branch_top_k_chunks": branch_top_k_chunks,
                "fused_top_k_chunks": fused_top_k_chunks,
                "rrf_k": rrf_k,
                "bm25_weight": bm25_weight,
                "dense_weight": dense_weight,
                "top10_bm25_chunk_ids": chunk_ids(results["bm25"][:10]),
                "top10_dense_chunk_ids": chunk_ids(results["dense"][:10]),
                "top10_hybrid_chunks": results["hybrid"][:10],
            }
        )
        traces.append(trace)
        if method_metrics["hybrid"]["is_valid"]:
            case_rows[trace["case_type"]].append(trace)
        complementarity_rows.append(
            build_complementarity_row(
                example.qid,
                example.article_ids,
                results,
                fused_top_k_chunks=fused_top_k_chunks,
            )
        )

    summaries = {
        method: summarize_chunk_metrics(
            dataset_name=dataset_name,
            records=len(qa_examples),
            valid_rows=[row for row in rows if row["is_valid"]],
            invalid_rows=[row for row in rows if not row["is_valid"]],
            ks=ks,
            top_k_chunks=fused_top_k_chunks,
        )
        for method, rows in method_rows.items()
    }
    summary = summaries["hybrid"]
    summary.update(
        {
            "retriever_type": "Hybrid RRF Retrieval",
            "retrieval_unit": "chunk",
            "model_name": model_name,
            "tokenizer_name": index_config.get("tokenizer_name"),
            "faiss_index_type": index_config.get("faiss_index_type", "IndexFlatIP"),
            "normalize_embeddings": bool(index_config.get("normalize_embeddings", True)),
            "chunk_size_tokens": index_config.get("chunk_size_tokens"),
            "chunk_overlap_tokens": index_config.get("chunk_overlap_tokens"),
            "chunk_records": len(kb_chunks),
            "chunk_article_records": len({chunk.article_id for chunk in kb_chunks}),
            "branch_top_k_chunks": branch_top_k_chunks,
            "fused_top_k_chunks": fused_top_k_chunks,
            "rrf_k": rrf_k,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "dense_worker_mode": dense_worker_mode,
            "dense_query_batch_size": dense_query_batch_size,
            f"avg_unique_articles_from_top{fused_top_k_chunks}_chunks": summary[
                f"unique_articles@{fused_top_k_chunks}_chunks"
            ],
            "hybrid_top10_rescued_gold_articles": sum(
                len(row["hybrid_top10_rescued_gold_article_ids"])
                for row in complementarity_rows
            ),
            "hybrid_top50_retained_gold_articles": sum(
                len(row["hybrid_top50_retained_gold_article_ids"])
                for row in complementarity_rows
            ),
            "avg_shared_candidate_chunks": average(
                len(row["shared_candidate_chunks"]) for row in complementarity_rows
            ),
        }
    )
    if fused_top_k_chunks >= 100:
        summary["hybrid_top100_retained_gold_articles"] = sum(
            len(row["hybrid_top100_retained_gold_article_ids"])
            for row in complementarity_rows
        )

    run_name = build_run_name(
        dataset_name=dataset_name,
        branch_top_k_chunks=branch_top_k_chunks,
        fused_top_k_chunks=fused_top_k_chunks,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )
    run_dir = ensure_dir(output_dir / run_name)
    write_json(run_dir / "metrics.json", summary)
    (run_dir / "metrics.md").write_text(
        render_hybrid_metrics_markdown(summary, traces),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "retrieval_traces.jsonl", traces)
    write_jsonl(run_dir / f"cases_{case_a}.jsonl", case_rows[case_a])
    write_jsonl(run_dir / f"cases_{case_b}.jsonl", case_rows[case_b])
    write_jsonl(run_dir / f"cases_{case_c}.jsonl", case_rows[case_c])
    write_jsonl(
        run_dir / "complementarity_analysis.jsonl",
        complementarity_rows,
    )
    (run_dir / "comparison.md").write_text(
        render_comparison_markdown(summary, summaries),
        encoding="utf-8",
    )

    print_summary(console, summary, run_dir)
    return summary


def validate_args(
    *,
    dataset_name: str,
    branch_top_k_chunks: int,
    fused_top_k_chunks: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
    dense_query_batch_size: int,
) -> None:
    if dataset_name not in VALID_QA_DATASETS:
        raise HybridRRFEvalError(
            f"Unsupported dataset {dataset_name!r}. Choose one of: {sorted(VALID_QA_DATASETS)}"
        )
    if branch_top_k_chunks <= 0:
        raise HybridRRFEvalError("--branch_top_k_chunks must be a positive integer.")
    if fused_top_k_chunks < 50:
        raise HybridRRFEvalError("--fused_top_k_chunks must be at least 50.")
    if fused_top_k_chunks > branch_top_k_chunks:
        raise HybridRRFEvalError(
            "--fused_top_k_chunks must be smaller than or equal to --branch_top_k_chunks."
        )
    if rrf_k <= 0:
        raise HybridRRFEvalError("--rrf_k must be a positive integer.")
    if not math.isfinite(bm25_weight) or bm25_weight <= 0:
        raise HybridRRFEvalError("--bm25_weight must be a positive finite number.")
    if not math.isfinite(dense_weight) or dense_weight <= 0:
        raise HybridRRFEvalError("--dense_weight must be a positive finite number.")
    if dense_query_batch_size <= 0:
        raise HybridRRFEvalError("--dense_query_batch_size must be a positive integer.")


def build_evaluation_row(
    *,
    example: Any,
    chunk_results: list[dict[str, Any]],
    metrics: dict[str, Any],
    top_k_chunks: int,
) -> dict[str, Any]:
    article_ids = result_article_ids(chunk_results)
    return {
        **metrics,
        "qid": example.qid,
        "is_multi_article": example.is_multi_article,
        "case_type": classify_case(example.article_ids, article_ids, top_k_chunks),
    }


def classify_case(
    gold_article_ids: list[str],
    chunk_article_ids: list[str],
    top_k_chunks: int,
) -> str:
    from src.evaluation.chunk_eval import classify_chunk_case

    return classify_chunk_case(gold_article_ids, chunk_article_ids, top_k_chunks)


def build_complementarity_row(
    qid: str,
    gold_article_ids: list[str],
    results: dict[str, list[dict[str, Any]]],
    *,
    fused_top_k_chunks: int,
) -> dict[str, Any]:
    gold_set = set(gold_article_ids)
    article_sets = {
        method: set(result_article_ids(chunk_results[:fused_top_k_chunks])) & gold_set
        for method, chunk_results in results.items()
    }
    bm25_top10 = set(result_article_ids(results["bm25"][:10]))
    dense_top10 = set(result_article_ids(results["dense"][:10]))
    hybrid_top10 = set(result_article_ids(results["hybrid"][:10]))
    hybrid_top50 = set(result_article_ids(results["hybrid"][:50]))
    missing_both_top10 = gold_set - bm25_top10 - dense_top10
    bm25_chunk_ids = set(chunk_ids(results["bm25"]))
    dense_chunk_ids = set(chunk_ids(results["dense"]))

    row: dict[str, Any] = {
        "qid": qid,
        "gold_article_ids": gold_article_ids,
        "gold_article_first_chunk_rank": {
            method: gold_article_first_chunk_rank(gold_article_ids, chunk_results)
            for method, chunk_results in results.items()
        },
        f"gold_article_ids_in_bm25_top{fused_top_k_chunks}": sorted(article_sets["bm25"]),
        f"gold_article_ids_in_dense_top{fused_top_k_chunks}": sorted(article_sets["dense"]),
        f"gold_article_ids_in_hybrid_top{fused_top_k_chunks}": sorted(article_sets["hybrid"]),
        "bm25_only_gold_article_ids": sorted(article_sets["bm25"] - article_sets["dense"]),
        "dense_only_gold_article_ids": sorted(article_sets["dense"] - article_sets["bm25"]),
        "missing_articles_after_hybrid": sorted(gold_set - article_sets["hybrid"]),
        "shared_candidate_chunks": sorted(bm25_chunk_ids & dense_chunk_ids),
        "bm25_only_candidate_chunks": sorted(bm25_chunk_ids - dense_chunk_ids),
        "dense_only_candidate_chunks": sorted(dense_chunk_ids - bm25_chunk_ids),
        "hybrid_top10_rescued_gold_article_ids": sorted(missing_both_top10 & hybrid_top10),
        "hybrid_top50_retained_gold_article_ids": sorted(missing_both_top10 & hybrid_top50),
    }
    if fused_top_k_chunks >= 100:
        hybrid_top100 = set(result_article_ids(results["hybrid"][:100]))
        row["hybrid_top100_retained_gold_article_ids"] = sorted(
            missing_both_top10 & hybrid_top100
        )
    return row


def gold_article_first_chunk_rank(
    gold_article_ids: list[str],
    chunk_results: list[dict[str, Any]],
) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {article_id: None for article_id in gold_article_ids}
    for fallback_rank, result in enumerate(chunk_results, start=1):
        article_id = result.get("article_id")
        if article_id in ranks and ranks[article_id] is None:
            ranks[article_id] = int(result.get("rank") or fallback_rank)
    return ranks


def chunk_ids(results: list[dict[str, Any]]) -> list[str]:
    return [result["chunk_id"] for result in results if result.get("chunk_id")]


def render_hybrid_metrics_markdown(
    summary: dict[str, Any],
    traces: list[dict[str, Any]],
) -> str:
    top_k_chunks = summary["fused_top_k_chunks"]
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    lines = [
        f"# Hybrid RRF Baseline：{summary['dataset_name']}",
        "",
        "## 基本信息",
        "",
    ]
    lines.extend(
        markdown_table(
            ["指标", "数值"],
            [
                ["retriever_type", summary["retriever_type"]],
                ["model_name", summary["model_name"]],
                ["tokenizer_name", summary["tokenizer_name"]],
                ["faiss_index_type", summary["faiss_index_type"]],
                ["records", summary["records"]],
                ["valid_records", summary["valid_records"]],
                ["chunk_records", summary["chunk_records"]],
                ["branch_top_k_chunks", summary["branch_top_k_chunks"]],
                ["fused_top_k_chunks", top_k_chunks],
                ["rrf_k", summary["rrf_k"]],
                ["bm25_weight", summary["bm25_weight"]],
                ["dense_weight", summary["dense_weight"]],
                ["dense_worker_mode", summary["dense_worker_mode"]],
                ["dense_query_batch_size", summary["dense_query_batch_size"]],
                ["hybrid_top10_rescued_gold_articles", summary["hybrid_top10_rescued_gold_articles"]],
                ["hybrid_top50_retained_gold_articles", summary["hybrid_top50_retained_gold_articles"]],
            ],
        )
    )
    lines.extend(["", "## 整体指标", ""])
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
    lines.extend(["", f"MRR: `{summary['mrr']:.4f}`", ""])
    lines.extend(render_group_metrics({**summary, "top_k_chunks": top_k_chunks}))
    lines.extend(["", "## Case 分布", ""])
    lines.extend(
        markdown_table(
            ["case_type", "数量"],
            [
                [case_a, summary["case_A_count"]],
                [case_b, summary["case_B_count"]],
                [case_c, summary["case_C_count"]],
            ],
        )
    )
    lines.extend(render_failed_examples(traces, case_c, top_k_chunks))
    return "\n".join(lines)


def render_comparison_markdown(
    hybrid_summary: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> str:
    cutoff = hybrid_summary["fused_top_k_chunks"]
    lines = [
        f"# Hybrid RRF 对比：{hybrid_summary['dataset_name']}",
        "",
        (
            f"同次运行对比：Chunk BM25 top{hybrid_summary['branch_top_k_chunks']} + "
            f"Dense FAISS top{hybrid_summary['branch_top_k_chunks']} -> Hybrid top{cutoff}，"
            f"RRF k={hybrid_summary['rrf_k']}，"
            f"BM25 weight={hybrid_summary['bm25_weight']:g}，"
            f"Dense weight={hybrid_summary['dense_weight']:g}。"
        ),
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "method",
                f"chunk_hit@{cutoff}",
                f"chunk_full_article_hit@{cutoff}",
                f"chunk_article_recall@{cutoff}",
                "mrr",
                f"unique_articles@{cutoff}_chunks",
                f"duplicate_article_ratio@{cutoff}_chunks",
            ],
            [
                [
                    method,
                    f"{summary[f'chunk_hit@{cutoff}']:.4f}",
                    f"{summary[f'chunk_full_article_hit@{cutoff}']:.4f}",
                    f"{summary[f'chunk_article_recall@{cutoff}']:.4f}",
                    f"{summary['mrr']:.4f}",
                    f"{summary[f'unique_articles@{cutoff}_chunks']:.2f}",
                    f"{summary[f'duplicate_article_ratio@{cutoff}_chunks']:.4f}",
                ]
                for method, summary in summaries.items()
            ],
        )
    )
    lines.extend(["", "## Top 10 对比", ""])
    lines.extend(
        markdown_table(
            ["method", "chunk_hit@10", "chunk_full_article_hit@10", "chunk_article_recall@10"],
            [
                [
                    method,
                    f"{summary['chunk_hit@10']:.4f}",
                    f"{summary['chunk_full_article_hit@10']:.4f}",
                    f"{summary['chunk_article_recall@10']:.4f}",
                ]
                for method, summary in summaries.items()
            ],
        )
    )
    return "\n".join(lines)


def print_summary(console: Console, summary: dict[str, Any], output_dir: Path) -> None:
    cutoff = summary["fused_top_k_chunks"]
    console.print()
    console.print("[bold green]Hybrid RRF Baseline Finished[/bold green]")
    console.print()
    console.print(f"Dataset: {summary['dataset_name']}")
    console.print(
        f"Candidates: {summary['branch_top_k_chunks']} + {summary['branch_top_k_chunks']} "
        f"-> {cutoff}"
    )
    console.print(f"RRF k: {summary['rrf_k']}")
    console.print(f"RRF weights: BM25={summary['bm25_weight']:g}, Dense={summary['dense_weight']:g}")
    console.print(f"Dense worker mode: {summary['dense_worker_mode']}")
    console.print()
    for key in ("chunk_hit@10", "chunk_full_article_hit@10", "chunk_article_recall@10", "mrr"):
        console.print(f"{key}: {summary.get(key, 0.0):.4f}")
    console.print(
        f"chunk_full_article_hit@{cutoff}: "
        f"{summary.get(f'chunk_full_article_hit@{cutoff}', 0.0):.4f}"
    )
    console.print()
    console.print(f"Results saved to {output_dir}/")


def average(values: Any) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(float(value) for value in value_list) / len(value_list)


def weight_slug(weight: float) -> str:
    return f"{weight:g}".replace(".", "p")


def build_run_name(
    *,
    dataset_name: str,
    branch_top_k_chunks: int,
    fused_top_k_chunks: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
) -> str:
    return (
        f"hybrid_rrf_b{branch_top_k_chunks}_f{fused_top_k_chunks}_k{rrf_k}"
        f"_bw{weight_slug(bm25_weight)}_dw{weight_slug(dense_weight)}_{dataset_name}"
    )
