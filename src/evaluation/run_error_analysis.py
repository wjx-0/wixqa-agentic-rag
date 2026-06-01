from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from src.data.schema import KBArticle
from src.evaluation.chunk_eval import chunk_case_labels
from src.evaluation.eval_utils import CASE_INVALID, BM25EvalError, load_kb_articles, markdown_table
from src.utils.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


DEFAULT_RERANK_RUN_DIR = (
    "outputs/rerank_baseline/"
    "hybrid_rrf_b100_f50_k60_bw1_dw2_wixqa_expertwritten/"
    "qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024"
)
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_OUTPUT_DIR = "outputs/error_analysis"
TOP10_CHUNKS = 10


class ErrorAnalysisError(RuntimeError):
    pass


def run_error_analysis(
    *,
    rerank_run_dir: str | Path = DEFAULT_RERANK_RUN_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_k_chunks: int | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    console = console or Console()
    rerank_run_dir = Path(rerank_run_dir)
    processed_dir = Path(processed_dir)
    output_dir = ensure_dir(output_dir)

    try:
        source_traces = load_rerank_traces(rerank_run_dir / "rerank_traces.jsonl")
        source_metrics = load_optional_json(rerank_run_dir / "metrics.json")
        top_k_chunks = infer_top_k_chunks(
            explicit_top_k_chunks=top_k_chunks,
            source_metrics=source_metrics,
            source_traces=source_traces,
        )
        article_lookup = {article.article_id: article for article in load_kb_articles(processed_dir)}
    except (BM25EvalError, OSError, ValueError) as exc:
        raise ErrorAnalysisError(str(exc)) from exc

    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    case_traces = [
        build_error_trace(trace, article_lookup, top_k_chunks=top_k_chunks)
        for trace in source_traces
    ]
    case_rows = {
        case_a: [trace for trace in case_traces if trace["case_type"] == case_a],
        case_b: [trace for trace in case_traces if trace["case_type"] == case_b],
        case_c: [trace for trace in case_traces if trace["case_type"] == case_c],
    }
    multi_case_b = [trace for trace in case_rows[case_b] if trace["is_multi_article"]]
    multi_case_c = [trace for trace in case_rows[case_c] if trace["is_multi_article"]]

    summary = summarize_error_traces(
        case_traces,
        top_k_chunks=top_k_chunks,
        source_rerank_run=str(rerank_run_dir),
    )

    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_error_summary_markdown(summary),
        encoding="utf-8",
    )
    write_jsonl(output_dir / f"case_traces_top{top_k_chunks}_chunks.jsonl", case_traces)
    write_jsonl(output_dir / f"cases_{case_a}.jsonl", case_rows[case_a])
    write_jsonl(output_dir / f"cases_{case_b}.jsonl", case_rows[case_b])
    write_jsonl(output_dir / f"cases_{case_c}.jsonl", case_rows[case_c])
    write_jsonl(output_dir / f"multi_cases_{case_b}.jsonl", multi_case_b)
    write_jsonl(output_dir / f"multi_cases_{case_c}.jsonl", multi_case_c)

    print_summary(console, summary, output_dir)
    return summary


def load_rerank_traces(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ErrorAnalysisError(f"Missing rerank trace file: {path}")
    traces = list(read_jsonl(path))
    if not traces:
        raise ErrorAnalysisError(f"No rerank traces found in {path}")
    return traces


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ErrorAnalysisError(f"Expected a JSON object in {path}.")
    return value


def infer_top_k_chunks(
    *,
    explicit_top_k_chunks: int | None,
    source_metrics: dict[str, Any],
    source_traces: list[dict[str, Any]],
) -> int:
    if explicit_top_k_chunks is not None:
        top_k_chunks = explicit_top_k_chunks
    else:
        first_trace = source_traces[0]
        top_k_chunks = int(
            source_metrics.get("candidate_top_k_chunks")
            or source_metrics.get("top_k_chunks")
            or first_trace.get("candidate_top_k_chunks")
            or first_trace.get("requested_top_k_chunks")
            or 50
        )
    if top_k_chunks <= TOP10_CHUNKS:
        raise ErrorAnalysisError("--top_k_chunks must be greater than 10 for A/B/C analysis.")
    return top_k_chunks


def build_error_trace(
    source_trace: dict[str, Any],
    article_lookup: dict[str, KBArticle],
    *,
    top_k_chunks: int,
) -> dict[str, Any]:
    gold_article_ids = clean_article_ids(source_trace.get("gold_article_ids") or [])
    top10_article_ids = trace_article_ids(source_trace, TOP10_CHUNKS)
    top_cutoff_article_ids = trace_article_ids(source_trace, top_k_chunks)
    case_type = classify_error_case(
        gold_article_ids,
        top10_article_ids,
        top_cutoff_article_ids,
        top_k_chunks=top_k_chunks,
    )

    trace: dict[str, Any] = {
        "qid": source_trace.get("qid"),
        "dataset_name": source_trace.get("dataset_name"),
        "question": source_trace.get("question"),
        "answer": source_trace.get("answer"),
        "is_multi_article": bool(source_trace.get("is_multi_article")),
        "num_gold_articles": int(source_trace.get("num_gold_articles") or len(gold_article_ids)),
        "gold_article_ids": gold_article_ids,
        "gold_article_titles": article_titles(gold_article_ids, article_lookup),
        "top10_chunks_article_ids": top10_article_ids,
        f"top{top_k_chunks}_chunks_article_ids": top_cutoff_article_ids,
        "top10_chunks_titles": article_titles(top10_article_ids, article_lookup),
        f"top{top_k_chunks}_chunks_titles": article_titles(top_cutoff_article_ids, article_lookup),
        "case_type": case_type,
        "gold_article_first_chunk_rank": first_chunk_ranks(
            gold_article_ids,
            top_cutoff_article_ids,
        ),
        "gold_article_first_chunk_rank_before_rerank": source_trace.get(
            "gold_article_first_chunk_rank_before_rerank"
        ),
        "gold_article_first_chunk_rank_after_rerank": source_trace.get(
            "gold_article_first_chunk_rank_after_rerank"
        ),
        "rerank_top10_rescued_gold_article_ids": clean_article_ids(
            source_trace.get("rerank_top10_rescued_gold_article_ids") or []
        ),
        "rerank_top10_dropped_gold_article_ids": clean_article_ids(
            source_trace.get("rerank_top10_dropped_gold_article_ids") or []
        ),
    }
    trace.update(article_coverage_metrics(gold_article_ids, top10_article_ids, TOP10_CHUNKS))
    trace.update(article_coverage_metrics(gold_article_ids, top_cutoff_article_ids, top_k_chunks))
    return trace


def trace_article_ids(source_trace: dict[str, Any], k: int) -> list[str]:
    for key in (
        f"top{k}_chunks_article_ids",
        f"top{k}_chunk_article_ids",
        f"top{k}_article_ids",
    ):
        if key in source_trace:
            return clean_article_ids(source_trace.get(key) or [])
    if k == TOP10_CHUNKS and "top10_reranked_chunks" in source_trace:
        return clean_article_ids(
            result.get("article_id") for result in source_trace.get("top10_reranked_chunks") or []
        )
    raise ErrorAnalysisError(
        f"Trace qid={source_trace.get('qid')} is missing top{k} chunk article ids."
    )


def clean_article_ids(values: Any) -> list[str]:
    output = []
    for value in values or []:
        if value is None:
            continue
        article_id = str(value).strip()
        if article_id:
            output.append(article_id)
    return output


def article_titles(
    article_ids: list[str],
    article_lookup: dict[str, KBArticle],
) -> list[str | None]:
    return [
        article_lookup[article_id].title if article_id in article_lookup else None
        for article_id in article_ids
    ]


def article_coverage_metrics(
    gold_article_ids: list[str],
    chunk_article_ids: list[str],
    k: int,
) -> dict[str, Any]:
    gold_set = set(gold_article_ids)
    retrieved_set = set(chunk_article_ids)
    covered = gold_set & retrieved_set
    returned_chunks = len(chunk_article_ids)
    unique_articles = len(retrieved_set)
    return {
        f"article_full_hit@{k}_chunks": int(bool(gold_set) and gold_set.issubset(retrieved_set)),
        f"article_recall@{k}_chunks": len(covered) / len(gold_set) if gold_set else 0.0,
        f"unique_articles@{k}_chunks": unique_articles,
        f"duplicate_article_ratio@{k}_chunks": (
            1.0 - unique_articles / returned_chunks if returned_chunks else 0.0
        ),
        f"missing_articles_at_{k}_chunks": sorted(gold_set - retrieved_set),
    }


def first_chunk_ranks(
    gold_article_ids: list[str],
    chunk_article_ids: list[str],
) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {article_id: None for article_id in gold_article_ids}
    for rank, article_id in enumerate(chunk_article_ids, start=1):
        if article_id in ranks and ranks[article_id] is None:
            ranks[article_id] = rank
    return ranks


def classify_error_case(
    gold_article_ids: list[str],
    top10_article_ids: list[str],
    top_cutoff_article_ids: list[str],
    *,
    top_k_chunks: int,
) -> str:
    gold_set = set(gold_article_ids)
    if not gold_set:
        return CASE_INVALID
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    if gold_set.issubset(set(top10_article_ids)):
        return case_a
    if gold_set.issubset(set(top_cutoff_article_ids)):
        return case_b
    return case_c


def summarize_error_traces(
    traces: list[dict[str, Any]],
    *,
    top_k_chunks: int,
    source_rerank_run: str,
) -> dict[str, Any]:
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    valid_traces = [trace for trace in traces if trace["case_type"] != CASE_INVALID]
    groups = {
        "all": valid_traces,
        "single": [trace for trace in valid_traces if not trace["is_multi_article"]],
        "multi": [trace for trace in valid_traces if trace["is_multi_article"]],
    }
    summary: dict[str, Any] = {
        "source_rerank_run": source_rerank_run,
        "top_k_chunks": top_k_chunks,
        "records": len(traces),
        "valid_records": len(valid_traces),
        "invalid_no_gold_articles": len(traces) - len(valid_traces),
        "case_labels": {"A": case_a, "B": case_b, "C": case_c},
        "case_descriptions": {
            "A": "top10 chunks already cover all gold articles",
            "B": f"top{top_k_chunks} chunks cover all gold articles but top10 chunks do not; keep for diagnosis without a dedicated selection stage",
            "C": f"top{top_k_chunks} chunks still miss at least one gold article; target second-hop retrieval",
        },
        "group_metrics": {
            group_name: group_summary(rows, top_k_chunks=top_k_chunks)
            for group_name, rows in groups.items()
        },
        "dataset_metrics": dataset_group_metrics(valid_traces, top_k_chunks=top_k_chunks),
    }

    for group_name, rows in groups.items():
        group_metrics = summary["group_metrics"][group_name]
        summary[f"{group_name}_records"] = len(rows)
        for key in (
            "article_full_hit@10_chunks",
            "article_recall@10_chunks",
            f"article_full_hit@{top_k_chunks}_chunks",
            f"article_recall@{top_k_chunks}_chunks",
        ):
            summary[f"{group_name}_{key}"] = group_metrics[key]

    for label, case_name in (("A", case_a), ("B", case_b), ("C", case_c)):
        count = sum(trace["case_type"] == case_name for trace in valid_traces)
        summary[f"case_{label}_count"] = count
        summary[f"case_{label}_ratio"] = safe_div(count, len(valid_traces))

    case_b_rows = [trace for trace in valid_traces if trace["case_type"] == case_b]
    case_c_rows = [trace for trace in valid_traces if trace["case_type"] == case_c]
    multi_case_b_count = sum(trace["is_multi_article"] for trace in case_b_rows)
    multi_case_c_count = sum(trace["is_multi_article"] for trace in case_c_rows)
    summary.update(
        {
            "multi_case_B_count": multi_case_b_count,
            "multi_case_B_ratio_within_case_B": safe_div(multi_case_b_count, len(case_b_rows)),
            "multi_case_C_count": multi_case_c_count,
            "multi_case_C_ratio_within_case_C": safe_div(multi_case_c_count, len(case_c_rows)),
            "avg_unique_articles@10_chunks": average(
                trace["unique_articles@10_chunks"] for trace in valid_traces
            ),
            "avg_duplicate_article_ratio@10_chunks": average(
                trace["duplicate_article_ratio@10_chunks"] for trace in valid_traces
            ),
            f"avg_unique_articles@{top_k_chunks}_chunks": average(
                trace[f"unique_articles@{top_k_chunks}_chunks"] for trace in valid_traces
            ),
            f"avg_duplicate_article_ratio@{top_k_chunks}_chunks": average(
                trace[f"duplicate_article_ratio@{top_k_chunks}_chunks"]
                for trace in valid_traces
            ),
            "multi_avg_missing_articles_at_10_chunks": average(
                len(trace["missing_articles_at_10_chunks"]) for trace in groups["multi"]
            ),
            f"multi_avg_missing_articles_at_{top_k_chunks}_chunks": average(
                len(trace[f"missing_articles_at_{top_k_chunks}_chunks"])
                for trace in groups["multi"]
            ),
            "rerank_top10_rescued_gold_articles": sum(
                len(trace.get("rerank_top10_rescued_gold_article_ids") or [])
                for trace in valid_traces
            ),
            "rerank_top10_dropped_gold_articles": sum(
                len(trace.get("rerank_top10_dropped_gold_article_ids") or [])
                for trace in valid_traces
            ),
        }
    )
    return summary


def group_summary(rows: list[dict[str, Any]], *, top_k_chunks: int) -> dict[str, Any]:
    return {
        "records": len(rows),
        "article_full_hit@10_chunks": average(
            row["article_full_hit@10_chunks"] for row in rows
        ),
        "article_recall@10_chunks": average(row["article_recall@10_chunks"] for row in rows),
        f"article_full_hit@{top_k_chunks}_chunks": average(
            row[f"article_full_hit@{top_k_chunks}_chunks"] for row in rows
        ),
        f"article_recall@{top_k_chunks}_chunks": average(
            row[f"article_recall@{top_k_chunks}_chunks"] for row in rows
        ),
        "avg_unique_articles@10_chunks": average(
            row["unique_articles@10_chunks"] for row in rows
        ),
        "avg_duplicate_article_ratio@10_chunks": average(
            row["duplicate_article_ratio@10_chunks"] for row in rows
        ),
        f"avg_unique_articles@{top_k_chunks}_chunks": average(
            row[f"unique_articles@{top_k_chunks}_chunks"] for row in rows
        ),
        f"avg_duplicate_article_ratio@{top_k_chunks}_chunks": average(
            row[f"duplicate_article_ratio@{top_k_chunks}_chunks"] for row in rows
        ),
    }


def dataset_group_metrics(
    rows: list[dict[str, Any]],
    *,
    top_k_chunks: int,
) -> dict[str, dict[str, Any]]:
    dataset_names = sorted({row["dataset_name"] for row in rows if row.get("dataset_name")})
    return {
        dataset_name: group_summary(
            [row for row in rows if row.get("dataset_name") == dataset_name],
            top_k_chunks=top_k_chunks,
        )
        for dataset_name in dataset_names
    }


def render_error_summary_markdown(summary: dict[str, Any]) -> str:
    top_k_chunks = summary["top_k_chunks"]
    case_a = summary["case_labels"]["A"]
    case_b = summary["case_labels"]["B"]
    case_c = summary["case_labels"]["C"]
    lines = [
        "# Error Analysis & Trace Logging",
        "",
        "## 基本信息",
        "",
    ]
    lines.extend(
        markdown_table(
            ["指标", "数值"],
            [
                ["source_rerank_run", summary["source_rerank_run"]],
                ["top_k_chunks", top_k_chunks],
                ["records", summary["records"]],
                ["valid_records", summary["valid_records"]],
                ["invalid_no_gold_articles", summary["invalid_no_gold_articles"]],
            ],
        )
    )
    lines.extend(["", "## Article Coverage", ""])
    lines.extend(
        markdown_table(
            [
                "group",
                "records",
                "article_full_hit@10_chunks",
                "article_recall@10_chunks",
                f"article_full_hit@{top_k_chunks}_chunks",
                f"article_recall@{top_k_chunks}_chunks",
            ],
            [
                [
                    group,
                    summary[f"{group}_records"],
                    f"{summary[f'{group}_article_full_hit@10_chunks']:.4f}",
                    f"{summary[f'{group}_article_recall@10_chunks']:.4f}",
                    f"{summary[f'{group}_article_full_hit@{top_k_chunks}_chunks']:.4f}",
                    f"{summary[f'{group}_article_recall@{top_k_chunks}_chunks']:.4f}",
                ]
                for group in ("all", "single", "multi")
            ],
        )
    )
    lines.extend(["", "## Case 分布", ""])
    lines.extend(
        markdown_table(
            ["case_type", "count", "ratio", "next_stage"],
            [
                [case_a, summary["case_A_count"], f"{summary['case_A_ratio']:.4f}", "done"],
                [
                    case_b,
                    summary["case_B_count"],
                    f"{summary['case_B_ratio']:.4f}",
                    "diagnosis only",
                ],
                [
                    case_c,
                    summary["case_C_count"],
                    f"{summary['case_C_ratio']:.4f}",
                    "Phase 7 second-hop retrieval",
                ],
            ],
        )
    )
    lines.extend(["", "## Diversity", ""])
    lines.extend(
        markdown_table(
            ["metric", "value"],
            [
                ["avg_unique_articles@10_chunks", f"{summary['avg_unique_articles@10_chunks']:.4f}"],
                [
                    "avg_duplicate_article_ratio@10_chunks",
                    f"{summary['avg_duplicate_article_ratio@10_chunks']:.4f}",
                ],
                [
                    f"avg_unique_articles@{top_k_chunks}_chunks",
                    f"{summary[f'avg_unique_articles@{top_k_chunks}_chunks']:.4f}",
                ],
                [
                    f"avg_duplicate_article_ratio@{top_k_chunks}_chunks",
                    f"{summary[f'avg_duplicate_article_ratio@{top_k_chunks}_chunks']:.4f}",
                ],
            ],
        )
    )
    lines.extend(["", "## Multi-article Focus", ""])
    lines.extend(
        markdown_table(
            ["metric", "value"],
            [
                ["multi_case_B_count", summary["multi_case_B_count"]],
                [
                    "multi_case_B_ratio_within_case_B",
                    f"{summary['multi_case_B_ratio_within_case_B']:.4f}",
                ],
                ["multi_case_C_count", summary["multi_case_C_count"]],
                [
                    "multi_case_C_ratio_within_case_C",
                    f"{summary['multi_case_C_ratio_within_case_C']:.4f}",
                ],
                [
                    "multi_avg_missing_articles_at_10_chunks",
                    f"{summary['multi_avg_missing_articles_at_10_chunks']:.4f}",
                ],
                [
                    f"multi_avg_missing_articles_at_{top_k_chunks}_chunks",
                    f"{summary[f'multi_avg_missing_articles_at_{top_k_chunks}_chunks']:.4f}",
                ],
            ],
        )
    )
    lines.extend(["", "## Reranker Diagnostics", ""])
    lines.extend(
        markdown_table(
            ["metric", "value"],
            [
                [
                    "rerank_top10_rescued_gold_articles",
                    summary["rerank_top10_rescued_gold_articles"],
                ],
                [
                    "rerank_top10_dropped_gold_articles",
                    summary["rerank_top10_dropped_gold_articles"],
                ],
            ],
        )
    )
    return "\n".join(lines)


def print_summary(console: Console, summary: dict[str, Any], output_dir: Path) -> None:
    top_k_chunks = summary["top_k_chunks"]
    console.print()
    console.print("[bold green]Error Analysis Finished[/bold green]")
    console.print()
    console.print(f"Source: {summary['source_rerank_run']}")
    console.print(f"Records: {summary['records']}")
    console.print(f"top_k_chunks: {top_k_chunks}")
    console.print()
    for key in (
        "all_article_full_hit@10_chunks",
        f"all_article_full_hit@{top_k_chunks}_chunks",
        "avg_duplicate_article_ratio@10_chunks",
    ):
        console.print(f"{key}: {summary.get(key, 0.0):.4f}")
    console.print()
    console.print(f"case_A_count: {summary['case_A_count']}")
    console.print(f"case_B_count: {summary['case_B_count']}")
    console.print(f"case_C_count: {summary['case_C_count']}")
    console.print()
    console.print(f"Results saved to {output_dir}/")


def average(values: Any) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(float(value) for value in value_list) / len(value_list)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
