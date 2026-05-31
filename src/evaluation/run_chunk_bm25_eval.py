from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.data.schema import KBChunk
from src.evaluation.chunk_eval import (
    build_chunk_trace,
    chunk_case_labels,
    compute_chunk_retrieval_metrics,
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
from src.retrievers.chunk_bm25_retriever import ChunkBM25Retriever
from src.utils.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.utils.text_utils import preview_text


def run_chunk_bm25_eval(
    *,
    processed_dir: str | Path,
    chunks_path: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    top_k_chunks: int = 100,
    console: Console | None = None,
) -> dict[str, Any]:
    if dataset_name not in VALID_QA_DATASETS:
        raise BM25EvalError(
            f"Unsupported dataset {dataset_name!r}. Choose one of: {sorted(VALID_QA_DATASETS)}"
        )
    if top_k_chunks <= 0:
        raise BM25EvalError("--top_k_chunks must be a positive integer.")

    console = console or Console()
    processed_dir = Path(processed_dir)
    output_dir = ensure_dir(output_dir)

    kb_articles = load_kb_articles(processed_dir)
    kb_chunks = load_kb_chunks(Path(chunks_path))
    qa_examples = load_qa_examples(processed_dir, dataset_name)
    retriever = ChunkBM25Retriever(kb_chunks)
    article_lookup = {article.article_id: article for article in kb_articles}
    ks = select_chunk_ks(top_k_chunks)
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    case_rows: dict[str, list[dict[str, Any]]] = {case_a: [], case_b: [], case_c: []}
    traces = []
    metric_rows = []

    for example in tqdm(qa_examples, desc=f"Chunk BM25 eval {dataset_name}"):
        chunk_results = retriever.search(example.question, top_k_chunks=top_k_chunks)
        metrics = compute_chunk_retrieval_metrics(example.article_ids, chunk_results, ks)
        trace = build_chunk_trace(
            example,
            chunk_results,
            metrics,
            article_lookup,
            top_k_chunks=top_k_chunks,
        )
        traces.append(trace)
        if metrics["is_valid"]:
            metric_rows.append(trace)
            case_rows[trace["case_type"]].append(trace)

    summary = summarize_chunk_metrics(
        dataset_name=dataset_name,
        records=len(qa_examples),
        valid_rows=metric_rows,
        invalid_rows=[trace for trace in traces if trace["case_type"] == CASE_INVALID],
        ks=ks,
        top_k_chunks=top_k_chunks,
    )
    summary.update(
        {
            "retrieval_unit": "chunk",
            "chunk_records": len(kb_chunks),
            "chunk_article_records": len({chunk.article_id for chunk in kb_chunks}),
            "top_k_chunks": top_k_chunks,
            f"avg_unique_articles_from_top{top_k_chunks}_chunks": summary[
                f"unique_articles@{top_k_chunks}_chunks"
            ],
        }
    )

    write_json(output_dir / f"{dataset_name}_metrics.json", summary)
    (output_dir / f"{dataset_name}_metrics.md").write_text(
        render_chunk_metrics_markdown(summary, traces),
        encoding="utf-8",
    )
    write_jsonl(output_dir / f"{dataset_name}_retrieval_traces.jsonl", traces)
    write_jsonl(output_dir / f"{dataset_name}_cases_{case_a}.jsonl", case_rows[case_a])
    write_jsonl(output_dir / f"{dataset_name}_cases_{case_b}.jsonl", case_rows[case_b])
    write_jsonl(output_dir / f"{dataset_name}_cases_{case_c}.jsonl", case_rows[case_c])

    print_summary(console, summary, output_dir)
    return summary


def load_kb_chunks(chunks_path: Path) -> list[KBChunk]:
    if not chunks_path.exists():
        raise BM25EvalError(
            f"Missing chunk file: {chunks_path}. Please run `python scripts/prepare_chunks.py` first."
        )
    chunks = [KBChunk(**row) for row in read_jsonl(chunks_path) if (row.get("text") or "").strip()]
    if not chunks:
        raise BM25EvalError(f"No usable KB chunks found in {chunks_path}.")
    return chunks


def render_chunk_metrics_markdown(summary: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    top_k_chunks = summary["top_k_chunks"]
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    lines = [
        f"# BM25 Chunk-level Baseline：{summary['dataset_name']}",
        "",
        "## 基本信息",
        "",
    ]
    lines.extend(
        markdown_table(
            ["指标", "数值"],
            [
                ["records", summary["records"]],
                ["valid_records", summary["valid_records"]],
                ["invalid_no_gold_articles", summary["invalid_no_gold_articles"]],
                ["chunk_records", summary["chunk_records"]],
                ["chunk_article_records", summary["chunk_article_records"]],
                ["top_k_chunks", top_k_chunks],
                [
                    f"avg_unique_articles_from_top{top_k_chunks}_chunks",
                    f"{summary[f'avg_unique_articles_from_top{top_k_chunks}_chunks']:.2f}",
                ],
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
    lines.extend(render_group_metrics(summary))
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


def metric_rows(summary: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            k,
            f"{summary[f'chunk_hit@{k}']:.4f}",
            f"{summary[f'chunk_full_article_hit@{k}']:.4f}",
            f"{summary[f'chunk_article_recall@{k}']:.4f}",
            f"{summary[f'chunk_gold_rate@{k}']:.4f}",
            f"{summary[f'unique_articles@{k}_chunks']:.2f}",
            f"{summary[f'duplicate_article_ratio@{k}_chunks']:.4f}",
        ]
        for k in summary["ks"]
    ]


def render_group_metrics(summary: dict[str, Any]) -> list[str]:
    top_k_chunks = summary["top_k_chunks"]
    return [
        "## 单文章 vs 多文章",
        "",
        *markdown_table(
            [
                "分组",
                "样本数",
                "chunk_full_article_hit@10",
                "chunk_article_recall@10",
                f"chunk_full_article_hit@{top_k_chunks}",
                f"chunk_article_recall@{top_k_chunks}",
            ],
            [
                [
                    group,
                    summary[f"{group}_article_records"],
                    f"{summary[f'{group}_chunk_full_article_hit@10']:.4f}",
                    f"{summary[f'{group}_chunk_article_recall@10']:.4f}",
                    f"{summary[f'{group}_chunk_full_article_hit@{top_k_chunks}']:.4f}",
                    f"{summary[f'{group}_chunk_article_recall@{top_k_chunks}']:.4f}",
                ]
                for group in ("single", "multi")
            ],
        ),
    ]


def render_failed_examples(
    traces: list[dict[str, Any]],
    case_c: str,
    top_k_chunks: int,
) -> list[str]:
    lines = ["", "## Top 5 失败样例", ""]
    failed = [trace for trace in traces if trace["case_type"] == case_c][:5]
    if not failed:
        return [*lines, f"没有 top{top_k_chunks} chunks 未完整命中的失败样例。"]
    for trace in failed:
        lines.extend(
            [
                f"### {trace['qid']}",
                "",
                f"- 问题：{preview_text(trace['question'], 500)}",
                f"- gold article_ids：`{trace['gold_article_ids']}`",
                f"- top10 chunk article_ids：`{trace['top10_chunk_article_ids']}`",
                f"- missing_articles_at_10_chunks：`{trace['missing_articles_at_10_chunks']}`",
                (
                    f"- missing_articles_at_{top_k_chunks}_chunks："
                    f"`{trace[f'missing_articles_at_{top_k_chunks}_chunks']}`"
                ),
                "",
            ]
        )
    return lines


def print_summary(console: Console, summary: dict[str, Any], output_dir: Path) -> None:
    top_k_chunks = summary["top_k_chunks"]
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    console.print()
    console.print("[bold green]Chunk BM25 Baseline Finished[/bold green]")
    console.print()
    console.print(f"Dataset: {summary['dataset_name']}")
    console.print(f"Records: {summary['records']}")
    console.print(f"Chunks: {summary['chunk_records']}")
    console.print(f"top_k_chunks: {top_k_chunks}")
    console.print()
    for key in ("chunk_hit@10", "chunk_full_article_hit@10", "chunk_article_recall@10", "mrr"):
        console.print(f"{key}: {summary.get(key, 0.0):.4f}")
    console.print(
        f"chunk_full_article_hit@{top_k_chunks}: "
        f"{summary.get(f'chunk_full_article_hit@{top_k_chunks}', 0.0):.4f}"
    )
    console.print()
    console.print(f"{case_a}: {summary['case_A_count']}")
    console.print(f"{case_b}: {summary['case_B_count']}")
    console.print(f"{case_c}: {summary['case_C_count']}")
    console.print()
    console.print(f"Results saved to {output_dir}/")
