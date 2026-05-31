from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.data.schema import KBArticle, KBChunk, QAExample
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.run_bm25_eval import (
    BM25EvalError,
    CASE_A,
    CASE_INVALID,
    VALID_QA_DATASETS,
    load_kb_articles,
    load_qa_examples,
    markdown_table,
    summarize_metrics,
)
from src.retrievers.chunk_bm25_retriever import ChunkBM25Retriever
from src.utils.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.utils.text_utils import preview_text


CHUNK_KS = [1, 3, 5, 10, 20, 30]
CASE_B_TOP30 = "B_top30_full_not_top10"
CASE_C_TOP30 = "C_top30_not_full"


def run_chunk_bm25_eval(
    *,
    processed_dir: str | Path,
    chunks_path: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    top_k_chunks: int = 100,
    top_k_articles: int = 30,
    console: Console | None = None,
) -> dict[str, Any]:
    if dataset_name not in VALID_QA_DATASETS:
        raise BM25EvalError(
            f"Unsupported dataset {dataset_name!r}. Choose one of: {sorted(VALID_QA_DATASETS)}"
        )
    if top_k_chunks <= 0:
        raise BM25EvalError("--top_k_chunks must be a positive integer.")
    if top_k_articles <= 0:
        raise BM25EvalError("--top_k_articles must be a positive integer.")
    if top_k_articles > top_k_chunks:
        raise BM25EvalError("--top_k_articles must be smaller than or equal to --top_k_chunks.")

    console = console or Console()
    processed_dir = Path(processed_dir)
    chunks_path = Path(chunks_path)
    output_dir = ensure_dir(output_dir)

    kb_articles = load_kb_articles(processed_dir)
    kb_chunks = load_kb_chunks(chunks_path)
    qa_examples = load_qa_examples(processed_dir, dataset_name)
    retriever = ChunkBM25Retriever(kb_chunks)
    article_lookup = {article.article_id: article for article in kb_articles}
    ks = [k for k in CHUNK_KS if k <= top_k_articles]

    traces = []
    metric_rows = []
    case_rows: dict[str, list[dict[str, Any]]] = {
        CASE_A: [],
        CASE_B_TOP30: [],
        CASE_C_TOP30: [],
    }

    for example in tqdm(qa_examples, desc=f"Chunk BM25 eval {dataset_name}"):
        chunk_results = retriever.search(example.question, top_k_chunks=top_k_chunks)
        article_results = aggregate_chunk_results(chunk_results, top_k_articles=top_k_articles)
        retrieved_ids = [result["article_id"] for result in article_results]
        metrics = compute_retrieval_metrics(example.article_ids, retrieved_ids, ks)
        trace = build_chunk_trace(
            example,
            chunk_results,
            article_results,
            metrics,
            article_lookup,
            top_k_articles=top_k_articles,
        )
        traces.append(trace)

        if metrics["is_valid"]:
            metric_rows.append(trace)
            if trace["case_type"] in case_rows:
                case_rows[trace["case_type"]].append(trace)

    summary = summarize_metrics(
        dataset_name=dataset_name,
        records=len(qa_examples),
        valid_rows=metric_rows,
        invalid_rows=[trace for trace in traces if trace["case_type"] == CASE_INVALID],
        ks=ks,
    )
    summary["case_A_count"] = sum(row["case_type"] == CASE_A for row in metric_rows)
    summary["case_B_count"] = sum(row["case_type"] == CASE_B_TOP30 for row in metric_rows)
    summary["case_C_count"] = sum(row["case_type"] == CASE_C_TOP30 for row in metric_rows)
    summary.update(
        {
            "retrieval_unit": "chunk",
            "chunk_records": len(kb_chunks),
            "chunk_article_records": len({chunk.article_id for chunk in kb_chunks}),
            "top_k_chunks": top_k_chunks,
            "top_k_articles": top_k_articles,
        }
    )

    write_json(output_dir / f"{dataset_name}_metrics.json", summary)
    (output_dir / f"{dataset_name}_metrics.md").write_text(
        render_chunk_metrics_markdown(summary, traces),
        encoding="utf-8",
    )
    write_jsonl(output_dir / f"{dataset_name}_retrieval_traces.jsonl", traces)
    write_jsonl(output_dir / f"{dataset_name}_cases_A_top10_full.jsonl", case_rows[CASE_A])
    write_jsonl(
        output_dir / f"{dataset_name}_cases_B_top30_full_not_top10.jsonl",
        case_rows[CASE_B_TOP30],
    )
    write_jsonl(output_dir / f"{dataset_name}_cases_C_top30_not_full.jsonl", case_rows[CASE_C_TOP30])

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


def aggregate_chunk_results(
    chunk_results: list[dict[str, Any]],
    *,
    top_k_articles: int,
) -> list[dict[str, Any]]:
    best_by_article: dict[str, dict[str, Any]] = {}
    for result in chunk_results:
        article_id = result.get("article_id")
        if not article_id:
            continue
        existing = best_by_article.get(article_id)
        if existing is None or is_better_chunk_result(result, existing):
            best_by_article[article_id] = result

    ranked = sorted(
        best_by_article.values(),
        key=lambda result: (-float(result["score"]), int(result["rank"])),
    )[:top_k_articles]

    article_results = []
    for rank, result in enumerate(ranked, start=1):
        article_results.append(
            {
                "rank": rank,
                "article_id": result["article_id"],
                "title": result.get("title"),
                "url": result.get("url"),
                "score": result["score"],
                "best_chunk_id": result["chunk_id"],
                "best_chunk_index": result["chunk_index"],
                "best_chunk_rank": result["rank"],
                "contents_preview": result.get("contents_preview"),
            }
        )
    return article_results


def is_better_chunk_result(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_score = float(candidate.get("score", 0.0))
    current_score = float(current.get("score", 0.0))
    if candidate_score != current_score:
        return candidate_score > current_score
    return int(candidate.get("rank", 0)) < int(current.get("rank", 0))


def build_chunk_trace(
    example: QAExample,
    chunk_results: list[dict[str, Any]],
    article_results: list[dict[str, Any]],
    metrics: dict[str, Any],
    article_lookup: dict[str, KBArticle],
    top_k_articles: int,
) -> dict[str, Any]:
    retrieved_ids = [result["article_id"] for result in article_results]
    retrieved_titles = [result.get("title") for result in article_results]
    gold_set = set(example.article_ids)

    trace: dict[str, Any] = {
        "qid": example.qid,
        "dataset_name": example.dataset_name,
        "question": example.question,
        "answer": example.answer,
        "gold_article_ids": example.article_ids,
        "gold_article_titles": [
            article_lookup[article_id].title
            for article_id in example.article_ids
            if article_id in article_lookup
        ],
        "num_gold_articles": example.num_gold_articles,
        "is_multi_article": example.is_multi_article,
        "top5_article_ids": retrieved_ids[:5],
        "top10_article_ids": retrieved_ids[:10],
        "top20_article_ids": retrieved_ids[:20],
        f"top{top_k_articles}_article_ids": retrieved_ids[:top_k_articles],
        "top10_chunk_ids": [result["chunk_id"] for result in chunk_results[:10]],
        "top10_chunk_article_ids": [result["article_id"] for result in chunk_results[:10]],
        "top10_chunk_scores": [result["score"] for result in chunk_results[:10]],
        "top10_titles": retrieved_titles[:10],
        f"top{top_k_articles}_titles": retrieved_titles[:top_k_articles],
        "top10_best_chunk_ids": [result["best_chunk_id"] for result in article_results[:10]],
        "top10_best_chunk_indexes": [
            result["best_chunk_index"] for result in article_results[:10]
        ],
        "missing_articles_at_10": sorted(gold_set - set(retrieved_ids[:10])),
        f"missing_articles_at_{top_k_articles}": sorted(
            gold_set - set(retrieved_ids[:top_k_articles])
        ),
    }
    trace.update({key: value for key, value in metrics.items() if key != "is_valid"})
    trace["case_type"] = classify_chunk_case(example.article_ids, retrieved_ids, top_k_articles)
    return trace


def classify_chunk_case(
    gold_article_ids: list[str],
    retrieved_article_ids: list[str],
    top_k_articles: int,
) -> str:
    gold_set = {article_id for article_id in gold_article_ids if article_id}
    if not gold_set:
        return CASE_INVALID
    top10 = set(retrieved_article_ids[:10])
    top_cutoff = set(retrieved_article_ids[:top_k_articles])
    if gold_set.issubset(top10):
        return CASE_A
    if gold_set.issubset(top_cutoff):
        return CASE_B_TOP30
    return CASE_C_TOP30


def render_chunk_metrics_markdown(summary: dict[str, Any], traces: list[dict[str, Any]]) -> str:
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
                ["top_k_chunks", summary["top_k_chunks"]],
                ["top_k_articles", summary["top_k_articles"]],
            ],
        )
    )
    lines.extend(["", "## 整体指标", ""])
    rows = []
    for k in summary["ks"]:
        rows.append(
            [
                k,
                f"{summary[f'article_hit@{k}']:.4f}",
                f"{summary[f'article_full_hit@{k}']:.4f}",
                f"{summary[f'article_recall@{k}']:.4f}",
                f"{summary[f'article_precision@{k}']:.4f}",
            ]
        )
    lines.extend(
        markdown_table(
            ["k", "article_hit", "article_full_hit", "article_recall", "article_precision"],
            rows,
        )
    )
    lines.extend(["", f"MRR: `{summary['mrr']:.4f}`", ""])

    lines.extend(["## 单文章 vs 多文章", ""])
    lines.extend(
        markdown_table(
            ["分组", "样本数", "article_full_hit@10", "article_recall@10"],
            [
                [
                    "single",
                    summary["single_article_records"],
                    f"{summary['single_article_article_full_hit@10']:.4f}",
                    f"{summary['single_article_article_recall@10']:.4f}",
                ],
                [
                    "multi",
                    summary["multi_article_records"],
                    f"{summary['multi_article_article_full_hit@10']:.4f}",
                    f"{summary['multi_article_article_recall@10']:.4f}",
                ],
            ],
        )
    )

    lines.extend(["", "## Case 分布", ""])
    lines.extend(
        markdown_table(
            ["case_type", "数量"],
            [
                [CASE_A, summary["case_A_count"]],
                [CASE_B_TOP30, summary["case_B_count"]],
                [CASE_C_TOP30, summary["case_C_count"]],
            ],
        )
    )

    failed = [trace for trace in traces if trace["case_type"] == CASE_C_TOP30][:5]
    lines.extend(["", "## Top 5 失败样例", ""])
    if not failed:
        lines.append("没有 top30 未完整命中的失败样例。")
    for trace in failed:
        lines.extend(
            [
                f"### {trace['qid']}",
                "",
                f"- 问题：{preview_text(trace['question'], 500)}",
                f"- gold article_ids：`{trace['gold_article_ids']}`",
                f"- top10 article_ids：`{trace['top10_article_ids']}`",
                f"- top10 chunk article_ids：`{trace['top10_chunk_article_ids']}`",
                f"- missing_articles_at_10：`{trace['missing_articles_at_10']}`",
                "",
            ]
        )
    return "\n".join(lines)


def print_summary(console: Console, summary: dict[str, Any], output_dir: Path) -> None:
    console.print()
    console.print("[bold green]Chunk BM25 Baseline Finished[/bold green]")
    console.print()
    console.print(f"Dataset: {summary['dataset_name']}")
    console.print(f"Records: {summary['records']}")
    console.print(f"Valid records: {summary['valid_records']}")
    console.print(f"Chunks: {summary['chunk_records']}")
    console.print()
    for key in ("article_hit@10", "article_full_hit@10", "article_recall@10", "mrr"):
        console.print(f"{key}: {summary.get(key, 0.0):.4f}")
    console.print()
    console.print(f"A_top10_full: {summary['case_A_count']}")
    console.print(f"{CASE_B_TOP30}: {summary['case_B_count']}")
    console.print(f"{CASE_C_TOP30}: {summary['case_C_count']}")
    console.print()
    console.print(f"Results saved to {output_dir}/")
