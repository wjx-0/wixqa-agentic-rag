from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.data.schema import KBArticle, QAExample
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
from src.retrievers.dense_faiss_retriever import (
    DEFAULT_DENSE_MODEL_NAME,
    DenseFaissRetriever,
    DenseFaissRetrieverError,
    aggregate_chunk_results,
)
from src.retrievers.faiss_store import FaissStoreError
from src.utils.io_utils import ensure_dir, read_json, write_json, write_jsonl
from src.utils.text_utils import preview_text


DENSE_KS = [1, 3, 5, 10, 20, 30]
CASE_B_TOP30 = "B_top30_full_not_top10"
CASE_C_TOP30 = "C_top30_not_full"


class DenseFaissEvalError(RuntimeError):
    pass


def run_dense_faiss_eval(
    *,
    processed_dir: str | Path,
    index_dir: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    model_name: str = DEFAULT_DENSE_MODEL_NAME,
    local_files_only: bool = True,
    top_k_chunks: int = 100,
    top_k_articles: int = 30,
    device: str | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    if dataset_name not in VALID_QA_DATASETS:
        raise DenseFaissEvalError(
            f"Unsupported dataset {dataset_name!r}. Choose one of: {sorted(VALID_QA_DATASETS)}"
        )
    if top_k_chunks <= 0:
        raise DenseFaissEvalError("--top_k_chunks must be a positive integer.")
    if top_k_articles != 30:
        raise DenseFaissEvalError("--top_k_articles must be 30 for the Dense FAISS top30 case split.")
    if top_k_articles > top_k_chunks:
        raise DenseFaissEvalError("--top_k_articles must be smaller than or equal to --top_k_chunks.")

    console = console or Console()
    processed_dir = Path(processed_dir)
    index_dir = Path(index_dir)
    output_dir = ensure_dir(output_dir)

    try:
        index_config = load_index_config(index_dir)
        retriever = DenseFaissRetriever(
            index_dir=index_dir,
            model_name=model_name,
            local_files_only=local_files_only,
            device=device,
        )
    except (DenseFaissRetrieverError, FaissStoreError) as exc:
        raise DenseFaissEvalError(str(exc)) from exc

    kb_articles = load_kb_articles(processed_dir)
    qa_examples = load_qa_examples(processed_dir, dataset_name)
    article_lookup = {article.article_id: article for article in kb_articles}
    ks = [k for k in DENSE_KS if k <= top_k_articles]

    traces = []
    metric_rows = []
    case_rows: dict[str, list[dict[str, Any]]] = {
        CASE_A: [],
        CASE_B_TOP30: [],
        CASE_C_TOP30: [],
    }

    for example in tqdm(qa_examples, desc=f"Dense FAISS eval {dataset_name}"):
        chunk_results = retriever.search_chunks(example.question, top_k_chunks=top_k_chunks)
        article_results = aggregate_chunk_results(chunk_results, top_k_articles=top_k_articles)
        retrieved_ids = [result["article_id"] for result in article_results]
        metrics = compute_retrieval_metrics(example.article_ids, retrieved_ids, ks)
        trace = build_dense_trace(
            example,
            chunk_results,
            article_results,
            metrics,
            article_lookup,
            top_k_chunks=top_k_chunks,
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
            "retriever_type": "Dense FAISS Retrieval",
            "retrieval_unit": "chunk",
            "model_name": model_name,
            "tokenizer_name": index_config.get("tokenizer_name"),
            "faiss_index_type": index_config.get("faiss_index_type", "IndexFlatIP"),
            "normalize_embeddings": bool(index_config.get("normalize_embeddings", True)),
            "chunk_size_tokens": index_config.get("chunk_size_tokens"),
            "chunk_overlap_tokens": index_config.get("chunk_overlap_tokens"),
            "chunk_records": retriever.store.ntotal,
            "chunk_article_records": len(
                {row.get("article_id") for row in retriever.store.metadata if row.get("article_id")}
            ),
            "top_k_chunks": top_k_chunks,
            "top_k_articles": top_k_articles,
            "avg_unique_articles_from_top100_chunks": average(
                row["unique_articles_from_top100_chunks"] for row in metric_rows
            ),
            "unique_articles_less_than_30_count": sum(
                bool(row["unique_articles_warning"]) for row in metric_rows
            ),
        }
    )

    prefix = f"dense_{model_slug(model_name)}_{dataset_name}"
    write_json(output_dir / f"{prefix}_metrics.json", summary)
    (output_dir / f"{prefix}_metrics.md").write_text(
        render_dense_metrics_markdown(summary, traces),
        encoding="utf-8",
    )
    write_jsonl(output_dir / f"{prefix}_retrieval_traces.jsonl", traces)
    write_jsonl(output_dir / f"{prefix}_cases_A_top10_full.jsonl", case_rows[CASE_A])
    write_jsonl(
        output_dir / f"{prefix}_cases_B_top30_full_not_top10.jsonl",
        case_rows[CASE_B_TOP30],
    )
    write_jsonl(output_dir / f"{prefix}_cases_C_top30_not_full.jsonl", case_rows[CASE_C_TOP30])

    print_summary(console, summary, output_dir)
    return summary


def load_index_config(index_dir: Path) -> dict[str, Any]:
    config_path = index_dir / "index_config.json"
    if not config_path.exists():
        raise DenseFaissEvalError(
            f"Missing FAISS index config: {config_path}. "
            "Please run `python scripts/build_faiss_index.py` first."
        )
    return read_json(config_path)


def build_dense_trace(
    example: QAExample,
    chunk_results: list[dict[str, Any]],
    article_results: list[dict[str, Any]],
    metrics: dict[str, Any],
    article_lookup: dict[str, KBArticle],
    *,
    top_k_chunks: int,
    top_k_articles: int,
) -> dict[str, Any]:
    retrieved_ids = [result["article_id"] for result in article_results]
    retrieved_titles = [result.get("title") for result in article_results]
    gold_set = set(example.article_ids)
    unique_articles = len({result["article_id"] for result in chunk_results if result.get("article_id")})

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
        "top30_article_ids": retrieved_ids[:30],
        "top10_titles": retrieved_titles[:10],
        "top30_titles": retrieved_titles[:30],
        "top10_chunk_ids": [result["chunk_id"] for result in chunk_results[:10]],
        "top10_chunk_article_ids": [result["article_id"] for result in chunk_results[:10]],
        "top10_chunk_scores": [result["score"] for result in chunk_results[:10]],
        "top_chunks_preview": [
            {
                "rank": result["rank"],
                "chunk_id": result["chunk_id"],
                "article_id": result["article_id"],
                "title": result.get("title"),
                "score": result["score"],
                "text_preview": result.get("text_preview"),
            }
            for result in chunk_results[:10]
        ],
        "top10_best_chunk_ids": [result["best_chunk_id"] for result in article_results[:10]],
        "top10_best_chunk_indexes": [
            result["best_chunk_index"] for result in article_results[:10]
        ],
        "unique_articles_from_top100_chunks": unique_articles,
        "unique_articles_warning": unique_articles < top_k_articles,
        "requested_top_k_chunks": top_k_chunks,
        "requested_top_k_articles": top_k_articles,
        "missing_articles_at_10": sorted(gold_set - set(retrieved_ids[:10])),
        "missing_articles_at_30": sorted(gold_set - set(retrieved_ids[:30])),
    }
    trace.update({key: value for key, value in metrics.items() if key != "is_valid"})
    trace["case_type"] = classify_dense_case(example.article_ids, retrieved_ids)
    return trace


def classify_dense_case(gold_article_ids: list[str], retrieved_article_ids: list[str]) -> str:
    gold_set = {article_id for article_id in gold_article_ids if article_id}
    if not gold_set:
        return CASE_INVALID
    top10 = set(retrieved_article_ids[:10])
    top30 = set(retrieved_article_ids[:30])
    if gold_set.issubset(top10):
        return CASE_A
    if gold_set.issubset(top30):
        return CASE_B_TOP30
    return CASE_C_TOP30


def render_dense_metrics_markdown(summary: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    lines = [
        f"# Dense FAISS Baseline：{summary['dataset_name']}",
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
                ["normalize_embeddings", summary["normalize_embeddings"]],
                ["records", summary["records"]],
                ["valid_records", summary["valid_records"]],
                ["invalid_no_gold_articles", summary["invalid_no_gold_articles"]],
                ["chunk_records", summary["chunk_records"]],
                ["chunk_article_records", summary["chunk_article_records"]],
                ["chunk_size_tokens", summary["chunk_size_tokens"]],
                ["chunk_overlap_tokens", summary["chunk_overlap_tokens"]],
                ["top_k_chunks", summary["top_k_chunks"]],
                ["top_k_articles", summary["top_k_articles"]],
                [
                    "avg_unique_articles_from_top100_chunks",
                    f"{summary['avg_unique_articles_from_top100_chunks']:.2f}",
                ],
                [
                    "unique_articles_less_than_30_count",
                    summary["unique_articles_less_than_30_count"],
                ],
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
                f"- missing_articles_at_10：`{trace['missing_articles_at_10']}`",
                "",
            ]
        )
    return "\n".join(lines)


def print_summary(console: Console, summary: dict[str, Any], output_dir: Path) -> None:
    console.print()
    console.print("[bold green]Dense FAISS Baseline Finished[/bold green]")
    console.print()
    console.print(f"Dataset: {summary['dataset_name']}")
    console.print(f"Model: {summary['model_name']}")
    console.print(f"Index: FAISS {summary['faiss_index_type']}")
    console.print(f"Chunks: {summary['chunk_records']}")
    console.print()
    console.print(f"top_k_chunks: {summary['top_k_chunks']}")
    console.print(f"top_k_articles: {summary['top_k_articles']}")
    console.print(
        "avg_unique_articles_from_top100_chunks: "
        f"{summary['avg_unique_articles_from_top100_chunks']:.2f}"
    )
    console.print(
        "unique_articles_less_than_30_count: "
        f"{summary['unique_articles_less_than_30_count']}"
    )
    console.print()
    for key in ("article_hit@10", "article_full_hit@10", "article_recall@10", "mrr"):
        console.print(f"{key}: {summary.get(key, 0.0):.4f}")
    console.print()
    console.print(f"A_top10_full: {summary['case_A_count']}")
    console.print(f"{CASE_B_TOP30}: {summary['case_B_count']}")
    console.print(f"{CASE_C_TOP30}: {summary['case_C_count']}")
    console.print()
    console.print(f"Results saved to {output_dir}/")


def average(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def model_slug(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1].replace(" ", "_")
