from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.data.schema import KBArticle, QAExample
from src.evaluation.retrieval_metrics import DEFAULT_KS, compute_retrieval_metrics
from src.retrievers.bm25_retriever import BM25Retriever
from src.utils.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.utils.text_utils import preview_text


VALID_QA_DATASETS = {"wixqa_expertwritten", "wixqa_simulated", "wixqa_synthetic"}
CASE_A = "A_top10_full"
CASE_B = "B_top50_full_not_top10"
CASE_C = "C_top50_not_full"
CASE_INVALID = "INVALID_NO_GOLD_ARTICLES"


class BM25EvalError(RuntimeError):
    pass


def run_bm25_eval(
    *,
    processed_dir: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    top_k: int = 50,
    console: Console | None = None,
) -> dict[str, Any]:
    if dataset_name not in VALID_QA_DATASETS:
        raise BM25EvalError(
            f"Unsupported dataset {dataset_name!r}. Choose one of: {sorted(VALID_QA_DATASETS)}"
        )
    if top_k <= 0:
        raise BM25EvalError("--top_k must be a positive integer.")

    console = console or Console()
    processed_dir = Path(processed_dir)
    output_dir = ensure_dir(output_dir)

    kb_articles = load_kb_articles(processed_dir)
    qa_examples = load_qa_examples(processed_dir, dataset_name)
    retriever = BM25Retriever(kb_articles)
    article_lookup = {article.article_id: article for article in kb_articles}
    ks = [k for k in DEFAULT_KS if k <= top_k]

    traces = []
    metric_rows = []
    case_rows: dict[str, list[dict[str, Any]]] = {CASE_A: [], CASE_B: [], CASE_C: []}

    for example in tqdm(qa_examples, desc=f"BM25 eval {dataset_name}"):
        results = retriever.search(example.question, top_k=top_k)
        retrieved_ids = [result["article_id"] for result in results]
        metrics = compute_retrieval_metrics(example.article_ids, retrieved_ids, ks)
        trace = build_trace(example, results, metrics, article_lookup, top_k)
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

    prefix = output_dir / dataset_name
    write_json(prefix.with_name(f"{dataset_name}_metrics.json"), summary)
    (prefix.with_name(f"{dataset_name}_metrics.md")).write_text(
        render_metrics_markdown(summary, traces),
        encoding="utf-8",
    )
    write_jsonl(prefix.with_name(f"{dataset_name}_retrieval_traces.jsonl"), traces)
    write_jsonl(prefix.with_name(f"{dataset_name}_cases_A_top10_full.jsonl"), case_rows[CASE_A])
    write_jsonl(
        prefix.with_name(f"{dataset_name}_cases_B_top50_full_not_top10.jsonl"),
        case_rows[CASE_B],
    )
    write_jsonl(prefix.with_name(f"{dataset_name}_cases_C_top50_not_full.jsonl"), case_rows[CASE_C])

    print_summary(console, summary, output_dir)
    return summary


def load_kb_articles(processed_dir: Path) -> list[KBArticle]:
    path = processed_dir / "wix_kb_corpus.jsonl"
    require_file(path)
    articles = []
    for row in read_jsonl(path):
        if not (row.get("contents") or "").strip():
            continue
        articles.append(KBArticle(**row))
    if not articles:
        raise BM25EvalError(f"No usable KB articles found in {path}.")
    return articles


def load_qa_examples(processed_dir: Path, dataset_name: str) -> list[QAExample]:
    path = processed_dir / f"{dataset_name}.jsonl"
    require_file(path)
    return [QAExample(**row) for row in read_jsonl(path)]


def require_file(path: Path) -> None:
    if not path.exists():
        raise BM25EvalError(
            f"Missing processed file: {path}. Please run `python scripts/prepare_wixqa.py` first."
        )


def build_trace(
    example: QAExample,
    results: list[dict],
    metrics: dict[str, Any],
    article_lookup: dict[str, KBArticle],
    top_k: int,
) -> dict[str, Any]:
    retrieved_ids = [result["article_id"] for result in results]
    retrieved_titles = [result.get("title") for result in results]
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
        "top50_article_ids": retrieved_ids[:50],
        "top10_titles": retrieved_titles[:10],
        "top50_titles": retrieved_titles[:50],
        "missing_articles_at_10": sorted(gold_set - set(retrieved_ids[:10])),
        "missing_articles_at_50": sorted(gold_set - set(retrieved_ids[:50])),
    }
    trace.update({key: value for key, value in metrics.items() if key != "is_valid"})
    trace["case_type"] = classify_case(example.article_ids, retrieved_ids)
    return trace


def classify_case(gold_article_ids: list[str], retrieved_article_ids: list[str]) -> str:
    gold_set = {article_id for article_id in gold_article_ids if article_id}
    if not gold_set:
        return CASE_INVALID
    top10 = set(retrieved_article_ids[:10])
    top50 = set(retrieved_article_ids[:50])
    if gold_set.issubset(top10):
        return CASE_A
    if gold_set.issubset(top50):
        return CASE_B
    return CASE_C


def summarize_metrics(
    *,
    dataset_name: str,
    records: int,
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    ks: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "records": records,
        "valid_records": len(valid_rows),
        "invalid_no_gold_articles": len(invalid_rows),
        "ks": ks,
    }

    for key in metric_keys(ks):
        summary[key] = average(row[key] for row in valid_rows)

    summary["case_A_count"] = sum(row["case_type"] == CASE_A for row in valid_rows)
    summary["case_B_count"] = sum(row["case_type"] == CASE_B for row in valid_rows)
    summary["case_C_count"] = sum(row["case_type"] == CASE_C for row in valid_rows)

    groups = group_rows(valid_rows)
    summary["single_article_records"] = len(groups["single"])
    summary["multi_article_records"] = len(groups["multi"])
    summary["single_article_article_full_hit@10"] = average(
        row.get("article_full_hit@10", 0) for row in groups["single"]
    )
    summary["multi_article_article_full_hit@10"] = average(
        row.get("article_full_hit@10", 0) for row in groups["multi"]
    )
    summary["single_article_article_recall@10"] = average(
        row.get("article_recall@10", 0.0) for row in groups["single"]
    )
    summary["multi_article_article_recall@10"] = average(
        row.get("article_recall@10", 0.0) for row in groups["multi"]
    )
    return summary


def metric_keys(ks: list[int]) -> list[str]:
    keys = []
    for prefix in ("article_hit", "article_full_hit", "article_recall", "article_precision"):
        keys.extend(f"{prefix}@{k}" for k in ks)
    keys.append("mrr")
    return keys


def average(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped["multi" if row.get("is_multi_article") else "single"].append(row)
    return grouped


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_metrics_markdown(summary: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    lines = [
        f"# BM25 Article-level Baseline：{summary['dataset_name']}",
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
                [CASE_B, summary["case_B_count"]],
                [CASE_C, summary["case_C_count"]],
            ],
        )
    )

    failed = [trace for trace in traces if trace["case_type"] == CASE_C][:5]
    lines.extend(["", "## Top 5 失败样例", ""])
    if not failed:
        lines.append("没有 top50 未完整命中的失败样例。")
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
    console.print("[bold green]BM25 Baseline Finished[/bold green]")
    console.print()
    console.print(f"Dataset: {summary['dataset_name']}")
    console.print(f"Records: {summary['records']}")
    console.print(f"Valid records: {summary['valid_records']}")
    console.print()
    for key in ("article_hit@10", "article_full_hit@10", "article_recall@10", "mrr"):
        console.print(f"{key}: {summary.get(key, 0.0):.4f}")
    console.print()
    console.print(f"A_top10_full: {summary['case_A_count']}")
    console.print(f"B_top50_full_not_top10: {summary['case_B_count']}")
    console.print(f"C_top50_not_full: {summary['case_C_count']}")
    console.print()
    console.print(f"Results saved to {output_dir}/")
