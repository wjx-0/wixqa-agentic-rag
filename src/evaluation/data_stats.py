from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from rich.console import Console

from src.utils.io_utils import ensure_dir, read_jsonl, write_json
from src.utils.text_utils import preview_text


QA_SUBSETS = ("wixqa_expertwritten", "wixqa_simulated", "wixqa_synthetic")


def avg(values: list[int]) -> float:
    return float(mean(values)) if values else 0.0


def med(values: list[int]) -> float:
    return float(median(values)) if values else 0.0


def load_processed(processed_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    kb_path = processed_dir / "wix_kb_corpus.jsonl"
    if not kb_path.exists():
        raise FileNotFoundError(f"Missing processed KB file: {kb_path}")
    kb_rows = list(read_jsonl(kb_path))

    qa_rows: dict[str, list[dict[str, Any]]] = {}
    for subset in QA_SUBSETS:
        path = processed_dir / f"{subset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed QA file: {path}")
        qa_rows[subset] = list(read_jsonl(path))
    return kb_rows, qa_rows


def kb_stats(kb_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(row.get("contents") or "") for row in kb_rows]
    return {
        "num_articles": len(kb_rows),
        "avg_contents_length": avg(lengths),
        "median_contents_length": med(lengths),
        "max_contents_length": max(lengths) if lengths else 0,
        "min_contents_length": min(lengths) if lengths else 0,
        "article_type_distribution": dict(Counter(row.get("article_type") or "__missing__" for row in kb_rows)),
        "missing_url_count": sum(not row.get("url") for row in kb_rows),
        "missing_title_count": sum(not row.get("title") for row in kb_rows),
        "empty_contents_count": sum(not (row.get("contents") or "").strip() for row in kb_rows),
    }


def qa_subset_stats(rows: list[dict[str, Any]], kb_ids: set[str]) -> dict[str, Any]:
    question_lengths = [len(row.get("question") or "") for row in rows]
    answer_lengths = [len(row.get("answer") or "") for row in rows]
    gold_counts = [len(row.get("article_ids") or []) for row in rows]
    missing_gold_mentions = 0
    missing_gold_ids: set[str] = set()
    total_gold_mentions = 0

    for row in rows:
        for article_id in row.get("article_ids") or []:
            total_gold_mentions += 1
            if article_id not in kb_ids:
                missing_gold_mentions += 1
                missing_gold_ids.add(article_id)

    coverage = (
        (total_gold_mentions - missing_gold_mentions) / total_gold_mentions
        if total_gold_mentions
        else 0.0
    )
    multi_count = sum(count >= 2 for count in gold_counts)

    return {
        "num_examples": len(rows),
        "avg_question_length": avg(question_lengths),
        "avg_answer_length": avg(answer_lengths),
        "avg_gold_article_count": avg(gold_counts),
        "max_gold_article_count": max(gold_counts) if gold_counts else 0,
        "single_article_count": sum(count == 1 for count in gold_counts),
        "multi_article_count": multi_count,
        "multi_article_ratio": multi_count / len(rows) if rows else 0.0,
        "empty_article_ids_count": sum(count == 0 for count in gold_counts),
        "missing_gold_article_id_mentions": missing_gold_mentions,
        "missing_unique_gold_article_ids": len(missing_gold_ids),
        "gold_article_id_coverage": coverage,
    }


def build_stats(kb_rows: list[dict[str, Any]], qa_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    kb_ids = {row["article_id"] for row in kb_rows}
    qa_stats = {subset: qa_subset_stats(rows, kb_ids) for subset, rows in qa_rows.items()}
    return {
        "kb_corpus": kb_stats(kb_rows),
        "qa_subsets": qa_stats,
        "multi_article_analysis": {
            subset: {"multi_article_count": qa_stats[subset]["multi_article_count"]}
            for subset in QA_SUBSETS
        },
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_markdown(
    stats: dict[str, Any],
    kb_rows: list[dict[str, Any]],
    qa_rows: dict[str, list[dict[str, Any]]],
) -> str:
    kb_lookup = {row["article_id"]: row for row in kb_rows}
    lines = ["# WixQA 数据统计", ""]

    kb = stats["kb_corpus"]
    lines.extend(["## 知识库 Corpus", ""])
    lines.extend(
        markdown_table(
            ["指标", "数值"],
            [
                ["文章总数", kb["num_articles"]],
                ["contents 平均长度", f"{kb['avg_contents_length']:.2f}"],
                ["contents 中位数长度", f"{kb['median_contents_length']:.2f}"],
                ["contents 最大长度", kb["max_contents_length"]],
                ["contents 最小长度", kb["min_contents_length"]],
                ["URL 缺失数量", kb["missing_url_count"]],
                ["标题缺失数量", kb["missing_title_count"]],
                ["contents 为空数量", kb["empty_contents_count"]],
            ],
        )
    )
    lines.extend(["", "### 文章类型分布", ""])
    lines.extend(
        markdown_table(
            ["article_type", "数量"],
            [[key, value] for key, value in kb["article_type_distribution"].items()],
        )
    )

    lines.extend(["", "## QA 子集", ""])
    qa_table = []
    for subset in QA_SUBSETS:
        subset_stats = stats["qa_subsets"][subset]
        qa_table.append(
            [
                subset,
                subset_stats["num_examples"],
                f"{subset_stats['avg_question_length']:.2f}",
                f"{subset_stats['avg_answer_length']:.2f}",
                f"{subset_stats['avg_gold_article_count']:.2f}",
                subset_stats["max_gold_article_count"],
                subset_stats["single_article_count"],
                subset_stats["multi_article_count"],
                f"{subset_stats['multi_article_ratio']:.2%}",
                subset_stats["empty_article_ids_count"],
                subset_stats["missing_gold_article_id_mentions"],
                f"{subset_stats['gold_article_id_coverage']:.2%}",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "子集",
                "样本数",
                "问题平均长度",
                "答案平均长度",
                "平均 gold 数",
                "最大 gold 数",
                "单文章",
                "多文章",
                "多文章比例",
                "空 article_ids",
                "缺失 gold 次数",
                "gold 覆盖率",
            ],
            qa_table,
        )
    )

    lines.extend(["", "## 多文章样例分析", ""])
    for subset in QA_SUBSETS:
        multi_examples = [row for row in qa_rows[subset] if row.get("is_multi_article")][:5]
        lines.extend([f"### {subset}", ""])
        lines.append(f"多文章样本数：{stats['qa_subsets'][subset]['multi_article_count']}")
        lines.append("")
        if not multi_examples:
            lines.extend(["未找到多文章样例。", ""])
            continue
        for row in multi_examples:
            lines.extend(
                [
                    f"#### {row.get('qid')}",
                    "",
                    f"- 问题：{preview_text(row.get('question'), 500)}",
                    f"- 答案：{preview_text(row.get('answer'), 300)}",
                    f"- 文章 ID：`{row.get('article_ids')}`",
                    "- 相关文章：",
                ]
            )
            for article_id in row.get("article_ids") or []:
                article = kb_lookup.get(article_id)
                if article is None:
                    lines.append(f"  - `{article_id}`：KB corpus 中缺失")
                else:
                    title = article.get("title") or "（标题缺失）"
                    url = article.get("url") or "（URL 缺失）"
                    lines.append(f"  - `{article_id}`: {title} | {url}")
            lines.append("")
    return "\n".join(lines)


def print_summary(console: Console, stats: dict[str, Any], processed_dir: Path, output_dir: Path) -> None:
    expert = stats["qa_subsets"]["wixqa_expertwritten"]
    simulated = stats["qa_subsets"]["wixqa_simulated"]
    synthetic = stats["qa_subsets"]["wixqa_synthetic"]

    console.print()
    console.print("[bold green]WixQA 数据统计完成[/bold green]")
    console.print()
    console.print(f"KB 文章数: {stats['kb_corpus']['num_articles']}")
    console.print(f"ExpertWritten QA: {expert['num_examples']}")
    console.print(f"Simulated QA: {simulated['num_examples']}")
    console.print(f"Synthetic QA: {synthetic['num_examples']}")
    console.print()
    console.print(f"ExpertWritten 多文章样本数: {expert['multi_article_count']}")
    console.print(f"Simulated 多文章样本数: {simulated['multi_article_count']}")
    console.print(f"Synthetic 多文章样本数: {synthetic['multi_article_count']}")
    console.print()
    console.print(f"Processed 文件已保存到 {processed_dir}/")
    console.print(f"统计文件已保存到 {output_dir}/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute WixQA processed data statistics.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_dir", default="data/stats")
    args = parser.parse_args()

    console = Console()
    processed_dir = Path(args.processed_dir)
    output_dir = ensure_dir(args.output_dir)

    try:
        kb_rows, qa_rows = load_processed(processed_dir)
        stats = build_stats(kb_rows, qa_rows)
        write_json(output_dir / "wixqa_data_stats.json", stats)
        (output_dir / "wixqa_data_stats.md").write_text(
            render_markdown(stats, kb_rows, qa_rows),
            encoding="utf-8",
        )
        print_summary(console, stats, processed_dir, output_dir)
    except Exception as exc:
        console.print(f"[bold red]WixQA stats failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
