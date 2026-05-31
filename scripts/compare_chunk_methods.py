#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.run_bm25_eval import VALID_QA_DATASETS, markdown_table
from src.utils.io_utils import ensure_dir, read_json


DEFAULT_DATASETS = ("wixqa_expertwritten", "wixqa_simulated", "wixqa_synthetic")
COMPARE_METRICS = (
    "article_hit@10",
    "article_full_hit@10",
    "article_recall@10",
    "mrr",
    "single_article_article_full_hit@10",
    "multi_article_article_full_hit@10",
    "single_article_article_recall@10",
    "multi_article_article_recall@10",
)


def load_metrics(metrics_dir: Path, dataset_name: str) -> dict[str, Any]:
    path = metrics_dir / f"{dataset_name}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return read_json(path)


def format_score(value: Any) -> str:
    return f"{float(value):.4f}"


def format_delta(value: float) -> str:
    return f"{value:+.4f}"


def render_compare_markdown(
    *,
    article_bm25_dir: Path,
    chunk_bm25_dir: Path,
    datasets: list[str],
) -> str:
    lines = [
        "# Chunk BM25 Method Comparison",
        "",
        "对比口径：chunk-level BM25 检索 chunk 后按 `article_id` 聚合，再与 article-level BM25 使用相同 article-level metrics。",
        "",
        f"- Article-level BM25: `{article_bm25_dir}`",
        f"- Chunk-level BM25: `{chunk_bm25_dir}`",
        "",
    ]

    for dataset_name in datasets:
        article_metrics = load_metrics(article_bm25_dir, dataset_name)
        chunk_metrics = load_metrics(chunk_bm25_dir, dataset_name)

        lines.extend([f"## {dataset_name}", ""])
        lines.extend(
            markdown_table(
                ["metric", "article_bm25", "chunk_bm25", "delta"],
                [
                    [
                        metric,
                        format_score(article_metrics.get(metric, 0.0)),
                        format_score(chunk_metrics.get(metric, 0.0)),
                        format_delta(
                            float(chunk_metrics.get(metric, 0.0))
                            - float(article_metrics.get(metric, 0.0))
                        ),
                    ]
                    for metric in COMPARE_METRICS
                ],
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare article-level and chunk-level BM25 metrics.")
    parser.add_argument("--article_bm25_dir", default="outputs/bm25_baseline")
    parser.add_argument("--chunk_bm25_dir", default="outputs/chunk_bm25_baseline")
    parser.add_argument("--output_path", default="outputs/chunk_bm25_baseline/compare_chunk_methods.md")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=sorted(VALID_QA_DATASETS),
    )
    args = parser.parse_args()

    console = Console()
    article_bm25_dir = Path(args.article_bm25_dir)
    chunk_bm25_dir = Path(args.chunk_bm25_dir)
    output_path = Path(args.output_path)

    try:
        report = render_compare_markdown(
            article_bm25_dir=article_bm25_dir,
            chunk_bm25_dir=chunk_bm25_dir,
            datasets=list(args.datasets),
        )
        ensure_dir(output_path.parent)
        output_path.write_text(report, encoding="utf-8")
    except FileNotFoundError as exc:
        console.print(f"[bold red]Comparison failed:[/bold red] {exc}")
        return 1

    console.print(f"[green]Wrote[/green] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
