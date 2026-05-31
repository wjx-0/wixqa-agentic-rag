#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.data.chunk_wixqa import (
    ChunkTokenizer,
    ChunkingError,
    build_chunk_tokenizer,
    chunk_article,
    validate_chunk_params,
)
from src.data.schema import KBArticle, KBChunk
from src.utils.io_utils import ensure_dir, model_to_dict, read_jsonl, write_json, write_jsonl


def load_kb_articles(processed_dir: Path) -> list[KBArticle]:
    path = processed_dir / "wix_kb_corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed KB file: {path}. Please run `python scripts/prepare_wixqa.py` first."
        )
    return [KBArticle(**row) for row in read_jsonl(path)]


def build_chunks(
    articles: list[KBArticle],
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    tokenizer: ChunkTokenizer,
) -> list[KBChunk]:
    chunks: list[KBChunk] = []
    for article in tqdm(articles, desc="prepare KB chunks"):
        chunks.extend(
            chunk_article(
                article,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                tokenizer=tokenizer,
            )
        )
    return chunks


def avg(values: list[int]) -> float:
    return float(mean(values)) if values else 0.0


def med(values: list[int]) -> float:
    return float(median(values)) if values else 0.0


def build_chunk_stats(
    *,
    articles: list[KBArticle],
    chunks: list[KBChunk],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    chunk_tokenizer: str,
) -> dict[str, Any]:
    counts_by_article = Counter(chunk.article_id for chunk in chunks)
    lengths = [chunk.num_tokens for chunk in chunks]
    duplicate_chunk_ids = len(chunks) - len({chunk.chunk_id for chunk in chunks})
    article_ids = {article.article_id for article in articles}
    empty_contents_count = sum(not (article.contents or "").strip() for article in articles)
    chunked_article_count = len(counts_by_article)
    missing_article_id_count = sum(not chunk.article_id for chunk in chunks)
    oversized_chunk_count = sum(chunk.num_tokens > chunk_size_tokens for chunk in chunks)

    starts_by_article: dict[str, list[int]] = defaultdict(list)
    for chunk in chunks:
        starts_by_article[chunk.article_id].append(chunk.start_token)

    stride = chunk_size_tokens - chunk_overlap_tokens
    observed_steps = []
    for starts in starts_by_article.values():
        starts = sorted(starts)
        observed_steps.extend(next_start - start for start, next_start in zip(starts, starts[1:]))

    chunks_per_article_distribution = Counter(counts_by_article.values())
    top_chunked_articles = sorted(
        counts_by_article.items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]

    return {
        "num_articles": len(articles),
        "num_article_ids": len(article_ids),
        "empty_contents_count": empty_contents_count,
        "num_articles_with_chunks": chunked_article_count,
        "num_articles_without_chunks": len(articles) - chunked_article_count,
        "num_chunks": len(chunks),
        "duplicate_chunk_id_count": duplicate_chunk_ids,
        "missing_chunk_article_id_count": missing_article_id_count,
        "oversized_chunk_count": oversized_chunk_count,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "chunk_tokenizer": chunk_tokenizer,
        "expected_stride_tokens": stride,
        "avg_chunk_tokens": avg(lengths),
        "median_chunk_tokens": med(lengths),
        "min_chunk_tokens": min(lengths) if lengths else 0,
        "max_chunk_tokens": max(lengths) if lengths else 0,
        "min_observed_stride_tokens": min(observed_steps) if observed_steps else 0,
        "max_observed_stride_tokens": max(observed_steps) if observed_steps else 0,
        "chunks_per_article_distribution": {
            str(num_chunks): count
            for num_chunks, count in sorted(chunks_per_article_distribution.items())
        },
        "top_chunked_articles": [
            {"article_id": article_id, "num_chunks": count}
            for article_id, count in top_chunked_articles
        ],
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_chunk_stats_markdown(stats: dict[str, Any]) -> str:
    lines = ["# WixQA KB Chunk Stats", ""]
    lines.extend(
        markdown_table(
            ["Metric", "Value"],
            [
                ["num_articles", stats["num_articles"]],
                ["num_article_ids", stats["num_article_ids"]],
                ["empty_contents_count", stats["empty_contents_count"]],
                ["num_articles_with_chunks", stats["num_articles_with_chunks"]],
                ["num_articles_without_chunks", stats["num_articles_without_chunks"]],
                ["num_chunks", stats["num_chunks"]],
                ["duplicate_chunk_id_count", stats["duplicate_chunk_id_count"]],
                ["missing_chunk_article_id_count", stats["missing_chunk_article_id_count"]],
                ["oversized_chunk_count", stats["oversized_chunk_count"]],
                ["chunk_size_tokens", stats["chunk_size_tokens"]],
                ["chunk_overlap_tokens", stats["chunk_overlap_tokens"]],
                ["chunk_tokenizer", stats["chunk_tokenizer"]],
                ["expected_stride_tokens", stats["expected_stride_tokens"]],
                ["avg_chunk_tokens", f"{stats['avg_chunk_tokens']:.2f}"],
                ["median_chunk_tokens", f"{stats['median_chunk_tokens']:.2f}"],
                ["min_chunk_tokens", stats["min_chunk_tokens"]],
                ["max_chunk_tokens", stats["max_chunk_tokens"]],
                ["min_observed_stride_tokens", stats["min_observed_stride_tokens"]],
                ["max_observed_stride_tokens", stats["max_observed_stride_tokens"]],
            ],
        )
    )

    lines.extend(["", "## Chunks Per Article", ""])
    lines.extend(
        markdown_table(
            ["num_chunks", "article_count"],
            [
                [num_chunks, count]
                for num_chunks, count in stats["chunks_per_article_distribution"].items()
            ],
        )
    )

    lines.extend(["", "## Top Chunked Articles", ""])
    lines.extend(
        markdown_table(
            ["article_id", "num_chunks"],
            [
                [row["article_id"], row["num_chunks"]]
                for row in stats["top_chunked_articles"]
            ],
        )
    )
    return "\n".join(lines)


def print_summary(console: Console, stats: dict[str, Any], output_path: Path, inspection_dir: Path) -> None:
    console.print()
    console.print("[bold green]WixQA chunk preparation finished[/bold green]")
    console.print()
    console.print(f"Articles: {stats['num_articles']}")
    console.print(f"Chunks: {stats['num_chunks']}")
    console.print(f"Chunk size / overlap: {stats['chunk_size_tokens']} / {stats['chunk_overlap_tokens']}")
    console.print(f"Chunk tokenizer: {stats['chunk_tokenizer']}")
    console.print(f"Max chunk tokens: {stats['max_chunk_tokens']}")
    console.print()
    console.print(f"Chunks saved to {output_path}")
    console.print(f"Chunk inspection saved to {inspection_dir}/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare token chunks for WixQA KB articles.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_path", default="data/processed/wix_kb_chunks.jsonl")
    parser.add_argument("--inspection_dir", default="outputs/data_inspection")
    parser.add_argument("--chunk_size_tokens", type=int, default=512)
    parser.add_argument("--chunk_overlap_tokens", type=int, default=128)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--sample_size", type=int, default=10)
    args = parser.parse_args()

    console = Console()
    processed_dir = Path(args.processed_dir)
    output_path = Path(args.output_path)
    inspection_dir = ensure_dir(args.inspection_dir)

    try:
        validate_chunk_params(args.chunk_size_tokens, args.chunk_overlap_tokens)
        tokenizer = build_chunk_tokenizer(local_files_only=args.local_files_only)
        articles = load_kb_articles(processed_dir)
        chunks = build_chunks(
            articles,
            chunk_size_tokens=args.chunk_size_tokens,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            tokenizer=tokenizer,
        )
        stats = build_chunk_stats(
            articles=articles,
            chunks=chunks,
            chunk_size_tokens=args.chunk_size_tokens,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            chunk_tokenizer=tokenizer.name,
        )

        write_jsonl(output_path, chunks)
        write_json(
            inspection_dir / "chunk_samples.json",
            [model_to_dict(chunk) for chunk in chunks[: args.sample_size]],
        )
        (inspection_dir / "chunk_stats.md").write_text(
            render_chunk_stats_markdown(stats),
            encoding="utf-8",
        )
        print_summary(console, stats, output_path, inspection_dir)
    except (ChunkingError, FileNotFoundError) as exc:
        console.print(f"[bold red]Chunk preparation failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
