#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.data.load_wixqa import (
    CANONICAL_CONFIGS,
    DATASET_NAME,
    WixQALoadError,
    discover_configs,
    load_config,
    map_actual_configs,
)
from src.data.preprocess_wixqa import (
    FieldMappingError,
    build_kb_field_mapping,
    build_qa_field_mapping,
    convert_kb_row,
    convert_qa_row,
)
from src.data.schema import KBArticle, QAExample
from src.utils.io_utils import ensure_dir, model_to_dict, write_json, write_jsonl
from src.utils.text_utils import preview_text


SAMPLE_FILE_NAMES = {
    "wix_kb_corpus": "kb_samples.json",
    "wixqa_expertwritten": "expertwritten_samples.json",
    "wixqa_simulated": "simulated_samples.json",
    "wixqa_synthetic": "synthetic_samples.json",
}


def dataset_features(dataset: Any) -> list[str]:
    return list(getattr(dataset, "features", {}).keys())


def process_kb_config(actual_config: str, splits: dict[str, Any]) -> list[KBArticle]:
    articles: list[KBArticle] = []
    for split_name, dataset in splits.items():
        mapping = build_kb_field_mapping(dataset_features(dataset))
        for row_index, row in enumerate(tqdm(dataset, desc=f"prepare {actual_config}/{split_name}")):
            articles.append(
                convert_kb_row(
                    row,
                    mapping=mapping,
                    source_config=actual_config,
                    split=split_name,
                    row_index=row_index,
                )
            )
    return articles


def process_qa_config(
    canonical_config: str,
    actual_config: str,
    splits: dict[str, Any],
) -> list[QAExample]:
    examples: list[QAExample] = []
    global_index = 0
    for split_name, dataset in splits.items():
        mapping = build_qa_field_mapping(dataset_features(dataset))
        for row_index, row in enumerate(tqdm(dataset, desc=f"prepare {actual_config}/{split_name}")):
            examples.append(
                convert_qa_row(
                    row,
                    mapping=mapping,
                    canonical_dataset_name=canonical_config,
                    source_config=actual_config,
                    split=split_name,
                    row_index=row_index,
                    global_index=global_index,
                )
            )
            global_index += 1
    return examples


def write_samples(
    inspection_dir: Path,
    processed: dict[str, list[KBArticle] | list[QAExample]],
    *,
    sample_size: int,
) -> None:
    ensure_dir(inspection_dir)
    for canonical_config, rows in processed.items():
        filename = SAMPLE_FILE_NAMES[canonical_config]
        write_json(
            inspection_dir / filename,
            [model_to_dict(row) for row in rows[:sample_size]],
        )


def write_multi_article_samples(
    inspection_dir: Path,
    kb_articles: list[KBArticle],
    qa_sets: dict[str, list[QAExample]],
    *,
    sample_size: int,
) -> None:
    kb_lookup = {article.article_id: article for article in kb_articles}
    lines = ["# 多文章样例", ""]
    for dataset_name, examples in qa_sets.items():
        lines.extend([f"## {dataset_name}", ""])
        multi_examples = [example for example in examples if example.is_multi_article][:sample_size]
        if not multi_examples:
            lines.extend(["未找到多文章样例。", ""])
            continue
        for example in multi_examples:
            lines.extend(
                [
                    f"### {example.qid}",
                    "",
                    f"- 问题：{preview_text(example.question, 500)}",
                    f"- 答案：{preview_text(example.answer, 300)}",
                    f"- 文章 ID：`{example.article_ids}`",
                    "- 相关文章：",
                ]
            )
            for article_id in example.article_ids:
                article = kb_lookup.get(article_id)
                if article is None:
                    lines.append(f"  - `{article_id}`：KB corpus 中缺失")
                else:
                    title = article.title or "（标题缺失）"
                    url = article.url or "（URL 缺失）"
                    lines.append(f"  - `{article_id}`: {title} | {url}")
            lines.append("")
    (inspection_dir / "multi_article_samples.md").write_text("\n".join(lines), encoding="utf-8")


def print_summary(console: Console, processed: dict[str, list]) -> None:
    kb_count = len(processed.get("wix_kb_corpus", []))
    expert = processed.get("wixqa_expertwritten", [])
    simulated = processed.get("wixqa_simulated", [])
    synthetic = processed.get("wixqa_synthetic", [])

    console.print()
    console.print("[bold green]WixQA Data Preparation Finished[/bold green]")
    console.print()
    console.print(f"KB articles: {kb_count}")
    console.print(f"ExpertWritten QA: {len(expert)}")
    console.print(f"Simulated QA: {len(simulated)}")
    console.print(f"Synthetic QA: {len(synthetic)}")
    console.print()
    console.print(
        f"ExpertWritten multi-article cases: {sum(example.is_multi_article for example in expert)}"
    )
    console.print(
        f"Simulated multi-article cases: {sum(example.is_multi_article for example in simulated)}"
    )
    console.print(
        f"Synthetic multi-article cases: {sum(example.is_multi_article for example in synthetic)}"
    )
    console.print()
    console.print("Processed files saved to data/processed/")
    console.print("Inspection samples saved to outputs/data_inspection/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare WixQA into project JSONL schemas.")
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--inspection_dir", default="outputs/data_inspection")
    parser.add_argument("--sample_size", type=int, default=10)
    args = parser.parse_args()

    console = Console()
    processed_dir = ensure_dir(args.processed_dir)
    inspection_dir = ensure_dir(args.inspection_dir)

    try:
        discovered_configs = discover_configs(args.dataset_name)
        mapped_configs = map_actual_configs(discovered_configs)
        missing = [config for config in CANONICAL_CONFIGS if config not in mapped_configs]
        if missing:
            console.print(f"[yellow]Warning:[/yellow] missing expected config(s): {missing}")
        console.print(f"[bold]Discovered configs:[/bold] {discovered_configs}")
        console.print(f"[bold]Mapped configs:[/bold] {mapped_configs}")

        processed: dict[str, list[KBArticle] | list[QAExample]] = {}
        for canonical_config in CANONICAL_CONFIGS:
            actual_config = mapped_configs.get(canonical_config)
            if actual_config is None:
                continue
            splits = load_config(actual_config, args.dataset_name)
            if canonical_config == "wix_kb_corpus":
                processed[canonical_config] = process_kb_config(actual_config, splits)
            else:
                processed[canonical_config] = process_qa_config(canonical_config, actual_config, splits)

            output_file = processed_dir / f"{canonical_config}.jsonl"
            count = write_jsonl(output_file, processed[canonical_config])
            console.print(f"[green]Wrote[/green] {count} rows -> {output_file}")

        write_samples(inspection_dir, processed, sample_size=args.sample_size)
        write_multi_article_samples(
            inspection_dir,
            processed.get("wix_kb_corpus", []),
            {
                "wixqa_expertwritten": processed.get("wixqa_expertwritten", []),
                "wixqa_simulated": processed.get("wixqa_simulated", []),
                "wixqa_synthetic": processed.get("wixqa_synthetic", []),
            },
            sample_size=5,
        )
        write_json(
            inspection_dir / "loaded_config_mapping.json",
            {"dataset_name": args.dataset_name, "discovered_configs": discovered_configs, "mapped_configs": mapped_configs},
        )
        print_summary(console, processed)
    except (WixQALoadError, FieldMappingError) as exc:
        console.print(f"[bold red]WixQA preparation failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
