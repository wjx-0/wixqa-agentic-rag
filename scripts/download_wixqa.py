#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from src.data.load_wixqa import DATASET_NAME, WixQALoadError, discover_configs, load_config, split_info
from src.utils.io_utils import ensure_dir, write_json, write_jsonl


def raw_jsonl_name(config: str, split_name: str) -> str:
    safe_split = split_name.replace("/", "_").replace("\\", "_")
    return f"{config}_{safe_split}.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/cache WixQA and export complete raw JSONL files.")
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--sample_size", type=int, default=10)
    args = parser.parse_args()

    console = Console()
    raw_dir = ensure_dir(args.raw_dir)
    samples_dir = ensure_dir(raw_dir / "samples")

    try:
        configs = discover_configs(args.dataset_name)
        summary = {"dataset_name": args.dataset_name, "configs": []}

        for config in configs:
            console.print(f"[cyan]Loading[/cyan] {args.dataset_name}/{config}")
            splits = load_config(config, args.dataset_name)
            config_payload = {"config": config, "splits": []}
            sample_path = samples_dir / f"{config}_samples.json"
            sample_payload = {"dataset_name": args.dataset_name, "config": config, "splits": {}}

            for split_name, dataset in splits.items():
                info = split_info(split_name, dataset)
                raw_path = raw_dir / raw_jsonl_name(config, split_name)
                rows = [
                    dict(row)
                    for row in tqdm(
                        dataset,
                        desc=f"export {config}/{split_name}",
                        total=info.num_rows,
                    )
                ]
                num_written = write_jsonl(raw_path, rows)

                config_payload["splits"].append(
                    {
                        "split": split_name,
                        "num_rows": num_written,
                        "features": info.features,
                        "raw_path": str(raw_path),
                        "sample_path": str(sample_path),
                    }
                )
                sample_payload["splits"][split_name] = rows[: args.sample_size]

            summary["configs"].append(config_payload)
            write_json(sample_path, sample_payload)

        write_json(raw_dir / "wixqa_raw_summary.json", summary)

        table = Table(title="WixQA raw JSONL exports")
        table.add_column("config")
        table.add_column("splits")
        table.add_column("rows", justify="right")
        table.add_column("raw files")
        for config_info in summary["configs"]:
            split_names = ", ".join(split["split"] for split in config_info["splits"])
            row_count = sum(split["num_rows"] for split in config_info["splits"])
            raw_files = ", ".join(split["raw_path"] for split in config_info["splits"])
            table.add_row(config_info["config"], split_names, str(row_count), raw_files)
        console.print(table)
        console.print(f"[green]完整 raw JSONL 已保存到[/green] {raw_dir}")
        console.print(f"[green]Raw 样例已保存到[/green] {samples_dir}")
        console.print(f"[green]Raw 摘要已保存到[/green] {raw_dir / 'wixqa_raw_summary.json'}")
    except WixQALoadError as exc:
        console.print(f"[bold red]WixQA download failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
