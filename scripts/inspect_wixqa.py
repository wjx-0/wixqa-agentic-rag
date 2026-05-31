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
from rich.pretty import Pretty
from rich.table import Table

from src.data.load_wixqa import DATASET_NAME, WixQALoadError, discover_configs, load_config, split_info
from src.utils.text_utils import preview_text


def row_preview(row: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            preview[key] = preview_text(value, 220)
        elif isinstance(value, list):
            preview[key] = value[:5]
        else:
            preview[key] = value
    return preview


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect WixQA configs, splits, fields, and samples.")
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--max_rows", type=int, default=2)
    args = parser.parse_args()

    console = Console()
    try:
        configs = discover_configs(args.dataset_name)
        console.print(f"[bold]Dataset:[/bold] {args.dataset_name}")
        console.print(f"[bold]Configs:[/bold] {configs}")

        for config in configs:
            console.rule(f"[bold cyan]{config}")
            splits = load_config(config, args.dataset_name)
            table = Table(title=f"{config} splits")
            table.add_column("split")
            table.add_column("rows", justify="right")
            table.add_column("fields")
            for split_name, dataset in splits.items():
                info = split_info(split_name, dataset)
                table.add_row(info.name, str(info.num_rows), ", ".join(info.features))
            console.print(table)

            for split_name, dataset in splits.items():
                console.print(f"[bold]Samples from {config}/{split_name}[/bold]")
                for index, row in enumerate(dataset):
                    if index >= args.max_rows:
                        break
                    console.print(Pretty(row_preview(row)))
    except WixQALoadError as exc:
        console.print(f"[bold red]WixQA inspection failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

