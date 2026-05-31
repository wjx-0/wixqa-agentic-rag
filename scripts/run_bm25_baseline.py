#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.run_bm25_eval import BM25EvalError, VALID_QA_DATASETS, run_bm25_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BM25 article-level retrieval baseline.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--dataset", required=True, choices=sorted(VALID_QA_DATASETS))
    parser.add_argument("--output_dir", default="outputs/bm25_baseline")
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    console = Console()
    try:
        run_bm25_eval(
            processed_dir=args.processed_dir,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            top_k=args.top_k,
            console=console,
        )
    except BM25EvalError as exc:
        console.print(f"[bold red]BM25 baseline failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

