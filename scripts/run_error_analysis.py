#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.run_error_analysis import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RERANK_RUN_DIR,
    ErrorAnalysisError,
    run_error_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Phase 6 error-analysis traces from saved reranker traces."
    )
    parser.add_argument("--rerank_run_dir", default=DEFAULT_RERANK_RUN_DIR)
    parser.add_argument("--processed_dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_k_chunks", type=int, default=None)
    args = parser.parse_args()

    try:
        run_error_analysis(
            rerank_run_dir=args.rerank_run_dir,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            top_k_chunks=args.top_k_chunks,
            console=Console(),
        )
    except ErrorAnalysisError as exc:
        Console().print(f"[bold red]Error analysis failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
