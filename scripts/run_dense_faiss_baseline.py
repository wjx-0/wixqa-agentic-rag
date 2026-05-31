#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.eval_utils import BM25EvalError, VALID_QA_DATASETS
from src.evaluation.run_dense_faiss_eval import DenseFaissEvalError, run_dense_faiss_eval
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Dense FAISS retrieval baseline.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--index_dir", default="indexes/faiss_bge_m3")
    parser.add_argument("--dataset", required=True, choices=sorted(VALID_QA_DATASETS))
    parser.add_argument("--output_dir", default="outputs/dense_faiss_baseline")
    parser.add_argument("--model_name", default=DEFAULT_DENSE_MODEL_NAME)
    parser.add_argument("--local_files_only", type=parse_bool, default=True)
    parser.add_argument("--top_k_chunks", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    console = Console()
    try:
        run_dense_faiss_eval(
            processed_dir=args.processed_dir,
            index_dir=args.index_dir,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
            top_k_chunks=args.top_k_chunks,
            device=args.device,
            console=console,
        )
    except (DenseFaissEvalError, BM25EvalError) as exc:
        console.print(f"[bold red]Dense FAISS baseline failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
