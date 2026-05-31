#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.run_hybrid_rrf_eval import HybridRRFEvalError, run_hybrid_rrf_eval
from src.evaluation.eval_utils import VALID_QA_DATASETS
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME
from src.retrievers.dense_worker_client import VALID_WORKER_MODES
from src.retrievers.rrf import DEFAULT_BM25_WEIGHT, DEFAULT_DENSE_WEIGHT


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
    parser = argparse.ArgumentParser(description="Run Hybrid BM25 + Dense RRF baseline.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--chunks_path", default="data/processed/wix_kb_chunks.jsonl")
    parser.add_argument("--index_dir", default="indexes/faiss_bge_m3")
    parser.add_argument("--dataset", required=True, choices=sorted(VALID_QA_DATASETS))
    parser.add_argument("--output_dir", default="outputs/hybrid_rrf_baseline")
    parser.add_argument("--model_name", default=DEFAULT_DENSE_MODEL_NAME)
    parser.add_argument("--local_files_only", type=parse_bool, default=True)
    parser.add_argument("--branch_top_k_chunks", type=int, default=50)
    parser.add_argument("--fused_top_k_chunks", type=int, default=50)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--bm25_weight", type=float, default=DEFAULT_BM25_WEIGHT)
    parser.add_argument("--dense_weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    parser.add_argument("--dense_worker_mode", choices=sorted(VALID_WORKER_MODES), default="full")
    parser.add_argument("--dense_query_batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        run_hybrid_rrf_eval(
            processed_dir=args.processed_dir,
            chunks_path=args.chunks_path,
            index_dir=args.index_dir,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
            branch_top_k_chunks=args.branch_top_k_chunks,
            fused_top_k_chunks=args.fused_top_k_chunks,
            rrf_k=args.rrf_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            dense_worker_mode=args.dense_worker_mode,
            dense_query_batch_size=args.dense_query_batch_size,
            device=args.device,
            console=Console(),
        )
    except HybridRRFEvalError as exc:
        Console().print(f"[bold red]Hybrid RRF baseline failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
