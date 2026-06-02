#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.run_rule_second_hop_eval import (
    DEFAULT_BASELINE_RERANK_RUN_DIR,
    DEFAULT_BRANCH_TOP_K_CHUNKS,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_FIRST_HOP_HYBRID_RUN_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_MAX_SEED_ARTICLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_SECOND_HOP_TOP_K_CHUNKS,
    RuleSecondHopEvalError,
    run_rule_second_hop_eval,
)
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_INSTRUCTION_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL_NAME,
)
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME
from src.retrievers.dense_worker_client import VALID_WORKER_MODES
from src.retrievers.rrf import DEFAULT_BM25_WEIGHT


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
    parser = argparse.ArgumentParser(description="Run rule-based second-hop retrieval.")
    parser.add_argument("--first_hop_hybrid_run_dir", default=DEFAULT_FIRST_HOP_HYBRID_RUN_DIR)
    parser.add_argument("--baseline_rerank_run_dir", default=DEFAULT_BASELINE_RERANK_RUN_DIR)
    parser.add_argument(
        "--top100_control_rerank_run_dir",
        default=None,
        help="Optional top100 reranker control run. Omit to skip top100 comparison.",
    )
    parser.add_argument("--chunks_path", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dense_model_name", default=DEFAULT_DENSE_MODEL_NAME)
    parser.add_argument("--reranker_model_name", default=DEFAULT_RERANK_MODEL_NAME)
    parser.add_argument("--local_files_only", type=parse_bool, default=True)
    parser.add_argument("--dense_worker_mode", choices=sorted(VALID_WORKER_MODES), default="full")
    parser.add_argument("--dense_query_batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rerank_batch_size", type=int, default=DEFAULT_RERANK_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--instruction_name", default=DEFAULT_INSTRUCTION_NAME)
    parser.add_argument("--instruction", default=DEFAULT_RERANK_INSTRUCTION)
    parser.add_argument("--max_seed_articles", type=int, default=DEFAULT_MAX_SEED_ARTICLES)
    parser.add_argument("--branch_top_k_chunks", type=int, default=DEFAULT_BRANCH_TOP_K_CHUNKS)
    parser.add_argument(
        "--second_hop_top_k_chunks",
        type=int,
        default=DEFAULT_SECOND_HOP_TOP_K_CHUNKS,
    )
    parser.add_argument("--rrf_k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--bm25_weight", type=float, default=DEFAULT_BM25_WEIGHT)
    parser.add_argument("--dense_weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    args = parser.parse_args()

    try:
        run_rule_second_hop_eval(
            first_hop_hybrid_run_dir=args.first_hop_hybrid_run_dir,
            baseline_rerank_run_dir=args.baseline_rerank_run_dir,
            top100_control_rerank_run_dir=args.top100_control_rerank_run_dir,
            chunks_path=args.chunks_path,
            index_dir=args.index_dir,
            output_dir=args.output_dir,
            dense_model_name=args.dense_model_name,
            reranker_model_name=args.reranker_model_name,
            local_files_only=args.local_files_only,
            dense_worker_mode=args.dense_worker_mode,
            dense_query_batch_size=args.dense_query_batch_size,
            device=args.device,
            rerank_batch_size=args.rerank_batch_size,
            max_length=args.max_length,
            instruction_name=args.instruction_name,
            instruction=args.instruction,
            max_seed_articles=args.max_seed_articles,
            branch_top_k_chunks=args.branch_top_k_chunks,
            second_hop_top_k_chunks=args.second_hop_top_k_chunks,
            rrf_k=args.rrf_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            console=Console(),
        )
    except RuleSecondHopEvalError as exc:
        Console().print(f"[bold red]Rule second-hop retrieval failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
