#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.run_evidence_context_eval import DEFAULT_RERANK_RUN_DIR
from src.evaluation.run_evidence_loop_eval import (
    DEFAULT_BRANCH_TOP_K_CHUNKS,
    DEFAULT_INDEX_DIR,
    DEFAULT_MAX_RAW_CHUNKS_PER_CHECKER_CALL,
    DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RRF_K,
    DEFAULT_SECOND_HOP_TOP_K_CHUNKS,
    EvidenceLoopEvalError,
    run_evidence_loop_eval,
)
from src.llm.evidence_checker import (
    DEFAULT_CHECKER_MAX_TOKENS,
    DEFAULT_CHECKER_TEMPERATURE,
    DEFAULT_CHECKER_TIMEOUT,
    DEFAULT_CONTEXT_PREVIEW_CHARS,
    DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
)
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_INSTRUCTION_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL_NAME,
)
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


def parse_qids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    qids = [item.strip() for item in value.split(",") if item.strip()]
    return qids or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run traceable EvidenceContext loop with 8B checker and 0.6B reranker."
    )
    parser.add_argument("--rerank_run_dir", default=DEFAULT_RERANK_RUN_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--dense_model_name", default=DEFAULT_DENSE_MODEL_NAME)
    parser.add_argument("--local_files_only", type=parse_bool, default=True)
    parser.add_argument("--dense_worker_mode", choices=sorted(VALID_WORKER_MODES), default="full")
    parser.add_argument("--dense_query_batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--branch_top_k_chunks", type=int, default=DEFAULT_BRANCH_TOP_K_CHUNKS)
    parser.add_argument("--second_hop_top_k_chunks", type=int, default=DEFAULT_SECOND_HOP_TOP_K_CHUNKS)
    parser.add_argument("--rrf_k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--bm25_weight", type=float, default=DEFAULT_BM25_WEIGHT)
    parser.add_argument("--dense_weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    parser.add_argument("--reranker_model_name", default=DEFAULT_RERANK_MODEL_NAME)
    parser.add_argument("--reranker_local_files_only", type=parse_bool, default=True)
    parser.add_argument("--reranker_batch_size", type=int, default=DEFAULT_RERANK_BATCH_SIZE)
    parser.add_argument("--reranker_max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--reranker_instruction_name", default=DEFAULT_INSTRUCTION_NAME)
    parser.add_argument("--reranker_instruction", default=DEFAULT_RERANK_INSTRUCTION)
    parser.add_argument("--llm_base_url", default=None)
    parser.add_argument("--llm_api_key", default=None)
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--llm_temperature", type=float, default=DEFAULT_CHECKER_TEMPERATURE)
    parser.add_argument("--llm_max_tokens", type=int, default=DEFAULT_CHECKER_MAX_TOKENS)
    parser.add_argument("--llm_timeout", type=float, default=DEFAULT_CHECKER_TIMEOUT)
    parser.add_argument("--context_preview_chars", type=int, default=DEFAULT_CONTEXT_PREVIEW_CHARS)
    parser.add_argument(
        "--model_context_window_tokens",
        type=int,
        default=DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS,
    )
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--max_queries_per_round", type=int, default=DEFAULT_TRACEABLE_MAX_NEXT_QUERIES)
    parser.add_argument(
        "--max_raw_chunks_per_checker_call",
        type=int,
        default=DEFAULT_MAX_RAW_CHUNKS_PER_CHECKER_CALL,
    )
    parser.add_argument("--max_new_raw_chunks_per_round", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qids", type=parse_qids, default=None)
    args = parser.parse_args()

    try:
        run_evidence_loop_eval(
            rerank_run_dir=args.rerank_run_dir,
            output_dir=args.output_dir,
            index_dir=args.index_dir,
            dense_model_name=args.dense_model_name,
            local_files_only=args.local_files_only,
            dense_worker_mode=args.dense_worker_mode,
            dense_query_batch_size=args.dense_query_batch_size,
            device=args.device,
            branch_top_k_chunks=args.branch_top_k_chunks,
            second_hop_top_k_chunks=args.second_hop_top_k_chunks,
            rrf_k=args.rrf_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            reranker_model_name=args.reranker_model_name,
            reranker_local_files_only=args.reranker_local_files_only,
            reranker_batch_size=args.reranker_batch_size,
            reranker_max_length=args.reranker_max_length,
            reranker_instruction_name=args.reranker_instruction_name,
            reranker_instruction=args.reranker_instruction,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            llm_timeout=args.llm_timeout,
            context_preview_chars=args.context_preview_chars,
            model_context_window_tokens=args.model_context_window_tokens,
            max_rounds=args.max_rounds,
            max_queries_per_round=args.max_queries_per_round,
            max_raw_chunks_per_checker_call=args.max_raw_chunks_per_checker_call,
            max_new_raw_chunks_per_round=args.max_new_raw_chunks_per_round,
            limit=args.limit,
            qids=args.qids,
            console=Console(),
        )
    except EvidenceLoopEvalError as exc:
        Console().print(f"[bold red]EvidenceContext loop eval failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
