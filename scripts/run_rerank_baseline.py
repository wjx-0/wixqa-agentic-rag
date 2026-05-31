#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.run_rerank_eval import RerankEvalError, run_rerank_eval
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_INSTRUCTION_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL_NAME,
)


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
    parser = argparse.ArgumentParser(
        description="Rerank saved Hybrid RRF chunk candidates with Qwen3."
    )
    parser.add_argument("--hybrid_run_dir", required=True)
    parser.add_argument("--chunks_path", default="data/processed/wix_kb_chunks.jsonl")
    parser.add_argument("--output_dir", default="outputs/rerank_baseline")
    parser.add_argument("--model_name", default=DEFAULT_RERANK_MODEL_NAME)
    parser.add_argument("--local_files_only", type=parse_bool, default=True)
    parser.add_argument("--rerank_batch_size", type=int, default=DEFAULT_RERANK_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--instruction_name", default=DEFAULT_INSTRUCTION_NAME)
    parser.add_argument("--instruction", default=DEFAULT_RERANK_INSTRUCTION)
    args = parser.parse_args()

    try:
        run_rerank_eval(
            hybrid_run_dir=args.hybrid_run_dir,
            chunks_path=args.chunks_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
            rerank_batch_size=args.rerank_batch_size,
            max_length=args.max_length,
            device=args.device,
            instruction_name=args.instruction_name,
            instruction=args.instruction,
            console=Console(),
        )
    except RerankEvalError as exc:
        Console().print(f"[bold red]Reranker baseline failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
