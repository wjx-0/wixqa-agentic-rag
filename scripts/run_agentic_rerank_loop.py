#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.run_agentic_rerank_eval import (
    DEFAULT_LLM_CHECKER_RUN_DIR,
    DEFAULT_OUTPUT_DIR,
    AgenticRerankEvalError,
    run_agentic_rerank_eval,
)
from src.evaluation.run_llm_evidence_checker_eval import DEFAULT_RULE_SECOND_HOP_RUN_DIR
from src.evaluation.run_rule_second_hop_eval import (
    DEFAULT_BASELINE_RERANK_RUN_DIR,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_FIRST_HOP_HYBRID_RUN_DIR,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_TOP100_CONTROL_RERANK_RUN_DIR,
)
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_INSTRUCTION_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL_NAME,
)
from src.retrievers.dense_worker_client import VALID_WORKER_MODES


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
        description="Run Phase 9 offline rerank over Phase 8 LLM merged pools."
    )
    parser.add_argument("--first_hop_hybrid_run_dir", default=DEFAULT_FIRST_HOP_HYBRID_RUN_DIR)
    parser.add_argument("--baseline_rerank_run_dir", default=DEFAULT_BASELINE_RERANK_RUN_DIR)
    parser.add_argument(
        "--top100_control_rerank_run_dir",
        default=DEFAULT_TOP100_CONTROL_RERANK_RUN_DIR,
    )
    parser.add_argument("--rule_second_hop_run_dir", default=DEFAULT_RULE_SECOND_HOP_RUN_DIR)
    parser.add_argument("--llm_checker_run_dir", default=DEFAULT_LLM_CHECKER_RUN_DIR)
    parser.add_argument("--chunks_path", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reranker_model_name", default=DEFAULT_RERANK_MODEL_NAME)
    parser.add_argument("--local_files_only", type=parse_bool, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rerank_batch_size", type=int, default=DEFAULT_RERANK_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--instruction_name", default=DEFAULT_INSTRUCTION_NAME)
    parser.add_argument("--instruction", default=DEFAULT_RERANK_INSTRUCTION)
    parser.add_argument(
        "--dense_worker_mode",
        choices=sorted(VALID_WORKER_MODES),
        default=None,
        help="Accepted for command compatibility; Phase 9 does not run Dense retrieval.",
    )
    args = parser.parse_args()

    try:
        run_agentic_rerank_eval(
            first_hop_hybrid_run_dir=args.first_hop_hybrid_run_dir,
            baseline_rerank_run_dir=args.baseline_rerank_run_dir,
            top100_control_rerank_run_dir=args.top100_control_rerank_run_dir,
            rule_second_hop_run_dir=args.rule_second_hop_run_dir,
            llm_checker_run_dir=args.llm_checker_run_dir,
            chunks_path=args.chunks_path,
            output_dir=args.output_dir,
            reranker_model_name=args.reranker_model_name,
            local_files_only=args.local_files_only,
            device=args.device,
            rerank_batch_size=args.rerank_batch_size,
            max_length=args.max_length,
            instruction_name=args.instruction_name,
            instruction=args.instruction,
            dense_worker_mode=args.dense_worker_mode,
            console=Console(),
        )
    except AgenticRerankEvalError as exc:
        Console().print(f"[bold red]Phase 9 agentic rerank failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
