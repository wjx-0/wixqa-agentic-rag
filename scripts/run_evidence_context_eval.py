#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.evaluation.run_evidence_context_eval import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RERANK_RUN_DIR,
    EvidenceContextEvalError,
    run_evidence_context_eval,
)
from src.agentic.context_budget import DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build traceable initial EvidenceContext objects from reranker artifacts."
    )
    parser.add_argument("--rerank_run_dir", default=DEFAULT_RERANK_RUN_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--active_top_k_chunks", type=int, default=10)
    parser.add_argument("--model_context_window_tokens", type=int, default=DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS)
    args = parser.parse_args()

    try:
        run_evidence_context_eval(
            rerank_run_dir=args.rerank_run_dir,
            output_dir=args.output_dir,
            active_top_k_chunks=args.active_top_k_chunks,
            model_context_window_tokens=args.model_context_window_tokens,
            console=Console(),
        )
    except EvidenceContextEvalError as exc:
        Console().print(f"[bold red]EvidenceContext eval failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
