#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic.runtime_factory import DialogueRuntimeConfig
from src.retrievers.dense_worker_client import VALID_WORKER_MODES
from src.web.server import build_server


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
    parser = argparse.ArgumentParser(description="Run the WixQA dialogue web console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=["real", "demo"], default="real")
    parser.add_argument("--reranker_provider", choices=["auto", "dashscope", "local"], default="auto")
    parser.add_argument("--dense_worker_mode", choices=sorted(VALID_WORKER_MODES), default="model_only")
    parser.add_argument("--local_files_only", type=parse_bool, default=True)
    parser.add_argument("--branch_top_k_chunks", type=int, default=50)
    parser.add_argument("--initial_rerank_top_k_chunks", type=int, default=50)
    parser.add_argument("--checker_max_tokens", type=int, default=1024)
    parser.add_argument("--checker_retry_attempts", type=int, default=2)
    parser.add_argument("--answer_max_tokens", type=int, default=1600)
    parser.add_argument("--enable_api_intent_gate", type=parse_bool, default=True)
    parser.add_argument("--intent_gate_max_tokens", type=int, default=512)
    parser.add_argument("--enable_api_query_rewrite", type=parse_bool, default=True)
    parser.add_argument("--query_rewrite_max_tokens", type=int, default=512)
    args = parser.parse_args()

    config = None
    if args.mode == "real":
        config = DialogueRuntimeConfig(
            root=ROOT,
            local_files_only=args.local_files_only,
            dense_worker_mode=args.dense_worker_mode,
            branch_top_k_chunks=args.branch_top_k_chunks,
            initial_rerank_top_k_chunks=args.initial_rerank_top_k_chunks,
            reranker_provider=args.reranker_provider,
            checker_max_tokens=args.checker_max_tokens,
            checker_retry_attempts=args.checker_retry_attempts,
            answer_max_tokens=args.answer_max_tokens,
            enable_api_intent_gate=args.enable_api_intent_gate,
            intent_gate_max_tokens=args.intent_gate_max_tokens,
            enable_api_query_rewrite=args.enable_api_query_rewrite,
            query_rewrite_max_tokens=args.query_rewrite_max_tokens,
        )
    server = build_server(host=args.host, port=args.port, mode=args.mode, config=config)
    url = f"http://{args.host}:{args.port}"
    print(f"WixQA dialogue web console: {url} ({args.mode})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping WixQA dialogue web console.")
    finally:
        server.runtime.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
