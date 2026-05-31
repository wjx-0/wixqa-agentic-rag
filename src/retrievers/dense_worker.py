from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.retrievers.faiss_store import FaissVectorStore


VALID_WORKER_MODES = {"full", "model_only"}


class DenseWorkerError(RuntimeError):
    pass


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def configure_offline_mode(local_files_only: bool) -> None:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_model(model_name: str, local_files_only: bool, device: str | None) -> Any:
    configure_offline_mode(local_files_only)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise DenseWorkerError(
            "Missing dependency `sentence-transformers`. Install project dependencies with "
            "`pip install -r requirements.txt` and retry."
        ) from exc

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if device:
        kwargs["device"] = device
    try:
        return SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        if local_files_only:
            raise DenseWorkerError(
                f"Could not load local embedding model {model_name!r}. "
                "Make sure it is cached locally, or rerun with `--local_files_only false`."
            ) from exc
        raise DenseWorkerError(f"Could not load embedding model {model_name!r}: {exc}") from exc


def run_worker(
    *,
    index_dir: Path,
    model_name: str,
    local_files_only: bool,
    device: str | None,
    mode: str,
) -> None:
    if mode not in VALID_WORKER_MODES:
        raise DenseWorkerError(f"Unsupported worker mode {mode!r}.")

    model = load_model(model_name, local_files_only, device)
    store = None
    if mode == "full":
        store = FaissVectorStore(
            index_dir / "faiss.index",
            index_dir / "chunk_metadata.jsonl",
        )

    send_message({"type": "ready", "mode": mode})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request: dict[str, Any] | None = None
        try:
            request = json.loads(line)
            request_type = request.get("type")
            if request_type == "close":
                send_message({"type": "closed"})
                return
            if request_type != "search":
                raise DenseWorkerError(f"Unsupported request type {request_type!r}.")
            response = search_batch(model=model, store=store, mode=mode, request=request)
            send_message(response)
        except Exception as exc:
            send_message(
                {
                    "type": "error",
                    "request_id": request.get("request_id") if request else None,
                    "message": str(exc),
                }
            )


def search_batch(
    *,
    model: Any,
    store: FaissVectorStore | None,
    mode: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    queries = request.get("queries") or []
    top_k_chunks = int(request.get("top_k_chunks") or 0)
    batch_size = int(request.get("batch_size") or 0)
    if not isinstance(queries, list) or not all(isinstance(query, str) for query in queries):
        raise DenseWorkerError("queries must be a list of strings.")
    if top_k_chunks <= 0:
        raise DenseWorkerError("top_k_chunks must be a positive integer.")
    if batch_size <= 0:
        raise DenseWorkerError("batch_size must be a positive integer.")

    embeddings = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if mode == "model_only":
        return {
            "type": "search_result",
            "request_id": request.get("request_id"),
            "embeddings": embeddings.tolist(),
        }
    if store is None:
        raise DenseWorkerError("FAISS store is not loaded in full worker mode.")
    return {
        "type": "search_result",
        "request_id": request.get("request_id"),
        "results": [store.search(embedding, top_k=top_k_chunks) for embedding in embeddings],
    }


def send_message(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Long-lived Dense FAISS retrieval worker.")
    parser.add_argument("--index_dir", default="indexes/faiss_bge_m3")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--local_files_only", type=parse_bool, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=sorted(VALID_WORKER_MODES), default="full")
    args = parser.parse_args()

    try:
        run_worker(
            index_dir=Path(args.index_dir),
            model_name=args.model_name,
            local_files_only=args.local_files_only,
            device=args.device,
            mode=args.mode,
        )
    except Exception as exc:
        print(f"Dense worker failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
