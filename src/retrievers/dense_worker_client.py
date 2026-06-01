from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.retrievers.faiss_store import FaissVectorStore


VALID_WORKER_MODES = {"full", "model_only"}


class DenseWorkerClientError(RuntimeError):
    pass


class DenseWorkerClient:
    def __init__(
        self,
        *,
        index_dir: str | Path,
        model_name: str,
        local_files_only: bool,
        device: str | None,
        mode: str,
    ):
        if mode not in VALID_WORKER_MODES:
            raise DenseWorkerClientError(
                f"Unsupported dense worker mode {mode!r}. Choose one of: {sorted(VALID_WORKER_MODES)}"
            )
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.device = device
        self.mode = mode
        self.process: subprocess.Popen[str] | None = None
        self.store: FaissVectorStore | None = None
        self.request_id = 0

    def __enter__(self) -> DenseWorkerClient:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return
        if self.mode == "model_only":
            self.store = FaissVectorStore(
                self.index_dir / "faiss.index",
                self.index_dir / "chunk_metadata.jsonl",
            )

        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        hf_home = Path("/root/rivermind-data/models/huggingface")
        if hf_home.exists():
            env["HF_HOME"] = str(hf_home)
            env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
            env["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
        python_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(root) if not python_path else f"{root}{os.pathsep}{python_path}"
        if self.local_files_only:
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")

        command = [
            sys.executable,
            "-m",
            "src.retrievers.dense_worker",
            "--index_dir",
            str(self.index_dir),
            "--model_name",
            self.model_name,
            "--local_files_only",
            "true" if self.local_files_only else "false",
            "--mode",
            self.mode,
        ]
        if self.device:
            command.extend(["--device", self.device])
        self.process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            message = self._read_message("while starting")
        except Exception:
            self.close()
            raise
        if message.get("type") != "ready":
            self.close()
            raise DenseWorkerClientError(
                f"Dense worker returned an unexpected startup message: {message!r}"
            )

    def search(
        self,
        queries: list[str],
        *,
        top_k_chunks: int,
        query_batch_size: int,
    ) -> list[list[dict[str, Any]]]:
        if top_k_chunks <= 0:
            raise DenseWorkerClientError("top_k_chunks must be a positive integer.")
        if query_batch_size <= 0:
            raise DenseWorkerClientError("query_batch_size must be a positive integer.")
        self.start()

        all_results: list[list[dict[str, Any]]] = []
        for start in range(0, len(queries), query_batch_size):
            query_batch = queries[start : start + query_batch_size]
            self.request_id += 1
            self._write_message(
                {
                    "type": "search",
                    "request_id": self.request_id,
                    "queries": query_batch,
                    "top_k_chunks": top_k_chunks,
                    "batch_size": query_batch_size,
                }
            )
            message = self._read_message("while searching")
            if message.get("type") == "error":
                raise DenseWorkerClientError(f"Dense worker search failed: {message.get('message')}")
            if message.get("type") != "search_result":
                raise DenseWorkerClientError(
                    f"Dense worker returned an unexpected search message: {message!r}"
                )
            if message.get("request_id") != self.request_id:
                raise DenseWorkerClientError(
                    "Dense worker response request_id does not match the current request."
                )
            all_results.extend(self._decode_results(message, top_k_chunks=top_k_chunks))
        return all_results

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write('{"type":"close"}\n')
                    process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()

    def _decode_results(
        self,
        message: dict[str, Any],
        *,
        top_k_chunks: int,
    ) -> list[list[dict[str, Any]]]:
        if self.mode == "full":
            results = message.get("results")
            if not isinstance(results, list):
                raise DenseWorkerClientError("Dense worker full mode did not return results.")
            return results

        embeddings = message.get("embeddings")
        if not isinstance(embeddings, list):
            raise DenseWorkerClientError("Dense worker model_only mode did not return embeddings.")
        if self.store is None:
            raise DenseWorkerClientError("FAISS store is not loaded for model_only mode.")
        return [self.store.search(embedding, top_k=top_k_chunks) for embedding in embeddings]

    def _write_message(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise DenseWorkerClientError("Dense worker is not running.")
        try:
            self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise self._worker_stopped_error("while sending a request") from exc

    def _read_message(self, context: str) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise DenseWorkerClientError("Dense worker is not running.")
        line = self.process.stdout.readline()
        if not line:
            raise self._worker_stopped_error(context)
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise DenseWorkerClientError(
                f"Dense worker emitted invalid JSON {context}: {line.strip()!r}"
            ) from exc

    def _worker_stopped_error(self, context: str) -> DenseWorkerClientError:
        return_code = self.process.poll() if self.process is not None else None
        hint = ""
        if self.mode == "full":
            hint = (
                " If this machine crashes when SentenceTransformer and FAISS share a process, "
                "rerun with `--dense_worker_mode model_only`."
            )
        return DenseWorkerClientError(
            f"Dense worker stopped unexpectedly {context}; returncode={return_code}.{hint}"
        )
