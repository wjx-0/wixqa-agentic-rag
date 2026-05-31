#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.data.chunk_wixqa import DEFAULT_CHUNK_TOKENIZER_NAME
from src.data.schema import KBChunk
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME
from src.utils.io_utils import ensure_dir, model_to_dict, read_jsonl, write_json, write_jsonl


class FaissIndexBuildError(RuntimeError):
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


def configure_hf_offline_mode(local_files_only: bool) -> None:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_chunks(chunks_path: Path) -> list[KBChunk]:
    if not chunks_path.exists():
        raise FaissIndexBuildError(
            f"Missing chunk file: {chunks_path}. Please run `python scripts/prepare_chunks.py` first."
        )
    chunks = [KBChunk(**row) for row in read_jsonl(chunks_path) if (row.get("text") or "").strip()]
    if not chunks:
        raise FaissIndexBuildError(f"No usable chunks found in {chunks_path}.")
    return chunks


def validate_chunks(
    chunks: list[KBChunk],
    *,
    model_name: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> None:
    tokenizers = {chunk.metadata.get("chunk_tokenizer") for chunk in chunks}
    if tokenizers != {model_name}:
        raise FaissIndexBuildError(
            "Chunk tokenizer does not match embedding model. "
            f"expected={model_name!r}, observed={sorted(str(value) for value in tokenizers)}. "
            "Please regenerate chunks with `python scripts/prepare_chunks.py`."
        )

    size_values = {chunk.metadata.get("chunk_size_tokens") for chunk in chunks}
    overlap_values = {chunk.metadata.get("chunk_overlap_tokens") for chunk in chunks}
    if size_values != {chunk_size_tokens} or overlap_values != {chunk_overlap_tokens}:
        raise FaissIndexBuildError(
            "Chunk size/overlap does not match the requested index config. "
            f"expected={chunk_size_tokens}/{chunk_overlap_tokens}, "
            f"observed_size={sorted(str(value) for value in size_values)}, "
            f"observed_overlap={sorted(str(value) for value in overlap_values)}. "
            "Please regenerate chunks with matching parameters."
        )

    duplicate_chunk_ids = len(chunks) - len({chunk.chunk_id for chunk in chunks})
    if duplicate_chunk_ids:
        raise FaissIndexBuildError(f"Found {duplicate_chunk_ids} duplicate chunk_id values.")


def load_embedding_model(
    *,
    model_name: str,
    local_files_only: bool,
    device: str | None,
) -> Any:
    configure_hf_offline_mode(local_files_only)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise FaissIndexBuildError(
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
            raise FaissIndexBuildError(
                f"Could not load local embedding model {model_name!r}. "
                "Make sure it is cached locally, or rerun with `--local_files_only false`."
            ) from exc
        raise FaissIndexBuildError(f"Could not load embedding model {model_name!r}: {exc}") from exc


def encode_chunks(
    model: Any,
    chunks: list[KBChunk],
    *,
    batch_size: int,
) -> Any:
    if batch_size <= 0:
        raise FaissIndexBuildError("--batch_size must be a positive integer.")

    texts = [chunk.text for chunk in tqdm(chunks, desc="collect chunk texts")]
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    except Exception as exc:
        raise FaissIndexBuildError(f"Could not encode chunk texts: {exc}") from exc

    try:
        import numpy as np
    except ImportError as exc:
        raise FaissIndexBuildError(
            "Missing dependency `numpy`. Install project dependencies with "
            "`pip install -r requirements.txt` and retry."
        ) from exc

    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise FaissIndexBuildError(
            "Unexpected embedding shape: "
            f"shape={tuple(embeddings.shape)}, chunks={len(chunks)}."
        )
    return embeddings


def write_faiss_index(embeddings: Any, index_path: Path) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise FaissIndexBuildError(
            "Missing dependency `faiss-cpu`. Install project dependencies with "
            "`pip install -r requirements.txt` and retry."
        ) from exc

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    faiss.write_index(index, str(index_path))


def build_index_config(
    *,
    model_name: str,
    chunks: list[KBChunk],
    chunks_path: Path,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "tokenizer_name": DEFAULT_CHUNK_TOKENIZER_NAME,
        "chunking_strategy": "token",
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "chunks_path": str(chunks_path),
        "num_chunks": len(chunks),
        "num_chunk_articles": len({chunk.article_id for chunk in chunks}),
        "embedding_dim": embedding_dim,
        "normalize_embeddings": True,
        "faiss_index_type": "IndexFlatIP",
    }


def build_faiss_index(
    *,
    chunks_path: str | Path,
    index_dir: str | Path,
    model_name: str,
    local_files_only: bool,
    batch_size: int,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    device: str | None,
    console: Console,
) -> dict[str, Any]:
    chunks_path = Path(chunks_path)
    index_dir = ensure_dir(index_dir)

    chunks = load_chunks(chunks_path)
    validate_chunks(
        chunks,
        model_name=model_name,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    model = load_embedding_model(
        model_name=model_name,
        local_files_only=local_files_only,
        device=device,
    )
    embeddings = encode_chunks(model, chunks, batch_size=batch_size)

    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "chunk_metadata.jsonl"
    config_path = index_dir / "index_config.json"

    write_faiss_index(embeddings, index_path)
    write_jsonl(metadata_path, (model_to_dict(chunk) for chunk in chunks))
    config = build_index_config(
        model_name=model_name,
        chunks=chunks,
        chunks_path=chunks_path,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        embedding_dim=int(embeddings.shape[1]),
    )
    write_json(config_path, config)

    print_summary(console, config, index_dir)
    return config


def print_summary(console: Console, config: dict[str, Any], index_dir: Path) -> None:
    console.print()
    console.print("[bold green]Dense FAISS index build finished[/bold green]")
    console.print()
    console.print(f"Model: {config['model_name']}")
    console.print(f"Index: FAISS {config['faiss_index_type']}")
    console.print(f"Chunks: {config['num_chunks']}")
    console.print(f"Embedding dim: {config['embedding_dim']}")
    console.print()
    console.print(f"Index saved to {index_dir}/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a FAISS index for WixQA dense retrieval.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--chunks_path", default="data/processed/wix_kb_chunks.jsonl")
    parser.add_argument("--index_dir", default="indexes/faiss_bge_m3")
    parser.add_argument("--model_name", default=DEFAULT_DENSE_MODEL_NAME)
    parser.add_argument("--local_files_only", type=parse_bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size_tokens", type=int, default=512)
    parser.add_argument("--chunk_overlap_tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    console = Console()
    chunks_path = Path(args.chunks_path)
    if args.chunks_path == parser.get_default("chunks_path"):
        chunks_path = Path(args.processed_dir) / "wix_kb_chunks.jsonl"

    try:
        build_faiss_index(
            chunks_path=chunks_path,
            index_dir=args.index_dir,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
            batch_size=args.batch_size,
            chunk_size_tokens=args.chunk_size_tokens,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            device=args.device,
            console=console,
        )
    except FaissIndexBuildError as exc:
        console.print(f"[bold red]Dense FAISS index build failed:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
