from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io_utils import read_jsonl
from src.utils.text_utils import preview_text


class FaissStoreError(RuntimeError):
    pass


class FaissVectorStore:
    def __init__(self, index_path: str | Path, metadata_path: str | Path):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = self._load_index(self.index_path)
        self.metadata = self._load_metadata(self.metadata_path)
        self._validate_metadata_count()

    @staticmethod
    def _load_index(index_path: Path) -> Any:
        if not index_path.exists():
            raise FaissStoreError(
                f"Missing FAISS index: {index_path}. Please run `python scripts/build_faiss_index.py` first."
            )
        try:
            import faiss
        except ImportError as exc:
            raise FaissStoreError(
                "Missing dependency `faiss-cpu`. Install project dependencies with "
                "`pip install -r requirements.txt` and retry."
            ) from exc
        try:
            return faiss.read_index(str(index_path))
        except Exception as exc:
            raise FaissStoreError(f"Could not read FAISS index {index_path}: {exc}") from exc

    @staticmethod
    def _load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
        if not metadata_path.exists():
            raise FaissStoreError(
                f"Missing FAISS metadata: {metadata_path}. Please run `python scripts/build_faiss_index.py` first."
            )
        rows = list(read_jsonl(metadata_path))
        if not rows:
            raise FaissStoreError(f"No chunk metadata found in {metadata_path}.")
        return rows

    def _validate_metadata_count(self) -> None:
        if self.index.ntotal != len(self.metadata):
            raise FaissStoreError(
                "FAISS index vector count does not match metadata count: "
                f"index={self.index.ntotal}, metadata={len(self.metadata)}."
            )

    @property
    def embedding_dim(self) -> int:
        return int(self.index.d)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def search(self, query_embedding: Any, top_k: int = 100) -> list[dict[str, Any]]:
        if top_k <= 0 or self.ntotal == 0:
            return []

        try:
            import numpy as np
        except ImportError as exc:
            raise FaissStoreError(
                "Missing dependency `numpy`. Install project dependencies with "
                "`pip install -r requirements.txt` and retry."
            ) from exc

        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise FaissStoreError(
                f"Query embedding must have shape [1, dim], got {tuple(query.shape)}."
            )
        if query.shape[1] != self.embedding_dim:
            raise FaissStoreError(
                f"Query embedding dim {query.shape[1]} does not match index dim {self.embedding_dim}."
            )

        search_k = min(top_k, self.ntotal)
        scores, indexes = self.index.search(query, search_k)
        results: list[dict[str, Any]] = []
        for rank, (score, row_index) in enumerate(zip(scores[0], indexes[0]), start=1):
            row_index = int(row_index)
            if row_index < 0:
                continue
            chunk = self.metadata[row_index]
            text = chunk.get("text") or ""
            contents = chunk.get("contents") or ""
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk.get("chunk_id"),
                    "article_id": chunk.get("article_id"),
                    "title": chunk.get("title"),
                    "url": chunk.get("url"),
                    "chunk_index": chunk.get("chunk_index"),
                    "text_preview": preview_text(text, 300),
                    "contents_preview": preview_text(contents, 300),
                    "metadata": chunk.get("metadata") or {},
                }
            )
        return results
