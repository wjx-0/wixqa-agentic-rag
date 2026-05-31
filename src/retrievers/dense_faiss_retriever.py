from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.retrievers.faiss_store import FaissStoreError, FaissVectorStore


DEFAULT_DENSE_MODEL_NAME = "BAAI/bge-m3"


class DenseFaissRetrieverError(RuntimeError):
    pass


class DenseFaissRetriever:
    def __init__(
        self,
        index_dir: str | Path,
        model_name: str = DEFAULT_DENSE_MODEL_NAME,
        local_files_only: bool = True,
        device: str | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.device = device
        self.model = self._load_model()
        self.store = FaissVectorStore(
            self.index_dir / "faiss.index",
            self.index_dir / "chunk_metadata.jsonl",
        )

    def _load_model(self) -> Any:
        configure_hf_offline_mode(self.local_files_only)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseFaissRetrieverError(
                "Missing dependency `sentence-transformers`. Install project dependencies with "
                "`pip install -r requirements.txt` and retry."
            ) from exc

        kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.device:
            kwargs["device"] = self.device

        try:
            return SentenceTransformer(self.model_name, **kwargs)
        except Exception as exc:
            if self.local_files_only:
                raise DenseFaissRetrieverError(
                    f"Could not load local embedding model {self.model_name!r}. "
                    "Make sure it is cached locally, or rerun with `--local_files_only false`."
                ) from exc
            raise DenseFaissRetrieverError(
                f"Could not load embedding model {self.model_name!r}: {exc}"
            ) from exc

    def search_chunks(self, query: str, top_k_chunks: int = 100) -> list[dict[str, Any]]:
        if top_k_chunks <= 0 or not (query or "").strip():
            return []
        query_embedding = self._encode_query(query)
        try:
            return self.store.search(query_embedding, top_k=top_k_chunks)
        except FaissStoreError as exc:
            raise DenseFaissRetrieverError(str(exc)) from exc

    def search_articles(
        self,
        query: str,
        top_k_articles: int = 30,
        top_k_chunks: int = 100,
    ) -> list[dict[str, Any]]:
        chunk_results = self.search_chunks(query, top_k_chunks=top_k_chunks)
        return aggregate_chunk_results(chunk_results, top_k_articles=top_k_articles)

    def _encode_query(self, query: str) -> Any:
        try:
            import numpy as np
        except ImportError as exc:
            raise DenseFaissRetrieverError(
                "Missing dependency `numpy`. Install project dependencies with "
                "`pip install -r requirements.txt` and retry."
            ) from exc

        embedding = self.model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embedding, dtype="float32")


def aggregate_chunk_results(
    chunk_results: list[dict[str, Any]],
    *,
    top_k_articles: int,
) -> list[dict[str, Any]]:
    best_by_article: dict[str, dict[str, Any]] = {}
    for result in chunk_results:
        article_id = result.get("article_id")
        if not article_id:
            continue
        existing = best_by_article.get(article_id)
        if existing is None or is_better_chunk_result(result, existing):
            best_by_article[article_id] = result

    ranked = sorted(
        best_by_article.values(),
        key=lambda result: (-float(result["score"]), int(result["rank"])),
    )[:top_k_articles]

    metadata: dict[str, Any] = {
        "unique_articles": len(best_by_article),
        "requested_top_k_articles": top_k_articles,
    }
    if len(best_by_article) < top_k_articles:
        metadata["warning"] = "unique_articles_less_than_requested"

    article_results = []
    for rank, result in enumerate(ranked, start=1):
        article_results.append(
            {
                "rank": rank,
                "article_id": result["article_id"],
                "title": result.get("title"),
                "url": result.get("url"),
                "score": result["score"],
                "best_chunk_id": result["chunk_id"],
                "best_chunk_index": result["chunk_index"],
                "best_chunk_rank": result["rank"],
                "best_chunk_preview": result.get("text_preview") or result.get("contents_preview"),
                "metadata": metadata,
            }
        )
    return article_results


def is_better_chunk_result(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_score = float(candidate.get("score", 0.0))
    current_score = float(current.get("score", 0.0))
    if candidate_score != current_score:
        return candidate_score > current_score
    return int(candidate.get("rank", 0)) < int(current.get("rank", 0))


def configure_hf_offline_mode(local_files_only: bool) -> None:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
