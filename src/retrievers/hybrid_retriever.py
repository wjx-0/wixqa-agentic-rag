from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.schema import KBChunk
from src.retrievers.chunk_bm25_retriever import ChunkBM25Retriever
from src.retrievers.dense_worker_client import DenseWorkerClient, DenseWorkerClientError
from src.retrievers.rrf import RRFFusionError, reciprocal_rank_fusion


class HybridRetrieverError(RuntimeError):
    pass


class HybridRetriever:
    def __init__(
        self,
        *,
        chunks: list[KBChunk],
        index_dir: str | Path,
        model_name: str,
        local_files_only: bool,
        device: str | None,
        dense_worker_mode: str,
    ):
        self.bm25 = ChunkBM25Retriever(chunks)
        self.dense = DenseWorkerClient(
            index_dir=index_dir,
            model_name=model_name,
            local_files_only=local_files_only,
            device=device,
            mode=dense_worker_mode,
        )

    def __enter__(self) -> HybridRetriever:
        try:
            self.dense.start()
        except Exception as exc:
            raise HybridRetrieverError(str(exc)) from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.dense.close()

    def search_batch(
        self,
        queries: list[str],
        *,
        branch_top_k_chunks: int,
        fused_top_k_chunks: int,
        rrf_k: int,
        bm25_weight: float,
        dense_weight: float,
        dense_query_batch_size: int,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        try:
            dense_results = self.dense.search(
                queries,
                top_k_chunks=branch_top_k_chunks,
                query_batch_size=dense_query_batch_size,
            )
        except DenseWorkerClientError as exc:
            raise HybridRetrieverError(str(exc)) from exc

        if len(dense_results) != len(queries):
            raise HybridRetrieverError(
                "Dense worker result count does not match query count: "
                f"results={len(dense_results)}, queries={len(queries)}."
            )

        batches = []
        for query, query_dense_results in zip(queries, dense_results):
            bm25_results = self.bm25.search(query, top_k_chunks=branch_top_k_chunks)
            try:
                hybrid_results = reciprocal_rank_fusion(
                    bm25_results,
                    query_dense_results,
                    rrf_k=rrf_k,
                    top_k_chunks=fused_top_k_chunks,
                    bm25_weight=bm25_weight,
                    dense_weight=dense_weight,
                )
            except RRFFusionError as exc:
                raise HybridRetrieverError(str(exc)) from exc
            batches.append(
                {
                    "bm25": bm25_results,
                    "dense": query_dense_results,
                    "hybrid": hybrid_results,
                }
            )
        return batches
