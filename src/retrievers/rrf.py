from __future__ import annotations

import math
from typing import Any


DEFAULT_BM25_WEIGHT = 1.0
DEFAULT_DENSE_WEIGHT = 1.0


class RRFFusionError(ValueError):
    pass


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
    top_k_chunks: int = 50,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
) -> list[dict[str, Any]]:
    if rrf_k <= 0:
        raise RRFFusionError("rrf_k must be a positive integer.")
    if top_k_chunks <= 0:
        raise RRFFusionError("top_k_chunks must be a positive integer.")
    validate_weight("bm25_weight", bm25_weight)
    validate_weight("dense_weight", dense_weight)

    fused: dict[str, dict[str, Any]] = {}
    add_branch_results(
        fused,
        bm25_results,
        source="bm25",
        source_weight=bm25_weight,
        rrf_k=rrf_k,
    )
    add_branch_results(
        fused,
        dense_results,
        source="dense",
        source_weight=dense_weight,
        rrf_k=rrf_k,
    )

    ranked = sorted(
        fused.values(),
        key=lambda result: (
            -float(result["rrf_score"]),
            int(result["best_source_rank"]),
            str(result["chunk_id"]),
        ),
    )[:top_k_chunks]

    results = []
    for rank, result in enumerate(ranked, start=1):
        results.append(
            {
                "rank": rank,
                "score": result["rrf_score"],
                "rrf_score": result["rrf_score"],
                "chunk_id": result["chunk_id"],
                "article_id": result.get("article_id"),
                "chunk_index": result.get("chunk_index"),
                "title": result.get("title"),
                "url": result.get("url"),
                "text_preview": result.get("text_preview"),
                "contents_preview": result.get("contents_preview"),
                "bm25_rank": result.get("bm25_rank"),
                "dense_rank": result.get("dense_rank"),
                "bm25_score": result.get("bm25_score"),
                "dense_score": result.get("dense_score"),
                "sources": sorted(result["sources"]),
            }
        )
    return results


def add_branch_results(
    fused: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    source: str,
    source_weight: float,
    rrf_k: int,
) -> None:
    seen_chunk_ids: set[str] = set()
    for fallback_rank, result in enumerate(results, start=1):
        chunk_id = result.get("chunk_id")
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        rank = int(result.get("rank") or fallback_rank)
        current = fused.setdefault(
            chunk_id,
            {
                **result,
                "chunk_id": chunk_id,
                "rrf_score": 0.0,
                "best_source_rank": rank,
                "sources": set(),
            },
        )
        current["rrf_score"] += source_weight / (rrf_k + rank)
        current["best_source_rank"] = min(int(current["best_source_rank"]), rank)
        current["sources"].add(source)
        current[f"{source}_rank"] = rank
        current[f"{source}_score"] = result.get("score")
        if not current.get("text_preview") and result.get("text_preview"):
            current["text_preview"] = result["text_preview"]
        if not current.get("contents_preview") and result.get("contents_preview"):
            current["contents_preview"] = result["contents_preview"]


def validate_weight(name: str, weight: float) -> None:
    if not math.isfinite(weight) or weight <= 0:
        raise RRFFusionError(f"{name} must be a positive finite number.")
