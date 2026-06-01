from __future__ import annotations

from typing import Any

from src.utils.text_utils import compact_text


DEFAULT_MAX_SEED_ARTICLES = 3
TITLE_EXPANSION_TEMPLATE = "{question} Related Wix Help Center topic: {title}"


class RuleSecondHopError(ValueError):
    pass


def build_title_expansion_queries(
    question: str,
    top_reranked_chunks: list[dict[str, Any]],
    *,
    max_seed_articles: int = DEFAULT_MAX_SEED_ARTICLES,
) -> list[dict[str, Any]]:
    normalized_question = compact_text(question)
    if not normalized_question:
        raise RuleSecondHopError("question must not be empty.")
    if max_seed_articles <= 0:
        raise RuleSecondHopError("max_seed_articles must be a positive integer.")

    queries = []
    seen_article_ids: set[str] = set()
    seen_query_texts: set[str] = set()
    for chunk in top_reranked_chunks:
        article_id = compact_text(chunk.get("article_id"))
        title = compact_text(chunk.get("title"))
        if not article_id or article_id in seen_article_ids or not title:
            continue
        seen_article_ids.add(article_id)
        query_text = TITLE_EXPANSION_TEMPLATE.format(
            question=normalized_question,
            title=title,
        )
        normalized_query_text = compact_text(query_text)
        dedup_key = normalized_query_text.casefold()
        if dedup_key in seen_query_texts:
            continue
        seen_query_texts.add(dedup_key)
        queries.append(
            {
                "query_id": f"title_expand_{len(queries) + 1}",
                "query_text": normalized_query_text,
                "seed_article_id": article_id,
                "seed_title": title,
            }
        )
        if len(queries) == max_seed_articles:
            break
    return queries


def merge_chunk_candidates(
    first_hop_candidates: list[dict[str, Any]],
    second_hop_batches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    indexes_by_chunk_id: dict[str, int] = {}

    for fallback_rank, candidate in enumerate(first_hop_candidates, start=1):
        chunk_id = require_chunk_id(candidate)
        if chunk_id in indexes_by_chunk_id:
            raise RuleSecondHopError(f"Duplicate first-hop chunk_id: {chunk_id}.")
        first_hop_rank = int(candidate.get("rank") or fallback_rank)
        indexes_by_chunk_id[chunk_id] = len(merged)
        merged.append(
            {
                **candidate,
                "first_hop_rank": first_hop_rank,
                "second_hop_query_ids": [],
                "second_hop_ranks": [],
            }
        )

    for batch in second_hop_batches:
        query_id = compact_text(batch.get("query_id"))
        if not query_id:
            raise RuleSecondHopError("Second-hop batch is missing query_id.")
        results = batch.get("results")
        if not isinstance(results, list):
            raise RuleSecondHopError(f"Second-hop batch {query_id} is missing results.")
        for fallback_rank, candidate in enumerate(results, start=1):
            chunk_id = require_chunk_id(candidate)
            second_hop_rank = int(candidate.get("rank") or fallback_rank)
            existing_index = indexes_by_chunk_id.get(chunk_id)
            if existing_index is None:
                existing_index = len(merged)
                indexes_by_chunk_id[chunk_id] = existing_index
                merged.append(
                    {
                        **candidate,
                        "first_hop_rank": None,
                        "second_hop_query_ids": [],
                        "second_hop_ranks": [],
                    }
                )
            current = merged[existing_index]
            if query_id not in current["second_hop_query_ids"]:
                current["second_hop_query_ids"].append(query_id)
                current["second_hop_ranks"].append(
                    {
                        "query_id": query_id,
                        "rank": second_hop_rank,
                    }
                )

    return [
        {
            **candidate,
            "rank": merged_rank,
            "merged_rank": merged_rank,
        }
        for merged_rank, candidate in enumerate(merged, start=1)
    ]


def require_chunk_id(candidate: dict[str, Any]) -> str:
    chunk_id = compact_text(candidate.get("chunk_id"))
    if not chunk_id:
        raise RuleSecondHopError("Chunk candidate is missing chunk_id.")
    return chunk_id
