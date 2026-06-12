from __future__ import annotations

from typing import Any

from src.agentic.evidence_context import EvidenceItem


def group_candidate_rows_by_query_id(
    candidate_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_query_id: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        query_id = row.get("second_hop_query_id")
        if query_id:
            rows_by_query_id.setdefault(str(query_id), []).append(row)
            continue
        for fallback_query_id in row.get("second_hop_query_ids") or []:
            rows_by_query_id.setdefault(str(fallback_query_id), []).append(row)
    return rows_by_query_id


def dedupe_ranked_candidate_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_chunk_id: dict[str, dict[str, Any]] = {}
    for sequence_index, row in enumerate(candidate_rows):
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        candidate = {**row, "_selection_sequence_index": sequence_index}
        current = best_by_chunk_id.get(chunk_id)
        if current is None or ranked_candidate_is_better(candidate, current):
            best_by_chunk_id[chunk_id] = candidate
    ranked_rows = sorted(best_by_chunk_id.values(), key=ranked_candidate_sort_key)
    output = []
    for rank, row in enumerate(ranked_rows, start=1):
        cleaned = {key: value for key, value in row.items() if key != "_selection_sequence_index"}
        output.append({**cleaned, "rank": rank})
    return output


def select_new_candidate_rows(
    candidate_rows: list[dict[str, Any]],
    *,
    config: Any,
) -> list[dict[str, Any]]:
    rows_by_query = group_candidate_rows_by_query_id(candidate_rows)
    query_groups = [
        (query_id, [row for row in rows if row in candidate_rows])
        for query_id, rows in rows_by_query.items()
    ]
    query_groups = [(query_id, rows) for query_id, rows in query_groups if rows]
    if len(query_groups) > 1:
        return select_new_candidate_rows_by_query_quota(
            candidate_rows,
            query_groups=query_groups,
            config=config,
        )
    return select_ranked_rows_with_article_limits(candidate_rows, config=config)


def select_new_candidate_rows_by_query_quota(
    candidate_rows: list[dict[str, Any]],
    *,
    query_groups: list[tuple[str, list[dict[str, Any]]]],
    config: Any,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_chunk_ids: set[str] = set()
    article_counts: dict[str, int] = {}
    per_query_quota = max(1, config.max_new_raw_chunks_per_round // len(query_groups))
    for _, rows in query_groups:
        select_rows_into_output(
            sorted(rows, key=retrieval_rank_sort_key),
            selected=selected,
            selected_chunk_ids=selected_chunk_ids,
            article_counts=article_counts,
            config=config,
            max_additions=1,
        )
        select_rows_into_output(
            rows,
            selected=selected,
            selected_chunk_ids=selected_chunk_ids,
            article_counts=article_counts,
            config=config,
            max_additions=per_query_quota,
        )
        if len(selected) == config.max_new_raw_chunks_per_round:
            return selected
    select_rows_into_output(
        candidate_rows,
        selected=selected,
        selected_chunk_ids=selected_chunk_ids,
        article_counts=article_counts,
        config=config,
        max_additions=config.max_new_raw_chunks_per_round - len(selected),
    )
    return selected


def retrieval_rank_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    try:
        retrieval_rank = int(row.get("second_hop_retrieval_rank") or 1_000_000)
    except (TypeError, ValueError):
        retrieval_rank = 1_000_000
    try:
        rerank_rank = int(row.get("rank") or 1_000_000)
    except (TypeError, ValueError):
        rerank_rank = 1_000_000
    return (retrieval_rank, rerank_rank, str(row.get("chunk_id") or ""))


def select_ranked_rows_with_article_limits(
    candidate_rows: list[dict[str, Any]],
    *,
    config: Any,
) -> list[dict[str, Any]]:
    selected = []
    article_counts: dict[str, int] = {}
    selected_chunk_ids: set[str] = set()
    select_rows_into_output(
        candidate_rows,
        selected=selected,
        selected_chunk_ids=selected_chunk_ids,
        article_counts=article_counts,
        config=config,
        max_additions=config.max_new_raw_chunks_per_round,
    )
    return selected


def select_rows_into_output(
    rows: list[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    selected_chunk_ids: set[str],
    article_counts: dict[str, int],
    config: Any,
    max_additions: int,
) -> None:
    additions = 0
    for row in rows:
        if additions >= max_additions or len(selected) == config.max_new_raw_chunks_per_round:
            break
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id or chunk_id in selected_chunk_ids:
            continue
        article_id = str(row.get("article_id") or "")
        if article_id:
            count = article_counts.get(article_id, 0)
            if count >= config.max_new_chunks_per_article:
                continue
            article_counts[article_id] = count + 1
        selected_chunk_ids.add(chunk_id)
        selected.append(row)
        additions += 1


def ranked_candidate_is_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    return ranked_candidate_sort_key(candidate) < ranked_candidate_sort_key(current)


def ranked_candidate_sort_key(row: dict[str, Any]) -> tuple[float, int, int, str]:
    return (
        -candidate_score(row),
        int(row.get("rank") or 1_000_000),
        int(row.get("_selection_sequence_index") or 0),
        str(row.get("chunk_id") or ""),
    )


def candidate_score(row: dict[str, Any]) -> float:
    score = row.get("rerank_score", row.get("score"))
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def pack_visible_items(
    *,
    previous_visible_items: list[EvidenceItem],
    selected_new_items: list[EvidenceItem],
    llm_call_id: str,
    config: Any,
    priority_chunk_ids: list[str] | None = None,
) -> list[EvidenceItem]:
    anchors = previous_visible_items[:10]
    older_context = previous_visible_items[10:]
    priority_chunk_id_set = set(priority_chunk_ids or [])
    priority_context = [item for item in older_context if item.chunk_id in priority_chunk_id_set]
    regular_context = [item for item in older_context if item.chunk_id not in priority_chunk_id_set]
    priority_new_items = [
        item for item in selected_new_items if item.chunk_id in priority_chunk_id_set
    ]
    regular_new_items = [
        item for item in selected_new_items if item.chunk_id not in priority_chunk_id_set
    ]
    visible_items = dedupe_items(
        [*anchors, *priority_context, *priority_new_items, *regular_new_items, *regular_context]
    )
    visible_items = select_diverse_items(
        visible_items,
        max_items=config.max_raw_chunks_per_checker_call,
        max_chunks_per_article=config.max_visible_chunks_per_article,
    )
    for item in visible_items:
        if llm_call_id not in item.visible_to_llm_call_ids:
            item.visible_to_llm_call_ids.append(llm_call_id)
    return visible_items


def dedupe_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    output = []
    seen = set()
    for item in items:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        output.append(item)
    return output


def select_diverse_items(
    items: list[EvidenceItem],
    *,
    max_items: int,
    max_chunks_per_article: int,
) -> list[EvidenceItem]:
    output = []
    article_counts: dict[str, int] = {}
    for item in items:
        count = article_counts.get(item.article_id, 0)
        if count >= max_chunks_per_article:
            continue
        article_counts[item.article_id] = count + 1
        output.append(item)
        if len(output) == max_items:
            break
    return output
