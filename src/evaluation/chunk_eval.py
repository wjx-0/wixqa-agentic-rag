from __future__ import annotations

from typing import Any

from src.data.schema import KBArticle, QAExample
from src.evaluation.eval_utils import CASE_INVALID
from src.utils.metrics import average


CHUNK_KS = [1, 3, 5, 10, 20, 30, 50, 100]
CASE_A_CHUNKS = "A_top10_chunks_full"


def select_chunk_ks(top_k_chunks: int) -> list[int]:
    return [k for k in CHUNK_KS if k <= top_k_chunks]


def chunk_case_labels(top_k_chunks: int) -> tuple[str, str, str]:
    return (
        CASE_A_CHUNKS,
        f"B_top{top_k_chunks}_chunks_full_not_top10_chunks",
        f"C_top{top_k_chunks}_chunks_not_full",
    )


def compute_chunk_retrieval_metrics(
    gold_article_ids: list[str],
    chunk_results: list[dict[str, Any]],
    ks: list[int],
) -> dict[str, Any]:
    gold_set = {article_id for article_id in gold_article_ids if article_id}
    chunk_article_ids = result_article_ids(chunk_results)
    metrics: dict[str, float | int | bool] = {"is_valid": bool(gold_set)}

    for k in ks:
        top_k_article_ids = chunk_article_ids[:k]
        hit_articles = set(top_k_article_ids) & gold_set
        gold_chunk_count = sum(article_id in gold_set for article_id in top_k_article_ids)
        returned_chunks = len(top_k_article_ids)
        unique_articles = len(set(top_k_article_ids))
        metrics[f"chunk_hit@{k}"] = int(bool(hit_articles))
        metrics[f"chunk_full_article_hit@{k}"] = int(
            bool(gold_set) and gold_set.issubset(set(top_k_article_ids))
        )
        metrics[f"chunk_article_recall@{k}"] = (
            len(hit_articles) / len(gold_set) if gold_set else 0.0
        )
        metrics[f"chunk_gold_rate@{k}"] = gold_chunk_count / k
        metrics[f"unique_articles@{k}_chunks"] = unique_articles
        metrics[f"duplicate_article_ratio@{k}_chunks"] = (
            1.0 - unique_articles / returned_chunks if returned_chunks else 0.0
        )

    reciprocal_rank = 0.0
    for rank, article_id in enumerate(chunk_article_ids, start=1):
        if article_id in gold_set:
            reciprocal_rank = 1.0 / rank
            break
    metrics["mrr"] = reciprocal_rank
    return metrics


def build_chunk_trace(
    example: QAExample,
    chunk_results: list[dict[str, Any]],
    metrics: dict[str, Any],
    article_lookup: dict[str, KBArticle],
    *,
    top_k_chunks: int,
) -> dict[str, Any]:
    chunk_article_ids = result_article_ids(chunk_results)
    gold_set = set(example.article_ids)
    top10_chunk_article_ids = chunk_article_ids[:10]
    top_cutoff_chunk_article_ids = chunk_article_ids[:top_k_chunks]

    trace: dict[str, Any] = {
        "qid": example.qid,
        "dataset_name": example.dataset_name,
        "question": example.question,
        "answer": example.answer,
        "gold_article_ids": example.article_ids,
        "gold_article_titles": [
            article_lookup[article_id].title
            for article_id in example.article_ids
            if article_id in article_lookup
        ],
        "num_gold_articles": example.num_gold_articles,
        "is_multi_article": example.is_multi_article,
        "top5_chunk_article_ids": chunk_article_ids[:5],
        "top10_chunk_article_ids": top10_chunk_article_ids,
        "top20_chunk_article_ids": chunk_article_ids[:20],
        "top30_chunk_article_ids": chunk_article_ids[:30],
        "top50_chunk_article_ids": chunk_article_ids[:50],
        "top100_chunk_article_ids": chunk_article_ids[:100],
        f"top{top_k_chunks}_chunk_article_ids": top_cutoff_chunk_article_ids,
        "top10_chunk_ids": [result["chunk_id"] for result in chunk_results[:10]],
        "top10_chunk_scores": [result.get("score") for result in chunk_results[:10]],
        "top_chunks_preview": [chunk_preview(result) for result in chunk_results[:10]],
        f"unique_articles_from_top{top_k_chunks}_chunks": len(
            set(top_cutoff_chunk_article_ids)
        ),
        "requested_top_k_chunks": top_k_chunks,
        "missing_articles_at_10_chunks": sorted(gold_set - set(top10_chunk_article_ids)),
        f"missing_articles_at_{top_k_chunks}_chunks": sorted(
            gold_set - set(top_cutoff_chunk_article_ids)
        ),
    }
    trace.update({key: value for key, value in metrics.items() if key != "is_valid"})
    trace["case_type"] = classify_chunk_case(
        example.article_ids,
        chunk_article_ids,
        top_k_chunks,
    )
    return trace


def classify_chunk_case(
    gold_article_ids: list[str],
    chunk_article_ids: list[str],
    top_k_chunks: int,
) -> str:
    gold_set = {article_id for article_id in gold_article_ids if article_id}
    if not gold_set:
        return CASE_INVALID
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    if gold_set.issubset(set(chunk_article_ids[:10])):
        return case_a
    if gold_set.issubset(set(chunk_article_ids[:top_k_chunks])):
        return case_b
    return case_c


def summarize_chunk_metrics(
    *,
    dataset_name: str,
    records: int,
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    ks: list[int],
    top_k_chunks: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "records": records,
        "valid_records": len(valid_rows),
        "invalid_no_gold_articles": len(invalid_rows),
        "ks": ks,
    }
    for key in chunk_metric_keys(ks):
        summary[key] = average(row[key] for row in valid_rows)

    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    summary["case_A_count"] = sum(row["case_type"] == case_a for row in valid_rows)
    summary["case_B_count"] = sum(row["case_type"] == case_b for row in valid_rows)
    summary["case_C_count"] = sum(row["case_type"] == case_c for row in valid_rows)

    groups = group_rows(valid_rows)
    summary["single_article_records"] = len(groups["single"])
    summary["multi_article_records"] = len(groups["multi"])
    for group_name, rows in groups.items():
        for k in sorted({10, top_k_chunks}):
            summary[f"{group_name}_chunk_full_article_hit@{k}"] = average(
                row.get(f"chunk_full_article_hit@{k}", 0) for row in rows
            )
            summary[f"{group_name}_chunk_article_recall@{k}"] = average(
                row.get(f"chunk_article_recall@{k}", 0.0) for row in rows
            )
    return summary


def chunk_metric_keys(ks: list[int]) -> list[str]:
    keys = []
    for prefix in (
        "chunk_hit",
        "chunk_full_article_hit",
        "chunk_article_recall",
        "chunk_gold_rate",
    ):
        keys.extend(f"{prefix}@{k}" for k in ks)
    for k in ks:
        keys.append(f"unique_articles@{k}_chunks")
        keys.append(f"duplicate_article_ratio@{k}_chunks")
    keys.append("mrr")
    return keys


def result_article_ids(chunk_results: list[dict[str, Any]]) -> list[str]:
    return [result["article_id"] for result in chunk_results if result.get("article_id")]


def chunk_preview(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": result["rank"],
        "chunk_id": result["chunk_id"],
        "article_id": result["article_id"],
        "chunk_index": result.get("chunk_index"),
        "title": result.get("title"),
        "score": result.get("score"),
        "preview": result.get("text_preview") or result.get("contents_preview"),
    }


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"single": [], "multi": []}
    for row in rows:
        grouped["multi" if row.get("is_multi_article") else "single"].append(row)
    return grouped
