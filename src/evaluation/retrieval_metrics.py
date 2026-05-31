from __future__ import annotations


DEFAULT_KS = [1, 3, 5, 10, 20, 50]


def compute_retrieval_metrics(
    gold_article_ids: list[str],
    retrieved_article_ids: list[str],
    ks: list[int],
) -> dict:
    gold = [article_id for article_id in gold_article_ids if article_id]
    retrieved = [article_id for article_id in retrieved_article_ids if article_id]
    gold_set = set(gold)

    metrics: dict[str, float | int | bool] = {"is_valid": bool(gold_set)}
    if not gold_set:
        for k in ks:
            metrics[f"article_hit@{k}"] = 0
            metrics[f"article_full_hit@{k}"] = 0
            metrics[f"article_recall@{k}"] = 0.0
            metrics[f"article_precision@{k}"] = 0.0
        metrics["mrr"] = 0.0
        return metrics

    for k in ks:
        top_k = retrieved[:k]
        hit_count = len(set(top_k) & gold_set)
        metrics[f"article_hit@{k}"] = int(hit_count > 0)
        metrics[f"article_full_hit@{k}"] = int(gold_set.issubset(set(top_k)))
        metrics[f"article_recall@{k}"] = hit_count / len(gold_set)
        metrics[f"article_precision@{k}"] = hit_count / k

    reciprocal_rank = 0.0
    for rank, article_id in enumerate(retrieved, start=1):
        if article_id in gold_set:
            reciprocal_rank = 1.0 / rank
            break
    metrics["mrr"] = reciprocal_rank
    return metrics

