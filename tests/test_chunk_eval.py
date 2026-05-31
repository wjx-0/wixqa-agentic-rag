from __future__ import annotations

import unittest

from src.evaluation.chunk_eval import (
    classify_chunk_case,
    compute_chunk_retrieval_metrics,
)
from src.evaluation.eval_utils import CASE_INVALID
from src.evaluation.run_hybrid_rrf_eval import gold_article_first_chunk_rank


def result(chunk_id: str, article_id: str, rank: int) -> dict:
    return {"chunk_id": chunk_id, "article_id": article_id, "rank": rank}


class ChunkEvalTest(unittest.TestCase):
    def test_metrics_include_unique_articles_and_duplicate_ratio(self) -> None:
        metrics = compute_chunk_retrieval_metrics(
            ["gold_a", "gold_b"],
            [
                result("a1", "gold_a", 1),
                result("a2", "gold_a", 2),
                result("c1", "other", 3),
                result("b1", "gold_b", 4),
            ],
            [3, 4],
        )

        self.assertEqual(metrics["chunk_hit@3"], 1)
        self.assertEqual(metrics["chunk_full_article_hit@3"], 0)
        self.assertEqual(metrics["chunk_full_article_hit@4"], 1)
        self.assertEqual(metrics["unique_articles@3_chunks"], 2)
        self.assertAlmostEqual(metrics["duplicate_article_ratio@3_chunks"], 1 - 2 / 3)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_duplicate_ratio_uses_actual_returned_chunks(self) -> None:
        metrics = compute_chunk_retrieval_metrics(
            ["gold"],
            [result("a1", "gold", 1), result("a2", "gold", 2)],
            [5],
        )

        self.assertAlmostEqual(metrics["duplicate_article_ratio@5_chunks"], 0.5)

    def test_dynamic_case_labels(self) -> None:
        self.assertEqual(
            classify_chunk_case(["gold"], ["gold"], 50),
            "A_top10_chunks_full",
        )
        self.assertEqual(
            classify_chunk_case(["gold"], ["other"] * 10 + ["gold"], 50),
            "B_top50_chunks_full_not_top10_chunks",
        )
        self.assertEqual(
            classify_chunk_case(["gold"], ["other"] * 100, 100),
            "C_top100_chunks_not_full",
        )
        self.assertEqual(classify_chunk_case([], [], 50), CASE_INVALID)

    def test_gold_article_first_rank(self) -> None:
        ranks = gold_article_first_chunk_rank(
            ["gold_a", "gold_b"],
            [
                result("x", "other", 1),
                result("a", "gold_a", 2),
                result("a2", "gold_a", 3),
            ],
        )

        self.assertEqual(ranks, {"gold_a": 2, "gold_b": None})


if __name__ == "__main__":
    unittest.main()
