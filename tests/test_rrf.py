from __future__ import annotations

import unittest

from src.retrievers.rrf import RRFFusionError, reciprocal_rank_fusion


def result(chunk_id: str, rank: int, score: float = 1.0) -> dict:
    return {
        "chunk_id": chunk_id,
        "article_id": f"article_{chunk_id}",
        "chunk_index": 0,
        "rank": rank,
        "score": score,
    }


class ReciprocalRankFusionTest(unittest.TestCase):
    def test_accumulates_scores_for_shared_chunks(self) -> None:
        fused = reciprocal_rank_fusion(
            [result("shared", 1), result("bm25", 2)],
            [result("dense", 1), result("shared", 2)],
            rrf_k=60,
            top_k_chunks=3,
        )

        self.assertEqual(fused[0]["chunk_id"], "shared")
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61 + 1 / 62)
        self.assertEqual(fused[0]["sources"], ["bm25", "dense"])
        self.assertEqual(fused[0]["bm25_rank"], 1)
        self.assertEqual(fused[0]["dense_rank"], 2)

    def test_applies_branch_weights(self) -> None:
        fused = reciprocal_rank_fusion(
            [result("bm25", 1)],
            [result("dense", 1)],
            rrf_k=60,
            top_k_chunks=2,
            bm25_weight=1.0,
            dense_weight=2.0,
        )

        self.assertEqual([row["chunk_id"] for row in fused], ["dense", "bm25"])
        self.assertAlmostEqual(fused[0]["rrf_score"], 2 / 61)

    def test_branch_duplicates_only_count_once(self) -> None:
        fused = reciprocal_rank_fusion(
            [result("shared", 1), result("shared", 2)],
            [],
            rrf_k=60,
            top_k_chunks=1,
        )

        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61)

    def test_ties_use_best_rank_then_chunk_id(self) -> None:
        fused = reciprocal_rank_fusion(
            [result("b", 1), result("a", 1)],
            [],
            rrf_k=60,
            top_k_chunks=2,
        )

        self.assertEqual([row["chunk_id"] for row in fused], ["a", "b"])

    def test_top_k_truncates_results(self) -> None:
        fused = reciprocal_rank_fusion(
            [result("a", 1), result("b", 2)],
            [result("c", 1)],
            top_k_chunks=2,
        )

        self.assertEqual(len(fused), 2)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(RRFFusionError):
            reciprocal_rank_fusion([], [], rrf_k=0)
        with self.assertRaises(RRFFusionError):
            reciprocal_rank_fusion([], [], top_k_chunks=0)
        with self.assertRaises(RRFFusionError):
            reciprocal_rank_fusion([], [], bm25_weight=0)
        with self.assertRaises(RRFFusionError):
            reciprocal_rank_fusion([], [], dense_weight=float("nan"))


if __name__ == "__main__":
    unittest.main()
