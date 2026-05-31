from __future__ import annotations

import unittest

from src.evaluation.run_hybrid_rrf_eval import (
    HybridRRFEvalError,
    build_run_name,
    build_complementarity_row,
    validate_args,
)


def result(chunk_id: str, article_id: str, rank: int) -> dict:
    return {"chunk_id": chunk_id, "article_id": article_id, "rank": rank}


class HybridEvalTest(unittest.TestCase):
    def test_build_run_name_tracks_configuration(self) -> None:
        self.assertEqual(
            build_run_name(
                dataset_name="wixqa_expertwritten",
                branch_top_k_chunks=100,
                fused_top_k_chunks=50,
                rrf_k=60,
                bm25_weight=1.0,
                dense_weight=2.0,
            ),
            "hybrid_rrf_b100_f50_k60_bw1_dw2_wixqa_expertwritten",
        )

    def test_rejects_invalid_candidate_configuration(self) -> None:
        with self.assertRaises(HybridRRFEvalError):
            validate_args(
                dataset_name="wixqa_expertwritten",
                branch_top_k_chunks=50,
                fused_top_k_chunks=100,
                rrf_k=60,
                bm25_weight=1.0,
                dense_weight=1.0,
                dense_query_batch_size=16,
            )
        with self.assertRaises(HybridRRFEvalError):
            validate_args(
                dataset_name="wixqa_expertwritten",
                branch_top_k_chunks=50,
                fused_top_k_chunks=50,
                rrf_k=0,
                bm25_weight=1.0,
                dense_weight=1.0,
                dense_query_batch_size=16,
            )
        with self.assertRaises(HybridRRFEvalError):
            validate_args(
                dataset_name="wixqa_expertwritten",
                branch_top_k_chunks=50,
                fused_top_k_chunks=50,
                rrf_k=60,
                bm25_weight=1.0,
                dense_weight=0,
                dense_query_batch_size=16,
            )

    def test_complementarity_tracks_rescued_and_retained_gold_articles(self) -> None:
        results = {
            "bm25": [result(f"bm25_{index}", "other", index) for index in range(1, 11)]
            + [result("gold_bm25", "gold", 11)],
            "dense": [result(f"dense_{index}", "other", index) for index in range(1, 11)]
            + [result("gold_dense", "gold", 11)],
            "hybrid": [result("gold_bm25", "gold", 1)],
        }

        row = build_complementarity_row(
            "qid",
            ["gold"],
            results,
            fused_top_k_chunks=50,
        )

        self.assertEqual(row["hybrid_top10_rescued_gold_article_ids"], ["gold"])
        self.assertEqual(row["hybrid_top50_retained_gold_article_ids"], ["gold"])
        self.assertEqual(row["gold_article_first_chunk_rank"]["bm25"], {"gold": 11})

    def test_standard_diagnostic_adds_top100_retained_field(self) -> None:
        row = build_complementarity_row(
            "qid",
            ["gold"],
            {
                "bm25": [
                    result(f"bm25_{index}", "other", index) for index in range(1, 11)
                ]
                + [result("bm25_gold", "gold", 11)],
                "dense": [],
                "hybrid": [
                    result(f"hybrid_{index}", "other", index) for index in range(1, 75)
                ]
                + [result("hybrid_gold", "gold", 75)],
            },
            fused_top_k_chunks=100,
        )

        self.assertEqual(row["hybrid_top100_retained_gold_article_ids"], ["gold"])


if __name__ == "__main__":
    unittest.main()
