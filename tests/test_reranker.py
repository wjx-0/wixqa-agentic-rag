from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rich.console import Console

from src.data.schema import KBChunk
from src.evaluation.run_rerank_eval import (
    RerankEvalError,
    build_rerank_diagnostics,
    build_reranker_run_name,
    load_hybrid_candidates,
    run_rerank_eval,
    validate_candidates,
)
from src.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.utils.io_utils import write_json, write_jsonl


def chunk(chunk_id: str, article_id: str) -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        article_id=article_id,
        chunk_index=0,
        text=f"title\ncontents {chunk_id}",
        contents=f"contents {chunk_id}",
        start_token=0,
        end_token=5,
        num_tokens=5,
    )


def candidate(chunk_id: str, article_id: str, rank: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "article_id": article_id,
        "rank": rank,
        "rrf_score": 0.1,
        "sources": ["dense"],
    }


class FakeModel:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


class FakeReranker:
    def rerank(self, question, candidates, chunk_lookup, *, batch_size):
        results = []
        for rank, candidate_row in enumerate(reversed(candidates), start=1):
            chunk_row = chunk_lookup[candidate_row["chunk_id"]]
            results.append(
                {
                    **candidate_row,
                    "hybrid_rank": candidate_row["rank"],
                    "rank": rank,
                    "score": float(100 - rank),
                    "rerank_score": float(100 - rank),
                    "chunk_index": chunk_row.chunk_index,
                    "title": chunk_row.title,
                }
            )
        return results


class CrossEncoderRerankerTest(unittest.TestCase):
    def test_reranks_by_score_then_hybrid_rank_then_chunk_id(self) -> None:
        model = FakeModel([0.1, 0.9, 0.9, 0.9])
        reranker = CrossEncoderReranker(model=model)
        lookup = {
            "a": chunk("a", "article_a"),
            "b": chunk("b", "article_b"),
            "c": chunk("c", "article_c"),
            "d": chunk("d", "article_d"),
        }

        ranked = reranker.rerank(
            "question",
            [
                candidate("a", "article_a", 1),
                candidate("d", "article_d", 2),
                candidate("c", "article_c", 3),
                candidate("b", "article_b", 3),
            ],
            lookup,
            batch_size=4,
        )

        self.assertEqual([row["chunk_id"] for row in ranked], ["d", "b", "c", "a"])
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3, 4])
        self.assertEqual(ranked[0]["hybrid_rank"], 2)
        self.assertEqual(model.calls[0][1]["prompt_name"], "query")

    def test_diagnostics_track_rescued_and_dropped_gold_articles(self) -> None:
        before = [
            candidate("a", "gold_a", 1),
            *[candidate(f"x{rank}", "other", rank) for rank in range(2, 11)],
            candidate("b", "gold_b", 11),
        ]
        after = [
            candidate("b", "gold_b", 1),
            *[candidate(f"x{rank}", "other", rank) for rank in range(2, 11)],
            candidate("a", "gold_a", 11),
        ]

        diagnostics = build_rerank_diagnostics(["gold_a", "gold_b"], before, after)

        self.assertEqual(diagnostics["rerank_top10_rescued_gold_article_ids"], ["gold_b"])
        self.assertEqual(diagnostics["rerank_top10_dropped_gold_article_ids"], ["gold_a"])
        self.assertEqual(diagnostics["missing_articles_after_rerank_at_10"], ["gold_a"])
        self.assertEqual(diagnostics["source_candidate_full_article_hit"], 1)

    def test_candidate_validation_rejects_duplicates_and_unknown_chunks(self) -> None:
        lookup = {"a": chunk("a", "article_a")}
        with self.assertRaises(RerankEvalError):
            validate_candidates(
                "qid",
                [candidate("a", "article_a", 1), candidate("a", "article_a", 2)],
                lookup,
            )
        with self.assertRaises(RerankEvalError):
            validate_candidates("qid", [candidate("missing", "article", 1)], lookup)

    def test_candidate_loader_rejects_duplicate_qids(self) -> None:
        lookup = {"a": chunk("a", "article_a")}
        row = {
            "qid": "qid",
            "fused_top_k_chunks": 1,
            "hybrid_candidates": [candidate("a", "article_a", 1)],
        }
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "candidates.jsonl"
            write_jsonl(path, [row, row])
            with self.assertRaises(RerankEvalError):
                load_hybrid_candidates(
                    path,
                    chunk_lookup=lookup,
                    expected_top_k_chunks=1,
                    expected_records=2,
                )

    def test_candidate_loader_rejects_wrong_candidate_count(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "candidates.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "qid": "qid",
                        "fused_top_k_chunks": 2,
                        "hybrid_candidates": [candidate("a", "article_a", 1)],
                    }
                ],
            )
            with self.assertRaises(RerankEvalError):
                load_hybrid_candidates(
                    path,
                    chunk_lookup={"a": chunk("a", "article_a")},
                    expected_top_k_chunks=2,
                    expected_records=1,
                )

    def test_build_run_name_records_model_instruction_and_length(self) -> None:
        self.assertEqual(
            build_reranker_run_name(
                model_name="Qwen/Qwen3-Reranker-0.6B",
                instruction_name="wixqa_help_center_v1",
                max_length=1024,
            ),
            "qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024",
        )

    def test_eval_writes_rerank_outputs_and_four_method_comparison(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", "gold" if index == 1 else f"article_{index}")
            for index in range(1, 51)
        ]
        candidates = [
            candidate(chunk_row.chunk_id, chunk_row.article_id, rank)
            for rank, chunk_row in enumerate(chunks, start=1)
        ]
        source_summary = {
            "chunk_hit@10": 1.0,
            "chunk_full_article_hit@10": 1.0,
            "chunk_article_recall@10": 1.0,
            "chunk_hit@50": 1.0,
            "chunk_full_article_hit@50": 1.0,
            "chunk_article_recall@50": 1.0,
            "mrr": 1.0,
            "unique_articles@50_chunks": 50.0,
            "duplicate_article_ratio@50_chunks": 0.0,
        }
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hybrid_run_dir = temp_path / "hybrid_run"
            hybrid_run_dir.mkdir()
            chunks_path = temp_path / "chunks.jsonl"
            write_jsonl(chunks_path, chunks)
            write_json(
                hybrid_run_dir / "metrics.json",
                {
                    "dataset_name": "wixqa_expertwritten",
                    "records": 1,
                    "fused_top_k_chunks": 50,
                },
            )
            write_json(
                hybrid_run_dir / "comparison.json",
                {
                    "source_run_name": "hybrid_run",
                    "fused_top_k_chunks": 50,
                    "methods": {
                        "bm25": source_summary,
                        "dense": source_summary,
                        "hybrid": source_summary,
                    },
                },
            )
            write_jsonl(
                hybrid_run_dir / "candidates.jsonl",
                [
                    {
                        "qid": "qid",
                        "dataset_name": "wixqa_expertwritten",
                        "question": "question",
                        "answer": "answer",
                        "gold_article_ids": ["gold"],
                        "num_gold_articles": 1,
                        "is_multi_article": False,
                        "fused_top_k_chunks": 50,
                        "hybrid_candidates": candidates,
                    }
                ],
            )

            summary = run_rerank_eval(
                hybrid_run_dir=hybrid_run_dir,
                chunks_path=chunks_path,
                output_dir=temp_path / "outputs",
                reranker=FakeReranker(),
                console=Console(file=None, quiet=True),
            )
            run_dir = (
                temp_path
                / "outputs"
                / "hybrid_run"
                / "qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024"
            )

            self.assertEqual(summary["rerank_top10_dropped_gold_articles"], 1)
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "rerank_traces.jsonl").exists())
            self.assertIn(
                "Hybrid RRF + Qwen3 Reranker",
                (run_dir / "comparison.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
