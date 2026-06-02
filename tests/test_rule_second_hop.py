from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rich.console import Console

from src.data.schema import KBChunk
from src.evaluation.run_rule_second_hop_eval import (
    RuleSecondHopEvalError,
    build_rescue_diagnostics,
    run_rule_second_hop_eval,
    validate_fair_top100_control,
    validate_qid_sets,
)
from src.retrievers.rule_second_hop import (
    RuleSecondHopError,
    build_title_expansion_queries,
    merge_chunk_candidates,
)
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


def chunk(chunk_id: str, article_id: str, *, title: str | None = None) -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        article_id=article_id,
        chunk_index=0,
        title=title,
        text=f"{title or article_id}\ncontents {chunk_id}",
        contents=f"contents {chunk_id}",
        start_token=0,
        end_token=5,
        num_tokens=5,
    )


def candidate(chunk_id: str, article_id: str, rank: int, *, title: str | None = None) -> dict:
    row = {
        "chunk_id": chunk_id,
        "article_id": article_id,
        "rank": rank,
        "rrf_score": 0.1,
        "sources": ["dense"],
    }
    if title is not None:
        row["title"] = title
    return row


class FakeRetriever:
    def __init__(self, result_by_question: dict[str, list[dict]]) -> None:
        self.result_by_question = result_by_question
        self.queries: list[str] = []

    def search_batch(self, queries, **kwargs):
        self.queries.extend(queries)
        batches = []
        for query in queries:
            question_name = query.split(" Related Wix Help Center topic:", 1)[0]
            batches.append(
                {
                    "bm25": [],
                    "dense": [],
                    "hybrid": self.result_by_question[question_name],
                }
            )
        return batches


class FakeReranker:
    def rerank(self, question, candidates, chunk_lookup, *, batch_size):
        ranked = sorted(
            candidates,
            key=lambda row: (
                0 if row["article_id"].startswith("gold") else 1,
                int(row["rank"]),
            ),
        )
        return [
            {
                **row,
                "rank": rank,
                "score": float(100 - rank),
                "rerank_score": float(100 - rank),
                "title": chunk_lookup[row["chunk_id"]].title,
            }
            for rank, row in enumerate(ranked, start=1)
        ]


class RuleSecondHopTest(unittest.TestCase):
    def test_builds_queries_from_first_three_unique_titled_articles(self) -> None:
        rows = [
            {"article_id": "a", "title": "  First   title "},
            {"article_id": "a", "title": "duplicate"},
            {"article_id": "b", "title": ""},
            {"article_id": "c", "title": "Third title"},
            {"article_id": "d", "title": "Fourth title"},
            {"article_id": "e", "title": "unused"},
        ]

        queries = build_title_expansion_queries(
            "  original   question ",
            rows,
            max_seed_articles=3,
        )

        self.assertEqual([row["seed_article_id"] for row in queries], ["a", "c", "d"])
        self.assertEqual(
            queries[0]["query_text"],
            "original question Related Wix Help Center topic: First title",
        )

    def test_query_builder_does_not_require_gold_fields(self) -> None:
        queries = build_title_expansion_queries(
            "question",
            [{"article_id": "a", "title": "Title"}],
        )
        self.assertEqual(len(queries), 1)
        with self.assertRaises(RuleSecondHopError):
            build_title_expansion_queries("", [])

    def test_merge_deduplicates_chunks_and_keeps_provenance(self) -> None:
        merged = merge_chunk_candidates(
            [candidate("first", "article_a", 1)],
            [
                {
                    "query_id": "q1",
                    "results": [
                        candidate("first", "article_a", 2),
                        candidate("new", "article_b", 1),
                    ],
                },
                {
                    "query_id": "q2",
                    "results": [candidate("new", "article_b", 3)],
                },
            ],
        )

        self.assertEqual([row["chunk_id"] for row in merged], ["first", "new"])
        self.assertEqual(merged[0]["first_hop_rank"], 1)
        self.assertEqual(merged[0]["second_hop_query_ids"], ["q1"])
        self.assertEqual(merged[1]["second_hop_query_ids"], ["q1", "q2"])
        self.assertEqual(merged[1]["merged_rank"], 2)

    def test_partial_pool_recovery_is_not_rescued(self) -> None:
        diagnostics = build_rescue_diagnostics(
            gold_article_ids=["gold_a", "gold_b"],
            first_hop_candidates=[candidate("other", "other", 1)],
            baseline_top10=[candidate("other", "other", 1)],
            merged_candidates=[
                candidate("other", "other", 1),
                candidate("gold_a", "gold_a", 2),
            ],
            final_top10=[candidate("gold_a", "gold_a", 1)],
            is_multi_article=True,
        )

        self.assertTrue(diagnostics["source_C"])
        self.assertFalse(diagnostics["C_pool_rescued"])
        self.assertFalse(diagnostics["C_top10_rescued"])
        self.assertEqual(diagnostics["pool_still_missing_gold_article_ids"], ["gold_b"])

    def test_qid_validation_rejects_control_mismatch(self) -> None:
        with self.assertRaises(RuleSecondHopEvalError):
            validate_qid_sets(
                [{"qid": "a"}],
                {"a": {}},
                {"different": {}},
            )

    def test_qid_validation_allows_missing_optional_control(self) -> None:
        validate_qid_sets(
            [{"qid": "a"}],
            {"a": {}},
            None,
        )

    def test_fair_top100_control_only_changes_fused_cutoff(self) -> None:
        first_hop = hybrid_metric_summary(fused_top_k_chunks=50)
        control = hybrid_metric_summary(fused_top_k_chunks=100)
        validate_fair_top100_control(
            first_hop_metrics=first_hop,
            control_hybrid_metrics=control,
        )
        control["dense_weight"] = 2.5
        with self.assertRaises(RuleSecondHopEvalError):
            validate_fair_top100_control(
                first_hop_metrics=first_hop,
                control_hybrid_metrics=control,
            )

    def test_eval_runs_second_hop_for_a_b_and_c_and_writes_rescue_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hybrid_dir = root / "hybrid"
            baseline_dir = root / "baseline"
            control_dir = root / "control"
            control_hybrid_dir = root / "control_hybrid"
            for path in (hybrid_dir, baseline_dir, control_dir, control_hybrid_dir):
                path.mkdir()
            chunks_path = root / "chunks.jsonl"

            all_chunks = []
            candidate_rows = []
            baseline_traces = []
            for qid, gold_article_id, gold_rank, is_multi in (
                ("a", "gold_a", 1, False),
                ("b", "gold_b", 20, False),
                ("c", "gold_c", None, True),
            ):
                question = f"{qid} question"
                candidates = []
                for rank in range(1, 51):
                    article_id = (
                        gold_article_id
                        if gold_rank == rank
                        else f"{qid}_article_{rank}"
                    )
                    chunk_id = f"{qid}_chunk_{rank}"
                    all_chunks.append(chunk(chunk_id, article_id, title=f"{qid} title {rank}"))
                    candidates.append(candidate(chunk_id, article_id, rank))
                candidate_rows.append(
                    {
                        "qid": qid,
                        "dataset_name": "wixqa_expertwritten",
                        "question": question,
                        "answer": "answer",
                        "gold_article_ids": [gold_article_id],
                        "num_gold_articles": 1,
                        "is_multi_article": is_multi,
                        "fused_top_k_chunks": 50,
                        "hybrid_candidates": candidates,
                    }
                )
                baseline_traces.append(
                    {
                        "qid": qid,
                        "top10_reranked_chunks": [
                            {
                                **row,
                                "title": f"{qid} title {row['rank']}",
                            }
                            for row in candidates[:10]
                        ],
                    }
                )

            all_chunks.extend(
                [
                    chunk("new_a", "new_a", title="new a"),
                    chunk("new_b", "new_b", title="new b"),
                    chunk("new_c", "gold_c", title="gold c"),
                ]
            )
            write_jsonl(chunks_path, all_chunks)
            write_json(
                hybrid_dir / "metrics.json",
                {**hybrid_metric_summary(fused_top_k_chunks=50), "records": 3},
            )
            write_jsonl(hybrid_dir / "candidates.jsonl", candidate_rows)
            write_json(
                baseline_dir / "metrics.json",
                metric_summary(candidate_top_k_chunks=50),
            )
            write_jsonl(baseline_dir / "rerank_traces.jsonl", baseline_traces)
            write_json(
                control_dir / "metrics.json",
                metric_summary(candidate_top_k_chunks=100),
            )
            write_json(
                control_dir / "run_config.json",
                {"source_hybrid_run_dir": str(control_hybrid_dir)},
            )
            write_json(
                control_hybrid_dir / "metrics.json",
                hybrid_metric_summary(fused_top_k_chunks=100),
            )
            write_jsonl(control_dir / "rerank_traces.jsonl", baseline_traces)
            retriever = FakeRetriever(
                {
                    "a question": [candidate("new_a", "new_a", 1)],
                    "b question": [candidate("new_b", "new_b", 1)],
                    "c question": [candidate("new_c", "gold_c", 1)],
                }
            )

            summary = run_rule_second_hop_eval(
                first_hop_hybrid_run_dir=hybrid_dir,
                baseline_rerank_run_dir=baseline_dir,
                top100_control_rerank_run_dir=control_dir,
                chunks_path=chunks_path,
                output_dir=root / "outputs",
                retriever=retriever,
                reranker=FakeReranker(),
                console=Console(file=None, quiet=True),
            )

            run_dir = root / "outputs" / "hybrid" / "baseline" / "title_expand_s3_h20"
            traces = list(read_jsonl(run_dir / "second_hop_traces.jsonl"))
            self.assertEqual(len(traces), 3)
            self.assertEqual(len(retriever.queries), 9)
            self.assertEqual(summary["C_pool_rescued_count"], 1)
            self.assertEqual(summary["C_top10_rescued_count"], 1)
            self.assertEqual(summary["multi_C_top10_rescued_count"], 1)
            self.assertEqual(summary["A_dropped_count"], 0)
            self.assertEqual(
                len(list(read_jsonl(run_dir / "cases_C_top10_rescued_by_second_hop.jsonl"))),
                1,
            )


def metric_summary(*, candidate_top_k_chunks: int) -> dict:
    return {
        "dataset_name": "wixqa_expertwritten",
        "candidate_top_k_chunks": candidate_top_k_chunks,
        "chunk_full_article_hit@10": 0.5,
        "chunk_article_recall@10": 0.5,
        "multi_chunk_full_article_hit@10": 0.5,
        "multi_chunk_article_recall@10": 0.5,
    }


def hybrid_metric_summary(*, fused_top_k_chunks: int) -> dict:
    return {
        "dataset_name": "wixqa_expertwritten",
        "branch_top_k_chunks": 100,
        "fused_top_k_chunks": fused_top_k_chunks,
        "rrf_k": 60,
        "bm25_weight": 1.0,
        "dense_weight": 2.0,
    }


if __name__ == "__main__":
    unittest.main()
