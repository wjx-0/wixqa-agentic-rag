from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rich.console import Console

from src.data.schema import KBChunk
from src.evaluation.run_agentic_rerank_eval import (
    AgenticRerankEvalError,
    DEFAULT_AGENTIC_RUN_NAME,
    GAP_AWARE_ALPHA,
    GAP_AWARE_BETA,
    build_agentic_rerank_diagnostics,
    rerank_gap_aware_candidates,
    run_agentic_rerank_eval,
    validate_qid_sets,
)
from src.utils.io_utils import read_json, read_jsonl, write_json, write_jsonl


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


def candidate(
    chunk_id: str,
    article_id: str,
    rank: int,
    *,
    title: str | None = None,
    phase9_rank_hint: int | None = None,
) -> dict:
    row = {
        "chunk_id": chunk_id,
        "article_id": article_id,
        "rank": rank,
        "rrf_score": 0.1,
        "sources": ["dense"],
    }
    if title is not None:
        row["title"] = title
    if phase9_rank_hint is not None:
        row["phase9_rank_hint"] = phase9_rank_hint
    return row


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(self, question, candidates, chunk_lookup, *, batch_size):
        self.calls.append((question, len(candidates)))
        ranked = sorted(
            candidates,
            key=lambda row: (
                int(row.get("phase9_rank_hint") or 1000 + int(row["rank"])),
                int(row["rank"]),
                row["chunk_id"],
            ),
        )
        return [
            {
                **row,
                "hybrid_rank": int(row.get("rank") or rank),
                "rank": rank,
                "score": fake_score(row),
                "rerank_score": fake_score(row),
                "title": chunk_lookup[row["chunk_id"]].title,
            }
            for rank, row in enumerate(ranked, start=1)
        ]


class QueryAwareFakeReranker:
    def __init__(self, scores_by_query: dict[str, dict[str, float]]) -> None:
        self.scores_by_query = scores_by_query
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, question, candidates, chunk_lookup, *, batch_size):
        self.calls.append((question, [row["chunk_id"] for row in candidates]))
        query_scores = self.scores_by_query.get(question, {})
        scored = []
        for fallback_rank, row in enumerate(candidates, start=1):
            score = float(query_scores.get(row["chunk_id"], 0.0))
            scored.append(
                {
                    **row,
                    "hybrid_rank": int(row.get("rank") or fallback_rank),
                    "title": chunk_lookup[row["chunk_id"]].title,
                    "score": score,
                    "rerank_score": score,
                }
            )
        return [
            {**row, "rank": rank}
            for rank, row in enumerate(
                sorted(
                    scored,
                    key=lambda result: (
                        -float(result["rerank_score"]),
                        int(result.get("hybrid_rank") or result.get("rank") or 0),
                        result["chunk_id"],
                    ),
                ),
                start=1,
            )
        ]


def fake_score(row: dict) -> float:
    return float(100 - int(row.get("phase9_rank_hint") or 1000 + int(row["rank"])))


class AgenticRerankTest(unittest.TestCase):
    def test_gap_aware_rerank_blends_pure_second_hop_only(self) -> None:
        chunk_lookup = {
            "first": chunk("first", "article_first", title="First"),
            "overlap": chunk("overlap", "article_overlap", title="Overlap"),
            "second": chunk("second", "article_second", title="Second"),
        }
        candidates = [
            {
                **candidate("first", "article_first", 1),
                "first_hop_rank": 1,
                "second_hop_query_ids": [],
                "second_hop_ranks": [],
                "merged_rank": 1,
            },
            {
                **candidate("overlap", "article_overlap", 2),
                "first_hop_rank": 2,
                "second_hop_query_ids": ["gap_query_1"],
                "second_hop_ranks": [{"query_id": "gap_query_1", "rank": 1}],
                "merged_rank": 2,
            },
            {
                **candidate("second", "article_second", 3),
                "first_hop_rank": None,
                "second_hop_query_ids": ["gap_query_1"],
                "second_hop_ranks": [{"query_id": "gap_query_1", "rank": 2}],
                "merged_rank": 3,
            },
        ]
        reranker = QueryAwareFakeReranker(
            {
                "original question": {"first": 10.0, "overlap": 8.0, "second": 5.0},
                "missing evidence query": {"second": 20.0},
            }
        )

        results = rerank_gap_aware_candidates(
            question="original question",
            checker_trace={
                "second_hop_queries": [
                    {"query_id": "gap_query_1", "query_text": "missing evidence query"}
                ]
            },
            candidates=candidates,
            reranker=reranker,
            chunk_lookup=chunk_lookup,
            batch_size=2,
        )

        results_by_chunk = {row["chunk_id"]: row for row in results}
        self.assertEqual(results[0]["chunk_id"], "second")
        self.assertIsNone(results_by_chunk["first"]["score_gap"])
        self.assertIsNone(results_by_chunk["overlap"]["score_gap"])
        self.assertEqual(results_by_chunk["second"]["score_original"], 5.0)
        self.assertEqual(results_by_chunk["second"]["score_gap"], 20.0)
        self.assertAlmostEqual(
            results_by_chunk["second"]["gap_aware_score"],
            GAP_AWARE_ALPHA * 5.0 + GAP_AWARE_BETA * 20.0,
        )
        self.assertEqual(
            reranker.calls,
            [
                ("original question", ["first", "overlap", "second"]),
                ("missing evidence query", ["second"]),
            ],
        )

    def test_gap_aware_rerank_uses_best_second_hop_rank_query(self) -> None:
        chunk_lookup = {
            "second": chunk("second", "article_second", title="Second"),
        }
        candidates = [
            {
                **candidate("second", "article_second", 1),
                "first_hop_rank": None,
                "second_hop_query_ids": ["gap_query_1", "gap_query_2"],
                "second_hop_ranks": [
                    {"query_id": "gap_query_1", "rank": 5},
                    {"query_id": "gap_query_2", "rank": 2},
                ],
                "merged_rank": 51,
            }
        ]
        reranker = QueryAwareFakeReranker(
            {
                "original": {"second": 1.0},
                "better gap": {"second": 11.0},
                "worse gap": {"second": 100.0},
            }
        )

        results = rerank_gap_aware_candidates(
            question="original",
            checker_trace={
                "second_hop_queries": [
                    {"query_id": "gap_query_1", "query_text": "worse gap"},
                    {"query_id": "gap_query_2", "query_text": "better gap"},
                ]
            },
            candidates=candidates,
            reranker=reranker,
            chunk_lookup=chunk_lookup,
            batch_size=1,
        )

        self.assertEqual(results[0]["gap_query_id"], "gap_query_2")
        self.assertEqual(results[0]["gap_query_text"], "better gap")
        self.assertEqual(
            reranker.calls,
            [
                ("original", ["second"]),
                ("better gap", ["second"]),
            ],
        )

    def test_gap_aware_rerank_rejects_missing_gap_query_mapping(self) -> None:
        chunk_lookup = {
            "second": chunk("second", "article_second", title="Second"),
        }
        candidates = [
            {
                **candidate("second", "article_second", 1),
                "first_hop_rank": None,
                "second_hop_query_ids": ["missing_query"],
                "second_hop_ranks": [{"query_id": "missing_query", "rank": 1}],
                "merged_rank": 51,
            }
        ]
        reranker = QueryAwareFakeReranker({"original": {"second": 1.0}})

        with self.assertRaises(AgenticRerankEvalError):
            rerank_gap_aware_candidates(
                question="original",
                checker_trace={"second_hop_queries": []},
                candidates=candidates,
                reranker=reranker,
                chunk_lookup=chunk_lookup,
                batch_size=1,
            )

    def test_diagnostics_require_full_top10_and_top20_coverage(self) -> None:
        diagnostics = build_agentic_rerank_diagnostics(
            gold_article_ids=["gold_a", "gold_b"],
            first_hop_candidates=[candidate("other", "other", 1)],
            baseline_top10=[candidate("other", "other", 1)],
            merged_candidates=[
                candidate("other", "other", 1),
                candidate("gold_a", "gold_a", 2),
            ],
            final_top10=[candidate("gold_a", "gold_a", 1)],
            final_top20=[candidate("gold_a", "gold_a", 1)],
            is_multi_article=True,
        )

        self.assertTrue(diagnostics["source_C"])
        self.assertFalse(diagnostics["LLM_C_pool_rescued"])
        self.assertFalse(diagnostics["LLM_C_top10_rescued"])
        self.assertFalse(diagnostics["LLM_C_top20_rescued"])
        self.assertEqual(diagnostics["pool_still_missing_gold_article_ids"], ["gold_b"])

    def test_top20_rescue_requires_pool_rescue(self) -> None:
        with self.assertRaises(AgenticRerankEvalError):
            build_agentic_rerank_diagnostics(
                gold_article_ids=["gold"],
                first_hop_candidates=[candidate("other", "other", 1)],
                baseline_top10=[candidate("other", "other", 1)],
                merged_candidates=[candidate("other", "other", 1)],
                final_top10=[],
                final_top20=[candidate("gold", "gold", 11)],
                is_multi_article=False,
            )

    def test_qid_validation_rejects_phase8_mismatch(self) -> None:
        with self.assertRaises(AgenticRerankEvalError):
            validate_qid_sets(
                [{"qid": "a"}],
                {"a": {}},
                {"a": {}},
                {"different": {}},
            )

    def test_eval_reconstructs_phase8_pool_and_writes_top10_top20_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hybrid_dir = root / "hybrid"
            baseline_dir = root / "baseline"
            control_dir = root / "control"
            control_hybrid_dir = root / "control_hybrid"
            rule_dir = root / "rule"
            checker_dir = root / "checker"
            for path in (
                hybrid_dir,
                baseline_dir,
                control_dir,
                control_hybrid_dir,
                rule_dir,
                checker_dir,
            ):
                path.mkdir()
            chunks_path = root / "chunks.jsonl"

            rows, baseline_traces, checker_traces, chunks = build_fixture_rows()
            write_jsonl(chunks_path, chunks)
            write_json(
                hybrid_dir / "metrics.json",
                {**hybrid_metric_summary(fused_top_k_chunks=50), "records": len(rows)},
            )
            write_jsonl(hybrid_dir / "candidates.jsonl", rows)
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
            write_json(rule_dir / "metrics.json", phase7_metric_summary())
            write_json(checker_dir / "metrics.json", phase8_metric_summary())
            write_jsonl(checker_dir / "checker_traces.jsonl", checker_traces)

            fake_reranker = FakeReranker()
            summary = run_agentic_rerank_eval(
                first_hop_hybrid_run_dir=hybrid_dir,
                baseline_rerank_run_dir=baseline_dir,
                top100_control_rerank_run_dir=control_dir,
                rule_second_hop_run_dir=rule_dir,
                llm_checker_run_dir=checker_dir,
                chunks_path=chunks_path,
                output_dir=root / "outputs",
                reranker=fake_reranker,
                console=Console(file=None, quiet=True),
            )

            run_dir = root / "outputs" / "hybrid" / "baseline" / "checker" / DEFAULT_AGENTIC_RUN_NAME
            traces = list(read_jsonl(run_dir / "agentic_rerank_traces.jsonl"))
            comparison = (run_dir / "comparison.md").read_text()
            run_config = read_json(run_dir / "run_config.json")

            self.assertEqual(len(fake_reranker.calls), 7)
            self.assertEqual(len(traces), 4)
            self.assertEqual(summary["LLM_C_pool_rescued_count"], 2)
            self.assertEqual(summary["LLM_C_top10_rescued_count"], 1)
            self.assertEqual(summary["LLM_C_top20_rescued_count"], 2)
            self.assertEqual(summary["multi_LLM_C_top20_rescued_count"], 1)
            self.assertEqual(summary["pool_rescued_but_not_top10_count"], 1)
            self.assertEqual(summary["pool_rescued_but_top20_only_count"], 1)
            self.assertEqual(summary["A_dropped@10_count"], 1)
            self.assertEqual(summary["A_dropped@20_count"], 1)
            self.assertIn("Top10 is the primary context budget; top20 is diagnostic only.", comparison)
            self.assertEqual(
                run_config["eval_scope"],
                "phase8_merged_pool_offline_gap_aware_rerank_no_llm_no_retrieval",
            )
            self.assertEqual(run_config["rerank_scoring_mode"], "gap_aware")
            self.assertEqual(run_config["gap_aware_alpha"], GAP_AWARE_ALPHA)
            self.assertEqual(run_config["gap_aware_beta"], GAP_AWARE_BETA)
            self.assertIn("score_original", traces[0]["final_top10_chunks"][0])
            self.assertIn("gap_aware_score", traces[0]["final_top10_chunks"][0])
            self.assertEqual(
                len(list(read_jsonl(run_dir / "cases_LLM_C_top10_rescued_by_rerank.jsonl"))),
                1,
            )
            self.assertEqual(
                len(list(read_jsonl(run_dir / "cases_LLM_C_top20_rescued_by_rerank.jsonl"))),
                2,
            )
            self.assertEqual(
                len(list(read_jsonl(run_dir / "cases_LLM_C_pool_rescued_but_top20_only.jsonl"))),
                1,
            )


def build_fixture_rows() -> tuple[list[dict], list[dict], list[dict], list[KBChunk]]:
    specs = [
        ("top10", ["gold_top10"], None, False, [("new_top10", "gold_top10", 1)]),
        ("top20", ["gold_top20"], None, True, [("new_top20", "gold_top20", 15)]),
        ("partial", ["gold_p1", "gold_p2"], None, True, [("new_p1", "gold_p1", 1)]),
        ("drop", ["gold_drop"], 1, False, []),
    ]
    rows = []
    baseline_traces = []
    checker_traces = []
    chunks = []
    for qid, gold_article_ids, gold_rank, is_multi, second_hop_items in specs:
        candidates = []
        for rank in range(1, 51):
            article_id = gold_article_ids[0] if gold_rank == rank else f"{qid}_article_{rank}"
            hint = rank
            if qid == "drop" and article_id == "gold_drop":
                hint = 25
            chunk_id = f"{qid}_chunk_{rank}"
            title = f"{qid} title {rank}"
            chunks.append(chunk(chunk_id, article_id, title=title))
            candidates.append(
                candidate(
                    chunk_id,
                    article_id,
                    rank,
                    title=title,
                    phase9_rank_hint=hint,
                )
            )
        rows.append(
            {
                "qid": qid,
                "dataset_name": "wixqa_expertwritten",
                "question": f"{qid} question",
                "answer": "answer",
                "gold_article_ids": gold_article_ids,
                "num_gold_articles": len(gold_article_ids),
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
        second_hop_results = []
        for index, (chunk_id, article_id, hint) in enumerate(second_hop_items, start=1):
            title = f"{qid} second hop {index}"
            chunks.append(chunk(chunk_id, article_id, title=title))
            second_hop_results.append(
                candidate(
                    chunk_id,
                    article_id,
                    index,
                    title=title,
                    phase9_rank_hint=hint,
                )
            )
        checker_traces.append(
            {
                "qid": qid,
                "checker_valid": True,
                "checker_sufficient": False,
                "llm_calls": 1,
                "next_queries": [f"{qid} missing evidence"] if second_hop_results else [],
                "second_hop_queries": [
                    {
                        "query_id": "gap_query_1",
                        "query_text": f"{qid} missing evidence",
                    }
                ] if second_hop_results else [],
                "second_hop_results": [
                    {
                        "query_id": "gap_query_1",
                        "query_text": f"{qid} missing evidence",
                        "results": second_hop_results,
                    }
                ] if second_hop_results else [],
                "retrieval_rounds": 1 + int(bool(second_hop_results)),
            }
        )
    return rows, baseline_traces, checker_traces, chunks


def metric_summary(*, candidate_top_k_chunks: int) -> dict:
    return {
        "dataset_name": "wixqa_expertwritten",
        "candidate_top_k_chunks": candidate_top_k_chunks,
        "chunk_full_article_hit@10": 0.5,
        "chunk_article_recall@10": 0.5,
        "multi_chunk_full_article_hit@10": 0.5,
        "multi_chunk_article_recall@10": 0.5,
        "chunk_full_article_hit@20": 0.5,
        "multi_chunk_full_article_hit@20": 0.5,
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


def phase7_metric_summary() -> dict:
    return {
        "chunk_full_article_hit@10": 0.805,
        "multi_chunk_full_article_hit@10": 0.5385,
        "C_pool_rescued_count": 3,
        "multi_C_pool_rescued_count": 1,
        "C_top10_rescued_count": 2,
    }


def phase8_metric_summary() -> dict:
    return {
        "dataset_name": "wixqa_expertwritten",
        "records": 4,
        "eval_scope": "pool_level_only_no_final_rerank",
        "merged_candidate_upper_bound": 110,
        "LLM_C_pool_rescued_count": 2,
        "multi_LLM_C_pool_rescued_count": 1,
    }


if __name__ == "__main__":
    unittest.main()
