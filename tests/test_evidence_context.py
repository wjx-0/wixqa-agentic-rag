from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rich.console import Console

from src.agentic.context_budget import ContextBudgetConfig, build_context_usage_snapshot
from src.agentic.evidence_context import build_initial_evidence_context
from src.agentic.evidence_loop import EvidenceLoopConfig, run_evidence_completion_loop
from src.data.schema import KBChunk
from src.evaluation.run_evidence_context_eval import run_evidence_context_eval
from src.evaluation.run_evidence_loop_eval import run_evidence_loop_eval
from src.utils.io_utils import model_to_dict, read_json, read_jsonl, write_json, write_jsonl


def chunk(chunk_id: str, article_id: str, rank: int) -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        article_id=article_id,
        chunk_index=rank - 1,
        title=f"title {rank}",
        text=f"title {rank}\ncontents for {chunk_id}",
        contents=f"contents for {chunk_id}",
        start_token=(rank - 1) * 10,
        end_token=rank * 10,
        num_tokens=10,
    )


def candidate(chunk_id: str, article_id: str, rank: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "article_id": article_id,
        "rank": rank,
        "score": float(100 - rank),
        "rrf_score": float(100 - rank),
        "sources": ["hybrid"],
    }


class EvidenceContextTest(unittest.TestCase):
    def test_builds_initial_context_from_hybrid_candidates_and_rerank_trace(self) -> None:
        chunks = [chunk(f"chunk_{index}", f"article_{index}", index) for index in range(1, 51)]
        chunk_lookup = {row.chunk_id: row for row in chunks}
        hybrid_candidates = [
            candidate(row.chunk_id, row.article_id, rank)
            for rank, row in enumerate(chunks, start=1)
        ]
        top10 = [
            {**row, "hybrid_rank": row["rank"], "rank": rank, "rerank_score": 1.0 / rank}
            for rank, row in enumerate(reversed(hybrid_candidates[:10]), start=1)
        ]
        context = build_initial_evidence_context(
            candidate_row={
                "qid": "qid",
                "dataset_name": "wixqa_expertwritten",
                "question": "question",
                "answer": "answer",
                "gold_article_ids": ["article_50"],
                "num_gold_articles": 1,
                "is_multi_article": False,
                "hybrid_candidates": hybrid_candidates,
            },
            rerank_trace={
                "qid": "qid",
                "top10_reranked_chunks": top10,
                "case_type": "B_top50_chunks_full_not_top10_chunks",
            },
            chunk_lookup=chunk_lookup,
        )

        self.assertEqual(len(context.candidate_items), 50)
        self.assertEqual(len(context.active_items), 10)
        self.assertEqual(
            context.prompt_manifests[0].input_chunk_ids,
            [item.chunk_id for item in context.active_items],
        )
        manifest_payload = model_to_dict(context.prompt_manifests[0])
        self.assertNotIn("gold_article_ids", manifest_payload)
        self.assertNotIn("article_50", str(manifest_payload))
        self.assertEqual(
            context.candidate_items[10].visible_to_llm_call_ids,
            [],
        )
        self.assertEqual(
            context.active_items[0].visible_to_llm_call_ids,
            [context.prompt_manifests[0].llm_call_id],
        )

    def test_context_usage_snapshot_calculates_ratio(self) -> None:
        snapshot = build_context_usage_snapshot(
            llm_call_id="call",
            phase="initial_context",
            qid="qid",
            input_texts=["abcdefghij"],
            input_chunk_ids=["chunk_1"],
            config=ContextBudgetConfig(
                model_context_window_tokens=100,
                reserved_output_ratio=0.10,
                safety_margin_ratio=0.05,
            ),
        )

        self.assertEqual(snapshot.projected_input_tokens, 3)
        self.assertEqual(snapshot.projected_output_tokens, 10)
        self.assertEqual(snapshot.projected_total_tokens, 18)
        self.assertAlmostEqual(snapshot.usage_ratio, 0.18)
        self.assertEqual(snapshot.budget_status, "healthy")

    def test_eval_writes_initial_context_artifacts(self) -> None:
        chunks = [chunk(f"chunk_{index}", "gold" if index == 1 else f"article_{index}", index) for index in range(1, 11)]
        candidates = [
            candidate(row.chunk_id, row.article_id, rank)
            for rank, row in enumerate(chunks, start=1)
        ]
        reranked_top10 = [
            {**row, "hybrid_rank": row["rank"], "rank": rank, "rerank_score": 1.0 / rank}
            for rank, row in enumerate(candidates, start=1)
        ]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_path = temp_path / "chunks.jsonl"
            hybrid_run_dir = temp_path / "hybrid_run"
            rerank_run_dir = temp_path / "rerank_run"
            hybrid_run_dir.mkdir()
            rerank_run_dir.mkdir()
            write_jsonl(chunks_path, chunks)
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
                        "fused_top_k_chunks": 10,
                        "hybrid_candidates": candidates,
                    }
                ],
            )
            write_json(
                rerank_run_dir / "run_config.json",
                {
                    "source_hybrid_run_dir": str(hybrid_run_dir),
                    "chunks_path": str(chunks_path),
                    "candidate_top_k_chunks": 10,
                },
            )
            write_json(
                rerank_run_dir / "metrics.json",
                {
                    "dataset_name": "wixqa_expertwritten",
                    "records": 1,
                    "candidate_top_k_chunks": 10,
                },
            )
            write_jsonl(
                rerank_run_dir / "rerank_traces.jsonl",
                [
                    {
                        "qid": "qid",
                        "dataset_name": "wixqa_expertwritten",
                        "question": "question",
                        "answer": "answer",
                        "gold_article_ids": ["gold"],
                        "num_gold_articles": 1,
                        "is_multi_article": False,
                        "top10_reranked_chunks": reranked_top10,
                        "case_type": "A_top10_chunks_full",
                    }
                ],
            )

            summary = run_evidence_context_eval(
                rerank_run_dir=rerank_run_dir,
                output_dir=temp_path / "outputs",
                console=Console(file=None, quiet=True),
            )
            run_dir = temp_path / "outputs" / "hybrid_run" / "rerank_run" / "initial_context"

            self.assertEqual(summary["records"], 1)
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "metrics.md").exists())
            self.assertTrue((run_dir / "evidence_context_traces.jsonl").exists())
            self.assertTrue((run_dir / "prompt_manifests.jsonl").exists())
            self.assertTrue((run_dir / "context_usage_snapshots.jsonl").exists())
            self.assertTrue((run_dir / "cases_B_context_not_full.jsonl").exists())
            self.assertTrue((run_dir / "cases_C_context_not_full.jsonl").exists())
            self.assertTrue((run_dir / "multi_cases_context_not_full.jsonl").exists())

            metrics = read_json(run_dir / "metrics.json")
            manifests = list(read_jsonl(run_dir / "prompt_manifests.jsonl"))
            self.assertEqual(metrics["prompt_manifest_count"], 1)
            self.assertEqual(manifests[0]["input_chunk_ids"], [row["chunk_id"] for row in reranked_top10])
            self.assertNotIn("gold_article_ids", manifests[0])

    def test_completion_loop_limits_visible_chunks_and_reranks_new_candidates(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 51)
        ]
        chunk_lookup = {row.chunk_id: row for row in chunks}
        initial_candidates = [
            candidate(row.chunk_id, row.article_id, rank)
            for rank, row in enumerate(chunks[:10], start=1)
        ]
        context = build_initial_evidence_context(
            candidate_row={
                "qid": "qid",
                "dataset_name": "wixqa_expertwritten",
                "question": "question",
                "answer": "answer",
                "gold_article_ids": ["article_50"],
                "num_gold_articles": 1,
                "is_multi_article": False,
                "hybrid_candidates": initial_candidates,
            },
            rerank_trace={
                "qid": "qid",
                "top10_reranked_chunks": initial_candidates,
                "case_type": "C_top10_chunks_not_full",
            },
            chunk_lookup=chunk_lookup,
        )
        retriever = FakeLoopRetriever(
            {
                "query one": [
                    candidate(row.chunk_id, row.article_id, rank)
                    for rank, row in enumerate(chunks[10:30], start=1)
                ],
                "query two": [
                    candidate(row.chunk_id, row.article_id, rank)
                    for rank, row in enumerate(chunks[30:50], start=1)
                ],
            }
        )
        checker = FakeLoopChecker()
        reranker = FakeLoopReranker()

        result = run_evidence_completion_loop(
            context=context,
            checker=checker,
            retriever=retriever,
            reranker=reranker,
            chunk_lookup=chunk_lookup,
            config=EvidenceLoopConfig(),
        )

        self.assertTrue(result.completed)
        self.assertEqual(len(result.checker_outputs), 2)
        self.assertEqual(len(result.prompt_manifests), 2)
        self.assertEqual(result.retrieval_rounds, 1)
        self.assertEqual(result.second_hop_query_count, 2)
        self.assertTrue(result.checker_outputs[0]["retrieval_executed"])
        self.assertEqual(result.checker_outputs[0]["retrieval_query_count"], 2)
        self.assertFalse(result.checker_outputs[1]["retrieval_executed"])
        self.assertTrue(
            all(
                len(row.input_chunk_ids) <= EvidenceLoopConfig().max_raw_chunks_per_checker_call
                for row in result.prompt_manifests
            )
        )
        self.assertEqual(len(result.context.candidate_items), 50)
        self.assertEqual([call[1] for call in retriever.calls], [20, 20])
        self.assertEqual(reranker.candidate_count, 40)
        self.assertEqual(len(result.prompt_manifests[0].input_chunk_ids), 10)
        self.assertEqual(len(result.prompt_manifests[1].input_chunk_ids), 15)
        self.assertIn("chunk_11", result.prompt_manifests[1].input_chunk_ids)
        self.assertEqual(len(result.usage_snapshots), 2)

    def test_completion_loop_does_not_count_unexecuted_final_round_queries(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 11)
        ]
        chunk_lookup = {row.chunk_id: row for row in chunks}
        initial_candidates = [
            candidate(row.chunk_id, row.article_id, rank)
            for rank, row in enumerate(chunks, start=1)
        ]
        context = build_initial_evidence_context(
            candidate_row={
                "qid": "qid",
                "dataset_name": "wixqa_expertwritten",
                "question": "question",
                "answer": "answer",
                "gold_article_ids": ["missing_article"],
                "num_gold_articles": 1,
                "is_multi_article": False,
                "hybrid_candidates": initial_candidates,
            },
            rerank_trace={
                "qid": "qid",
                "top10_reranked_chunks": initial_candidates,
                "case_type": "C_top10_chunks_not_full",
            },
            chunk_lookup=chunk_lookup,
        )
        retriever = FakeLoopRetriever({"query one": []})

        result = run_evidence_completion_loop(
            context=context,
            checker=FakeLoopChecker(),
            retriever=retriever,
            reranker=FakeLoopReranker(),
            chunk_lookup=chunk_lookup,
            config=EvidenceLoopConfig(max_rounds=0),
        )

        self.assertFalse(result.completed)
        self.assertEqual(result.retrieval_rounds, 0)
        self.assertEqual(result.second_hop_query_count, 0)
        self.assertEqual(retriever.calls, [])
        self.assertFalse(result.checker_outputs[0]["retrieval_executed"])
        self.assertEqual(result.checker_outputs[0]["retrieval_query_count"], 0)

    def test_loop_eval_writes_outputs_with_fake_components(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", "gold" if index == 11 else f"article_{index}", index)
            for index in range(1, 31)
        ]
        initial_candidates = [
            candidate(row.chunk_id, row.article_id, rank)
            for rank, row in enumerate(chunks[:10], start=1)
        ]
        reranked_top10 = [
            {**row, "hybrid_rank": row["rank"], "rank": rank, "rerank_score": 1.0 / rank}
            for rank, row in enumerate(initial_candidates, start=1)
        ]

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks_path = root / "chunks.jsonl"
            hybrid_run_dir = root / "hybrid_run"
            rerank_run_dir = root / "rerank_run"
            hybrid_run_dir.mkdir()
            rerank_run_dir.mkdir()
            write_jsonl(chunks_path, chunks)
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
                        "fused_top_k_chunks": 10,
                        "hybrid_candidates": initial_candidates,
                    }
                ],
            )
            write_json(
                rerank_run_dir / "run_config.json",
                {
                    "source_hybrid_run_dir": str(hybrid_run_dir),
                    "chunks_path": str(chunks_path),
                    "candidate_top_k_chunks": 10,
                },
            )
            write_json(
                rerank_run_dir / "metrics.json",
                {
                    "dataset_name": "wixqa_expertwritten",
                    "records": 1,
                    "candidate_top_k_chunks": 10,
                    "chunk_full_article_hit@10": 0.0,
                },
            )
            write_jsonl(
                rerank_run_dir / "rerank_traces.jsonl",
                [
                    {
                        "qid": "qid",
                        "dataset_name": "wixqa_expertwritten",
                        "question": "question",
                        "answer": "answer",
                        "gold_article_ids": ["gold"],
                        "num_gold_articles": 1,
                        "is_multi_article": False,
                        "top10_reranked_chunks": reranked_top10,
                        "case_type": "C_top10_chunks_not_full",
                    }
                ],
            )
            retriever = FakeLoopRetriever(
                {
                    "query one": [
                        candidate(row.chunk_id, row.article_id, rank)
                        for rank, row in enumerate(chunks[10:30], start=1)
                    ],
                    "query two": [],
                }
            )

            summary = run_evidence_loop_eval(
                rerank_run_dir=rerank_run_dir,
                output_dir=root / "outputs",
                checker=FakeLoopChecker(),
                retriever=retriever,
                reranker=FakeLoopReranker(),
                model_context_window_tokens=16384,
                console=Console(file=None, quiet=True),
            )
            run_dir = (
                root
                / "outputs"
                / "hybrid_run"
                / "rerank_run"
                / "loop_custom_checker_h20_qwen-qwen3-reranker-0p6b_w30_ctx16k"
            )

            self.assertEqual(summary["records"], 1)
            self.assertEqual(summary["final_chunk_full_article_hit@10"], 0.0)
            self.assertEqual(summary["final_context_chunk_full_article_hit"], 1.0)
            self.assertEqual(summary["delta_chunk_full_article_hit@10"], 0.0)
            self.assertEqual(summary["delta_context_chunk_full_article_hit"], 1.0)
            self.assertEqual(summary["sample_source_rerank_chunk_full_article_hit@10"], 0.0)
            self.assertEqual(summary["sample_delta_chunk_full_article_hit@10"], 0.0)
            self.assertEqual(summary["sample_delta_context_chunk_full_article_hit"], 1.0)
            self.assertEqual(summary["avg_retrieval_rounds"], 1.0)
            self.assertEqual(summary["avg_second_hop_queries"], 2.0)
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "evidence_loop_traces.jsonl").exists())
            self.assertTrue((run_dir / "prompt_manifests.jsonl").exists())
            self.assertTrue((run_dir / "context_usage_snapshots.jsonl").exists())


class FakeLoopChecker:
    def __init__(self) -> None:
        self.calls = []

    def check(self, *, context, visible_items, manifest, round_index):
        self.calls.append([item.chunk_id for item in visible_items])
        if round_index == 0:
            return {
                "sufficient": False,
                "seen_chunk_ids": [item.chunk_id for item in visible_items],
                "known_facts": [],
                "covered_facets": [],
                "missing_facets": [
                    {
                        "facet_id": "facet_missing",
                        "description": "missing details",
                        "inferred_from_chunk_ids": [visible_items[0].chunk_id],
                        "blocking": True,
                    }
                ],
                "next_queries": [
                    {
                        "query_id": "q1",
                        "query_text": "query one",
                        "target_missing_facet_id": "facet_missing",
                        "derived_from_chunk_ids": [visible_items[0].chunk_id],
                    },
                    {
                        "query_id": "q2",
                        "query_text": "query two",
                        "target_missing_facet_id": "facet_missing",
                        "derived_from_chunk_ids": [visible_items[1].chunk_id],
                    },
                    {
                        "query_id": "q3",
                        "query_text": "ignored query",
                        "target_missing_facet_id": "facet_missing",
                        "derived_from_chunk_ids": [visible_items[2].chunk_id],
                    },
                ],
            }
        return {
            "sufficient": True,
            "seen_chunk_ids": [item.chunk_id for item in visible_items],
            "known_facts": ["enough now"],
            "covered_facets": [],
            "missing_facets": [],
            "next_queries": [],
        }


class FakeLoopRetriever:
    def __init__(self, rows_by_query):
        self.rows_by_query = rows_by_query
        self.calls = []

    def search(self, query_text, *, top_k_chunks):
        self.calls.append((query_text, top_k_chunks))
        return self.rows_by_query[query_text][:top_k_chunks]


class FakeLoopReranker:
    def __init__(self) -> None:
        self.candidate_count = 0

    def rerank(self, question, candidates, chunk_lookup, *, batch_size):
        self.candidate_count = len(candidates)
        return [
            {**row, "rank": rank, "rerank_score": float(100 - rank)}
            for rank, row in enumerate(candidates, start=1)
        ]


if __name__ == "__main__":
    unittest.main()
