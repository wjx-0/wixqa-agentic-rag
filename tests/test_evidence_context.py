from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rich.console import Console

from src.agentic.context_budget import ContextBudgetConfig, build_context_usage_snapshot
from src.agentic.evidence_context import build_initial_evidence_context
from src.agentic.evidence_loop import (
    EvidenceLoopConfig,
    build_facet_audit_query,
    drop_one_visible_item,
    pack_visible_items,
    run_evidence_completion_loop,
    select_new_candidate_rows,
)
from src.data.schema import KBChunk
from src.evaluation.run_evidence_context_eval import run_evidence_context_eval
from src.evaluation.run_evidence_loop_eval import run_evidence_loop_eval
from src.evaluation import run_evidence_loop_eval as evidence_loop_eval_module
from src.retrievers.rrf import DEFAULT_BM25_WEIGHT, DEFAULT_DENSE_WEIGHT
from src.utils.env_utils import DEFAULT_DEEPSEEK_BASE_URL, load_project_env
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
    def test_evidence_loop_second_hop_defaults_are_light_equal_weight_hybrid(self) -> None:
        self.assertEqual(evidence_loop_eval_module.DEFAULT_BRANCH_TOP_K_CHUNKS, 50)
        self.assertEqual(evidence_loop_eval_module.DEFAULT_SECOND_HOP_TOP_K_CHUNKS, 20)
        self.assertEqual(DEFAULT_BM25_WEIGHT, 1.0)
        self.assertEqual(DEFAULT_DENSE_WEIGHT, 1.0)

    def test_project_env_loader_reads_env_copy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("DEEPSEEK_MODEL=old-model\n", encoding="utf-8")
            (root / ".env copy").write_text(
                "DEEPSEEK_MODEL=copy-model\n"
                "DEEPSEEK_BASE_URL=https://api.deepseek.com\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded = load_project_env(root)

                self.assertEqual([path.name for path in loaded], [".env", ".env copy"])
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "copy-model")
                self.assertEqual(os.environ["DEEPSEEK_BASE_URL"], "https://api.deepseek.com")

    def test_evidence_loop_resolves_deepseek_env_aliases(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "fake-key",
                "DEEPSEEK_MODEL": "deepseek-test",
            },
            clear=True,
        ):
            config = evidence_loop_eval_module.resolve_llm_config(
                llm_base_url=None,
                llm_api_key=None,
                llm_model=None,
                checker=None,
                checker_client=None,
            )

        self.assertEqual(config["base_url"], DEFAULT_DEEPSEEK_BASE_URL)
        self.assertEqual(config["api_key"], "fake-key")
        self.assertEqual(config["model"], "deepseek-test")

    def test_evidence_loop_builds_dashscope_reranker_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "fake-dashscope-key",
                "DASHSCOPE_RERANK_URL": "https://dashscope.example/reranks",
                "DASHSCOPE_RERANK_MODEL": "qwen3-rerank-test",
            },
            clear=True,
        ):
            config = evidence_loop_eval_module.build_loop_reranker(
                reranker=None,
                provider="dashscope",
                model_name="unused-local-model",
                local_files_only=True,
                device=None,
                instruction="instruction",
                max_length=1024,
                dashscope_url=None,
                dashscope_api_key=None,
                dashscope_model=None,
                dashscope_timeout=30,
            )

        self.assertEqual(config["provider"], "dashscope")
        self.assertEqual(config["model_name"], "dashscope/qwen3-rerank-test")
        self.assertEqual(config["dashscope_url"], "https://dashscope.example/reranks")
        self.assertEqual(config["dashscope_model"], "qwen3-rerank-test")
        self.assertTrue(config["dashscope_api_key_provided"])

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

    def test_facet_audit_query_uses_action_terms_instead_of_product_anchor(self) -> None:
        query = build_facet_audit_query(
            "How do I customize automated emails in Wix Bookings?",
            [
                {
                    "facet_id": "facet",
                    "facet_type": "task",
                    "user_need": "Customize email content with text, images, videos, and buttons",
                }
            ],
        )

        self.assertIn("customize", query)
        self.assertIn("email", query)
        self.assertIn("campaign", query)
        self.assertNotIn("bookings", query)

    def test_facet_audit_query_uses_special_social_share_bridge(self) -> None:
        query = build_facet_audit_query(
            "I updated the picture that shows up when sharing my website link, but it did not work.",
            [],
        )

        self.assertEqual(
            query,
            "social share settings pages website link picture image facebook debugger update preview",
        )

    def test_pack_visible_items_keeps_new_items_when_window_is_full(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 33)
        ]
        lookup = {row.chunk_id: row for row in chunks}
        previous_items = [
            build_initial_evidence_context(
                candidate_row={
                    "qid": f"qid_{index}",
                    "dataset_name": "wixqa_expertwritten",
                    "question": "question",
                    "answer": "answer",
                    "gold_article_ids": [],
                    "num_gold_articles": 0,
                    "is_multi_article": False,
                    "hybrid_candidates": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                },
                rerank_trace={
                    "qid": f"qid_{index}",
                    "top10_reranked_chunks": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                    "case_type": "A_top10_chunks_full",
                },
                chunk_lookup=lookup,
            ).active_items[0]
            for index in range(1, 30)
        ]
        selected_items = [
            build_initial_evidence_context(
                candidate_row={
                    "qid": f"new_{index}",
                    "dataset_name": "wixqa_expertwritten",
                    "question": "question",
                    "answer": "answer",
                    "gold_article_ids": [],
                    "num_gold_articles": 0,
                    "is_multi_article": False,
                    "hybrid_candidates": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                },
                rerank_trace={
                    "qid": f"new_{index}",
                    "top10_reranked_chunks": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                    "case_type": "A_top10_chunks_full",
                },
                chunk_lookup=lookup,
            ).active_items[0]
            for index in (30, 31)
        ]

        packed = pack_visible_items(
            previous_visible_items=previous_items,
            selected_new_items=selected_items,
            llm_call_id="call",
            config=EvidenceLoopConfig(max_raw_chunks_per_checker_call=30),
        )

        packed_ids = [item.chunk_id for item in packed]
        self.assertEqual(len(packed), 30)
        self.assertIn("chunk_30", packed_ids)
        self.assertIn("chunk_31", packed_ids)

    def test_pack_visible_items_keeps_checker_seen_priority_items(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 35)
        ]
        lookup = {row.chunk_id: row for row in chunks}

        def item_for(index: int):
            return build_initial_evidence_context(
                candidate_row={
                    "qid": f"qid_{index}",
                    "dataset_name": "wixqa_expertwritten",
                    "question": "question",
                    "answer": "answer",
                    "gold_article_ids": [],
                    "num_gold_articles": 0,
                    "is_multi_article": False,
                    "hybrid_candidates": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                },
                rerank_trace={
                    "qid": f"qid_{index}",
                    "top10_reranked_chunks": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                    "case_type": "A_top10_chunks_full",
                },
                chunk_lookup=lookup,
            ).active_items[0]

        previous_items = [item_for(index) for index in range(1, 31)]
        selected_items = [item_for(index) for index in range(31, 34)]

        packed = pack_visible_items(
            previous_visible_items=previous_items,
            selected_new_items=selected_items,
            llm_call_id="call",
            config=EvidenceLoopConfig(max_raw_chunks_per_checker_call=30),
            priority_chunk_ids=["chunk_30"],
        )

        packed_ids = [item.chunk_id for item in packed]
        self.assertIn("chunk_30", packed_ids)
        self.assertIn("chunk_31", packed_ids)
        self.assertIn("chunk_32", packed_ids)
        self.assertIn("chunk_33", packed_ids)

    def test_pack_visible_items_keeps_old_priority_items_before_new_fillers(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 41)
        ]
        lookup = {row.chunk_id: row for row in chunks}

        def item_for(index: int):
            return build_initial_evidence_context(
                candidate_row={
                    "qid": f"qid_{index}",
                    "dataset_name": "wixqa_expertwritten",
                    "question": "question",
                    "answer": "answer",
                    "gold_article_ids": [],
                    "num_gold_articles": 0,
                    "is_multi_article": False,
                    "hybrid_candidates": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                },
                rerank_trace={
                    "qid": f"qid_{index}",
                    "top10_reranked_chunks": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                    "case_type": "A_top10_chunks_full",
                },
                chunk_lookup=lookup,
            ).active_items[0]

        previous_items = [item_for(index) for index in range(1, 31)]
        selected_items = [item_for(index) for index in range(31, 41)]

        packed = pack_visible_items(
            previous_visible_items=previous_items,
            selected_new_items=selected_items,
            llm_call_id="call",
            config=EvidenceLoopConfig(max_raw_chunks_per_checker_call=30),
            priority_chunk_ids=[f"chunk_{index}" for index in range(20, 31)],
        )

        packed_ids = [item.chunk_id for item in packed]
        self.assertEqual(len(packed), 30)
        self.assertIn("chunk_30", packed_ids)

    def test_drop_one_visible_item_uses_importance_not_fixed_index(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 13)
        ]
        lookup = {row.chunk_id: row for row in chunks}
        items = [
            build_initial_evidence_context(
                candidate_row={
                    "qid": f"qid_{index}",
                    "dataset_name": "wixqa_expertwritten",
                    "question": "question",
                    "answer": "answer",
                    "gold_article_ids": [],
                    "num_gold_articles": 0,
                    "is_multi_article": False,
                    "hybrid_candidates": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                },
                rerank_trace={
                    "qid": f"qid_{index}",
                    "top10_reranked_chunks": [candidate(chunks[index - 1].chunk_id, chunks[index - 1].article_id, 1)],
                    "case_type": "A_top10_chunks_full",
                },
                chunk_lookup=lookup,
            ).active_items[0]
            for index in range(1, 13)
        ]
        items[10].second_hop_query_ids.append("query_selected")
        items[10].rerank_score = 99.0
        items[11].rerank_score = 0.1

        kept = drop_one_visible_item(items)
        kept_ids = [item.chunk_id for item in kept]

        self.assertIn("chunk_11", kept_ids)
        self.assertNotIn("chunk_12", kept_ids)

    def test_compact_uses_injected_summary_and_records_provenance(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 11)
        ]
        chunks[3].text = "title 4\nDropped fact: the special setup requires manual approval."
        chunks[3].contents = chunks[3].text
        chunk_lookup = {row.chunk_id: row for row in chunks}
        candidates = [
            candidate(row.chunk_id, row.article_id, rank)
            for rank, row in enumerate(chunks, start=1)
        ]
        context = build_initial_evidence_context(
            candidate_row={
                "qid": "qid_compact",
                "dataset_name": "wixqa_expertwritten",
                "question": "question",
                "answer": "answer",
                "gold_article_ids": ["article_10"],
                "num_gold_articles": 1,
                "is_multi_article": False,
                "hybrid_candidates": candidates,
            },
            rerank_trace={
                "qid": "qid_compact",
                "top10_reranked_chunks": candidates,
                "case_type": "A_top10_chunks_full",
            },
            chunk_lookup=chunk_lookup,
        )
        summarizer = FakeContextSummarizer(
            "Compressed fact from chunk_4: the special setup requires manual approval."
        )

        result = run_evidence_completion_loop(
            context=context,
            checker=AlwaysSufficientChecker(),
            retriever=FakeLoopRetriever({}),
            reranker=FakeLoopReranker(),
            chunk_lookup=chunk_lookup,
            context_summarizer=summarizer,
            config=EvidenceLoopConfig(
                max_rounds=0,
                max_raw_chunks_per_checker_call=3,
                max_new_raw_chunks_per_round=3,
            ),
        )

        self.assertTrue(result.completed)
        self.assertIn("manual approval", result.context.compressed_summary or "")
        self.assertEqual(result.context.compressed_context_ids, ["qid_compact:compact:1"])
        boundary = result.compact_boundaries[0]
        self.assertIn("chunk_4", boundary.dropped_chunk_ids)
        self.assertEqual(boundary.source_to_summary_map["chunk_4"], "qid_compact:compact:1")
        self.assertEqual(boundary.compressed_context_ids, ["qid_compact:compact:1"])
        self.assertTrue(boundary.compression_used_api)
        self.assertFalse(boundary.compression_fallback_used)
        self.assertIn("chunk_4", summarizer.calls[0]["dropped_chunk_ids"])
        self.assertEqual(
            result.prompt_manifests[0].input_compressed_context_ids,
            ["qid_compact:compact:1"],
        )

    def test_compact_without_summarizer_records_drops_without_fake_summary(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 11)
        ]
        chunk_lookup = {row.chunk_id: row for row in chunks}
        candidates = [
            candidate(row.chunk_id, row.article_id, rank)
            for rank, row in enumerate(chunks, start=1)
        ]
        context = build_initial_evidence_context(
            candidate_row={
                "qid": "qid_compact_fallback",
                "dataset_name": "wixqa_expertwritten",
                "question": "question",
                "answer": "answer",
                "gold_article_ids": ["article_10"],
                "num_gold_articles": 1,
                "is_multi_article": False,
                "hybrid_candidates": candidates,
            },
            rerank_trace={
                "qid": "qid_compact_fallback",
                "top10_reranked_chunks": candidates,
                "case_type": "A_top10_chunks_full",
            },
            chunk_lookup=chunk_lookup,
        )

        result = run_evidence_completion_loop(
            context=context,
            checker=AlwaysSufficientChecker(),
            retriever=FakeLoopRetriever({}),
            reranker=FakeLoopReranker(),
            chunk_lookup=chunk_lookup,
            config=EvidenceLoopConfig(
                max_rounds=0,
                max_raw_chunks_per_checker_call=3,
                max_new_raw_chunks_per_round=3,
            ),
        )

        boundary = result.compact_boundaries[0]
        self.assertIsNone(result.context.compressed_summary)
        self.assertEqual(result.context.compressed_context_ids, [])
        self.assertEqual(boundary.compressed_context_ids, [])
        self.assertTrue(boundary.dropped_chunk_ids)
        self.assertEqual(boundary.source_to_summary_map, {})
        self.assertTrue(boundary.compression_fallback_used)
        self.assertEqual(boundary.compression_error, "context_summarizer_not_configured")

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
        self.assertEqual(retriever.batch_calls, [(["query one", "query two"], 20)])
        self.assertEqual([call[1] for call in retriever.calls], [20, 20])
        self.assertEqual([call[0] for call in retriever.calls], ["query one", "query two"])
        self.assertEqual([call[0] for call in reranker.calls], ["query one", "query two"])
        self.assertEqual([call[1] for call in reranker.calls], [20, 20])
        self.assertEqual(reranker.candidate_count, 40)
        self.assertEqual(len(result.prompt_manifests[0].input_chunk_ids), 10)
        self.assertEqual(len(result.prompt_manifests[1].input_chunk_ids), 15)
        self.assertIn("chunk_11", result.prompt_manifests[1].input_chunk_ids)
        self.assertEqual(len(result.usage_snapshots), 2)

    def test_select_new_candidate_rows_keeps_per_query_retrieval_top_candidate(self) -> None:
        rows = [
            {
                "chunk_id": f"q1_chunk_{index}",
                "article_id": f"q1_article_{index}",
                "rank": index,
                "rerank_score": 100.0 - index,
                "second_hop_query_id": "q1",
                "second_hop_retrieval_rank": index,
            }
            for index in range(1, 8)
        ]
        rows.extend(
            [
                {
                    "chunk_id": "q2_low_rerank_top_retrieval",
                    "article_id": "q2_gold_like_article",
                    "rank": 99,
                    "rerank_score": 1.0,
                    "second_hop_query_id": "q2",
                    "second_hop_retrieval_rank": 1,
                },
                {
                    "chunk_id": "q2_high_rerank_low_retrieval",
                    "article_id": "q2_other_article",
                    "rank": 1,
                    "rerank_score": 99.0,
                    "second_hop_query_id": "q2",
                    "second_hop_retrieval_rank": 20,
                },
            ]
        )

        selected = select_new_candidate_rows(
            rows,
            config=EvidenceLoopConfig(max_new_raw_chunks_per_round=5),
        )

        self.assertIn(
            "q2_low_rerank_top_retrieval",
            [row["chunk_id"] for row in selected],
        )

    def test_completion_loop_can_promote_existing_inactive_hybrid_candidate(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", "gold" if index == 11 else f"article_{index}", index)
            for index in range(1, 16)
        ]
        chunk_lookup = {row.chunk_id: row for row in chunks}
        hybrid_candidates = [
            candidate(row.chunk_id, row.article_id, rank)
            for rank, row in enumerate(chunks, start=1)
        ]
        context = build_initial_evidence_context(
            candidate_row={
                "qid": "qid",
                "dataset_name": "wixqa_expertwritten",
                "question": "original question",
                "answer": "answer",
                "gold_article_ids": ["gold"],
                "num_gold_articles": 1,
                "is_multi_article": False,
                "hybrid_candidates": hybrid_candidates,
            },
            rerank_trace={
                "qid": "qid",
                "top10_reranked_chunks": hybrid_candidates[:10],
                "case_type": "B_candidate_pool_full_top10_not_full",
            },
            chunk_lookup=chunk_lookup,
        )
        retriever = FakeLoopRetriever(
            {
                "query one": [candidate("chunk_11", "gold", 1)],
                "query two": [],
            }
        )
        reranker = FakeLoopReranker()

        result = run_evidence_completion_loop(
            context=context,
            checker=FakeLoopChecker(),
            retriever=retriever,
            reranker=reranker,
            chunk_lookup=chunk_lookup,
            config=EvidenceLoopConfig(max_queries_per_round=2),
        )

        self.assertTrue(result.completed)
        self.assertEqual(len(result.context.candidate_items), 15)
        self.assertEqual([call[0] for call in reranker.calls], ["query one"])
        self.assertIn("chunk_11", result.prompt_manifests[1].input_chunk_ids)
        promoted_item = next(item for item in result.context.active_items if item.chunk_id == "chunk_11")
        self.assertTrue(promoted_item.active)
        self.assertEqual(promoted_item.second_hop_query_ids, ["round_0_query_1"])
        self.assertIn("chunk_11", result.context.eval_info["selected_second_hop_chunk_ids"])
        self.assertEqual(
            result.context.gap_query_provenance[0].retrieved_chunk_ids,
            ["chunk_11"],
        )
        self.assertEqual(
            result.context.gap_query_provenance[0].selected_chunk_ids,
            ["chunk_11"],
        )

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

    def test_completion_loop_default_stops_when_checker_is_sufficient(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 12)
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
                "question": "original question",
                "answer": "answer",
                "gold_article_ids": ["article_11"],
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
        retriever = FakeLoopRetriever({"original question": [candidate("chunk_11", "article_11", 1)]})

        result = run_evidence_completion_loop(
            context=context,
            checker=AlwaysSufficientChecker(),
            retriever=retriever,
            reranker=FakeLoopReranker(),
            chunk_lookup=chunk_lookup,
            config=EvidenceLoopConfig(),
        )

        self.assertTrue(result.completed)
        self.assertEqual(result.retrieval_rounds, 0)
        self.assertEqual(result.second_hop_query_count, 0)
        self.assertEqual(retriever.calls, [])
        self.assertFalse(result.checker_outputs[0]["minimum_retrieval_forced"])

    def test_completion_loop_min_retrieval_round_forces_audit_query(self) -> None:
        chunks = [
            chunk(f"chunk_{index}", f"article_{index}", index)
            for index in range(1, 12)
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
                "question": "original question",
                "answer": "answer",
                "gold_article_ids": ["article_11"],
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
        retriever = FakeLoopRetriever({"original question": [candidate("chunk_11", "article_11", 1)]})

        result = run_evidence_completion_loop(
            context=context,
            checker=AlwaysSufficientChecker(),
            retriever=retriever,
            reranker=FakeLoopReranker(),
            chunk_lookup=chunk_lookup,
            config=EvidenceLoopConfig(
                max_rounds=2,
                min_retrieval_rounds=1,
                max_queries_per_round=1,
            ),
        )

        self.assertTrue(result.completed)
        self.assertEqual(result.retrieval_rounds, 1)
        self.assertEqual(result.second_hop_query_count, 1)
        self.assertEqual([call[0] for call in retriever.calls], ["original question"])
        self.assertEqual(len(result.checker_outputs), 2)
        self.assertTrue(result.checker_outputs[0]["minimum_retrieval_forced"])
        self.assertFalse(result.checker_outputs[1]["minimum_retrieval_forced"])
        self.assertIn("chunk_11", result.prompt_manifests[1].input_chunk_ids)

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
            trace = list(read_jsonl(run_dir / "evidence_loop_traces.jsonl"))[0]
            self.assertEqual(trace["failure_stage"], "rescued")
            self.assertEqual(trace["second_hop_retrieved_missing_article_ids"], ["gold"])
            self.assertEqual(trace["second_hop_selected_missing_article_ids"], ["gold"])
            self.assertTrue((run_dir / "cases_rescued.jsonl").exists())


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


class AlwaysSufficientChecker:
    def __init__(self) -> None:
        self.calls = []

    def check(self, *, context, visible_items, manifest, round_index):
        self.calls.append([item.chunk_id for item in visible_items])
        return {
            "sufficient": True,
            "seen_chunk_ids": [item.chunk_id for item in visible_items],
            "required_facets": [
                {
                    "facet_id": "facet_1",
                    "facet_type": "task",
                    "user_need": "answer the original question",
                    "evidence_status": "covered",
                    "exact_object_match": True,
                    "supporting_chunk_ids": [visible_items[0].chunk_id],
                }
            ],
            "known_facts": ["enough"],
            "covered_facets": [],
            "missing_facets": [],
            "next_queries": [],
        }


class FakeLoopRetriever:
    def __init__(self, rows_by_query):
        self.rows_by_query = rows_by_query
        self.calls = []
        self.batch_calls = []

    def search(self, query_text, *, top_k_chunks):
        self.calls.append((query_text, top_k_chunks))
        return self.rows_by_query[query_text][:top_k_chunks]

    def search_batch(self, queries, *, top_k_chunks):
        self.batch_calls.append((list(queries), top_k_chunks))
        self.calls.extend((query_text, top_k_chunks) for query_text in queries)
        return [
            self.rows_by_query[query_text][:top_k_chunks]
            for query_text in queries
        ]


class FakeContextSummarizer:
    uses_api = True

    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.calls = []

    def summarize(self, *, context, dropped_items, existing_summary, compact_id):
        self.calls.append(
            {
                "qid": context.qid,
                "existing_summary": existing_summary,
                "compact_id": compact_id,
                "dropped_chunk_ids": [item.chunk_id for item in dropped_items],
            }
        )
        return self.summary


class FakeLoopReranker:
    def __init__(self) -> None:
        self.candidate_count = 0
        self.calls = []

    def rerank(self, question, candidates, chunk_lookup, *, batch_size):
        self.calls.append((question, len(candidates)))
        self.candidate_count += len(candidates)
        return [
            {**row, "rank": rank, "rerank_score": float(100 - rank)}
            for rank, row in enumerate(candidates, start=1)
        ]


if __name__ == "__main__":
    unittest.main()
