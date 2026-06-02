from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rich.console import Console

from src.data.schema import KBChunk
from src.evaluation.run_llm_evidence_checker_eval import (
    LLMEvidenceCheckerEvalError,
    build_pool_diagnostics,
    run_llm_evidence_checker_eval,
    validate_candidate_baseline_qids,
)
from src.llm.evidence_checker import (
    OpenAICompatibleChatClient,
    build_evidence_checker_messages,
    build_traceable_evidence_checker_messages,
    parse_checker_response,
    parse_traceable_checker_response,
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


class FakeLLMClient:
    model = "Qwen/Qwen3-8B"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages, *, temperature, max_tokens):
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLMClient received more calls than expected.")
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self, result_by_query: dict[str, list[dict]]) -> None:
        self.result_by_query = result_by_query
        self.queries: list[str] = []

    def search_batch(self, queries, **kwargs):
        self.queries.extend(queries)
        return [
            {
                "bm25": [],
                "dense": [],
                "hybrid": self.result_by_query.get(query, []),
            }
            for query in queries
        ]


class LLMEvidenceCheckerTest(unittest.TestCase):
    def test_prompt_hides_gold_fields_and_requires_missing_evidence_queries(self) -> None:
        messages = build_evidence_checker_messages(
            "How do I connect GA4?",
            [
                {
                    "rank": 1,
                    "article_id": "article_a",
                    "title": "Google Analytics",
                    "text_preview": "Connect analytics to a Wix site.",
                }
            ],
        )
        prompt = "\n".join(message["content"] for message in messages)

        self.assertNotIn("gold_article_ids", prompt)
        self.assertNotIn("case_type", prompt)
        self.assertNotIn("missing_articles", prompt)
        self.assertIn("next_queries must target blocking_missing_evidence", prompt)
        self.assertIn("not merely rewrite the original question", prompt)
        self.assertIn("Do not trigger retrieval for nice-to-have details", prompt)
        self.assertIn("decompose the question into required facets", prompt)
        self.assertIn("top10 evidence must directly cover every required facet", prompt)
        self.assertIn("Do not downgrade a missing required facet", prompt)

    def test_parse_fenced_json_and_normalizes_queries(self) -> None:
        parsed = parse_checker_response(
            """
            ```json
            {
              "sufficient": false,
              "known_facts": ["fact"],
              "blocking_missing_evidence": ["missing setup step"],
              "nice_to_have_missing_evidence": ["extra screenshot"],
              "next_queries": ["Wix GA4 setup", " ", "wix ga4 setup", "Wix Tag Manager"],
              "reason": "Need more evidence"
            }
            ```
            """,
            max_next_queries=3,
        )

        self.assertFalse(parsed["sufficient"])
        self.assertEqual(parsed["blocking_missing_evidence"], ["missing setup step"])
        self.assertEqual(parsed["nice_to_have_missing_evidence"], ["extra screenshot"])
        self.assertEqual(parsed["next_queries"], ["Wix GA4 setup", "Wix Tag Manager"])

    def test_traceable_checker_schema_cites_visible_chunk_ids(self) -> None:
        messages = build_traceable_evidence_checker_messages(
            "How do I connect GA4?",
            [
                {
                    "rank": 1,
                    "chunk_id": "chunk_a",
                    "snippet_id": "chunk_a:tokens:0-10",
                    "title": "Google Analytics",
                    "text_preview": "Connect analytics to a Wix site.",
                }
            ],
        )
        prompt = "\n".join(message["content"] for message in messages)

        self.assertIn("Chunk ID: chunk_a", prompt)
        self.assertIn("supporting_chunk_ids", prompt)
        self.assertIn("derived_from_chunk_ids", prompt)
        self.assertNotIn("gold_article_ids", prompt)

        parsed = parse_traceable_checker_response(
            """
            {
              "sufficient": false,
              "seen_chunk_ids": ["chunk_a"],
              "known_facts": ["GA4 is mentioned"],
              "covered_facets": [
                {
                  "facet_id": "facet_1",
                  "description": "analytics connection",
                  "supporting_chunk_ids": ["chunk_a"]
                }
              ],
              "missing_facets": [
                {
                  "facet_id": "facet_2",
                  "description": "property setup",
                  "inferred_from_chunk_ids": ["chunk_a"],
                  "blocking": true
                }
              ],
              "next_queries": [
                {
                  "query_text": "Wix GA4 property setup",
                  "target_missing_facet_id": "facet_2",
                  "derived_from_chunk_ids": ["chunk_a"]
                },
                {
                  "query_text": "Wix GA4 measurement ID",
                  "target_missing_facet_id": "facet_2",
                  "derived_from_chunk_ids": ["missing_chunk"]
                },
                {
                  "query_text": "ignored third query",
                  "target_missing_facet_id": "facet_2",
                  "derived_from_chunk_ids": ["chunk_a"]
                }
              ],
              "reason": "Need setup details"
            }
            """,
            allowed_chunk_ids=["chunk_a"],
            max_next_queries=2,
        )

        self.assertFalse(parsed["sufficient"])
        self.assertEqual(len(parsed["next_queries"]), 2)
        self.assertFalse(parsed["provenance_valid"])
        self.assertEqual(parsed["invalid_provenance_chunk_ids"], ["missing_chunk"])

    def test_parse_legacy_missing_evidence_as_blocking_for_compatibility(self) -> None:
        parsed = parse_checker_response(
            '{"sufficient": false, "known_facts": [], '
            '"missing_evidence": ["legacy blocking gap"], '
            '"next_queries": ["Wix legacy gap"], "reason": "legacy"}',
        )

        self.assertEqual(parsed["blocking_missing_evidence"], ["legacy blocking gap"])
        self.assertEqual(parsed["missing_evidence"], ["legacy blocking gap"])
        self.assertEqual(parsed["next_queries"], ["Wix legacy gap"])

    def test_non_blocking_missing_evidence_clears_queries(self) -> None:
        parsed = parse_checker_response(
            '{"sufficient": false, "known_facts": ["enough"], '
            '"blocking_missing_evidence": [], '
            '"nice_to_have_missing_evidence": ["extra example"], '
            '"next_queries": ["Wix extra example"], "reason": "minor"}',
        )

        self.assertFalse(parsed["sufficient"])
        self.assertEqual(parsed["blocking_missing_evidence"], [])
        self.assertEqual(parsed["nice_to_have_missing_evidence"], ["extra example"])
        self.assertEqual(parsed["next_queries"], [])

    def test_sufficient_response_clears_queries(self) -> None:
        parsed = parse_checker_response(
            '{"sufficient": true, "known_facts": [], "missing_evidence": [], '
            '"next_queries": ["Should be ignored"], "reason": "Enough"}',
        )

        self.assertTrue(parsed["sufficient"])
        self.assertEqual(parsed["next_queries"], [])

    def test_openai_client_accepts_base_url_or_full_completion_url(self) -> None:
        root_client = OpenAICompatibleChatClient(
            base_url="http://localhost:8000/v1",
            model="qwen3",
        )
        full_client = OpenAICompatibleChatClient(
            base_url="http://localhost:8000/v1/chat/completions",
            model="qwen3",
        )

        self.assertEqual(root_client.completions_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(full_client.completions_url, "http://localhost:8000/v1/chat/completions")

    def test_pool_diagnostics_requires_full_recovery(self) -> None:
        diagnostics = build_pool_diagnostics(
            gold_article_ids=["gold_a", "gold_b"],
            first_hop_candidates=[candidate("other", "other", 1)],
            baseline_top10=[candidate("other", "other", 1)],
            merged_candidates=[
                candidate("other", "other", 1),
                candidate("gold_a", "gold_a", 2),
            ],
            is_multi_article=True,
        )

        self.assertTrue(diagnostics["source_C"])
        self.assertFalse(diagnostics["LLM_C_pool_rescued"])
        self.assertEqual(diagnostics["pool_still_missing_gold_article_ids"], ["gold_b"])

    def test_qid_validation_rejects_baseline_mismatch(self) -> None:
        with self.assertRaises(LLMEvidenceCheckerEvalError):
            validate_candidate_baseline_qids(
                [{"qid": "a"}],
                {"different": {}},
            )

    def test_eval_writes_pool_level_outputs_with_fake_llm_and_retriever(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hybrid_dir = root / "hybrid"
            baseline_dir = root / "baseline"
            control_dir = root / "control"
            control_hybrid_dir = root / "control_hybrid"
            rule_dir = root / "rule"
            for path in (hybrid_dir, baseline_dir, control_dir, control_hybrid_dir, rule_dir):
                path.mkdir()
            chunks_path = root / "chunks.jsonl"

            rows, traces, chunks = build_fixture_rows()
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
            write_jsonl(baseline_dir / "rerank_traces.jsonl", traces)
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
            write_json(rule_dir / "metrics.json", phase7_metric_summary())

            checker = FakeLLMClient(
                [
                    '{"sufficient": true, "known_facts": ["covered"], '
                    '"blocking_missing_evidence": [], '
                    '"nice_to_have_missing_evidence": ["extra"], '
                    '"next_queries": ["ignored"], "reason": "ok"}',
                    '{"sufficient": false, "known_facts": [], '
                    '"blocking_missing_evidence": ["Need Wix CMS table connection setup"], '
                    '"nice_to_have_missing_evidence": [], '
                    '"next_queries": ["Wix CMS table connect collection"], "reason": "missing"}',
                    '{"sufficient": false, "known_facts": [], '
                    '"blocking_missing_evidence": ["Need Wix Stores tax setup"], '
                    '"nice_to_have_missing_evidence": [], '
                    '"next_queries": ["Wix Stores tax setup"], "reason": "partial"}',
                    'not json',
                ]
            )
            retriever = FakeRetriever(
                {
                    "Wix CMS table connect collection": [candidate("new_c", "gold_c", 1)],
                    "Wix Stores tax setup": [candidate("new_p1", "gold_p1", 1)],
                }
            )

            summary = run_llm_evidence_checker_eval(
                first_hop_hybrid_run_dir=hybrid_dir,
                baseline_rerank_run_dir=baseline_dir,
                rule_second_hop_run_dir=rule_dir,
                chunks_path=chunks_path,
                output_dir=root / "outputs",
                checker_client=checker,
                retriever=retriever,
                console=Console(file=None, quiet=True),
            )

            run_dir = root / "outputs" / "hybrid" / "baseline" / "qwen3_8b_s3_h20_pool_eval"
            traces_out = list(read_jsonl(run_dir / "checker_traces.jsonl"))
            comparison = (run_dir / "comparison.md").read_text()
            run_config = read_json(run_dir / "run_config.json")

            self.assertEqual(len(checker.messages), 4)
            self.assertEqual(retriever.queries, [
                "Wix CMS table connect collection",
                "Wix Stores tax setup",
            ])
            self.assertEqual(len(traces_out), 4)
            self.assertEqual(summary["checker_valid_count"], 3)
            self.assertEqual(summary["invalid_json_count"], 1)
            self.assertEqual(summary["checker_sufficient_count"], 1)
            self.assertEqual(summary["checker_insufficient_count"], 2)
            self.assertEqual(summary["source_A_insufficient_count"], 0)
            self.assertEqual(summary["source_A_unnecessary_retrieval_count"], 0)
            self.assertEqual(summary["LLM_C_pool_rescued_count"], 1)
            self.assertEqual(summary["multi_LLM_C_pool_rescued_count"], 1)
            self.assertEqual(summary["pool_new_gold_articles_count"], 2)
            self.assertAlmostEqual(summary["source_C_insufficient_rate"], 2 / 3)
            self.assertIn("Phase 8 is pool-level evaluation only.", comparison)
            self.assertIn("does not rerank the merged pool", comparison)
            self.assertIn("Top100 Hybrid + Qwen3 control", comparison)
            self.assertIn("N/A", comparison)
            self.assertIsNone(summary["top100_control_metrics"])
            self.assertEqual(run_config["eval_scope"], "pool_level_only_no_final_rerank")
            self.assertIsNone(run_config["top100_control_rerank_run_dir"])
            self.assertEqual(
                len(list(read_jsonl(run_dir / "cases_invalid_checker_json.jsonl"))),
                1,
            )
            self.assertEqual(
                len(list(read_jsonl(run_dir / "multi_cases_C_pool_rescued_by_llm_checker.jsonl"))),
                1,
            )

    def test_eval_does_not_retrieve_for_nice_to_have_only_gaps(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hybrid_dir = root / "hybrid"
            baseline_dir = root / "baseline"
            control_dir = root / "control"
            control_hybrid_dir = root / "control_hybrid"
            rule_dir = root / "rule"
            for path in (hybrid_dir, baseline_dir, control_dir, control_hybrid_dir, rule_dir):
                path.mkdir()
            chunks_path = root / "chunks.jsonl"

            rows, traces, chunks = build_fixture_rows()
            rows = rows[:1]
            traces = traces[:1]
            chunks = chunks[:50]
            write_jsonl(chunks_path, chunks)
            write_json(
                hybrid_dir / "metrics.json",
                {**hybrid_metric_summary(fused_top_k_chunks=50), "records": 1},
            )
            write_jsonl(hybrid_dir / "candidates.jsonl", rows)
            write_json(
                baseline_dir / "metrics.json",
                metric_summary(candidate_top_k_chunks=50),
            )
            write_jsonl(baseline_dir / "rerank_traces.jsonl", traces)
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
            write_json(rule_dir / "metrics.json", phase7_metric_summary())

            checker = FakeLLMClient(
                [
                    '{"sufficient": false, "known_facts": ["answer supported"], '
                    '"blocking_missing_evidence": [], '
                    '"nice_to_have_missing_evidence": ["could include more examples"], '
                    '"next_queries": ["Wix more examples"], "reason": "not blocking"}',
                ]
            )
            retriever = FakeRetriever({"Wix more examples": [candidate("new", "new", 1)]})

            summary = run_llm_evidence_checker_eval(
                first_hop_hybrid_run_dir=hybrid_dir,
                baseline_rerank_run_dir=baseline_dir,
                rule_second_hop_run_dir=rule_dir,
                chunks_path=chunks_path,
                output_dir=root / "outputs",
                checker_client=checker,
                retriever=retriever,
                console=Console(file=None, quiet=True),
            )

            run_dir = root / "outputs" / "hybrid" / "baseline" / "qwen3_8b_s3_h20_pool_eval"
            trace = next(read_jsonl(run_dir / "checker_traces.jsonl"))

            self.assertEqual(retriever.queries, [])
            self.assertFalse(trace["retrieval_triggered"])
            self.assertEqual(trace["second_hop_queries"], [])
            self.assertEqual(trace["nice_to_have_missing_evidence"], ["could include more examples"])
            self.assertEqual(summary["source_A_unnecessary_retrieval_count"], 0)


def build_fixture_rows() -> tuple[list[dict], list[dict], list[KBChunk]]:
    fixture = [
        ("a", ["gold_a"], 1, False),
        ("c", ["gold_c"], None, True),
        ("partial", ["gold_p1", "gold_p2"], None, True),
        ("invalid", ["gold_invalid"], None, False),
    ]
    rows = []
    traces = []
    chunks = []
    for qid, gold_article_ids, gold_rank, is_multi in fixture:
        candidates = []
        for rank in range(1, 51):
            article_id = gold_article_ids[0] if gold_rank == rank else f"{qid}_article_{rank}"
            chunk_id = f"{qid}_chunk_{rank}"
            title = f"{qid} title {rank}"
            chunks.append(chunk(chunk_id, article_id, title=title))
            candidates.append(candidate(chunk_id, article_id, rank, title=title))
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
        traces.append(
            {
                "qid": qid,
                "top10_reranked_chunks": [
                    {
                        **row,
                        "title": f"{qid} title {row['rank']}",
                        "text_preview": f"preview {qid} {row['rank']}",
                    }
                    for row in candidates[:10]
                ],
            }
        )
    return rows, traces, chunks


def metric_summary(*, candidate_top_k_chunks: int) -> dict:
    return {
        "dataset_name": "wixqa_expertwritten",
        "candidate_top_k_chunks": candidate_top_k_chunks,
        "chunk_full_article_hit@10": 0.5,
        "multi_chunk_full_article_hit@10": 0.5,
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


if __name__ == "__main__":
    unittest.main()
