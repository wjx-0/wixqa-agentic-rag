from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rich.console import Console

from src.data.schema import KBArticle
from src.evaluation.run_error_analysis import (
    build_error_trace,
    run_error_analysis,
    summarize_error_traces,
)
from src.utils.io_utils import read_json, read_jsonl, write_json, write_jsonl


def article(article_id: str, title: str) -> KBArticle:
    return KBArticle(article_id=article_id, title=title, url=None, contents=f"{title} body")


def trace_row(
    qid: str,
    gold_article_ids: list[str],
    top10_article_ids: list[str],
    top50_article_ids: list[str],
    *,
    is_multi_article: bool,
    extra: dict | None = None,
) -> dict:
    row = {
        "qid": qid,
        "dataset_name": "wixqa_expertwritten",
        "question": f"question {qid}",
        "answer": f"answer {qid}",
        "gold_article_ids": gold_article_ids,
        "num_gold_articles": len(gold_article_ids),
        "is_multi_article": is_multi_article,
        "top10_chunk_article_ids": top10_article_ids,
        "top50_chunk_article_ids": top50_article_ids,
        "candidate_top_k_chunks": 50,
    }
    if extra:
        row.update(extra)
    return row


class ErrorAnalysisTest(unittest.TestCase):
    def test_build_trace_tracks_coverage_diversity_titles_and_optional_fields(self) -> None:
        lookup = {
            "gold_a": article("gold_a", "Gold A"),
            "gold_b": article("gold_b", "Gold B"),
            "other": article("other", "Other"),
        }
        source = trace_row(
            "qid_b",
            ["gold_a", "gold_b"],
            ["gold_a", "gold_a", "other"],
            ["gold_a", "gold_a", "other", "gold_b"],
            is_multi_article=True,
        )

        result = build_error_trace(source, lookup, top_k_chunks=50)

        self.assertEqual(result["case_type"], "B_top50_chunks_full_not_top10_chunks")
        self.assertEqual(result["gold_article_titles"], ["Gold A", "Gold B"])
        self.assertEqual(result["top10_chunks_titles"], ["Gold A", "Gold A", "Other"])
        self.assertEqual(result["article_full_hit@10_chunks"], 0)
        self.assertAlmostEqual(result["article_recall@10_chunks"], 0.5)
        self.assertEqual(result["article_full_hit@50_chunks"], 1)
        self.assertEqual(result["missing_articles_at_10_chunks"], ["gold_b"])
        self.assertEqual(
            result["gold_article_first_chunk_rank"],
            {"gold_a": 1, "gold_b": 4},
        )
        self.assertEqual(result["unique_articles@10_chunks"], 2)
        self.assertAlmostEqual(result["duplicate_article_ratio@10_chunks"], 1 - 2 / 3)
        self.assertIsNone(result["gold_article_first_chunk_rank_before_rerank"])
        self.assertIsNone(result["gold_article_first_chunk_rank_after_rerank"])
        self.assertEqual(result["rerank_top10_rescued_gold_article_ids"], [])
        self.assertEqual(result["rerank_top10_dropped_gold_article_ids"], [])

    def test_summary_tracks_groups_cases_multi_focus_and_optional_reranker_counts(self) -> None:
        rows = [
            build_error_trace(
                trace_row(
                    "qid_a",
                    ["gold_a"],
                    ["gold_a", "other"],
                    ["gold_a", "other"],
                    is_multi_article=False,
                    extra={"rerank_top10_rescued_gold_article_ids": ["gold_a"]},
                ),
                {"gold_a": article("gold_a", "Gold A"), "other": article("other", "Other")},
                top_k_chunks=50,
            ),
            build_error_trace(
                trace_row(
                    "qid_b",
                    ["gold_b", "gold_c"],
                    ["gold_b", "other"],
                    ["gold_b", "other", "gold_c"],
                    is_multi_article=True,
                    extra={"rerank_top10_dropped_gold_article_ids": ["gold_c"]},
                ),
                {
                    "gold_b": article("gold_b", "Gold B"),
                    "gold_c": article("gold_c", "Gold C"),
                    "other": article("other", "Other"),
                },
                top_k_chunks=50,
            ),
            build_error_trace(
                trace_row(
                    "qid_c",
                    ["gold_d", "gold_e"],
                    ["gold_d", "other"],
                    ["gold_d", "other"],
                    is_multi_article=True,
                ),
                {
                    "gold_d": article("gold_d", "Gold D"),
                    "gold_e": article("gold_e", "Gold E"),
                    "other": article("other", "Other"),
                },
                top_k_chunks=50,
            ),
        ]

        summary = summarize_error_traces(
            rows,
            top_k_chunks=50,
            source_rerank_run="rerank_run",
        )

        self.assertEqual(summary["case_A_count"], 1)
        self.assertEqual(summary["case_B_count"], 1)
        self.assertEqual(summary["case_C_count"], 1)
        self.assertEqual(summary["multi_case_B_count"], 1)
        self.assertEqual(summary["multi_case_C_count"], 1)
        self.assertAlmostEqual(summary["all_article_full_hit@10_chunks"], 1 / 3)
        self.assertAlmostEqual(summary["multi_article_recall@10_chunks"], 0.5)
        self.assertAlmostEqual(summary["multi_avg_missing_articles_at_50_chunks"], 0.5)
        self.assertEqual(summary["rerank_top10_rescued_gold_articles"], 1)
        self.assertEqual(summary["rerank_top10_dropped_gold_articles"], 1)

    def test_run_error_analysis_writes_case_and_multi_case_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            processed_dir = temp_path / "processed"
            rerank_run_dir = temp_path / "rerank_run"
            output_dir = temp_path / "error_analysis"
            processed_dir.mkdir()
            rerank_run_dir.mkdir()
            write_jsonl(
                processed_dir / "wix_kb_corpus.jsonl",
                [
                    article("gold_a", "Gold A"),
                    article("gold_b", "Gold B"),
                    article("gold_c", "Gold C"),
                    article("gold_d", "Gold D"),
                    article("gold_e", "Gold E"),
                    article("other", "Other"),
                ],
            )
            write_json(rerank_run_dir / "metrics.json", {"candidate_top_k_chunks": 50})
            write_jsonl(
                rerank_run_dir / "rerank_traces.jsonl",
                [
                    trace_row(
                        "qid_a",
                        ["gold_a"],
                        ["gold_a", "other"],
                        ["gold_a", "other"],
                        is_multi_article=False,
                    ),
                    trace_row(
                        "qid_b",
                        ["gold_b", "gold_c"],
                        ["gold_b", "other"],
                        ["gold_b", "other", "gold_c"],
                        is_multi_article=True,
                    ),
                    trace_row(
                        "qid_c",
                        ["gold_d", "gold_e"],
                        ["gold_d", "other"],
                        ["gold_d", "other"],
                        is_multi_article=True,
                    ),
                ],
            )

            summary = run_error_analysis(
                rerank_run_dir=rerank_run_dir,
                processed_dir=processed_dir,
                output_dir=output_dir,
                console=Console(file=None, quiet=True),
            )

            self.assertEqual(summary["case_A_count"], 1)
            self.assertEqual(summary["case_B_count"], 1)
            self.assertEqual(summary["case_C_count"], 1)
            self.assertTrue((output_dir / "case_traces_top50_chunks.jsonl").exists())
            self.assertEqual(
                len(list(read_jsonl(output_dir / "cases_A_top10_chunks_full.jsonl"))),
                1,
            )
            self.assertEqual(
                len(
                    list(
                        read_jsonl(
                            output_dir
                            / "multi_cases_B_top50_chunks_full_not_top10_chunks.jsonl"
                        )
                    )
                ),
                1,
            )
            self.assertEqual(
                len(
                    list(read_jsonl(output_dir / "multi_cases_C_top50_chunks_not_full.jsonl"))
                ),
                1,
            )
            saved_summary = read_json(output_dir / "summary.json")
            self.assertEqual(
                saved_summary["case_descriptions"]["B"],
                "top50 chunks cover all gold articles but top10 chunks do not; keep for diagnosis without a dedicated selection stage",
            )
            self.assertIn("Phase 7 second-hop retrieval", (output_dir / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
