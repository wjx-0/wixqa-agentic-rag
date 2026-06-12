from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.data.schema import KBChunk, QAExample
from src.evaluation.chunk_eval import (
    build_chunk_trace,
    chunk_case_labels,
    compute_chunk_retrieval_metrics,
    result_article_ids,
    select_chunk_ks,
    summarize_chunk_metrics,
)
from src.evaluation.eval_utils import CASE_INVALID, markdown_table
from src.evaluation.run_chunk_bm25_eval import (
    load_kb_chunks,
    metric_rows,
    render_failed_examples,
    render_group_metrics,
)
from src.evaluation.run_hybrid_rrf_eval import gold_article_first_chunk_rank
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_INSTRUCTION_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL_NAME,
    CrossEncoderReranker,
    CrossEncoderRerankerError,
)
from src.utils.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from src.utils.metrics import average


SUPPORTED_CANDIDATE_CUTOFFS = {50, 100}
METHOD_LABELS = {
    "bm25": "Chunk BM25",
    "dense": "Dense FAISS",
    "hybrid": "Hybrid RRF",
    "hybrid_reranker": "Hybrid RRF + Qwen3 Reranker",
}


class RerankEvalError(RuntimeError):
    pass


def run_rerank_eval(
    *,
    hybrid_run_dir: str | Path,
    chunks_path: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_RERANK_MODEL_NAME,
    local_files_only: bool = True,
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str | None = None,
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
    instruction_name: str = DEFAULT_INSTRUCTION_NAME,
    console: Console | None = None,
    reranker: Any | None = None,
) -> dict[str, Any]:
    validate_args(
        rerank_batch_size=rerank_batch_size,
        max_length=max_length,
        instruction=instruction,
        instruction_name=instruction_name,
    )
    console = console or Console()
    hybrid_run_dir = Path(hybrid_run_dir)
    output_dir = Path(output_dir)

    source_metrics = load_required_json(hybrid_run_dir / "metrics.json")
    source_comparison = load_required_json(hybrid_run_dir / "comparison.json")
    candidate_top_k_chunks = validate_source_artifacts(
        hybrid_run_dir=hybrid_run_dir,
        source_metrics=source_metrics,
        source_comparison=source_comparison,
    )

    try:
        chunks = load_kb_chunks(Path(chunks_path))
        chunk_lookup = build_chunk_lookup(chunks)
        candidate_rows = load_hybrid_candidates(
            hybrid_run_dir / "candidates.jsonl",
            chunk_lookup=chunk_lookup,
            expected_top_k_chunks=candidate_top_k_chunks,
            expected_records=int(source_metrics["records"]),
            expected_dataset_name=source_metrics["dataset_name"],
        )
        reranker = reranker or CrossEncoderReranker(
            model_name=model_name,
            local_files_only=local_files_only,
            device=device,
            instruction=instruction,
            max_length=max_length,
        )
    except Exception as exc:
        if isinstance(exc, RerankEvalError):
            raise
        raise RerankEvalError(str(exc)) from exc

    ks = select_chunk_ks(candidate_top_k_chunks)
    case_a, case_b, case_c = chunk_case_labels(candidate_top_k_chunks)
    case_rows: dict[str, list[dict[str, Any]]] = {case_a: [], case_b: [], case_c: []}
    traces = []
    valid_rows = []

    for candidate_row in tqdm(candidate_rows, desc=f"Rerank eval {source_metrics['dataset_name']}"):
        try:
            reranked_results = reranker.rerank(
                candidate_row["question"],
                candidate_row["hybrid_candidates"],
                chunk_lookup,
                batch_size=rerank_batch_size,
            )
        except CrossEncoderRerankerError as exc:
            raise RerankEvalError(f"Reranking failed for qid={candidate_row['qid']}: {exc}") from exc

        example = candidate_row_to_example(candidate_row)
        metrics = compute_chunk_retrieval_metrics(example.article_ids, reranked_results, ks)
        trace = build_chunk_trace(
            example,
            reranked_results,
            metrics,
            {},
            top_k_chunks=candidate_top_k_chunks,
        )
        trace.update(
            build_rerank_diagnostics(
                example.article_ids,
                candidate_row["hybrid_candidates"],
                reranked_results,
            )
        )
        trace.update(
            {
                "source_hybrid_run": hybrid_run_dir.name,
                "candidate_top_k_chunks": candidate_top_k_chunks,
                "reranker_model_name": model_name,
                "instruction_name": instruction_name,
                "top10_reranked_chunks": reranked_results[:10],
            }
        )
        traces.append(trace)
        if metrics["is_valid"]:
            valid_rows.append(trace)
            case_rows[trace["case_type"]].append(trace)

    summary = summarize_chunk_metrics(
        dataset_name=source_metrics["dataset_name"],
        records=len(candidate_rows),
        valid_rows=valid_rows,
        invalid_rows=[trace for trace in traces if trace["case_type"] == CASE_INVALID],
        ks=ks,
        top_k_chunks=candidate_top_k_chunks,
    )
    summary.update(
        {
            "retriever_type": "Hybrid RRF + Qwen3 Reranker",
            "retrieval_unit": "chunk",
            "source_hybrid_run": hybrid_run_dir.name,
            "candidate_top_k_chunks": candidate_top_k_chunks,
            "reranker_model_name": model_name,
            "local_files_only": local_files_only,
            "rerank_batch_size": rerank_batch_size,
            "max_length": max_length,
            "device": device,
            "instruction_name": instruction_name,
            "instruction": instruction,
            "source_candidate_full_article_hit": average(
                trace["source_candidate_full_article_hit"] for trace in valid_rows
            ),
            "rerank_top10_rescued_gold_articles": sum(
                len(trace["rerank_top10_rescued_gold_article_ids"]) for trace in valid_rows
            ),
            "rerank_top10_dropped_gold_articles": sum(
                len(trace["rerank_top10_dropped_gold_article_ids"]) for trace in valid_rows
            ),
        }
    )

    run_dir = ensure_dir(
        output_dir
        / hybrid_run_dir.name
        / build_reranker_run_name(
            model_name=model_name,
            instruction_name=instruction_name,
            max_length=max_length,
        )
    )
    run_config = {
        "source_hybrid_run": hybrid_run_dir.name,
        "source_hybrid_run_dir": str(hybrid_run_dir),
        "chunks_path": str(chunks_path),
        "candidate_top_k_chunks": candidate_top_k_chunks,
        "model_name": model_name,
        "local_files_only": local_files_only,
        "rerank_batch_size": rerank_batch_size,
        "max_length": max_length,
        "device": device,
        "instruction_name": instruction_name,
        "instruction": instruction,
    }
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "metrics.json", summary)
    (run_dir / "metrics.md").write_text(
        render_rerank_metrics_markdown(summary, traces),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "rerank_traces.jsonl", traces)
    write_jsonl(run_dir / f"cases_{case_a}.jsonl", case_rows[case_a])
    write_jsonl(run_dir / f"cases_{case_b}.jsonl", case_rows[case_b])
    write_jsonl(run_dir / f"cases_{case_c}.jsonl", case_rows[case_c])
    (run_dir / "comparison.md").write_text(
        render_rerank_comparison_markdown(
            source_comparison=source_comparison,
            rerank_summary=summary,
        ),
        encoding="utf-8",
    )

    print_summary(console, summary, run_dir)
    return summary


def validate_args(
    *,
    rerank_batch_size: int,
    max_length: int,
    instruction: str,
    instruction_name: str,
) -> None:
    if rerank_batch_size <= 0:
        raise RerankEvalError("--rerank_batch_size must be a positive integer.")
    if max_length <= 0:
        raise RerankEvalError("--max_length must be a positive integer.")
    if not instruction.strip():
        raise RerankEvalError("--instruction must not be empty.")
    if not instruction_name.strip():
        raise RerankEvalError("--instruction_name must not be empty.")


def load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RerankEvalError(
            f"Missing Hybrid artifact: {path}. Rerun `python scripts/run_hybrid_rrf_baseline.py` "
            "to generate reusable reranker candidates."
        )
    value = read_json(path)
    if not isinstance(value, dict):
        raise RerankEvalError(f"Expected a JSON object in {path}.")
    return value


def validate_source_artifacts(
    *,
    hybrid_run_dir: Path,
    source_metrics: dict[str, Any],
    source_comparison: dict[str, Any],
) -> int:
    if not (hybrid_run_dir / "candidates.jsonl").exists():
        raise RerankEvalError(
            f"Missing Hybrid artifact: {hybrid_run_dir / 'candidates.jsonl'}. "
            "Rerun `python scripts/run_hybrid_rrf_baseline.py` to generate reusable "
            "reranker candidates."
        )
    candidate_top_k_chunks = int(source_metrics.get("fused_top_k_chunks") or 0)
    if candidate_top_k_chunks not in SUPPORTED_CANDIDATE_CUTOFFS:
        raise RerankEvalError(
            "Reranker candidate pool must use Hybrid fused_top_k_chunks 50 or 100, "
            f"got {candidate_top_k_chunks}."
        )
    if source_comparison.get("fused_top_k_chunks") != candidate_top_k_chunks:
        raise RerankEvalError(
            "Hybrid metrics.json and comparison.json disagree on fused_top_k_chunks."
        )
    methods = source_comparison.get("methods")
    required_methods = {"bm25", "dense", "hybrid"}
    if not isinstance(methods, dict) or not required_methods.issubset(methods):
        raise RerankEvalError(
            "Hybrid comparison.json must contain bm25, dense, and hybrid method summaries."
        )
    return candidate_top_k_chunks


def build_chunk_lookup(chunks: list[KBChunk]) -> dict[str, KBChunk]:
    lookup = {}
    for chunk in chunks:
        if chunk.chunk_id in lookup:
            raise RerankEvalError(f"Duplicate chunk_id in chunk file: {chunk.chunk_id}.")
        lookup[chunk.chunk_id] = chunk
    return lookup


def load_hybrid_candidates(
    candidates_path: Path,
    *,
    chunk_lookup: dict[str, KBChunk],
    expected_top_k_chunks: int,
    expected_records: int,
    expected_dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    seen_qids = set()
    for row in read_jsonl(candidates_path):
        qid = row.get("qid")
        if not qid:
            raise RerankEvalError(f"Candidate row in {candidates_path} is missing qid.")
        if qid in seen_qids:
            raise RerankEvalError(f"Duplicate qid in {candidates_path}: {qid}.")
        seen_qids.add(qid)
        if expected_dataset_name is not None and row.get("dataset_name") != expected_dataset_name:
            raise RerankEvalError(
                f"Candidate row {qid} does not match dataset_name={expected_dataset_name!r}."
            )
        if not (row.get("question") or "").strip():
            raise RerankEvalError(f"Candidate row {qid} is missing question text.")
        if row.get("fused_top_k_chunks") != expected_top_k_chunks:
            raise RerankEvalError(
                f"Candidate row {qid} does not match fused_top_k_chunks={expected_top_k_chunks}."
            )
        candidates = row.get("hybrid_candidates")
        if not isinstance(candidates, list):
            raise RerankEvalError(f"Candidate row {qid} is missing hybrid_candidates.")
        if len(candidates) != expected_top_k_chunks:
            raise RerankEvalError(
                f"Candidate row {qid} has {len(candidates)} chunks; "
                f"expected {expected_top_k_chunks}."
            )
        validate_candidates(qid, candidates, chunk_lookup)
        rows.append(row)

    if len(rows) != expected_records:
        raise RerankEvalError(
            f"Candidate row count does not match Hybrid metrics: "
            f"rows={len(rows)}, expected={expected_records}."
        )
    return rows


def validate_candidates(
    qid: str,
    candidates: list[dict[str, Any]],
    chunk_lookup: dict[str, KBChunk],
) -> None:
    seen_chunk_ids = set()
    for expected_rank, candidate in enumerate(candidates, start=1):
        chunk_id = candidate.get("chunk_id")
        if not chunk_id:
            raise RerankEvalError(f"Candidate row {qid} contains a chunk without chunk_id.")
        if chunk_id in seen_chunk_ids:
            raise RerankEvalError(f"Candidate row {qid} contains duplicate chunk_id: {chunk_id}.")
        seen_chunk_ids.add(chunk_id)
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            raise RerankEvalError(f"Candidate row {qid} references unknown chunk_id: {chunk_id}.")
        if candidate.get("rank") != expected_rank:
            raise RerankEvalError(
                f"Candidate row {qid} has non-sequential ranks at chunk_id={chunk_id}."
            )
        if candidate.get("article_id") != chunk.article_id:
            raise RerankEvalError(
                f"Candidate row {qid} disagrees with chunk metadata for chunk_id={chunk_id}."
            )


def candidate_row_to_example(row: dict[str, Any]) -> QAExample:
    return QAExample(
        qid=row["qid"],
        dataset_name=row["dataset_name"],
        question=row["question"],
        answer=row.get("answer"),
        article_ids=row.get("gold_article_ids") or [],
        num_gold_articles=int(row.get("num_gold_articles") or 0),
        is_multi_article=bool(row.get("is_multi_article")),
    )


def build_rerank_diagnostics(
    gold_article_ids: list[str],
    before_results: list[dict[str, Any]],
    after_results: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_set = set(gold_article_ids)
    before_top10 = set(result_article_ids(before_results[:10])) & gold_set
    after_top10 = set(result_article_ids(after_results[:10])) & gold_set
    before_all = set(result_article_ids(before_results)) & gold_set
    return {
        "gold_article_first_chunk_rank_before_rerank": gold_article_first_chunk_rank(
            gold_article_ids,
            before_results,
        ),
        "gold_article_first_chunk_rank_after_rerank": gold_article_first_chunk_rank(
            gold_article_ids,
            after_results,
        ),
        "rerank_top10_rescued_gold_article_ids": sorted(after_top10 - before_top10),
        "rerank_top10_dropped_gold_article_ids": sorted(before_top10 - after_top10),
        "missing_articles_after_rerank_at_10": sorted(gold_set - after_top10),
        "source_candidate_full_article_hit": int(bool(gold_set) and gold_set.issubset(before_all)),
    }


def render_rerank_metrics_markdown(
    summary: dict[str, Any],
    traces: list[dict[str, Any]],
) -> str:
    top_k_chunks = summary["candidate_top_k_chunks"]
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    lines = [
        f"# Qwen3 Reranker Baseline: {summary['dataset_name']}",
        "",
        "## 基本信息",
        "",
    ]
    lines.extend(
        markdown_table(
            ["指标", "数值"],
            [
                ["source_hybrid_run", summary["source_hybrid_run"]],
                ["reranker_model_name", summary["reranker_model_name"]],
                ["instruction_name", summary["instruction_name"]],
                ["candidate_top_k_chunks", top_k_chunks],
                ["rerank_batch_size", summary["rerank_batch_size"]],
                ["max_length", summary["max_length"]],
                ["records", summary["records"]],
                ["valid_records", summary["valid_records"]],
                ["source_candidate_full_article_hit", f"{summary['source_candidate_full_article_hit']:.4f}"],
                ["rerank_top10_rescued_gold_articles", summary["rerank_top10_rescued_gold_articles"]],
                ["rerank_top10_dropped_gold_articles", summary["rerank_top10_dropped_gold_articles"]],
            ],
        )
    )
    lines.extend(["", "## 整体指标", ""])
    lines.extend(
        markdown_table(
            [
                "top_k_chunks",
                "chunk_hit",
                "chunk_full_article_hit",
                "chunk_article_recall",
                "chunk_gold_rate",
                "unique_articles",
                "duplicate_article_ratio",
            ],
            metric_rows(summary),
        )
    )
    lines.extend(["", f"MRR: `{summary['mrr']:.4f}`", ""])
    lines.extend(render_group_metrics({**summary, "top_k_chunks": top_k_chunks}))
    lines.extend(["", "## Case 分布", ""])
    lines.extend(
        markdown_table(
            ["case_type", "数量"],
            [
                [case_a, summary["case_A_count"]],
                [case_b, summary["case_B_count"]],
                [case_c, summary["case_C_count"]],
            ],
        )
    )
    lines.extend(render_failed_examples(traces, case_c, top_k_chunks))
    return "\n".join(lines)


def render_rerank_comparison_markdown(
    *,
    source_comparison: dict[str, Any],
    rerank_summary: dict[str, Any],
) -> str:
    cutoff = rerank_summary["candidate_top_k_chunks"]
    methods = {**source_comparison["methods"], "hybrid_reranker": rerank_summary}
    lines = [
        f"# Qwen3 Reranker 对比: {rerank_summary['dataset_name']}",
        "",
        (
            f"复用 `{source_comparison['source_run_name']}` 的 Hybrid top{cutoff} chunks，"
            f"使用 `{rerank_summary['reranker_model_name']}` 重排。"
        ),
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "method",
                f"chunk_hit@{cutoff}",
                f"chunk_full_article_hit@{cutoff}",
                f"chunk_article_recall@{cutoff}",
                "mrr",
                f"unique_articles@{cutoff}_chunks",
                f"duplicate_article_ratio@{cutoff}_chunks",
            ],
            [
                [
                    METHOD_LABELS[method],
                    f"{summary[f'chunk_hit@{cutoff}']:.4f}",
                    f"{summary[f'chunk_full_article_hit@{cutoff}']:.4f}",
                    f"{summary[f'chunk_article_recall@{cutoff}']:.4f}",
                    f"{summary['mrr']:.4f}",
                    f"{summary[f'unique_articles@{cutoff}_chunks']:.2f}",
                    f"{summary[f'duplicate_article_ratio@{cutoff}_chunks']:.4f}",
                ]
                for method, summary in methods.items()
            ],
        )
    )
    lines.extend(["", "## Top 10 对比", ""])
    lines.extend(
        markdown_table(
            ["method", "chunk_hit@10", "chunk_full_article_hit@10", "chunk_article_recall@10"],
            [
                [
                    METHOD_LABELS[method],
                    f"{summary['chunk_hit@10']:.4f}",
                    f"{summary['chunk_full_article_hit@10']:.4f}",
                    f"{summary['chunk_article_recall@10']:.4f}",
                ]
                for method, summary in methods.items()
            ],
        )
    )
    return "\n".join(lines)


def print_summary(console: Console, summary: dict[str, Any], output_dir: Path) -> None:
    cutoff = summary["candidate_top_k_chunks"]
    console.print()
    console.print("[bold green]Qwen3 Reranker Baseline Finished[/bold green]")
    console.print()
    console.print(f"Dataset: {summary['dataset_name']}")
    console.print(f"Source Hybrid run: {summary['source_hybrid_run']}")
    console.print(f"Reranker model: {summary['reranker_model_name']}")
    console.print(f"Candidates: top{cutoff} chunks")
    console.print()
    for key in ("chunk_hit@10", "chunk_full_article_hit@10", "chunk_article_recall@10", "mrr"):
        console.print(f"{key}: {summary.get(key, 0.0):.4f}")
    console.print()
    console.print(f"Results saved to {output_dir}/")


def build_reranker_run_name(
    *,
    model_name: str,
    instruction_name: str,
    max_length: int,
) -> str:
    return (
        f"{slug(model_name.rsplit('/', 1)[-1])}"
        f"_inst-{slug(instruction_name, allow_underscore=True)}"
        f"_ml{max_length}"
    )


def slug(value: str, *, allow_underscore: bool = False) -> str:
    normalized = value.strip().lower().replace(".", "p")
    pattern = r"[^a-z0-9_-]+" if allow_underscore else r"[^a-z0-9-]+"
    return re.sub(pattern, "-", normalized).strip("-")
