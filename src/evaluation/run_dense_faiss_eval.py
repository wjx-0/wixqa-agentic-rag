from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from src.evaluation.chunk_eval import (
    build_chunk_trace,
    chunk_case_labels,
    compute_chunk_retrieval_metrics,
    select_chunk_ks,
    summarize_chunk_metrics,
)
from src.evaluation.eval_utils import (
    BM25EvalError,
    CASE_INVALID,
    VALID_QA_DATASETS,
    load_kb_articles,
    load_qa_examples,
    markdown_table,
)
from src.evaluation.run_chunk_bm25_eval import (
    metric_rows,
    render_failed_examples,
    render_group_metrics,
)
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME
from src.retrievers.faiss_store import FaissStoreError, FaissVectorStore
from src.utils.io_utils import ensure_dir, read_json, write_json, write_jsonl


class DenseFaissEvalError(RuntimeError):
    pass


def run_dense_faiss_eval(
    *,
    processed_dir: str | Path,
    index_dir: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    model_name: str = DEFAULT_DENSE_MODEL_NAME,
    local_files_only: bool = True,
    top_k_chunks: int = 100,
    device: str | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    if dataset_name not in VALID_QA_DATASETS:
        raise DenseFaissEvalError(
            f"Unsupported dataset {dataset_name!r}. Choose one of: {sorted(VALID_QA_DATASETS)}"
        )
    if top_k_chunks <= 0:
        raise DenseFaissEvalError("--top_k_chunks must be a positive integer.")

    console = console or Console()
    processed_dir = Path(processed_dir)
    index_dir = Path(index_dir)
    output_dir = ensure_dir(output_dir)

    try:
        index_config = load_index_config(index_dir)
        kb_articles = load_kb_articles(processed_dir)
        qa_examples = load_qa_examples(processed_dir, dataset_name)
        query_embeddings = encode_queries_in_subprocess(
            questions=[example.question for example in qa_examples],
            model_name=model_name,
            local_files_only=local_files_only,
            device=device,
        )
        store = FaissVectorStore(
            index_dir / "faiss.index",
            index_dir / "chunk_metadata.jsonl",
        )
    except (BM25EvalError, DenseFaissEvalError, FaissStoreError) as exc:
        raise DenseFaissEvalError(str(exc)) from exc

    article_lookup = {article.article_id: article for article in kb_articles}
    ks = select_chunk_ks(top_k_chunks)
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    case_rows: dict[str, list[dict[str, Any]]] = {case_a: [], case_b: [], case_c: []}
    traces = []
    metric_rows_list = []

    for example, query_embedding in tqdm(
        zip(qa_examples, query_embeddings),
        total=len(qa_examples),
        desc=f"Dense FAISS eval {dataset_name}",
    ):
        chunk_results = store.search(query_embedding, top_k=top_k_chunks)
        metrics = compute_chunk_retrieval_metrics(example.article_ids, chunk_results, ks)
        trace = build_chunk_trace(
            example,
            chunk_results,
            metrics,
            article_lookup,
            top_k_chunks=top_k_chunks,
        )
        traces.append(trace)
        if metrics["is_valid"]:
            metric_rows_list.append(trace)
            case_rows[trace["case_type"]].append(trace)

    summary = summarize_chunk_metrics(
        dataset_name=dataset_name,
        records=len(qa_examples),
        valid_rows=metric_rows_list,
        invalid_rows=[trace for trace in traces if trace["case_type"] == CASE_INVALID],
        ks=ks,
        top_k_chunks=top_k_chunks,
    )
    summary.update(
        {
            "retriever_type": "Dense FAISS Retrieval",
            "retrieval_unit": "chunk",
            "model_name": model_name,
            "tokenizer_name": index_config.get("tokenizer_name"),
            "faiss_index_type": index_config.get("faiss_index_type", "IndexFlatIP"),
            "normalize_embeddings": bool(index_config.get("normalize_embeddings", True)),
            "chunk_size_tokens": index_config.get("chunk_size_tokens"),
            "chunk_overlap_tokens": index_config.get("chunk_overlap_tokens"),
            "chunk_records": store.ntotal,
            "chunk_article_records": len(
                {row.get("article_id") for row in store.metadata if row.get("article_id")}
            ),
            "top_k_chunks": top_k_chunks,
            f"avg_unique_articles_from_top{top_k_chunks}_chunks": summary[
                f"unique_articles@{top_k_chunks}_chunks"
            ],
        }
    )

    prefix = f"dense_{model_slug(model_name)}_{dataset_name}"
    write_json(output_dir / f"{prefix}_metrics.json", summary)
    (output_dir / f"{prefix}_metrics.md").write_text(
        render_dense_metrics_markdown(summary, traces),
        encoding="utf-8",
    )
    write_jsonl(output_dir / f"{prefix}_retrieval_traces.jsonl", traces)
    write_jsonl(output_dir / f"{prefix}_cases_{case_a}.jsonl", case_rows[case_a])
    write_jsonl(output_dir / f"{prefix}_cases_{case_b}.jsonl", case_rows[case_b])
    write_jsonl(output_dir / f"{prefix}_cases_{case_c}.jsonl", case_rows[case_c])

    print_summary(console, summary, output_dir)
    return summary


def load_index_config(index_dir: Path) -> dict[str, Any]:
    config_path = index_dir / "index_config.json"
    if not config_path.exists():
        raise DenseFaissEvalError(
            f"Missing FAISS index config: {config_path}. "
            "Please run `python scripts/build_faiss_index.py` first."
        )
    return read_json(config_path)


def encode_queries_in_subprocess(
    *,
    questions: list[str],
    model_name: str,
    local_files_only: bool,
    device: str | None,
) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise DenseFaissEvalError(
            "Missing dependency `numpy`. Install project dependencies with "
            "`pip install -r requirements.txt` and retry."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="wixqa_dense_queries_") as temp_dir:
        temp_path = Path(temp_dir)
        questions_path = temp_path / "questions.json"
        embeddings_path = temp_path / "query_embeddings.npy"
        write_json(questions_path, questions)

        command = [
            sys.executable,
            "-c",
            QUERY_ENCODER_CODE,
            str(questions_path),
            str(embeddings_path),
            model_name,
            "true" if local_files_only else "false",
            device or "",
        ]
        env = os.environ.copy()
        if local_files_only:
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")

        result = subprocess.run(command, env=env, check=False)
        if result.returncode != 0:
            raise DenseFaissEvalError(
                "Query embedding subprocess failed. If the local model is missing, "
                "cache BAAI/bge-m3 first or rerun with `--local_files_only false`."
            )

        embeddings = np.load(embeddings_path).astype("float32")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(questions):
            raise DenseFaissEvalError(
                "Unexpected query embedding shape: "
                f"shape={tuple(embeddings.shape)}, questions={len(questions)}."
            )
        return embeddings


QUERY_ENCODER_CODE = r"""
import os
import sys

from src.utils.io_utils import read_json

questions_path, embeddings_path, model_name, local_files_only_value, device = sys.argv[1:]
local_files_only = local_files_only_value == "true"
if local_files_only:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit(f"Missing dense query dependency: {exc}")

kwargs = {"local_files_only": local_files_only}
if device:
    kwargs["device"] = device

try:
    model = SentenceTransformer(model_name, **kwargs)
except Exception as exc:
    if local_files_only:
        raise SystemExit(
            f"Could not load local embedding model {model_name!r}. "
            "Make sure it is cached locally, or rerun with --local_files_only false."
        ) from exc
    raise

questions = read_json(questions_path)
embeddings = model.encode(
    questions,
    batch_size=16,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)
np.save(embeddings_path, np.asarray(embeddings, dtype="float32"))
"""


def render_dense_metrics_markdown(summary: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    top_k_chunks = summary["top_k_chunks"]
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    lines = [
        f"# Dense FAISS Baseline：{summary['dataset_name']}",
        "",
        "## 基本信息",
        "",
    ]
    lines.extend(
        markdown_table(
            ["指标", "数值"],
            [
                ["retriever_type", summary["retriever_type"]],
                ["model_name", summary["model_name"]],
                ["tokenizer_name", summary["tokenizer_name"]],
                ["faiss_index_type", summary["faiss_index_type"]],
                ["normalize_embeddings", summary["normalize_embeddings"]],
                ["records", summary["records"]],
                ["valid_records", summary["valid_records"]],
                ["invalid_no_gold_articles", summary["invalid_no_gold_articles"]],
                ["chunk_records", summary["chunk_records"]],
                ["chunk_article_records", summary["chunk_article_records"]],
                ["chunk_size_tokens", summary["chunk_size_tokens"]],
                ["chunk_overlap_tokens", summary["chunk_overlap_tokens"]],
                ["top_k_chunks", top_k_chunks],
                [
                    f"avg_unique_articles_from_top{top_k_chunks}_chunks",
                    f"{summary[f'avg_unique_articles_from_top{top_k_chunks}_chunks']:.2f}",
                ],
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
    lines.extend(render_group_metrics(summary))
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


def print_summary(console: Console, summary: dict[str, Any], output_dir: Path) -> None:
    top_k_chunks = summary["top_k_chunks"]
    case_a, case_b, case_c = chunk_case_labels(top_k_chunks)
    console.print()
    console.print("[bold green]Dense FAISS Baseline Finished[/bold green]")
    console.print()
    console.print(f"Dataset: {summary['dataset_name']}")
    console.print(f"Model: {summary['model_name']}")
    console.print(f"Index: FAISS {summary['faiss_index_type']}")
    console.print(f"Chunks: {summary['chunk_records']}")
    console.print(f"top_k_chunks: {top_k_chunks}")
    console.print()
    for key in ("chunk_hit@10", "chunk_full_article_hit@10", "chunk_article_recall@10", "mrr"):
        console.print(f"{key}: {summary.get(key, 0.0):.4f}")
    console.print(
        f"chunk_full_article_hit@{top_k_chunks}: "
        f"{summary.get(f'chunk_full_article_hit@{top_k_chunks}', 0.0):.4f}"
    )
    console.print()
    console.print(f"{case_a}: {summary['case_A_count']}")
    console.print(f"{case_b}: {summary['case_B_count']}")
    console.print(f"{case_c}: {summary['case_C_count']}")
    console.print()
    console.print(f"Results saved to {output_dir}/")


def model_slug(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1].replace(" ", "_")
