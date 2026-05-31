from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.schema import KBArticle, QAExample
from src.utils.io_utils import read_jsonl


VALID_QA_DATASETS = {"wixqa_expertwritten", "wixqa_simulated", "wixqa_synthetic"}
CASE_INVALID = "INVALID_NO_GOLD_ARTICLES"


class BM25EvalError(RuntimeError):
    pass


def load_kb_articles(processed_dir: Path) -> list[KBArticle]:
    path = processed_dir / "wix_kb_corpus.jsonl"
    require_file(path)
    articles = []
    for row in read_jsonl(path):
        if not (row.get("contents") or "").strip():
            continue
        articles.append(KBArticle(**row))
    if not articles:
        raise BM25EvalError(f"No usable KB articles found in {path}.")
    return articles


def load_qa_examples(processed_dir: Path, dataset_name: str) -> list[QAExample]:
    path = processed_dir / f"{dataset_name}.jsonl"
    require_file(path)
    return [QAExample(**row) for row in read_jsonl(path)]


def require_file(path: Path) -> None:
    if not path.exists():
        raise BM25EvalError(
            f"Missing processed file: {path}. Please run `python scripts/prepare_wixqa.py` first."
        )


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines
