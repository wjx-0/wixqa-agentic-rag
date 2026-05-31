from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.data.schema import KBArticle, QAExample
from src.utils.text_utils import clean_text, stable_id, to_string_list


class FieldMappingError(ValueError):
    """Raised when an input row cannot be mapped into the project schema."""


@dataclass(frozen=True)
class KBFieldMapping:
    article_id: str | None
    title: str | None
    url: str | None
    contents: str
    article_type: str | None


@dataclass(frozen=True)
class QAFieldMapping:
    qid: str | None
    question: str
    answer: str | None
    article_ids: str | None


FIELD_CANDIDATES = {
    "article_id": ("article_id", "articleid", "article id", "doc_id", "document_id", "id"),
    "title": ("title", "article_title", "name", "heading"),
    "url": ("url", "article_url", "source_url", "link"),
    "contents": ("contents", "content", "text", "body", "article_text", "document"),
    "article_type": ("article_type", "type", "document_type", "category"),
    "qid": ("qid", "question_id", "sample_id", "example_id", "id"),
    "question": ("question", "query", "user_query", "utterance", "prompt"),
    "answer": ("answer", "response", "expert_answer", "final_answer", "gold_answer"),
    "article_ids": (
        "article_ids",
        "gold_article_ids",
        "relevant_article_ids",
        "document_ids",
        "doc_ids",
        "article_id",
    ),
}


def _normalize_field_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _resolve_field(
    fields: Iterable[str],
    role: str,
    *,
    required: bool,
    contains_any: tuple[str, ...] = (),
    avoid: tuple[str, ...] = (),
) -> str | None:
    available = list(fields)
    normalized_lookup = {_normalize_field_name(field): field for field in available}

    for candidate in FIELD_CANDIDATES[role]:
        normalized = _normalize_field_name(candidate)
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]

    for field in available:
        normalized = _normalize_field_name(field)
        if avoid and any(token in normalized for token in avoid):
            continue
        if any(token in normalized for token in contains_any):
            return field

    if required:
        raise FieldMappingError(
            f"Could not infer required field for role {role!r}. Available fields: {available}"
        )
    return None


def build_kb_field_mapping(fields: Iterable[str]) -> KBFieldMapping:
    available = list(fields)
    return KBFieldMapping(
        article_id=_resolve_field(available, "article_id", required=True),
        title=_resolve_field(available, "title", required=False),
        url=_resolve_field(available, "url", required=False),
        contents=_resolve_field(
            available,
            "contents",
            required=True,
            contains_any=("content", "contents", "text", "body", "document"),
        ),
        article_type=_resolve_field(available, "article_type", required=False),
    )


def build_qa_field_mapping(fields: Iterable[str]) -> QAFieldMapping:
    available = list(fields)
    return QAFieldMapping(
        qid=_resolve_field(available, "qid", required=False),
        question=_resolve_field(
            available,
            "question",
            required=True,
            contains_any=("question", "query", "utterance"),
        ),
        answer=_resolve_field(
            available,
            "answer",
            required=False,
            contains_any=("answer", "response"),
            avoid=("article", "document"),
        ),
        article_ids=_resolve_field(
            available,
            "article_ids",
            required=False,
            contains_any=("article_ids", "document_ids", "doc_ids", "gold"),
        ),
    )


def field_mapping_to_dict(mapping: KBFieldMapping | QAFieldMapping) -> dict[str, str | None]:
    return {name: getattr(mapping, name) for name in mapping.__dataclass_fields__}


def convert_kb_row(
    row: dict[str, Any],
    *,
    mapping: KBFieldMapping,
    source_config: str,
    split: str,
    row_index: int,
) -> KBArticle:
    article_id = clean_text(row.get(mapping.article_id)) if mapping.article_id else ""
    if not article_id:
        article_id = stable_id("kb_article", row_index + 1)

    return KBArticle(
        article_id=article_id,
        title=clean_text(row.get(mapping.title)) or None if mapping.title else None,
        url=clean_text(row.get(mapping.url)) or None if mapping.url else None,
        contents=clean_text(row.get(mapping.contents)),
        article_type=clean_text(row.get(mapping.article_type)) or None if mapping.article_type else None,
        metadata={
            "source_config": source_config,
            "source_split": split,
            "source_row_index": row_index,
            "field_mapping": field_mapping_to_dict(mapping),
        },
    )


def convert_qa_row(
    row: dict[str, Any],
    *,
    mapping: QAFieldMapping,
    canonical_dataset_name: str,
    source_config: str,
    split: str,
    row_index: int,
    global_index: int,
) -> QAExample:
    qid = clean_text(row.get(mapping.qid)) if mapping.qid else ""
    if not qid:
        qid_prefix = canonical_dataset_name.replace("wixqa_", "")
        qid = stable_id(qid_prefix, global_index + 1)

    article_ids = to_string_list(row.get(mapping.article_ids)) if mapping.article_ids else []

    return QAExample(
        qid=qid,
        dataset_name=canonical_dataset_name,
        question=clean_text(row.get(mapping.question)),
        answer=clean_text(row.get(mapping.answer)) or None if mapping.answer else None,
        article_ids=article_ids,
        num_gold_articles=len(article_ids),
        is_multi_article=len(article_ids) >= 2,
        metadata={
            "source_config": source_config,
            "source_split": split,
            "source_row_index": row_index,
            "field_mapping": field_mapping_to_dict(mapping),
        },
    )

