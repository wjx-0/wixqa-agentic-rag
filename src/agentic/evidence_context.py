from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from src.agentic.provenance import AnswerProvenance, GapQueryProvenance, PromptManifest
from src.data.schema import KBChunk
from src.utils.text_utils import preview_text


DEFAULT_INITIAL_PHASE = "initial_context"
DEFAULT_ACTIVE_TOP_K_CHUNKS = 10
DEFAULT_TEXT_PREVIEW_CHARS = 700


class EvidenceContextError(RuntimeError):
    pass


class EvidenceItem(BaseModel):
    chunk_id: str
    article_id: str
    snippet_id: str
    title: str | None = None
    text_preview: str = ""
    rank: int | None = None
    score: float | None = None
    source: str
    first_hop_rank: int | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None
    second_hop_query_ids: list[str] = Field(default_factory=list)
    token_span: dict[str, int] | None = None
    content_hash: str | None = None
    active: bool = False
    visible_to_llm_call_ids: list[str] = Field(default_factory=list)


class EvidenceContext(BaseModel):
    qid: str
    dataset_name: str
    question: str
    answer: str | None = None
    round_index: int = 0
    candidate_items: list[EvidenceItem] = Field(default_factory=list)
    active_items: list[EvidenceItem] = Field(default_factory=list)
    packed_items: list[EvidenceItem] = Field(default_factory=list)
    compressed_summary: str | None = None
    compressed_context_ids: list[str] = Field(default_factory=list)
    query_history: list[dict[str, Any]] = Field(default_factory=list)
    known_facts: list[dict[str, Any]] = Field(default_factory=list)
    required_facets: list[dict[str, Any]] = Field(default_factory=list)
    covered_facets: list[dict[str, Any]] = Field(default_factory=list)
    missing_facets: list[dict[str, Any]] = Field(default_factory=list)
    prompt_manifests: list[PromptManifest] = Field(default_factory=list)
    gap_query_provenance: list[GapQueryProvenance] = Field(default_factory=list)
    answer_provenance: AnswerProvenance | None = None
    eval_info: dict[str, Any] = Field(default_factory=dict)


def build_initial_llm_call_id(qid: str) -> str:
    return f"{qid}:initial_context:1"


def build_initial_evidence_context(
    *,
    candidate_row: dict[str, Any],
    rerank_trace: dict[str, Any],
    chunk_lookup: dict[str, KBChunk],
    active_top_k_chunks: int = DEFAULT_ACTIVE_TOP_K_CHUNKS,
    text_preview_chars: int = DEFAULT_TEXT_PREVIEW_CHARS,
) -> EvidenceContext:
    qid = require_matching_qid(candidate_row, rerank_trace)
    llm_call_id = build_initial_llm_call_id(qid)
    active_chunk_ids = [
        row["chunk_id"] for row in rerank_trace.get("top10_reranked_chunks", [])[:active_top_k_chunks]
    ]
    if not active_chunk_ids:
        raise EvidenceContextError(f"Rerank trace qid={qid} has no top10_reranked_chunks.")

    active_chunk_id_set = set(active_chunk_ids)
    candidate_items = [
        build_evidence_item(
            row,
            chunk_lookup,
            source="hybrid_candidate",
            active=row["chunk_id"] in active_chunk_id_set,
            llm_call_id=llm_call_id if row["chunk_id"] in active_chunk_id_set else None,
            text_preview_chars=text_preview_chars,
        )
        for row in candidate_row.get("hybrid_candidates", [])
    ]
    active_items = [
        build_evidence_item(
            row,
            chunk_lookup,
            source="reranker_top10",
            active=True,
            llm_call_id=llm_call_id,
            text_preview_chars=text_preview_chars,
        )
        for row in rerank_trace.get("top10_reranked_chunks", [])[:active_top_k_chunks]
    ]
    manifest = build_initial_prompt_manifest(
        qid=qid,
        llm_call_id=llm_call_id,
        active_items=active_items,
    )
    return EvidenceContext(
        qid=qid,
        dataset_name=candidate_row["dataset_name"],
        question=candidate_row["question"],
        answer=candidate_row.get("answer"),
        candidate_items=candidate_items,
        active_items=active_items,
        packed_items=active_items,
        prompt_manifests=[manifest],
        eval_info={
            "gold_article_ids": candidate_row.get("gold_article_ids") or [],
            "num_gold_articles": int(candidate_row.get("num_gold_articles") or 0),
            "is_multi_article": bool(candidate_row.get("is_multi_article")),
            "source_case_type": rerank_trace.get("case_type"),
        },
    )


def build_initial_prompt_manifest(
    *,
    qid: str,
    llm_call_id: str,
    active_items: list[EvidenceItem],
) -> PromptManifest:
    return PromptManifest(
        llm_call_id=llm_call_id,
        phase=DEFAULT_INITIAL_PHASE,
        qid=qid,
        input_chunk_ids=[item.chunk_id for item in active_items],
        input_snippet_ids=[item.snippet_id for item in active_items],
    )


def build_evidence_item(
    row: dict[str, Any],
    chunk_lookup: dict[str, KBChunk],
    *,
    source: str,
    active: bool,
    llm_call_id: str | None = None,
    text_preview_chars: int = DEFAULT_TEXT_PREVIEW_CHARS,
) -> EvidenceItem:
    chunk_id = row.get("chunk_id")
    if not chunk_id:
        raise EvidenceContextError("Evidence row is missing chunk_id.")
    chunk = chunk_lookup.get(chunk_id)
    if chunk is None:
        raise EvidenceContextError(f"Unknown chunk_id in evidence row: {chunk_id}.")

    text = chunk.text or chunk.contents
    rank = row.get("rank")
    score = first_present_float(row, ("score", "rerank_score", "rrf_score"))
    rerank_score = first_present_float(row, ("rerank_score", "score"))
    return EvidenceItem(
        chunk_id=chunk.chunk_id,
        article_id=chunk.article_id,
        snippet_id=f"{chunk.chunk_id}:tokens:{chunk.start_token}-{chunk.end_token}",
        title=row.get("title") or chunk.title,
        text_preview=preview_text(text, limit=text_preview_chars),
        rank=int(rank) if rank is not None else None,
        score=score,
        source=source,
        first_hop_rank=optional_int(row.get("hybrid_rank", row.get("rank"))),
        rerank_rank=optional_int(row.get("rank")) if source.startswith("reranker") else None,
        rerank_score=rerank_score if source.startswith("reranker") else None,
        token_span={"start": chunk.start_token, "end": chunk.end_token},
        content_hash=hash_text(text),
        active=active,
        visible_to_llm_call_ids=[llm_call_id] if llm_call_id else [],
    )


def require_matching_qid(candidate_row: dict[str, Any], rerank_trace: dict[str, Any]) -> str:
    qid = candidate_row.get("qid")
    if not qid:
        raise EvidenceContextError("Candidate row is missing qid.")
    trace_qid = rerank_trace.get("qid")
    if trace_qid != qid:
        raise EvidenceContextError(f"qid mismatch: candidate={qid!r}, rerank_trace={trace_qid!r}.")
    return qid


def first_present_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
