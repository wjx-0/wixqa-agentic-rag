from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


PROVENANCE_SCHEMA_VERSION = "evidence_context_v1"


class PromptManifest(BaseModel):
    llm_call_id: str
    phase: str
    qid: str | None = None
    conversation_id: str | None = None
    turn_id: str | None = None
    input_chunk_ids: list[str] = Field(default_factory=list)
    input_article_ids: list[str] = Field(default_factory=list)
    input_snippet_ids: list[str] = Field(default_factory=list)
    input_compressed_context_ids: list[str] = Field(default_factory=list)
    input_query_history_ids: list[str] = Field(default_factory=list)
    output_object_id: str | None = None
    prompt_token_count: int = 0
    output_token_count: int = 0
    schema_version: str = PROVENANCE_SCHEMA_VERSION


class GapQueryProvenance(BaseModel):
    query_id: str
    target_missing_facet_id: str | None = None
    query_text: str
    derived_from_chunk_ids: list[str] = Field(default_factory=list)
    seen_chunk_ids: list[str] = Field(default_factory=list)
    checker_call_id: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_article_ids: list[str] = Field(default_factory=list)
    candidate_chunk_ids: list[str] = Field(default_factory=list)
    candidate_article_ids: list[str] = Field(default_factory=list)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    selected_article_ids: list[str] = Field(default_factory=list)
    schema_version: str = PROVENANCE_SCHEMA_VERSION


class AnswerProvenance(BaseModel):
    answer_call_id: str
    seen_chunk_ids: list[str] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    supporting_compressed_context_ids: list[str] = Field(default_factory=list)
    supporting_facet_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = PROVENANCE_SCHEMA_VERSION


class CompactBoundary(BaseModel):
    compact_id: str
    qid: str | None = None
    phase: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    kept_raw_chunk_ids: list[str] = Field(default_factory=list)
    compressed_context_ids: list[str] = Field(default_factory=list)
    dropped_chunk_ids: list[str] = Field(default_factory=list)
    source_to_summary_map: dict[str, str] = Field(default_factory=dict)
    compression_used_api: bool = False
    compression_fallback_used: bool = False
    compression_error: str | None = None
    before_usage_ratio: float
    after_usage_ratio: float | None = None
    schema_version: str = PROVENANCE_SCHEMA_VERSION
