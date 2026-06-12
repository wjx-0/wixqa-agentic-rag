from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agentic.evidence_context import EvidenceContext
from src.utils.text_utils import compact_text


PROFILE_FAST = "fast_profile"
PROFILE_FULL_EVAL = "full_eval_profile"
VALID_PROFILES = {PROFILE_FAST, PROFILE_FULL_EVAL, "fast", "full_eval"}
IntentRoute = Literal["retrieve", "clarify", "direct_answer"]

AgentEventType = Literal[
    "status",
    "dialogue_state",
    "query",
    "evidence",
    "answer",
    "verification",
    "final_answer",
    "clarification_request",
    "abstention",
    "still_verifying",
]

TurnType = Literal[
    "new_question",
    "follow_up_question",
    "clarification_answer",
    "correction_or_constraint",
    "topic_shift",
    "small_talk",
    "needs_clarification",
]

VerifierStatus = Literal["ready_to_answer", "insufficient_evidence", "unsupported_answer"]


class LatencyConfig(BaseModel):
    first_event_target_ms: int = 500
    fast_answer_target_ms: int = 5000
    max_online_gap_rounds: int = 1
    max_online_queries_per_round: int = 1
    online_second_hop_top_k_chunks: int = 3
    online_max_raw_chunks_per_checker_call: int = 10
    online_max_new_raw_chunks_per_round: int = 2
    online_checker_mode: Literal["compact", "traceable"] = "compact"
    online_llm_max_tokens: int = 1024
    max_online_verifier_repair_rounds: int = 2
    enable_answer_cache: bool = True
    enable_evidence_context_reuse: bool = True


class PipelineProfileConfig(BaseModel):
    name: str
    checker_mode: Literal["compact", "traceable"] = "traceable"
    llm_max_tokens: int = 2048
    max_gap_rounds: int
    max_queries_per_round: int
    per_query_retrieve_top_k_chunks: int
    max_raw_chunks_per_checker_call: int
    max_new_raw_chunks_per_round: int
    max_new_chunks_per_article: int = 1
    max_visible_chunks_per_article: int = 2
    max_verifier_repair_rounds: int = 1
    enable_answer_cache: bool = True
    enable_evidence_context_reuse: bool = True


class AgentEvent(BaseModel):
    event_id: str
    conversation_id: str
    turn_id: str
    agent_name: str
    event_type: AgentEventType
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    profile: str
    is_final: bool = False


class DialogueTurn(BaseModel):
    turn_id: str
    role: Literal["user", "assistant"]
    message: str
    turn_type: TurnType | None = None
    standalone_question: str | None = None
    answer: str | None = None
    citation_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class DialogueState(BaseModel):
    conversation_id: str
    turns: list[DialogueTurn] = Field(default_factory=list)
    active_question: str | None = None
    rewritten_standalone_question: str | None = None
    evidence_context: EvidenceContext | None = None
    context_memory_summary: str | None = None
    context_memory_summary_turn_count: int = 0
    context_memory_summary_fallback_used: bool = False
    context_memory_summary_error: str | None = None
    answer_history: list[dict[str, Any]] = Field(default_factory=list)
    citation_history: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_user_constraints: list[str] = Field(default_factory=list)

    def add_user_turn(
        self,
        *,
        turn_id: str,
        message: str,
        turn_type: TurnType,
    ) -> DialogueTurn:
        turn = DialogueTurn(
            turn_id=turn_id,
            role="user",
            message=message,
            turn_type=turn_type,
        )
        self.turns.append(turn)
        return turn

    def add_assistant_turn(
        self,
        *,
        turn_id: str,
        message: str,
        standalone_question: str | None = None,
        citation_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DialogueTurn:
        turn = DialogueTurn(
            turn_id=turn_id,
            role="assistant",
            message=message,
            standalone_question=standalone_question,
            answer=message,
            citation_ids=list(citation_ids or []),
            metadata=dict(metadata or {}),
        )
        self.turns.append(turn)
        self.answer_history.append(
            {
                "turn_id": turn_id,
                "answer": message,
                "standalone_question": standalone_question,
                "citation_ids": list(citation_ids or []),
            }
        )
        return turn

    def remember_constraint(self, text: str) -> None:
        constraint = compact_text(text)
        if constraint and constraint not in self.unresolved_user_constraints:
            self.unresolved_user_constraints.append(constraint)

    def clear_topic_memory(self) -> None:
        self.evidence_context = None
        self.context_memory_summary = None
        self.context_memory_summary_turn_count = 0
        self.context_memory_summary_fallback_used = False
        self.context_memory_summary_error = None
        self.unresolved_user_constraints = []


class DialogueAgentResult(BaseModel):
    turn_id: str
    turn_type: TurnType
    user_message: str
    active_question: str
    topic_shift: bool = False
    constraints: list[str] = Field(default_factory=list)
    intent_route: IntentRoute = "retrieve"
    intent_reason: str = ""
    intent_confidence: float | None = None
    clarification_question: str | None = None
    intent_used_api: bool = False
    intent_fallback_used: bool = False
    intent_error: str | None = None


class QueryAgentResult(BaseModel):
    standalone_question: str
    route: Literal["retrieve", "reuse_context", "direct_answer", "clarify"]
    reused_context: bool = False
    turn_type: TurnType
    context_reuse_score: float = 0.0
    context_reuse_reason: str = ""
    memory_summary_used: bool = False
    memory_summary_fallback_used: bool = False
    memory_summary_error: str | None = None
    rewrite_used_api: bool = False
    rewrite_fallback_used: bool = False
    rewrite_error: str | None = None


class EvidenceAgentResult(BaseModel):
    context: EvidenceContext
    sufficient: bool = False
    completed: bool = False
    retrieval_rounds: int = 0
    second_hop_query_count: int = 0
    total_latency_ms: float = 0.0
    checker_outputs: list[dict[str, Any]] = Field(default_factory=list)
    profile_config: PipelineProfileConfig


class Citation(BaseModel):
    citation_id: str
    article_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    compressed_context_ids: list[str] = Field(default_factory=list)
    title: str | None = None
    url: str | None = None


class AnswerClaim(BaseModel):
    claim_id: str
    claim: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    supporting_compressed_context_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)


class AnswerTrace(BaseModel):
    answer: str
    seen_chunk_ids: list[str] = Field(default_factory=list)
    seen_compressed_context_ids: list[str] = Field(default_factory=list)
    answer_claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    prompt_manifest_id: str | None = None
    generation_mode: str = ""
    generator_error: str | None = None


class VerifierTrace(BaseModel):
    status: VerifierStatus
    reason: str
    verifier_seen_chunk_ids: list[str] = Field(default_factory=list)
    checked_claim_ids: list[str] = Field(default_factory=list)
    unsupported_claims: list[dict[str, Any]] = Field(default_factory=list)
    missing_facets: list[dict[str, Any]] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)


def normalize_profile(profile: str) -> str:
    value = compact_text(profile)
    if value not in VALID_PROFILES:
        raise ValueError(f"Unknown dialogue pipeline profile: {profile!r}.")
    if value == "fast":
        return PROFILE_FAST
    if value == "full_eval":
        return PROFILE_FULL_EVAL
    return value


def profile_config(profile: str, latency: LatencyConfig | None = None) -> PipelineProfileConfig:
    latency = latency or LatencyConfig()
    normalized = normalize_profile(profile)
    if normalized == PROFILE_FAST:
        return PipelineProfileConfig(
            name=PROFILE_FAST,
            checker_mode=latency.online_checker_mode,
            llm_max_tokens=latency.online_llm_max_tokens,
            max_gap_rounds=latency.max_online_gap_rounds,
            max_queries_per_round=latency.max_online_queries_per_round,
            per_query_retrieve_top_k_chunks=latency.online_second_hop_top_k_chunks,
            max_raw_chunks_per_checker_call=latency.online_max_raw_chunks_per_checker_call,
            max_new_raw_chunks_per_round=latency.online_max_new_raw_chunks_per_round,
            max_verifier_repair_rounds=latency.max_online_verifier_repair_rounds,
            enable_answer_cache=latency.enable_answer_cache,
            enable_evidence_context_reuse=latency.enable_evidence_context_reuse,
        )
    return PipelineProfileConfig(
        name=PROFILE_FULL_EVAL,
        checker_mode="traceable",
        llm_max_tokens=2048,
        max_gap_rounds=4,
        max_queries_per_round=3,
        per_query_retrieve_top_k_chunks=20,
        max_raw_chunks_per_checker_call=30,
        max_new_raw_chunks_per_round=5,
        max_verifier_repair_rounds=3,
        enable_answer_cache=False,
        enable_evidence_context_reuse=True,
    )
