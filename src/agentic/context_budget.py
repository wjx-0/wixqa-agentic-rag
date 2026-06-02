from __future__ import annotations

import math
from typing import Iterable

from pydantic import BaseModel, Field


DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS = 16384
DEFAULT_RESERVED_OUTPUT_RATIO = 0.12
DEFAULT_SAFETY_MARGIN_RATIO = 0.03
TARGET_AFTER_COMPACT_RATIO = 0.60

HEALTHY_UNTIL_RATIO = 0.70
WATCH_UNTIL_RATIO = 0.80
SOFT_AUTO_COMPACT_RATIO = 0.80
HARD_AUTO_COMPACT_RATIO = 0.90
EMERGENCY_COMPACT_RATIO = 0.95


class ContextBudgetConfig(BaseModel):
    model_context_window_tokens: int = DEFAULT_MODEL_CONTEXT_WINDOW_TOKENS
    reserved_output_ratio: float = DEFAULT_RESERVED_OUTPUT_RATIO
    safety_margin_ratio: float = DEFAULT_SAFETY_MARGIN_RATIO
    target_after_compact_ratio: float = TARGET_AFTER_COMPACT_RATIO


class ContextUsageSnapshot(BaseModel):
    llm_call_id: str
    phase: str
    qid: str | None = None
    model_context_window_tokens: int
    projected_input_tokens: int
    projected_output_tokens: int
    projected_total_tokens: int
    reserved_output_ratio: float
    safety_margin_ratio: float
    usage_ratio: float
    budget_status: str
    input_chunk_ids: list[str] = Field(default_factory=list)


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_texts_tokens(texts: Iterable[str | None]) -> int:
    return sum(estimate_tokens(text) for text in texts)


def classify_usage_ratio(usage_ratio: float) -> str:
    if usage_ratio >= EMERGENCY_COMPACT_RATIO:
        return "emergency_compact"
    if usage_ratio >= HARD_AUTO_COMPACT_RATIO:
        return "hard_auto_compact"
    if usage_ratio >= SOFT_AUTO_COMPACT_RATIO:
        return "soft_auto_compact"
    if usage_ratio >= HEALTHY_UNTIL_RATIO:
        return "watch"
    return "healthy"


def build_context_usage_snapshot(
    *,
    llm_call_id: str,
    phase: str,
    qid: str | None = None,
    input_texts: Iterable[str | None] = (),
    input_chunk_ids: list[str] | None = None,
    config: ContextBudgetConfig | None = None,
) -> ContextUsageSnapshot:
    config = config or ContextBudgetConfig()
    if config.model_context_window_tokens <= 0:
        raise ValueError("model_context_window_tokens must be positive.")
    if config.reserved_output_ratio < 0 or config.safety_margin_ratio < 0:
        raise ValueError("reserved_output_ratio and safety_margin_ratio must be non-negative.")

    projected_input_tokens = estimate_texts_tokens(input_texts)
    projected_output_tokens = math.ceil(
        config.model_context_window_tokens * config.reserved_output_ratio
    )
    safety_margin_tokens = math.ceil(
        config.model_context_window_tokens * config.safety_margin_ratio
    )
    projected_total_tokens = (
        projected_input_tokens + projected_output_tokens + safety_margin_tokens
    )
    usage_ratio = projected_total_tokens / config.model_context_window_tokens
    return ContextUsageSnapshot(
        llm_call_id=llm_call_id,
        phase=phase,
        qid=qid,
        model_context_window_tokens=config.model_context_window_tokens,
        projected_input_tokens=projected_input_tokens,
        projected_output_tokens=projected_output_tokens,
        projected_total_tokens=projected_total_tokens,
        reserved_output_ratio=config.reserved_output_ratio,
        safety_margin_ratio=config.safety_margin_ratio,
        usage_ratio=usage_ratio,
        budget_status=classify_usage_ratio(usage_ratio),
        input_chunk_ids=list(input_chunk_ids or []),
    )
