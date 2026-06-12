from __future__ import annotations

import re
from typing import Any

from src.agentic.dialogue_state import DialogueAgentResult, DialogueState, TurnType
from src.agentic.dialogue_relevance import (
    has_question_cue,
    is_context_related,
    looks_like_follow_up,
    looks_like_standalone_question,
)
from src.agentic.intent_gate import IntentGateDecision, RuleBasedIntentGate
from src.utils.text_utils import compact_text


CONSTRAINT_PATTERNS = (
    re.compile(r"\bi use\b.+", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:site|account|plan|domain|business)\b.+", re.IGNORECASE),
)


class DialogueAgent:
    name = "dialogue_agent"

    def __init__(self, intent_gate: Any | None = None) -> None:
        self.intent_gate = intent_gate or RuleBasedIntentGate()

    def run(
        self,
        *,
        state: DialogueState,
        user_message: str,
        turn_id: str,
    ) -> DialogueAgentResult:
        message = compact_text(user_message)
        if not message:
            raise ValueError("user_message must not be empty.")
        intent = self.intent_gate.run(state=state, message=message)
        turn_type = classify_turn_type(state, message, intent)
        constraints = extract_constraints(message)

        if turn_type in {"topic_shift", "needs_clarification"}:
            state.clear_topic_memory()
        for constraint in constraints:
            state.remember_constraint(constraint)

        if turn_type == "small_talk":
            active_question = state.active_question or ""
        elif turn_type == "needs_clarification":
            active_question = message
        elif turn_type in {"new_question", "topic_shift"}:
            active_question = message
        else:
            active_question = state.active_question or message
        state.add_user_turn(turn_id=turn_id, message=message, turn_type=turn_type)
        if turn_type != "small_talk":
            state.active_question = active_question
        return DialogueAgentResult(
            turn_id=turn_id,
            turn_type=turn_type,
            user_message=message,
            active_question=active_question,
            topic_shift=turn_type == "topic_shift",
            constraints=constraints,
            intent_route=intent.route,
            intent_reason=intent.reason,
            intent_confidence=intent.confidence,
            clarification_question=intent.clarification_question,
            intent_used_api=intent.used_api,
            intent_fallback_used=intent.fallback_used,
            intent_error=intent.error,
        )


def classify_turn_type(
    state: DialogueState,
    message: str,
    intent: IntentGateDecision,
) -> TurnType:
    if extract_constraints(message) and not has_question_cue(message):
        return "correction_or_constraint"
    if intent.route == "direct_answer":
        return "small_talk"
    if intent.route == "clarify":
        return "needs_clarification"
    if last_assistant_asked_clarification(state):
        return "clarification_answer"
    if not state.turns or not state.active_question:
        return "new_question"

    context_texts = active_context_texts(state)
    if looks_like_follow_up(message):
        return "follow_up_question"
    if looks_like_standalone_question(message):
        return (
            "follow_up_question"
            if is_context_related(message, context_texts)
            else "topic_shift"
        )
    if has_question_cue(message) and is_context_related(message, context_texts):
        return "follow_up_question"
    return "topic_shift"


def last_assistant_asked_clarification(state: DialogueState) -> bool:
    for turn in reversed(state.turns):
        if turn.role == "assistant":
            return turn.metadata.get("route") == "clarify"
    return False


def extract_constraints(message: str) -> list[str]:
    constraints = []
    for pattern in CONSTRAINT_PATTERNS:
        match = pattern.search(message)
        if match:
            constraints.append(compact_text(match.group(0)))
    return constraints


def active_context_texts(state: DialogueState) -> list[str]:
    texts = [compact_text(state.active_question), compact_text(state.rewritten_standalone_question)]
    if state.evidence_context is not None:
        texts.append(compact_text(state.evidence_context.question))
        texts.extend(compact_text(item.title) for item in state.evidence_context.active_items[:5])
    return [text for text in texts if text]
