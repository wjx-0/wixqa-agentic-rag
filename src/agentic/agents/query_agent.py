from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.agentic.dialogue_relevance import (
    context_relevance_score,
    has_bridge_overlap,
)
from src.agentic.dialogue_state import DialogueAgentResult, DialogueState, QueryAgentResult
from src.llm.structured import (
    StructuredLLMCaller,
    parse_json_object,
)
from src.utils.text_utils import compact_text


DEFAULT_QUERY_REWRITE_MAX_TOKENS = 512
DEFAULT_QUERY_REWRITE_TEMPERATURE = 0.0
DEFAULT_RECENT_TURNS_FOR_REWRITE = 6
# Reuse only when the current message shares a meaningful fraction of concrete
# terms with the active context; generic follow-up words alone should not pass.
DEFAULT_CONTEXT_REUSE_MIN_SCORE = 0.25
DEFAULT_MEMORY_SUMMARY_MAX_TOKENS = 512
DEFAULT_MEMORY_SUMMARY_MAX_CHARS = 1200


class QueryRewriteError(RuntimeError):
    pass


class QueryRewriteOutput(BaseModel):
    standalone_question: str
    used_api: bool = False
    fallback_used: bool = False
    error: str | None = None


class QueryRewritePayload(BaseModel):
    standalone_question: str
    reason: Any = ""


class ConversationMemoryPayload(BaseModel):
    summary: str


class ContextReuseDecision(BaseModel):
    reuse: bool
    score: float = 0.0
    reason: str = ""


class LLMQueryRewriter:
    uses_api = True

    def __init__(
        self,
        *,
        client: Any,
        temperature: float = DEFAULT_QUERY_REWRITE_TEMPERATURE,
        max_tokens: int = DEFAULT_QUERY_REWRITE_MAX_TOKENS,
    ) -> None:
        if max_tokens <= 0:
            raise QueryRewriteError("query rewrite max_tokens must be positive.")
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.caller = StructuredLLMCaller(
            client=client,
            schema=QueryRewritePayload,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_min_tokens=512,
        )

    def rewrite(
        self,
        *,
        state: DialogueState,
        dialogue: DialogueAgentResult,
    ) -> QueryRewriteOutput:
        messages = build_query_rewrite_messages(state=state, dialogue=dialogue)
        payload = self.caller.call(messages)
        standalone = normalize_query_rewrite_payload(payload)
        return QueryRewriteOutput(standalone_question=standalone, used_api=True)


class LLMConversationMemorySummarizer:
    uses_api = True

    def __init__(
        self,
        *,
        client: Any,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MEMORY_SUMMARY_MAX_TOKENS,
    ) -> None:
        if max_tokens <= 0:
            raise QueryRewriteError("memory summary max_tokens must be positive.")
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.caller = StructuredLLMCaller(
            client=client,
            schema=ConversationMemoryPayload,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_min_tokens=max_tokens,
        )

    def summarize(
        self,
        *,
        existing_summary: str | None,
        turns: list[Any],
        constraints: list[str],
    ) -> str:
        payload = self.caller.call(
            build_memory_summary_messages(
                existing_summary=existing_summary,
                turns=turns,
                constraints=constraints,
            )
        )
        return compact_text(payload.summary)[:DEFAULT_MEMORY_SUMMARY_MAX_CHARS]


class QueryAgent:
    name = "query_agent"

    def __init__(
        self,
        rewriter: Any | None = None,
        *,
        memory_summarizer: Any | None = None,
        recent_turns_for_rewrite: int = DEFAULT_RECENT_TURNS_FOR_REWRITE,
    ) -> None:
        self.rewriter = rewriter
        self.memory_summarizer = memory_summarizer
        self.recent_turns_for_rewrite = recent_turns_for_rewrite

    def run(
        self,
        *,
        state: DialogueState,
        dialogue: DialogueAgentResult,
    ) -> QueryAgentResult:
        if dialogue.turn_type == "small_talk":
            return QueryAgentResult(
                standalone_question=compact_text(dialogue.user_message),
                route="direct_answer",
                reused_context=False,
                turn_type=dialogue.turn_type,
                memory_summary_used=bool(state.context_memory_summary),
                memory_summary_fallback_used=state.context_memory_summary_fallback_used,
                memory_summary_error=state.context_memory_summary_error,
            )
        if dialogue.turn_type == "needs_clarification":
            return QueryAgentResult(
                standalone_question=compact_text(dialogue.user_message),
                route="clarify",
                reused_context=False,
                turn_type=dialogue.turn_type,
                memory_summary_used=bool(state.context_memory_summary),
                memory_summary_fallback_used=state.context_memory_summary_fallback_used,
                memory_summary_error=state.context_memory_summary_error,
            )

        ensure_context_memory_summary(
            state,
            dialogue=dialogue,
            summarizer=self.memory_summarizer,
            max_recent_turns=self.recent_turns_for_rewrite,
        )
        rewrite = rewrite_query(self.rewriter, state=state, dialogue=dialogue)
        standalone = rewrite.standalone_question
        reuse = should_reuse_context(state, dialogue, standalone_question=standalone)
        route = "reuse_context" if reuse.reuse else "retrieve"
        state.rewritten_standalone_question = standalone
        return QueryAgentResult(
            standalone_question=standalone,
            route=route,
            reused_context=route == "reuse_context",
            turn_type=dialogue.turn_type,
            context_reuse_score=reuse.score,
            context_reuse_reason=reuse.reason,
            memory_summary_used=bool(state.context_memory_summary),
            memory_summary_fallback_used=state.context_memory_summary_fallback_used,
            memory_summary_error=state.context_memory_summary_error,
            rewrite_used_api=rewrite.used_api,
            rewrite_fallback_used=rewrite.fallback_used,
            rewrite_error=rewrite.error,
        )


def rewrite_query(
    rewriter: Any | None,
    *,
    state: DialogueState,
    dialogue: DialogueAgentResult,
) -> QueryRewriteOutput:
    fallback = rewrite_with_rules(state, dialogue)
    if rewriter is None:
        return QueryRewriteOutput(standalone_question=fallback)

    try:
        raw_result = rewriter.rewrite(state=state, dialogue=dialogue)
        result = normalize_rewrite_result(raw_result, rewriter=rewriter)
        if not result.standalone_question:
            raise QueryRewriteError("query rewriter returned an empty standalone_question.")
        return result
    except Exception as exc:
        return QueryRewriteOutput(
            standalone_question=fallback,
            fallback_used=True,
            error=compact_text(str(exc))[:240],
        )


def normalize_rewrite_result(raw_result: Any, *, rewriter: Any) -> QueryRewriteOutput:
    if isinstance(raw_result, QueryRewriteOutput):
        raw_result.standalone_question = compact_text(raw_result.standalone_question)
        return raw_result
    return QueryRewriteOutput(
        standalone_question=compact_text(raw_result),
        used_api=bool(getattr(rewriter, "uses_api", False)),
    )


def rewrite_with_rules(state: DialogueState, dialogue: DialogueAgentResult) -> str:
    message = compact_text(dialogue.user_message)
    if dialogue.turn_type in {"new_question", "topic_shift"}:
        return message
    base = compact_text(state.active_question) or message
    summary = compact_text(state.context_memory_summary)
    constraints = "; ".join(state.unresolved_user_constraints[-3:])
    parts = [base]
    if summary:
        parts.append(f"Conversation memory: {summary}")
    if message != base:
        parts.append(f"Follow-up: {message}")
    if constraints:
        parts.append(f"User constraints: {constraints}")
    return " ".join(parts)


def should_reuse_context(
    state: DialogueState,
    dialogue: DialogueAgentResult,
    *,
    standalone_question: str,
    min_relevance_score: float = DEFAULT_CONTEXT_REUSE_MIN_SCORE,
) -> ContextReuseDecision:
    if state.evidence_context is None:
        return ContextReuseDecision(reuse=False, reason="no evidence context")
    if dialogue.turn_type not in {
        "follow_up_question",
        "clarification_answer",
        "correction_or_constraint",
    }:
        return ContextReuseDecision(reuse=False, reason=f"turn type {dialogue.turn_type}")

    context_texts = context_texts_for_reuse(state)
    message = compact_text(dialogue.user_message)
    standalone = compact_text(standalone_question)
    message_score = context_relevance_score(message, context_texts)
    standalone_score = context_relevance_score(standalone, context_texts)
    bridge_overlap = has_bridge_overlap(message, context_texts)
    score = max(message_score, standalone_score if message_score > 0.0 else 0.0)
    meets_threshold = message_score >= min_relevance_score or (
        message_score > 0.0 and standalone_score >= min_relevance_score
    )
    if dialogue.turn_type in {"clarification_answer", "correction_or_constraint"}:
        return ContextReuseDecision(
            reuse=True,
            score=score,
            reason=f"{dialogue.turn_type} updates active context",
        )
    if meets_threshold or bridge_overlap:
        return ContextReuseDecision(
            reuse=True,
            score=score,
            reason=(
                f"relevance score {score:.2f} >= {min_relevance_score:.2f}"
                if meets_threshold
                else "message bridges active evidence context"
            ),
        )
    return ContextReuseDecision(
        reuse=False,
        score=score,
        reason=f"relevance score {score:.2f} < {min_relevance_score:.2f}",
    )


def context_texts_for_reuse(state: DialogueState) -> list[str]:
    texts = [
        compact_text(state.active_question),
        compact_text(state.rewritten_standalone_question),
    ]
    context = state.evidence_context
    if context is not None:
        texts.append(compact_text(context.question))
        texts.extend(compact_text(item.title) for item in context.active_items[:5])
        texts.extend(compact_text(item.text_preview) for item in context.active_items[:2])
        texts.append(compact_text(context.compressed_summary))
    return [text for text in texts if text]


def ensure_context_memory_summary(
    state: DialogueState,
    *,
    dialogue: DialogueAgentResult,
    summarizer: Any | None,
    max_recent_turns: int = DEFAULT_RECENT_TURNS_FOR_REWRITE,
) -> None:
    if max_recent_turns <= 0:
        raise QueryRewriteError("max_recent_turns must be positive.")
    if dialogue.turn_type in {"new_question", "topic_shift", "needs_clarification", "small_talk"}:
        return
    older_turns = state.turns[:-max_recent_turns]
    if not older_turns:
        return
    if state.context_memory_summary_turn_count >= len(older_turns):
        return
    turns_to_summarize = older_turns[state.context_memory_summary_turn_count :]
    try:
        if summarizer is None:
            raise QueryRewriteError("memory summarizer is not configured")
        summary = summarize_with_injected_summarizer(
            summarizer,
            existing_summary=state.context_memory_summary,
            turns=turns_to_summarize,
            constraints=state.unresolved_user_constraints,
        )
        if not summary:
            raise QueryRewriteError("memory summarizer returned an empty summary")
        state.context_memory_summary = summary
        state.context_memory_summary_fallback_used = False
        state.context_memory_summary_error = None
    except Exception as exc:
        state.context_memory_summary = build_rule_memory_summary(
            existing_summary=state.context_memory_summary,
            turns=turns_to_summarize,
            constraints=state.unresolved_user_constraints,
        )
        state.context_memory_summary_fallback_used = True
        state.context_memory_summary_error = compact_text(str(exc))[:240]
    state.context_memory_summary_turn_count = len(older_turns)


def summarize_with_injected_summarizer(
    summarizer: Any,
    *,
    existing_summary: str | None,
    turns: list[Any],
    constraints: list[str],
) -> str:
    if hasattr(summarizer, "summarize"):
        return compact_text(
            summarizer.summarize(
                existing_summary=existing_summary,
                turns=turns,
                constraints=constraints,
            )
        )[:DEFAULT_MEMORY_SUMMARY_MAX_CHARS]
    return compact_text(summarizer(existing_summary, turns, constraints))[:DEFAULT_MEMORY_SUMMARY_MAX_CHARS]


def build_rule_memory_summary(
    *,
    existing_summary: str | None,
    turns: list[Any],
    constraints: list[str],
) -> str:
    lines = []
    if existing_summary:
        lines.append(compact_text(existing_summary))
    for turn in turns[-8:]:
        role = compact_text(getattr(turn, "role", "turn"))
        message = compact_text(getattr(turn, "message", ""))
        if message:
            lines.append(f"{role}: {message}")
    if constraints:
        lines.append("Constraints: " + "; ".join(compact_text(value) for value in constraints[-3:]))
    return compact_text(" | ".join(line for line in lines if line))[:DEFAULT_MEMORY_SUMMARY_MAX_CHARS]


def build_memory_summary_messages(
    *,
    existing_summary: str | None,
    turns: list[Any],
    constraints: list[str],
) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        [
            "Summarize older Wix support dialogue for future query rewriting.",
            "Keep durable user goals, product names, constraints, errors, and resolved context.",
            "Do not include gold labels, evaluation metadata, or facts not present in the dialogue.",
            "Return one compact JSON object only: {\"summary\":\"...\"}.",
        ]
    )
    turn_lines = [
        f"{compact_text(getattr(turn, 'role', 'turn'))}: {compact_text(getattr(turn, 'message', ''))}"
        for turn in turns
        if compact_text(getattr(turn, "message", ""))
    ]
    user_prompt = "\n\n".join(
        [
            f"Existing summary:\n{compact_text(existing_summary) or '(none)'}",
            "Older turns to fold in:\n" + ("\n".join(turn_lines) if turn_lines else "(none)"),
            "Current unresolved constraints:\n"
            + ("; ".join(compact_text(value) for value in constraints[-5:]) if constraints else "(none)"),
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_query_rewrite_messages(
    *,
    state: DialogueState,
    dialogue: DialogueAgentResult,
    max_recent_turns: int = DEFAULT_RECENT_TURNS_FOR_REWRITE,
) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        [
            "You rewrite Wix Help Center support messages into standalone retrieval queries.",
            "Return one compact JSON object only. Do not use markdown. Do not answer the user.",
            "The standalone_question must be a single clear question suitable for BM25/vector retrieval.",
            "Use the active support question and recent conversation only to resolve follow-ups.",
            "Preserve Wix product names, user constraints, error messages, plan/domain/payment details, and the user's language when possible.",
            "If the current message is already standalone, keep it mostly unchanged.",
            "Do not invent facts, account details, or unsupported products.",
            "JSON schema: {\"standalone_question\":\"question for retrieval\",\"reason\":\"short reason\"}",
        ]
    )
    recent_turns = state.turns[-max_recent_turns:]
    recent_lines = [
        f"{turn.role}: {compact_text(turn.message)}"
        for turn in recent_turns
        if compact_text(turn.message)
    ]
    constraints = "; ".join(state.unresolved_user_constraints[-3:])
    user_prompt = "\n\n".join(
        [
            f"Turn type:\n{dialogue.turn_type}",
            f"Active support question:\n{compact_text(state.active_question) or '(none)'}",
            f"Conversation memory summary:\n{compact_text(state.context_memory_summary) or '(none)'}",
            "Recent conversation:\n" + ("\n".join(recent_lines) if recent_lines else "(none)"),
            f"Remembered user constraints:\n{constraints or '(none)'}",
            f"Current user message:\n{compact_text(dialogue.user_message)}",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_query_rewrite_response(raw_text: str) -> str:
    try:
        payload = QueryRewritePayload(**parse_json_object(raw_text))
        return normalize_query_rewrite_payload(payload)
    except Exception as exc:
        raise QueryRewriteError(f"query rewriter returned invalid JSON: {exc}") from exc


def normalize_query_rewrite_payload(payload: QueryRewritePayload) -> str:
    standalone = compact_text(payload.standalone_question)
    if not standalone:
        raise QueryRewriteError("query rewriter JSON is missing standalone_question.")
    return standalone
