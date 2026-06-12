from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.agentic.dialogue_relevance import (
    contains_any_term,
    has_issue_signal,
    has_question_cue,
    is_context_related,
    looks_like_follow_up,
    support_intent_score,
)
from src.agentic.dialogue_state import DialogueState, IntentRoute
from src.llm.structured import (
    StructuredLLMCaller,
    parse_json_object,
)
from src.utils.text_utils import compact_text


VALID_INTENT_ROUTES = {"retrieve", "clarify", "direct_answer"}


class IntentGateDecision(BaseModel):
    route: IntentRoute
    reason: str = ""
    confidence: float | None = None
    clarification_question: str | None = None
    used_api: bool = False
    fallback_used: bool = False
    error: str | None = None


class IntentGatePayload(BaseModel):
    route: Any
    reason: Any = ""
    confidence: Any = None
    clarification_question: Any = None


class RuleBasedIntentGateConfig(BaseModel):
    domain_name: str = "Wix Help Center"
    domain_terms: tuple[str, ...] = Field(default=("wix",))
    retrieve_score_threshold: int = 4
    clarify_score_threshold: int = 2


class RuleBasedIntentGate:
    name = "rule_based_intent_gate"

    def __init__(self, config: RuleBasedIntentGateConfig | None = None) -> None:
        self.config = config or RuleBasedIntentGateConfig()

    def run(self, *, state: DialogueState, message: str) -> IntentGateDecision:
        text = compact_text(message)
        if not text:
            return IntentGateDecision(route="direct_answer", reason="empty message")

        score = support_intent_score(text, domain_terms=self.config.domain_terms)
        active_context = [compact_text(state.active_question)]
        if (
            state.active_question
            and looks_like_follow_up(text)
            and score >= self.config.clarify_score_threshold
        ):
            return IntentGateDecision(
                route="retrieve",
                reason="contextual follow-up to active support question",
                confidence=0.75,
            )
        if (
            state.active_question
            and has_question_cue(text)
            and is_context_related(
                text,
                active_context,
                domain_terms=self.config.domain_terms,
            )
        ):
            return IntentGateDecision(
                route="retrieve",
                reason="question is related to active support context",
                confidence=0.72,
            )
        can_retrieve_without_clarification = (
            has_question_cue(text)
            or has_issue_signal(text)
            or contains_any_term(text.casefold(), self.config.domain_terms)
        )
        if score >= self.config.retrieve_score_threshold and can_retrieve_without_clarification:
            return IntentGateDecision(
                route="retrieve",
                reason="support intent score passed retrieval threshold",
                confidence=min(0.95, 0.45 + score / 10),
            )
        if score >= self.config.clarify_score_threshold:
            return IntentGateDecision(
                route="clarify",
                reason="support intent is plausible but underspecified",
                confidence=min(0.7, 0.35 + score / 10),
                clarification_question=default_clarification_question(
                    text,
                    domain_name=self.config.domain_name,
                ),
            )
        return IntentGateDecision(route="direct_answer", reason="no support intent detected")


class LLMIntentGate:
    name = "llm_intent_gate"

    def __init__(
        self,
        *,
        client: Any,
        fallback: RuleBasedIntentGate | None = None,
        domain_name: str = "Wix Help Center",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("intent gate max_tokens must be positive.")
        self.client = client
        self.fallback = fallback or RuleBasedIntentGate()
        self.domain_name = compact_text(domain_name) or "support knowledge base"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.caller = StructuredLLMCaller(
            client=client,
            schema=IntentGatePayload,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_min_tokens=512,
        )

    def run(self, *, state: DialogueState, message: str) -> IntentGateDecision:
        messages = build_intent_gate_messages(
            state=state,
            message=message,
            domain_name=self.domain_name,
        )
        try:
            payload = self.caller.call(messages)
            decision = intent_gate_decision_from_payload(payload)
            decision = apply_route_overrides(message, decision)
            decision.used_api = True
            return decision
        except Exception as exc:
            last_error = exc
        fallback = self.fallback.run(state=state, message=message)
        fallback.fallback_used = True
        fallback.error = compact_text(str(last_error))[:240] if last_error else "intent gate failed"
        return fallback


def build_intent_gate_messages(
    *,
    state: DialogueState,
    message: str,
    domain_name: str = "Wix Help Center",
    max_recent_turns: int = 6,
) -> list[dict[str, str]]:
    domain = compact_text(domain_name) or "support knowledge base"
    system_prompt = "\n".join(
        [
            f"You are a fast intent gate before a {domain} RAG pipeline.",
            "Classify whether the current user message needs knowledge-base retrieval.",
            "Return one compact JSON object only. Do not use markdown. Do not answer the user's support question.",
            "Routes:",
            f"- retrieve: the user asks a concrete {domain}/support question, or a follow-up to the active support question.",
            f"- clarify: the user likely wants {domain}/support help but the object, product, or action is too ambiguous to retrieve safely.",
            f"- direct_answer: greetings, thanks, small talk, bot meta questions, or anything that does not need {domain} evidence.",
            "Use conversation context, but do not treat a new greeting as a follow-up.",
            "Short underspecified needs such as '我想收款', 'I want to get paid', or '怎么弄' should be clarify unless the active question already specifies the product/task.",
            "Concrete payment failure questions such as '支付不了', '付款失败', 'payment failed', or 'payment declined' should be retrieve.",
            "If route=clarify, write one short clarification_question in the user's language.",
            "JSON schema: {\"route\":\"retrieve|clarify|direct_answer\",\"confidence\":0.0,\"reason\":\"short reason\",\"clarification_question\":string|null}",
        ]
    )
    recent_turns = state.turns[-max_recent_turns:]
    recent_lines = [
        f"{turn.role}: {compact_text(turn.message)}"
        for turn in recent_turns
        if compact_text(turn.message)
    ]
    user_prompt = "\n\n".join(
        [
            f"Active support question:\n{compact_text(state.active_question) or '(none)'}",
            "Recent conversation:\n" + ("\n".join(recent_lines) if recent_lines else "(none)"),
            f"Current user message:\n{compact_text(message)}",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_intent_gate_response(raw_text: str) -> IntentGateDecision:
    try:
        payload = IntentGatePayload(**parse_json_object(raw_text))
        return intent_gate_decision_from_payload(payload)
    except Exception as exc:
        raise ValueError(f"intent gate returned invalid JSON: {exc}") from exc


def intent_gate_decision_from_payload(payload: IntentGatePayload) -> IntentGateDecision:
    route = normalize_route(payload.route)
    reason = compact_text(payload.reason)
    clarification_question = compact_text(payload.clarification_question) or None
    confidence = coerce_confidence(payload.confidence)
    if route == "clarify" and not clarification_question:
        clarification_question = default_clarification_question("")
    return IntentGateDecision(
        route=route,
        reason=reason,
        confidence=confidence,
        clarification_question=clarification_question,
    )


def apply_route_overrides(message: str, decision: IntentGateDecision) -> IntentGateDecision:
    if decision.route == "clarify" and is_concrete_payment_failure(message):
        decision.route = "retrieve"
        decision.reason = compact_text(
            f"{decision.reason}; concrete payment failure question"
        ).lstrip("; ")
        decision.clarification_question = None
    return decision


def is_concrete_payment_failure(message: str) -> bool:
    lowered = compact_text(message).casefold()
    phrases = (
        "支付不了",
        "付款失败",
        "支付失败",
        "付款不了",
        "扣款失败",
        "payment failed",
        "payment declined",
        "declined payment",
        "can't pay",
        "cannot pay",
        "failed payment",
    )
    return any(phrase in lowered for phrase in phrases)


def normalize_route(value: Any) -> IntentRoute:
    route = compact_text(value).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "search": "retrieve",
        "rag": "retrieve",
        "knowledge_base": "retrieve",
        "ask_clarification": "clarify",
        "clarification": "clarify",
        "direct": "direct_answer",
        "small_talk": "direct_answer",
        "no_retrieval": "direct_answer",
    }
    route = aliases.get(route, route)
    if route not in VALID_INTENT_ROUTES:
        raise ValueError(f"unsupported intent route: {route!r}")
    return route  # type: ignore[return-value]


def coerce_confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def default_clarification_question(message: str, *, domain_name: str = "Wix") -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in message):
        return f"你想咨询 {domain_name} 的哪个具体功能或操作？"
    return f"Which {domain_name} feature or task do you want help with?"
