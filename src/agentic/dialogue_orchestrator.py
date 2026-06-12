from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from src.agentic.agents import (
    AnswerAgent,
    DialogueAgent,
    EvidenceAgent,
    QueryAgent,
    VerifierAgent,
)
from src.agentic.dialogue_state import (
    AgentEvent,
    AnswerTrace,
    DialogueState,
    LatencyConfig,
    VerifierTrace,
    normalize_profile,
    profile_config,
)
from src.utils.io_utils import model_to_dict
from src.utils.text_utils import compact_text, contains_cjk
from src.utils.time_utils import elapsed_ms


class DialogueOrchestrator:
    def __init__(
        self,
        *,
        dialogue_agent: DialogueAgent | None = None,
        query_agent: QueryAgent | None = None,
        evidence_agent: EvidenceAgent,
        answer_agent: AnswerAgent | None = None,
        verifier_agent: VerifierAgent | None = None,
        latency: LatencyConfig | None = None,
    ) -> None:
        self.dialogue_agent = dialogue_agent or DialogueAgent()
        self.query_agent = query_agent or QueryAgent()
        self.evidence_agent = evidence_agent
        self.answer_agent = answer_agent or AnswerAgent()
        self.verifier_agent = verifier_agent or VerifierAgent()
        self.latency = latency or LatencyConfig()
        self.states: dict[str, DialogueState] = {}
        self._states_lock = threading.RLock()
        self._conversation_locks: dict[str, threading.RLock] = {}

    def run_turn(
        self,
        conversation_id: str,
        user_message: str,
        *,
        user_id: str | None = None,
        profile: str = "fast",
    ) -> Iterator[AgentEvent]:
        lock = self.conversation_lock(conversation_id)
        with lock:
            yield from self._run_turn_unlocked(
                conversation_id,
                user_message,
                user_id=user_id,
                profile=profile,
            )

    def conversation_lock(self, conversation_id: str) -> threading.RLock:
        with self._states_lock:
            lock = self._conversation_locks.get(conversation_id)
            if lock is None:
                lock = threading.RLock()
                self._conversation_locks[conversation_id] = lock
            return lock

    def _run_turn_unlocked(
        self,
        conversation_id: str,
        user_message: str,
        *,
        user_id: str | None = None,
        profile: str = "fast",
    ) -> Iterator[AgentEvent]:
        normalized_profile = normalize_profile(profile)
        config = profile_config(normalized_profile, self.latency)
        started_at = time.monotonic()
        state = self.states.setdefault(conversation_id, DialogueState(conversation_id=conversation_id))
        turn_id = build_turn_id(state)
        event_index = 1

        yield self.event(
            conversation_id=conversation_id,
            turn_id=turn_id,
            event_index=event_index,
            agent_name="orchestrator",
            event_type="status",
            message="正在处理你的消息。",
            payload={"user_id": user_id},
            started_at=started_at,
            profile=normalized_profile,
        )
        event_index += 1

        stage_started_at = time.monotonic()
        dialogue = self.dialogue_agent.run(
            state=state,
            user_message=user_message,
            turn_id=turn_id,
        )
        dialogue_latency_ms = elapsed_ms(stage_started_at)
        dialogue_payload = model_to_dict(dialogue)
        dialogue_payload["stage_latency_ms"] = {"dialogue_agent": dialogue_latency_ms}
        yield self.event(
            conversation_id=conversation_id,
            turn_id=turn_id,
            event_index=event_index,
            agent_name=self.dialogue_agent.name,
            event_type="dialogue_state",
            message=f"Turn classified as {dialogue.turn_type}.",
            payload=dialogue_payload,
            started_at=started_at,
            profile=normalized_profile,
        )
        event_index += 1

        stage_started_at = time.monotonic()
        query = self.query_agent.run(state=state, dialogue=dialogue)
        query_latency_ms = elapsed_ms(stage_started_at)
        query_payload = model_to_dict(query)
        query_payload["stage_latency_ms"] = {"query_agent": query_latency_ms}
        yield self.event(
            conversation_id=conversation_id,
            turn_id=turn_id,
            event_index=event_index,
            agent_name=self.query_agent.name,
            event_type="query",
            message="Query prepared.",
            payload=query_payload,
            started_at=started_at,
            profile=normalized_profile,
        )
        event_index += 1

        if query.route in {"direct_answer", "clarify"}:
            stage_started_at = time.monotonic()
            answer = AnswerTrace(
                answer=non_retrieval_answer_for(dialogue, query.route),
                generation_mode=query.route,
            )
            answer_latency_ms = elapsed_ms(stage_started_at)
            yield self.event(
                conversation_id=conversation_id,
                turn_id=turn_id,
                event_index=event_index,
                agent_name=self.answer_agent.name,
                event_type="answer",
                message="Non-retrieval response prepared.",
                payload={
                    "answer": model_to_dict(answer),
                    "route": query.route,
                    "stage_latency_ms": {"answer_agent": answer_latency_ms},
                },
                started_at=started_at,
                profile=normalized_profile,
            )
            event_index += 1

            verifier = VerifierTrace(
                status="insufficient_evidence" if query.route == "clarify" else "ready_to_answer",
                reason="Intent gate requested clarification."
                if query.route == "clarify"
                else "Direct response does not need Wix Help Center evidence.",
            )
            state.add_assistant_turn(
                turn_id=turn_id,
                message=answer.answer,
                standalone_question=None,
                citation_ids=[],
                metadata={
                    "verifier_status": verifier.status,
                    "profile": normalized_profile,
                    "route": query.route,
                },
            )
            yield self.event(
                conversation_id=conversation_id,
                turn_id=turn_id,
                event_index=event_index,
                agent_name="orchestrator",
                event_type="clarification_request" if query.route == "clarify" else "final_answer",
                message=answer.answer,
                payload={
                    "answer": model_to_dict(answer),
                    "verifier": model_to_dict(verifier),
                    "route": query.route,
                },
                started_at=started_at,
                profile=normalized_profile,
                is_final=True,
            )
            return

        stage_started_at = time.monotonic()
        evidence = self.evidence_agent.run(state=state, query=query, profile=config)
        evidence_latency_ms = elapsed_ms(stage_started_at)
        yield self.event(
            conversation_id=conversation_id,
            turn_id=turn_id,
            event_index=event_index,
            agent_name=self.evidence_agent.name,
            event_type="evidence",
            message="Evidence context updated.",
            payload=evidence_event_payload(evidence, evidence_latency_ms),
            started_at=started_at,
            profile=normalized_profile,
        )
        event_index += 1

        stage_started_at = time.monotonic()
        answer = self.answer_agent.run(
            context=evidence.context,
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        answer_latency_ms = elapsed_ms(stage_started_at)
        yield self.event(
            conversation_id=conversation_id,
            turn_id=turn_id,
            event_index=event_index,
            agent_name=self.answer_agent.name,
            event_type="answer",
            message="Answer drafted.",
            payload={
                "answer": model_to_dict(answer),
                "stage_latency_ms": {"answer_agent": answer_latency_ms},
            },
            started_at=started_at,
            profile=normalized_profile,
        )
        event_index += 1

        stage_started_at = time.monotonic()
        verifier = self.verifier_agent.run(answer=answer, context=evidence.context)
        verifier_latency_ms = elapsed_ms(stage_started_at)
        verifier_payload = model_to_dict(verifier)
        verifier_payload["stage_latency_ms"] = {"verifier_agent": verifier_latency_ms}
        yield self.event(
            conversation_id=conversation_id,
            turn_id=turn_id,
            event_index=event_index,
            agent_name=self.verifier_agent.name,
            event_type="verification",
            message=f"Verifier status: {verifier.status}.",
            payload=verifier_payload,
            started_at=started_at,
            profile=normalized_profile,
        )
        event_index += 1

        repair_round = 0
        while should_attempt_verifier_repair(verifier, repair_round, config):
            repair_round += 1
            if verifier.status == "insufficient_evidence":
                stage_started_at = time.monotonic()
                evidence = self.evidence_agent.run(
                    state=state,
                    query=query,
                    profile=config,
                    force_refresh=True,
                    suggested_queries=verifier.suggested_queries,
                )
                evidence_latency_ms = elapsed_ms(stage_started_at)
                yield self.event(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    event_index=event_index,
                    agent_name=self.evidence_agent.name,
                    event_type="evidence",
                    message="Evidence context refreshed.",
                    payload=evidence_event_payload(
                        evidence,
                        evidence_latency_ms,
                        refresh_reason=verifier.reason,
                        repair_round=repair_round,
                    ),
                    started_at=started_at,
                    profile=normalized_profile,
                )
                event_index += 1

                stage_started_at = time.monotonic()
                answer = self.answer_agent.run(
                    context=evidence.context,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                )
                answer_latency_ms = elapsed_ms(stage_started_at)
                yield self.event(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    event_index=event_index,
                    agent_name=self.answer_agent.name,
                    event_type="answer",
                    message="Answer redrafted.",
                    payload={
                        "answer": model_to_dict(answer),
                        "stage_latency_ms": {"answer_agent": answer_latency_ms},
                        "verifier_repair_round": repair_round,
                    },
                    started_at=started_at,
                    profile=normalized_profile,
                )
                event_index += 1
            elif verifier.status == "unsupported_answer":
                stage_started_at = time.monotonic()
                answer = self.answer_agent.run(
                    context=evidence.context,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    revision_request=verifier.reason,
                )
                answer_latency_ms = elapsed_ms(stage_started_at)
                yield self.event(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    event_index=event_index,
                    agent_name=self.answer_agent.name,
                    event_type="answer",
                    message="Answer redrafted.",
                    payload={
                        "answer": model_to_dict(answer),
                        "stage_latency_ms": {"answer_agent": answer_latency_ms},
                        "revision_request": verifier.reason,
                        "verifier_repair_round": repair_round,
                    },
                    started_at=started_at,
                    profile=normalized_profile,
                )
                event_index += 1
            else:
                break

            stage_started_at = time.monotonic()
            verifier = self.verifier_agent.run(answer=answer, context=evidence.context)
            verifier_latency_ms = elapsed_ms(stage_started_at)
            verifier_payload = model_to_dict(verifier)
            verifier_payload["stage_latency_ms"] = {"verifier_agent": verifier_latency_ms}
            verifier_payload["verifier_repair_round"] = repair_round
            yield self.event(
                conversation_id=conversation_id,
                turn_id=turn_id,
                event_index=event_index,
                agent_name=self.verifier_agent.name,
                event_type="verification",
                message=f"Verifier status: {verifier.status}.",
                payload=verifier_payload,
                started_at=started_at,
                profile=normalized_profile,
            )
            event_index += 1

        final_event = build_final_event_type(verifier)
        final_message = final_message_for(answer, verifier)
        state.add_assistant_turn(
            turn_id=turn_id,
            message=final_message,
            standalone_question=query.standalone_question,
            citation_ids=[citation.citation_id for citation in answer.citations],
            metadata={
                "verifier_status": verifier.status,
                "profile": normalized_profile,
            },
        )
        state.citation_history.extend(model_to_dict(citation) for citation in answer.citations)
        yield self.event(
            conversation_id=conversation_id,
            turn_id=turn_id,
            event_index=event_index,
            agent_name="orchestrator",
            event_type=final_event,
            message=final_message,
            payload={
                "answer": model_to_dict(answer),
                "verifier": model_to_dict(verifier),
            },
            started_at=started_at,
            profile=normalized_profile,
            is_final=True,
        )

    def event(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        event_index: int,
        agent_name: str,
        event_type: Any,
        message: str,
        payload: dict[str, Any],
        started_at: float,
        profile: str,
        is_final: bool = False,
    ) -> AgentEvent:
        return AgentEvent(
            event_id=f"{turn_id}:event_{event_index:03d}",
            conversation_id=conversation_id,
            turn_id=turn_id,
            agent_name=agent_name,
            event_type=event_type,
            message=message,
            payload=payload,
            latency_ms=(time.monotonic() - started_at) * 1000,
            profile=profile,
            is_final=is_final,
        )


def build_turn_id(state: DialogueState) -> str:
    user_turn_count = sum(1 for turn in state.turns if turn.role == "user")
    return f"turn_{user_turn_count + 1:04d}"


def evidence_event_payload(
    evidence: Any,
    stage_latency_ms: float,
    *,
    refresh_reason: str | None = None,
    repair_round: int | None = None,
) -> dict[str, Any]:
    checker_outputs = list(evidence.checker_outputs or [])
    payload = {
        "completed": evidence.completed,
        "retrieval_rounds": evidence.retrieval_rounds,
        "second_hop_query_count": evidence.second_hop_query_count,
        "total_latency_ms": evidence.total_latency_ms,
        "active_chunk_ids": [item.chunk_id for item in evidence.context.active_items],
        "checker_outputs": checker_outputs,
        "stage_latency_ms": {
            "evidence_agent": stage_latency_ms,
            "evidence_loop": evidence.total_latency_ms,
            "checker": sum(float(row.get("checker_latency_ms") or 0.0) for row in checker_outputs),
            "retrieval": sum(float(row.get("retrieval_latency_ms") or 0.0) for row in checker_outputs),
            "rerank": sum(float(row.get("rerank_latency_ms") or 0.0) for row in checker_outputs),
        },
    }
    if refresh_reason:
        payload["refresh_reason"] = refresh_reason
    if repair_round is not None:
        payload["verifier_repair_round"] = repair_round
    return payload


def should_attempt_verifier_repair(
    verifier: VerifierTrace,
    repair_rounds_completed: int,
    config: Any,
) -> bool:
    max_rounds = max(0, int(getattr(config, "max_verifier_repair_rounds", 1) or 0))
    if repair_rounds_completed >= max_rounds:
        return False
    if verifier.status == "unsupported_answer":
        return True
    if verifier.status != "insufficient_evidence":
        return False
    if repair_rounds_completed == 0:
        return True
    return bool(verifier.suggested_queries)


def build_final_event_type(verifier: VerifierTrace) -> str:
    if verifier.status == "ready_to_answer":
        return "final_answer"
    if verifier.status == "insufficient_evidence":
        return "clarification_request"
    return "abstention"


def final_message_for(answer: AnswerTrace, verifier: VerifierTrace) -> str:
    if verifier.status == "ready_to_answer":
        return answer.answer
    if verifier.status == "insufficient_evidence":
        return "I need more evidence before I can answer this reliably."
    return "I could not verify the answer against the available Wix Help Center evidence."


def non_retrieval_answer_for(dialogue: Any, route: str) -> str:
    if route == "clarify":
        return clarification_answer_for(dialogue)
    return direct_answer_for(dialogue.user_message)


def clarification_answer_for(dialogue: Any) -> str:
    question = compact_text(getattr(dialogue, "clarification_question", ""))
    if question:
        return question
    message = compact_text(getattr(dialogue, "user_message", ""))
    if contains_cjk(message):
        return "你想咨询 Wix 的哪个具体功能或操作？"
    return "Which Wix feature or task do you want help with?"


def direct_answer_for(message: str) -> str:
    if contains_cjk(message):
        return "我在。你可以直接问我 Wix 支付、网站设置、Bookings、Stores 等帮助中心问题。"
    return "I'm here. Ask me a Wix Help Center question and I'll check the evidence before answering."
