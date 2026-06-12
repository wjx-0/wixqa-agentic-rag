from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import unittest
from unittest.mock import patch

from src.agentic.agents.answer_agent import AnswerAgent
from src.agentic.agents.dialogue_agent import DialogueAgent
from src.agentic.agents.evidence_agent import EvidenceAgent
from src.agentic.agents.query_agent import (
    LLMQueryRewriter,
    QueryAgent,
    build_query_rewrite_messages,
)
from src.agentic.agents.verifier_agent import LLMClaimChecker, VerifierAgent
from src.agentic.dialogue_orchestrator import DialogueOrchestrator
from src.agentic.dialogue_state import (
    AnswerClaim,
    AnswerTrace,
    Citation,
    DialogueState,
    IntentRoute,
    LatencyConfig,
    PROFILE_FAST,
    PROFILE_FULL_EVAL,
    VerifierTrace,
    profile_config,
)
from src.agentic.evidence_context import build_initial_evidence_context
from src.agentic.evidence_loop import EvidenceCompletionLoopResult
from src.agentic.intent_gate import (
    IntentGateDecision,
    LLMIntentGate,
    RuleBasedIntentGate,
    RuleBasedIntentGateConfig,
)
from src.agentic.runtime_factory import DialogueRuntimeConfig, build_real_dialogue_runtime
from src.data.schema import KBChunk
from src.llm.answer_generator import LLMAnswerGenerator


def chunk(chunk_id: str, article_id: str, rank: int = 1) -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        article_id=article_id,
        chunk_index=rank - 1,
        title=f"Wix title {rank}",
        text=f"Wix title {rank}\nGuidance for {chunk_id}",
        contents=f"Guidance for {chunk_id}",
        start_token=(rank - 1) * 10,
        end_token=rank * 10,
        num_tokens=10,
    )


def candidate(row: KBChunk, rank: int) -> dict:
    return {
        "chunk_id": row.chunk_id,
        "article_id": row.article_id,
        "title": row.title,
        "rank": rank,
        "score": float(100 - rank),
    }


def context_for_question(question: str, qid: str = "qid"):
    chunks = [chunk(f"chunk_{index}", f"article_{index}", index) for index in range(1, 4)]
    lookup = {row.chunk_id: row for row in chunks}
    rows = [candidate(row, index) for index, row in enumerate(chunks, start=1)]
    return build_initial_evidence_context(
        candidate_row={
            "qid": qid,
            "dataset_name": "wixqa_expertwritten",
            "question": question,
            "answer": None,
            "gold_article_ids": [],
            "num_gold_articles": 0,
            "is_multi_article": False,
            "hybrid_candidates": rows,
        },
        rerank_trace={
            "qid": qid,
            "top10_reranked_chunks": rows[:2],
            "case_type": "A_top10_chunks_full",
        },
        chunk_lookup=lookup,
    )


class RecordingCompletionEngine:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, context, config):
        self.calls.append(config)
        return EvidenceCompletionLoopResult(
            context=context,
            completed=True,
            rounds_completed=context.round_index,
            retrieval_rounds=config.max_rounds,
            second_hop_query_count=config.max_queries_per_round,
        )


class EmptyAnswerAgent:
    name = "answer_agent"

    def run(self, *, context, conversation_id, turn_id, revision_request=None):
        return AnswerTrace(answer="No supported answer.")


class InvalidAnswerAgent:
    name = "answer_agent"

    def run(self, *, context, conversation_id, turn_id, revision_request=None):
        trace = AnswerAgent().run(
            context=context,
            conversation_id=conversation_id,
            turn_id=turn_id,
            revision_request=revision_request,
        )
        trace.answer_claims[0].supporting_chunk_ids = ["missing_chunk"]
        return trace


class FakeAnswerClient:
    model = "fake-answer-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = []

    def complete(self, messages, *, temperature, max_tokens):
        self.messages.append(messages)
        return self.response


class FailingAnswerClient:
    model = "failing-answer-model"

    def complete(self, messages, *, temperature, max_tokens):
        raise RuntimeError("answer endpoint timeout")


class FakeAnswerSequenceClient:
    model = "fake-answer-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.max_tokens_seen = []

    def complete(self, messages, *, temperature, max_tokens):
        self.max_tokens_seen.append(max_tokens)
        if not self.responses:
            return ""
        return self.responses.pop(0)


class FakeClaimCheckerClient:
    model = "fake-claim-checker"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = []

    def complete(self, messages, *, temperature, max_tokens):
        self.messages.append(messages)
        return self.response


class FakeQueryRewriteClient:
    model = "fake-query-rewrite-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = []

    def complete(self, messages, *, temperature, max_tokens):
        self.messages.append(messages)
        return self.response


class FakeMemorySummarizer:
    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.calls = []

    def summarize(self, *, existing_summary, turns, constraints):
        self.calls.append(
            {
                "existing_summary": existing_summary,
                "turns": list(turns),
                "constraints": list(constraints),
            }
        )
        return self.summary


class FailingQueryRewriteClient:
    model = "failing-query-rewrite-model"

    def complete(self, messages, *, temperature, max_tokens):
        raise RuntimeError("query rewrite endpoint timeout")


class FakeHybridRetriever:
    def __init__(self, **kwargs) -> None:
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class SequentialGateClient:
    model = "fake-gate-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.max_tokens_seen = []

    def complete(self, messages, *, temperature, max_tokens):
        self.max_tokens_seen.append(max_tokens)
        if not self.responses:
            return ""
        return self.responses.pop(0)


class StaticIntentGate:
    def __init__(
        self,
        route: IntentRoute,
        *,
        clarification_question: str | None = None,
    ) -> None:
        self.decision = IntentGateDecision(
            route=route,
            reason="test gate",
            confidence=0.99,
            clarification_question=clarification_question,
            used_api=True,
        )
        self.calls = []

    def run(self, *, state, message):
        self.calls.append(message)
        return self.decision


class SequenceIntentGate:
    def __init__(self, decisions: list[IntentGateDecision]) -> None:
        self.decisions = list(decisions)
        self.calls = []

    def run(self, *, state, message):
        self.calls.append(message)
        if not self.decisions:
            raise AssertionError("No intent gate decision left.")
        return self.decisions.pop(0)


class SequenceVerifierAgent:
    name = "verifier_agent"

    def __init__(self, traces: list[VerifierTrace]) -> None:
        self.traces = list(traces)
        self.calls = []

    def run(self, *, answer, context):
        self.calls.append((answer, context))
        if not self.traces:
            return VerifierTrace(status="ready_to_answer", reason="verified")
        return self.traces.pop(0)


class DialoguePipelineTest(unittest.TestCase):
    def build_orchestrator(
        self,
        *,
        engine: RecordingCompletionEngine | None = None,
        answer_agent=None,
        intent_gate=None,
        query_agent=None,
        verifier_agent=None,
        latency=None,
    ) -> DialogueOrchestrator:
        engine = engine or RecordingCompletionEngine()

        def build_context(question: str, state):
            return context_for_question(question, qid=f"{state.conversation_id}_qid")

        return DialogueOrchestrator(
            dialogue_agent=DialogueAgent(intent_gate=intent_gate) if intent_gate else None,
            query_agent=query_agent,
            evidence_agent=EvidenceAgent(
                context_builder=build_context,
                completion_engine=engine,
            ),
            answer_agent=answer_agent,
            verifier_agent=verifier_agent,
            latency=latency,
        )

    def test_first_event_is_emitted_before_evidence_work(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(engine=engine)

        events = orchestrator.run_turn("conv_fast", "How do I connect PayPal?")
        first = next(events)

        self.assertEqual(first.event_type, "status")
        self.assertEqual(first.profile, PROFILE_FAST)
        self.assertFalse(first.is_final)
        self.assertEqual(engine.calls, [])

    def test_same_conversation_turns_are_serialized(self) -> None:
        orchestrator = self.build_orchestrator()
        first_generator = orchestrator.run_turn("conv_serial", "How do I connect PayPal?")
        first_event = next(first_generator)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: list(orchestrator.run_turn("conv_serial", "What about services?"))
            )
            with self.assertRaises(TimeoutError):
                future.result(timeout=0.05)
            remaining_first_events = list(first_generator)
            second_events = future.result(timeout=2)

        self.assertEqual(first_event.turn_id, "turn_0001")
        self.assertEqual(remaining_first_events[-1].turn_id, "turn_0001")
        self.assertEqual(second_events[0].turn_id, "turn_0002")

    def test_fast_profile_limits_evidence_loop_budget(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(engine=engine)
        profile = profile_config(PROFILE_FAST)

        events = list(orchestrator.run_turn("conv_budget", "How do I connect PayPal?"))

        self.assertEqual(events[-1].event_type, "final_answer")
        self.assertEqual(profile.checker_mode, "compact")
        self.assertEqual(profile.llm_max_tokens, 1024)
        self.assertEqual(engine.calls[0].max_rounds, 1)
        self.assertEqual(engine.calls[0].max_queries_per_round, 1)
        self.assertEqual(engine.calls[0].per_query_retrieve_top_k_chunks, 3)
        self.assertEqual(engine.calls[0].max_raw_chunks_per_checker_call, 10)
        self.assertEqual(engine.calls[0].max_new_raw_chunks_per_round, 2)
        self.assertIn("answer", [event.event_type for event in events])
        evidence_event = next(event for event in events if event.event_type == "evidence")
        self.assertIn("stage_latency_ms", evidence_event.payload)

    def test_full_eval_profile_keeps_deeper_evidence_loop_budget(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(engine=engine)
        profile = profile_config(PROFILE_FULL_EVAL)

        events = list(
            orchestrator.run_turn(
                "conv_full",
                "How do I connect PayPal?",
                profile="full_eval",
            )
        )

        self.assertEqual(events[-1].profile, PROFILE_FULL_EVAL)
        self.assertEqual(profile.checker_mode, "traceable")
        self.assertEqual(profile.llm_max_tokens, 2048)
        self.assertEqual(engine.calls[0].max_rounds, 4)
        self.assertEqual(engine.calls[0].max_queries_per_round, 3)
        self.assertEqual(engine.calls[0].per_query_retrieve_top_k_chunks, 20)
        self.assertEqual(engine.calls[0].max_raw_chunks_per_checker_call, 30)

    def test_follow_up_reuses_dialogue_context(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(engine=engine)

        list(orchestrator.run_turn("conv_multi", "Can I sell products with Wix Stores?"))
        state = orchestrator.states["conv_multi"]
        first_context_qid = state.evidence_context.qid
        first_manifest_id = state.evidence_context.prompt_manifests[0].llm_call_id
        list(orchestrator.run_turn("conv_multi", "What about services?"))

        self.assertIn("Follow-up: What about services?", state.rewritten_standalone_question)
        self.assertEqual(state.evidence_context.question, state.rewritten_standalone_question)
        self.assertNotEqual(state.evidence_context.qid, first_context_qid)
        self.assertEqual(state.evidence_context.eval_info["reused_from_qid"], first_context_qid)
        self.assertNotEqual(
            state.evidence_context.prompt_manifests[0].llm_call_id,
            first_manifest_id,
        )
        self.assertEqual(len([turn for turn in state.turns if turn.role == "user"]), 2)
        self.assertTrue(state.answer_history)
        self.assertTrue(state.citation_history)

    def test_adjacent_standalone_question_does_not_reuse_previous_context(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(engine=engine)

        list(orchestrator.run_turn("conv_topic_shift", "Can I sell products with Wix Stores?"))
        state = orchestrator.states["conv_topic_shift"]
        events = list(orchestrator.run_turn("conv_topic_shift", "How do I connect PayPal?"))

        dialogue_event = next(event for event in events if event.event_type == "dialogue_state")
        query_event = next(event for event in events if event.event_type == "query")
        self.assertEqual(dialogue_event.payload["turn_type"], "topic_shift")
        self.assertEqual(query_event.payload["route"], "retrieve")
        self.assertFalse(query_event.payload["reused_context"])
        self.assertEqual(state.evidence_context.question, "How do I connect PayPal?")
        self.assertNotIn("reused_from_qid", state.evidence_context.eval_info)

    def test_follow_up_without_relevance_does_not_reuse_context(self) -> None:
        state = DialogueState(
            conversation_id="conv_reuse_threshold",
            active_question="How do I connect PayPal?",
            evidence_context=context_for_question("How do I connect PayPal?"),
        )
        dialogue = type(
            "Dialogue",
            (),
            {
                "turn_id": "turn_0002",
                "turn_type": "follow_up_question",
                "user_message": "What about SEO menu labels?",
                "active_question": state.active_question,
            },
        )()

        result = QueryAgent().run(state=state, dialogue=dialogue)

        self.assertEqual(result.route, "retrieve")
        self.assertFalse(result.reused_context)
        self.assertLess(result.context_reuse_score, 0.25)

    def test_follow_up_uses_llm_query_rewriter_when_configured(self) -> None:
        engine = RecordingCompletionEngine()
        client = FakeQueryRewriteClient(
            '{"standalone_question":"Can Wix Stores sell services as well as physical products?",'
            '"reason":"rewrite follow-up"}'
        )
        orchestrator = self.build_orchestrator(
            engine=engine,
            query_agent=QueryAgent(rewriter=LLMQueryRewriter(client=client)),
        )

        list(orchestrator.run_turn("conv_llm_rewrite", "Can I sell products with Wix Stores?"))
        events = list(orchestrator.run_turn("conv_llm_rewrite", "What about services?"))

        query_event = next(event for event in events if event.event_type == "query")
        self.assertEqual(
            query_event.payload["standalone_question"],
            "Can Wix Stores sell services as well as physical products?",
        )
        self.assertTrue(query_event.payload["rewrite_used_api"])
        self.assertFalse(query_event.payload["rewrite_fallback_used"])
        self.assertIsNone(query_event.payload["rewrite_error"])
        self.assertIn("Active support question", client.messages[-1][1]["content"])
        self.assertEqual(
            orchestrator.states["conv_llm_rewrite"].evidence_context.question,
            "Can Wix Stores sell services as well as physical products?",
        )

    def test_query_rewriter_falls_back_to_rules_on_llm_error(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(
            engine=engine,
            query_agent=QueryAgent(
                rewriter=LLMQueryRewriter(client=FailingQueryRewriteClient())
            ),
        )

        list(orchestrator.run_turn("conv_rewrite_fallback", "Can I sell products with Wix Stores?"))
        events = list(orchestrator.run_turn("conv_rewrite_fallback", "What about services?"))

        query_event = next(event for event in events if event.event_type == "query")
        self.assertIn("Follow-up: What about services?", query_event.payload["standalone_question"])
        self.assertFalse(query_event.payload["rewrite_used_api"])
        self.assertTrue(query_event.payload["rewrite_fallback_used"])
        self.assertIn("query rewrite endpoint timeout", query_event.payload["rewrite_error"])

    def test_long_dialogue_summary_is_written_and_used_in_rewrite_prompt(self) -> None:
        client = FakeQueryRewriteClient(
            '{"standalone_question":"Can I keep using the remembered domain detail?",'
            '"reason":"uses memory"}'
        )
        summarizer = FakeMemorySummarizer("Earlier domain detail: user uses example.com.")
        agent = QueryAgent(
            rewriter=LLMQueryRewriter(client=client),
            memory_summarizer=summarizer,
        )
        state = DialogueState(conversation_id="conv_memory", active_question="How do I connect my domain?")
        for index in range(1, 8):
            state.add_user_turn(
                turn_id=f"user_{index}",
                message=f"Older turn {index} about example.com",
                turn_type="follow_up_question" if index > 1 else "new_question",
            )
        dialogue = state.add_user_turn(
            turn_id="user_current",
            message="Does that still apply?",
            turn_type="follow_up_question",
        )

        result = agent.run(
            state=state,
            dialogue=type(
                "Dialogue",
                (),
                {
                    "turn_id": dialogue.turn_id,
                    "turn_type": "follow_up_question",
                    "user_message": dialogue.message,
                    "active_question": state.active_question,
                },
            )(),
        )

        self.assertTrue(result.memory_summary_used)
        self.assertFalse(result.memory_summary_fallback_used)
        self.assertEqual(state.context_memory_summary, "Earlier domain detail: user uses example.com.")
        self.assertTrue(summarizer.calls)
        self.assertIn("Conversation memory summary", client.messages[-1][1]["content"])
        self.assertIn("example.com", client.messages[-1][1]["content"])

    def test_memory_summary_falls_back_without_llm(self) -> None:
        agent = QueryAgent()
        state = DialogueState(conversation_id="conv_memory_fallback", active_question="How do I connect my domain?")
        for index in range(1, 8):
            state.add_user_turn(
                turn_id=f"user_{index}",
                message=f"Older turn {index} about fallback-domain.com",
                turn_type="follow_up_question" if index > 1 else "new_question",
            )
        dialogue = state.add_user_turn(
            turn_id="user_current",
            message="What about renewals?",
            turn_type="follow_up_question",
        )

        result = agent.run(
            state=state,
            dialogue=type(
                "Dialogue",
                (),
                {
                    "turn_id": dialogue.turn_id,
                    "turn_type": "follow_up_question",
                    "user_message": dialogue.message,
                    "active_question": state.active_question,
                },
            )(),
        )

        self.assertTrue(result.memory_summary_used)
        self.assertTrue(result.memory_summary_fallback_used)
        self.assertIn("fallback-domain.com", state.context_memory_summary or "")
        self.assertIn("memory summarizer is not configured", result.memory_summary_error or "")

    def test_topic_shift_clears_old_constraints_from_rewrite_prompt(self) -> None:
        state = DialogueState(
            conversation_id="conv_constraint",
            active_question="How do I manage my domain?",
        )
        state.add_user_turn(
            turn_id="turn_0001",
            message="How do I manage my domain?",
            turn_type="new_question",
        )
        state.remember_constraint("my site uses example.com")
        state.context_memory_summary = "Old topic summary"
        agent = DialogueAgent(
            intent_gate=StaticIntentGate("retrieve")
        )

        dialogue = agent.run(
            state=state,
            user_message="How do I connect PayPal?",
            turn_id="turn_0002",
        )
        messages = QueryAgent().run(state=state, dialogue=dialogue)
        prompt = "\n".join(
            message["content"]
            for message in build_query_rewrite_messages(state=state, dialogue=dialogue)
        )

        self.assertEqual(messages.route, "retrieve")
        self.assertTrue(dialogue.topic_shift)
        self.assertEqual(state.unresolved_user_constraints, [])
        self.assertIsNone(state.context_memory_summary)
        self.assertNotIn("example.com", prompt)
        self.assertNotIn("Old topic summary", prompt)

    def test_non_support_message_does_not_call_evidence_agent(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(engine=engine)

        events = list(orchestrator.run_turn("conv_small_talk", "下午好"))

        self.assertEqual(events[-1].event_type, "final_answer")
        self.assertEqual(engine.calls, [])
        self.assertNotIn("evidence", [event.event_type for event in events])
        query_event = next(event for event in events if event.event_type == "query")
        self.assertEqual(query_event.payload["route"], "direct_answer")
        self.assertIn("Wix", events[-1].message)
        self.assertIsNone(orchestrator.states["conv_small_talk"].active_question)

    def test_non_support_message_does_not_overwrite_active_question(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(engine=engine)

        list(orchestrator.run_turn("conv_keep_context", "Can I sell products with Wix Stores?"))
        state = orchestrator.states["conv_keep_context"]
        active_question = state.active_question
        call_count = len(engine.calls)

        events = list(orchestrator.run_turn("conv_keep_context", "good afternoon"))

        self.assertEqual(state.active_question, active_question)
        self.assertEqual(len(engine.calls), call_count)
        self.assertNotIn("evidence", [event.event_type for event in events])
        self.assertEqual(events[-1].payload["route"], "direct_answer")

    def test_rule_gate_ambiguous_support_need_asks_clarification(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(engine=engine)

        events = list(orchestrator.run_turn("conv_rule_clarify", "我想收款"))

        self.assertEqual(engine.calls, [])
        self.assertNotIn("evidence", [event.event_type for event in events])
        self.assertEqual(events[-1].event_type, "clarification_request")
        self.assertEqual(events[-1].payload["route"], "clarify")
        self.assertEqual(orchestrator.states["conv_rule_clarify"].active_question, "我想收款")

    def test_rule_gate_can_be_configured_for_another_support_domain(self) -> None:
        gate = RuleBasedIntentGate(
            RuleBasedIntentGateConfig(
                domain_name="Acme Docs",
                domain_terms=("acme",),
            )
        )

        decision = gate.run(
            state=DialogueState(conversation_id="conv_acme"),
            message="How do I reset my Acme account password?",
        )
        ambiguous = gate.run(
            state=DialogueState(conversation_id="conv_acme_ambiguous"),
            message="I want to upgrade",
        )

        self.assertEqual(decision.route, "retrieve")
        self.assertEqual(ambiguous.route, "clarify")
        self.assertIn("Acme Docs", ambiguous.clarification_question or "")

    def test_llm_intent_gate_retries_empty_response(self) -> None:
        client = SequentialGateClient(
            [
                "",
                (
                    '{"route":"clarify","confidence":0.9,'
                    '"reason":"underspecified",'
                    '"clarification_question":"你想咨询哪个 Wix 收款功能？"}'
                ),
            ]
        )
        gate = LLMIntentGate(client=client, max_tokens=180)

        decision = gate.run(
            state=DialogueState(conversation_id="conv_gate_retry"),
            message="我想收款",
        )

        self.assertEqual(decision.route, "clarify")
        self.assertTrue(decision.used_api)
        self.assertEqual(client.max_tokens_seen, [180, 512])

    def test_llm_intent_gate_overrides_payment_failure_to_retrieve(self) -> None:
        client = SequentialGateClient(
            [
                (
                    '{"route":"clarify","confidence":0.8,'
                    '"reason":"needs more detail",'
                    '"clarification_question":"是哪种支付方式？"}'
                ),
            ]
        )
        gate = LLMIntentGate(client=client, max_tokens=512)

        decision = gate.run(
            state=DialogueState(conversation_id="conv_gate_payment_failure"),
            message="请问我为什么支付不了",
        )

        self.assertEqual(decision.route, "retrieve")
        self.assertIsNone(decision.clarification_question)
        self.assertTrue(decision.used_api)

    def test_api_intent_gate_direct_answer_skips_evidence_agent(self) -> None:
        engine = RecordingCompletionEngine()
        gate = StaticIntentGate("direct_answer")
        orchestrator = self.build_orchestrator(engine=engine, intent_gate=gate)

        events = list(orchestrator.run_turn("conv_api_direct", "good afternoon"))

        self.assertEqual(engine.calls, [])
        self.assertNotIn("evidence", [event.event_type for event in events])
        self.assertEqual(events[-1].event_type, "final_answer")
        dialogue_event = next(event for event in events if event.event_type == "dialogue_state")
        self.assertEqual(dialogue_event.payload["intent_route"], "direct_answer")

    def test_api_intent_gate_clarify_skips_evidence_agent(self) -> None:
        engine = RecordingCompletionEngine()
        gate = StaticIntentGate("clarify", clarification_question="你想咨询 Wix 的哪个支付功能？")
        orchestrator = self.build_orchestrator(engine=engine, intent_gate=gate)

        events = list(orchestrator.run_turn("conv_api_clarify", "我想收款"))
        state = orchestrator.states["conv_api_clarify"]

        self.assertEqual(engine.calls, [])
        self.assertNotIn("evidence", [event.event_type for event in events])
        self.assertEqual(events[-1].event_type, "clarification_request")
        self.assertEqual(events[-1].payload["route"], "clarify")
        self.assertIn("支付功能", events[-1].message)
        self.assertEqual(state.active_question, "我想收款")

    def test_clarification_answer_can_retrieve_on_next_turn(self) -> None:
        engine = RecordingCompletionEngine()
        gate = SequenceIntentGate(
            [
                IntentGateDecision(
                    route="clarify",
                    reason="ambiguous support need",
                    clarification_question="你想咨询 Wix 的哪个具体功能？",
                    used_api=True,
                ),
                IntentGateDecision(
                    route="retrieve",
                    reason="clarified Wix product",
                    confidence=0.98,
                    used_api=True,
                ),
            ]
        )
        orchestrator = self.build_orchestrator(engine=engine, intent_gate=gate)

        list(orchestrator.run_turn("conv_api_followup", "我想收款"))
        events = list(orchestrator.run_turn("conv_api_followup", "Wix Stores"))

        self.assertEqual(len(engine.calls), 1)
        self.assertIn("evidence", [event.event_type for event in events])
        query_event = next(event for event in events if event.event_type == "query")
        self.assertEqual(query_event.payload["route"], "retrieve")
        self.assertIn("Follow-up: Wix Stores", query_event.payload["standalone_question"])

    def test_normal_support_question_still_uses_evidence_agent(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(engine=engine)

        events = list(orchestrator.run_turn("conv_support_gate", "How do I connect PayPal?"))

        self.assertEqual(events[-1].event_type, "final_answer")
        self.assertEqual(len(engine.calls), 1)
        self.assertIn("evidence", [event.event_type for event in events])
        query_event = next(event for event in events if event.event_type == "query")
        self.assertEqual(query_event.payload["route"], "retrieve")

    def test_answer_agent_outputs_visible_citations_only(self) -> None:
        context = context_for_question("How do I connect PayPal?")
        answer = AnswerAgent().run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )
        visible_chunk_ids = {item.chunk_id for item in context.active_items}

        self.assertTrue(answer.answer_claims)
        self.assertTrue(answer.citations)
        self.assertTrue(set(answer.seen_chunk_ids).issubset(visible_chunk_ids))
        for claim in answer.answer_claims:
            self.assertTrue(set(claim.supporting_chunk_ids).issubset(visible_chunk_ids))

    def test_extractive_answer_uses_evidence_text_not_placeholder(self) -> None:
        context = context_for_question("How do I connect PayPal?")

        answer = AnswerAgent().run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )

        self.assertIn("Guidance for chunk_1", answer.answer)
        self.assertNotIn("I found relevant Wix Help Center evidence", answer.answer)
        self.assertNotIn("Use the cited sources", answer.answer)

    def test_llm_answer_generator_falls_back_from_generic_greeting(self) -> None:
        context = context_for_question("How do I connect PayPal?")
        client = FakeAnswerClient(
            '{"answer": "Hello! How can I assist you today? [src_1]", '
            '"claims": [{"claim": "Greeting", "supporting_chunk_ids": ["chunk_1"], '
            '"citation_ids": ["src_1"]}]}'
        )

        answer = AnswerAgent(generator=LLMAnswerGenerator(client=client)).run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )

        self.assertIn("Guidance for chunk_1", answer.answer)
        self.assertNotIn("How can I assist you today", answer.answer)
        self.assertEqual(answer.generation_mode, "extractive_fallback")

    def test_llm_answer_generator_keeps_chinese_answer_without_english_title_terms(self) -> None:
        context = context_for_question("请问我为什么支付不了")
        client = FakeAnswerClient(
            '{"answer": "你可以先检查 Wix 账户里的付款方式是否失败或被拒绝；如果续订付款失败，'
            '可以更新付款方式后重试。 [src_1]", '
            '"claims": [{"claim": "续订付款失败时可以更新付款方式", '
            '"supporting_chunk_ids": ["chunk_1"], "citation_ids": ["src_1"]}]}'
        )

        answer = AnswerAgent(generator=LLMAnswerGenerator(client=client)).run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )

        self.assertIn("付款方式", answer.answer)
        self.assertNotIn("Guidance for chunk_1", answer.answer)
        self.assertEqual(answer.generation_mode, "llm")

    def test_llm_answer_generator_marks_invalid_json_fallback(self) -> None:
        context = context_for_question("How do I connect PayPal?")
        client = FakeAnswerClient("not json")

        answer = AnswerAgent(generator=LLMAnswerGenerator(client=client)).run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )

        self.assertEqual(answer.generation_mode, "extractive_fallback")
        self.assertIn("invalid_llm_json", answer.generator_error or "")

    def test_llm_answer_generator_falls_back_on_client_error(self) -> None:
        context = context_for_question("How do I connect PayPal?")

        answer = AnswerAgent(generator=LLMAnswerGenerator(client=FailingAnswerClient())).run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )

        self.assertEqual(answer.generation_mode, "extractive_fallback")
        self.assertIn("llm_answer_failed", answer.generator_error or "")
        self.assertIn("Guidance for chunk_1", answer.answer)

    def test_llm_answer_generator_retries_truncated_json(self) -> None:
        context = context_for_question("请问我为什么支付不了")
        client = FakeAnswerSequenceClient(
            [
                '{"answer": "截断',
                (
                    '{"answer": "如果付款失败，可以先检查付款方式是否被拒绝，'
                    '然后更新 Wix 账户里的付款方式后重试。 [src_1]", '
                    '"claims": [{"claim": "付款失败后可以更新付款方式", '
                    '"supporting_chunk_ids": ["chunk_1"], "citation_ids": ["src_1"]}]}'
                ),
            ]
        )

        answer = AnswerAgent(generator=LLMAnswerGenerator(client=client, max_tokens=900)).run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )

        self.assertEqual(answer.generation_mode, "llm")
        self.assertEqual(client.max_tokens_seen, [900, 2400])
        self.assertIn("付款失败", answer.answer)

    def test_llm_answer_generator_tracks_compressed_context_support(self) -> None:
        context = context_for_question("Does the special setup require manual approval?")
        context.compressed_summary = "Compressed fact: the special setup requires manual approval."
        context.compressed_context_ids = ["qid:compact:1"]
        client = FakeAnswerClient(
            '{"answer": "The special setup requires manual approval. [ctx_1]", '
            '"claims": [{"claim": "The special setup requires manual approval.", '
            '"supporting_compressed_context_ids": ["qid:compact:1"], '
            '"citation_ids": ["ctx_1"]}]}'
        )

        answer = AnswerAgent(generator=LLMAnswerGenerator(client=client)).run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )
        verifier = VerifierAgent().run(answer=answer, context=context)

        compressed_citation = next(
            citation for citation in answer.citations if citation.citation_id == "ctx_1"
        )
        self.assertEqual(answer.generation_mode, "llm")
        self.assertEqual(compressed_citation.compressed_context_ids, ["qid:compact:1"])
        self.assertEqual(
            answer.answer_claims[0].supporting_compressed_context_ids,
            ["qid:compact:1"],
        )
        self.assertEqual(
            context.answer_provenance.supporting_compressed_context_ids,
            ["qid:compact:1"],
        )
        self.assertEqual(verifier.status, "ready_to_answer")
        self.assertIn("Compressed Context ID: qid:compact:1", client.messages[-1][1]["content"])

    def test_extractive_answer_uses_compressed_summary(self) -> None:
        context = context_for_question("Does the special setup require manual approval?")
        context.compressed_summary = "Compressed fact: the special setup requires manual approval."
        context.compressed_context_ids = ["qid:compact:1"]

        answer = AnswerAgent().run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )
        verifier = VerifierAgent().run(answer=answer, context=context)

        compressed_claim = next(
            claim for claim in answer.answer_claims if claim.supporting_compressed_context_ids
        )
        self.assertIn("manual approval", answer.answer)
        self.assertIn("[ctx_1]", answer.answer)
        self.assertEqual(compressed_claim.supporting_compressed_context_ids, ["qid:compact:1"])
        self.assertEqual(verifier.status, "ready_to_answer")

    def test_verifier_rejects_claims_that_reference_hidden_chunks(self) -> None:
        context = context_for_question("How do I connect PayPal?")
        answer = AnswerAgent().run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )
        answer.answer_claims[0].supporting_chunk_ids = ["not_visible"]

        verifier = VerifierAgent().run(answer=answer, context=context)

        self.assertEqual(verifier.status, "unsupported_answer")
        self.assertEqual(verifier.unsupported_claims[0]["reason"], "supporting_chunk_ids_not_visible")

    def test_verifier_rejects_claims_that_reference_hidden_compressed_contexts(self) -> None:
        context = context_for_question("Does the special setup require manual approval?")
        context.compressed_summary = "Compressed fact: the special setup requires manual approval."
        context.compressed_context_ids = ["qid:compact:1"]
        answer = AnswerAgent().run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )
        compressed_claim = next(
            claim for claim in answer.answer_claims if claim.supporting_compressed_context_ids
        )
        compressed_claim.supporting_compressed_context_ids = ["missing:compact:1"]

        verifier = VerifierAgent().run(answer=answer, context=context)

        self.assertEqual(verifier.status, "unsupported_answer")
        self.assertEqual(
            verifier.unsupported_claims[0]["reason"],
            "supporting_compressed_context_ids_not_visible",
        )

    def test_verifier_rejects_generic_answer_even_with_valid_citation(self) -> None:
        context = context_for_question("How do I connect PayPal?")
        answer = AnswerTrace(
            answer="Hello! How can I assist you today? [src_1]",
            seen_chunk_ids=["chunk_1"],
            answer_claims=[
                AnswerClaim(
                    claim_id="claim_1",
                    claim="Generic greeting",
                    supporting_chunk_ids=["chunk_1"],
                    citation_ids=["src_1"],
                )
            ],
            citations=[
                Citation(
                    citation_id="src_1",
                    article_id="article_1",
                    chunk_ids=["chunk_1"],
                    title="Wix title 1",
                )
            ],
        )

        verifier = VerifierAgent().run(answer=answer, context=context)

        self.assertEqual(verifier.status, "unsupported_answer")
        self.assertEqual(verifier.unsupported_claims[0]["reason"], "answer_not_responsive")

    def test_llm_claim_checker_rejects_semantically_unsupported_claim(self) -> None:
        context = context_for_question("How do I connect PayPal?")
        answer = AnswerTrace(
            answer="You can cancel your Premium plan from the PayPal dashboard. [src_1]",
            seen_chunk_ids=["chunk_1"],
            answer_claims=[
                AnswerClaim(
                    claim_id="claim_1",
                    claim="Premium plans can be canceled from the PayPal dashboard.",
                    supporting_chunk_ids=["chunk_1"],
                    citation_ids=["src_1"],
                )
            ],
            citations=[
                Citation(
                    citation_id="src_1",
                    article_id="article_1",
                    chunk_ids=["chunk_1"],
                    title="Wix title 1",
                )
            ],
        )
        client = FakeClaimCheckerClient(
            '{"status":"unsupported_answer","reason":"PayPal dashboard cancellation is not supported",'
            '"unsupported_claims":[{"claim_id":"claim_1","reason":"not stated",'
            '"supporting_chunk_ids":["chunk_1"]}],"missing_facets":[],"suggested_queries":[]}'
        )

        verifier = VerifierAgent(claim_checker=LLMClaimChecker(client=client)).run(
            answer=answer,
            context=context,
        )

        self.assertEqual(verifier.status, "unsupported_answer")
        self.assertEqual(verifier.unsupported_claims[0]["claim_id"], "claim_1")
        self.assertIn("Visible evidence chunks", client.messages[0][1]["content"])

    def test_llm_claim_checker_sees_compressed_context_summary(self) -> None:
        context = context_for_question("Does the special setup require manual approval?")
        context.compressed_summary = "Compressed fact: the special setup requires manual approval."
        context.compressed_context_ids = ["qid:compact:1"]
        answer = AnswerAgent().run(
            context=context,
            conversation_id="conv",
            turn_id="turn_0001",
        )
        client = FakeClaimCheckerClient(
            '{"status":"ready_to_answer","reason":"compressed context supports the claim",'
            '"unsupported_claims":[],"missing_facets":[],"suggested_queries":[]}'
        )

        verifier = VerifierAgent(claim_checker=LLMClaimChecker(client=client)).run(
            answer=answer,
            context=context,
        )

        prompt = client.messages[0][1]["content"]
        self.assertEqual(verifier.status, "ready_to_answer")
        self.assertIn("Visible compressed context summaries", prompt)
        self.assertIn("Compressed Context ID: qid:compact:1", prompt)
        self.assertIn("Supporting compressed context IDs: qid:compact:1", prompt)

    def test_real_runtime_wires_semantic_claim_checker(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"LLM_BASE_URL": "https://llm.example/v1", "LLM_MODEL": "fake-model"},
                clear=True,
            ),
            patch("src.agentic.runtime_factory.load_project_env"),
            patch(
                "src.agentic.runtime_factory.load_chunks",
                return_value=[chunk("chunk_1", "article_1")],
            ),
            patch("src.agentic.runtime_factory.HybridRetriever", FakeHybridRetriever),
            patch("src.agentic.runtime_factory.build_reranker", return_value=object()),
        ):
            runtime = build_real_dialogue_runtime(
                DialogueRuntimeConfig(enable_api_intent_gate=False)
            )
        try:
            claim_checker = runtime.orchestrator.verifier_agent.claim_checker
            self.assertIsInstance(claim_checker, LLMClaimChecker)
            self.assertIsInstance(runtime.orchestrator.query_agent.rewriter, LLMQueryRewriter)
        finally:
            runtime.close()

    def test_verifier_insufficient_retries_evidence_before_clarification(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(
            engine=engine,
            answer_agent=EmptyAnswerAgent(),
        )

        events = list(orchestrator.run_turn("conv_insufficient", "How do I connect PayPal?"))

        self.assertEqual(events[-1].event_type, "clarification_request")
        self.assertEqual(len(engine.calls), 2)

    def test_verifier_suggested_queries_can_trigger_multiple_refresh_rounds(self) -> None:
        engine = RecordingCompletionEngine()
        verifier_agent = SequenceVerifierAgent(
            [
                VerifierTrace(
                    status="insufficient_evidence",
                    reason="Need payment provider setup evidence.",
                    suggested_queries=["payment provider setup"],
                ),
                VerifierTrace(
                    status="insufficient_evidence",
                    reason="Need region compatibility evidence.",
                    suggested_queries=["payment provider region compatibility"],
                    missing_facets=[{"description": "region compatibility"}],
                ),
                VerifierTrace(status="ready_to_answer", reason="refreshed evidence is enough"),
            ]
        )
        orchestrator = self.build_orchestrator(
            engine=engine,
            verifier_agent=verifier_agent,
            latency=LatencyConfig(max_online_verifier_repair_rounds=2),
        )

        events = list(orchestrator.run_turn("conv_verifier_loop", "How do I connect PayPal?"))

        evidence_events = [event for event in events if event.event_type == "evidence"]
        repair_evidence_events = [
            event for event in evidence_events if "verifier_repair_round" in event.payload
        ]
        self.assertEqual(events[-1].event_type, "final_answer")
        self.assertEqual(len(engine.calls), 3)
        self.assertEqual(len(repair_evidence_events), 2)
        self.assertEqual(repair_evidence_events[-1].payload["verifier_repair_round"], 2)

    def test_unsupported_answer_rewrites_until_repair_budget_then_abstains(self) -> None:
        engine = RecordingCompletionEngine()
        orchestrator = self.build_orchestrator(
            engine=engine,
            answer_agent=InvalidAnswerAgent(),
        )

        events = list(orchestrator.run_turn("conv_bad_answer", "How do I connect PayPal?"))

        self.assertEqual(events[-1].event_type, "abstention")
        self.assertEqual(len(engine.calls), 1)


if __name__ == "__main__":
    unittest.main()
