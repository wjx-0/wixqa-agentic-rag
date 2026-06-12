from __future__ import annotations

from typing import Any, Callable

from src.agentic.dialogue_state import (
    DialogueState,
    EvidenceAgentResult,
    PipelineProfileConfig,
    QueryAgentResult,
)
from src.agentic.evidence_context import (
    EvidenceContext,
    build_initial_llm_call_id,
    build_initial_prompt_manifest,
)
from src.agentic.evidence_loop import EvidenceLoopConfig, run_evidence_completion_loop
from src.data.schema import KBChunk


ContextBuilder = Callable[[str, DialogueState], EvidenceContext]
CompletionEngine = Callable[[EvidenceContext, EvidenceLoopConfig], Any]


class EvidenceAgent:
    name = "evidence_agent"

    def __init__(
        self,
        *,
        context_builder: ContextBuilder | None = None,
        completion_engine: CompletionEngine | None = None,
        checker: Any | None = None,
        retriever: Any | None = None,
        reranker: Any | None = None,
        chunk_lookup: dict[str, KBChunk] | None = None,
        context_summarizer: Any | None = None,
    ) -> None:
        self.context_builder = context_builder
        self.completion_engine = completion_engine
        self.checker = checker
        self.retriever = retriever
        self.reranker = reranker
        self.chunk_lookup = chunk_lookup
        self.context_summarizer = context_summarizer

    def run(
        self,
        *,
        state: DialogueState,
        query: QueryAgentResult,
        profile: PipelineProfileConfig,
        force_refresh: bool = False,
        suggested_queries: list[str] | None = None,
    ) -> EvidenceAgentResult:
        context = self.resolve_context(
            state=state,
            query=query,
            profile=profile,
            force_refresh=force_refresh,
        )
        if suggested_queries:
            existing_verifier_query_count = sum(
                1
                for row in context.query_history
                if row.get("source") == "verifier"
            )
            for index, query_text in enumerate(suggested_queries, start=1):
                context.query_history.append(
                    {
                        "query_id": f"verifier_suggested_{existing_verifier_query_count + index}",
                        "query_text": query_text,
                        "source": "verifier",
                    }
                )
        loop_config = build_loop_config(profile)
        result = self.run_completion(context, loop_config)
        state.evidence_context = result.context
        return EvidenceAgentResult(
            context=result.context,
            sufficient=bool(result.completed),
            completed=bool(result.completed),
            retrieval_rounds=int(getattr(result, "retrieval_rounds", 0)),
            second_hop_query_count=int(getattr(result, "second_hop_query_count", 0)),
            total_latency_ms=float(getattr(result, "total_latency_ms", 0.0)),
            checker_outputs=list(getattr(result, "checker_outputs", []) or []),
            profile_config=profile,
        )

    def resolve_context(
        self,
        *,
        state: DialogueState,
        query: QueryAgentResult,
        profile: PipelineProfileConfig,
        force_refresh: bool,
    ) -> EvidenceContext:
        can_reuse = (
            profile.enable_evidence_context_reuse
            and query.reused_context
            and state.evidence_context is not None
            and not force_refresh
        )
        if can_reuse:
            return clone_context_for_reuse(
                state.evidence_context,
                state=state,
                question=query.standalone_question,
            )
        if self.context_builder is not None:
            return self.context_builder(query.standalone_question, state)
        if state.evidence_context is not None:
            return state.evidence_context
        raise RuntimeError(
            "EvidenceAgent needs an existing EvidenceContext or a context_builder."
        )

    def run_completion(self, context: EvidenceContext, config: EvidenceLoopConfig) -> Any:
        if self.completion_engine is not None:
            return self.completion_engine(context, config)
        if not all([self.checker, self.retriever, self.reranker, self.chunk_lookup]):
            raise RuntimeError(
                "EvidenceAgent needs completion_engine or checker/retriever/reranker/chunk_lookup."
            )
        return run_evidence_completion_loop(
            context=context,
            checker=self.checker,
            retriever=self.retriever,
            reranker=self.reranker,
            chunk_lookup=self.chunk_lookup,
            context_summarizer=self.context_summarizer,
            config=config,
        )


def build_loop_config(profile: PipelineProfileConfig) -> EvidenceLoopConfig:
    return EvidenceLoopConfig(
        max_rounds=profile.max_gap_rounds,
        max_queries_per_round=profile.max_queries_per_round,
        per_query_retrieve_top_k_chunks=profile.per_query_retrieve_top_k_chunks,
        max_raw_chunks_per_checker_call=profile.max_raw_chunks_per_checker_call,
        max_new_raw_chunks_per_round=profile.max_new_raw_chunks_per_round,
        max_new_chunks_per_article=profile.max_new_chunks_per_article,
        max_visible_chunks_per_article=profile.max_visible_chunks_per_article,
    )


def clone_context_for_reuse(
    context: EvidenceContext,
    *,
    state: DialogueState,
    question: str,
) -> EvidenceContext:
    if hasattr(context, "model_copy"):
        reused = context.model_copy(deep=True)
    else:
        reused = context.copy(deep=True)
    previous_qid = reused.qid
    reused.qid = build_reused_context_qid(state)
    reused.question = question
    reused.round_index = 0
    reused.gap_query_provenance = []
    reused.answer_provenance = None
    llm_call_id = build_initial_llm_call_id(reused.qid)
    for item in reused.active_items:
        item.active = True
        if llm_call_id not in item.visible_to_llm_call_ids:
            item.visible_to_llm_call_ids.append(llm_call_id)
    reused.packed_items = list(reused.active_items)
    reused.prompt_manifests = [
        build_initial_prompt_manifest(
            qid=reused.qid,
            llm_call_id=llm_call_id,
            active_items=reused.active_items,
        )
    ]
    reused.eval_info["reused_from_qid"] = previous_qid
    return reused


def build_reused_context_qid(state: DialogueState) -> str:
    user_turn_count = sum(1 for turn in state.turns if turn.role == "user")
    return f"{state.conversation_id}:turn_{user_turn_count:04d}"
