from __future__ import annotations

from typing import Any

from src.agentic.dialogue_state import AnswerTrace
from src.agentic.evidence_context import EvidenceContext
from src.agentic.provenance import AnswerProvenance, PromptManifest
from src.llm.answer_fallback import build_extractive_answer
from src.utils.text_utils import compact_text


class AnswerAgent:
    name = "answer_agent"

    def __init__(self, generator: Any | None = None, max_citations: int = 5) -> None:
        self.generator = generator
        self.max_citations = max_citations

    def run(
        self,
        *,
        context: EvidenceContext,
        conversation_id: str,
        turn_id: str,
        revision_request: str | None = None,
    ) -> AnswerTrace:
        manifest = build_answer_manifest(
            context=context,
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        context.prompt_manifests.append(manifest)
        if self.generator is not None:
            trace = self.generator.generate(
                context=context,
                manifest=manifest,
                revision_request=revision_request,
            )
            if isinstance(trace, AnswerTrace):
                answer_trace = trace
            else:
                answer_trace = AnswerTrace(**trace)
        else:
            answer_trace = build_extractive_answer(context, manifest, self.max_citations)
        context.answer_provenance = AnswerProvenance(
            answer_call_id=manifest.llm_call_id,
            seen_chunk_ids=answer_trace.seen_chunk_ids,
            supporting_chunk_ids=unique_chunk_ids(
                chunk_id
                for claim in answer_trace.answer_claims
                for chunk_id in claim.supporting_chunk_ids
            ),
            supporting_compressed_context_ids=unique_chunk_ids(
                context_id
                for claim in answer_trace.answer_claims
                for context_id in claim.supporting_compressed_context_ids
            ),
            metadata={
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "revision_request": revision_request,
            },
        )
        return answer_trace


def build_answer_manifest(
    *,
    context: EvidenceContext,
    conversation_id: str,
    turn_id: str,
) -> PromptManifest:
    llm_call_id = f"{context.qid}:answer:{turn_id}"
    return PromptManifest(
        llm_call_id=llm_call_id,
        phase="answer",
        qid=context.qid,
        conversation_id=conversation_id,
        turn_id=turn_id,
        input_chunk_ids=[item.chunk_id for item in context.active_items],
        input_article_ids=unique_chunk_ids(item.article_id for item in context.active_items),
        input_snippet_ids=[item.snippet_id for item in context.active_items],
        input_compressed_context_ids=list(context.compressed_context_ids),
    )


def unique_chunk_ids(values: Any) -> list[str]:
    output = []
    for value in values:
        text = compact_text(value)
        if text and text not in output:
            output.append(text)
    return output
