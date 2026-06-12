from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.agentic.dialogue_state import AnswerClaim, AnswerTrace, Citation
from src.agentic.evidence_context import EvidenceContext, EvidenceItem
from src.agentic.provenance import PromptManifest
from src.llm.answer_fallback import build_extractive_answer
from src.llm.structured import (
    StructuredLLMCaller,
    StructuredLLMError,
    parse_json_object,
)
from src.utils.answer_quality import answer_is_unhelpful, significant_terms
from src.utils.text_utils import compact_text, preview_text


DEFAULT_ANSWER_MAX_TOKENS = 1600
DEFAULT_ANSWER_TEMPERATURE = 0.1
DEFAULT_MAX_ANSWER_CITATIONS = 3


class LLMAnswerGeneratorError(RuntimeError):
    pass


class AnswerPayload(BaseModel):
    answer: Any = ""
    claims: Any = Field(default_factory=list)


class LLMAnswerGenerator:
    def __init__(
        self,
        *,
        client: Any,
        temperature: float = DEFAULT_ANSWER_TEMPERATURE,
        max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
        max_citations: int = DEFAULT_MAX_ANSWER_CITATIONS,
    ) -> None:
        if max_tokens <= 0:
            raise LLMAnswerGeneratorError("max_tokens must be positive.")
        if max_citations <= 0:
            raise LLMAnswerGeneratorError("max_citations must be positive.")
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_citations = max_citations
        self.model = compact_text(getattr(client, "model", "")) or "answer_generator"
        self.caller = StructuredLLMCaller(
            client=client,
            schema=AnswerPayload,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_min_tokens=2400,
        )

    def generate(
        self,
        *,
        context: EvidenceContext,
        manifest: PromptManifest,
        revision_request: str | None = None,
    ) -> AnswerTrace:
        answer_items = select_answer_items(context, self.max_citations)
        citations = build_citations(
            answer_items,
            compressed_context_ids=context.compressed_context_ids
            if compact_text(context.compressed_summary)
            else [],
        )
        messages = build_answer_messages(
            question=context.question,
            items=answer_items,
            citations=citations,
            compressed_summary=context.compressed_summary,
            revision_request=revision_request,
        )
        payload, parse_error = complete_answer_payload(self, messages)
        if payload is None:
            error_prefix = (
                "llm_answer_failed"
                if parse_error.startswith("llm_call_failed")
                else "invalid_llm_json"
            )
            return build_fallback_trace(
                context,
                manifest,
                self.max_citations,
                error=f"{error_prefix}: {parse_error}",
            )

        answer = compact_text(payload.get("answer")) or fallback_answer_payload(
            context.question,
            citations,
        )["answer"]
        if answer_is_unhelpful(
            answer,
            question=context.question,
            evidence_texts=[item.title for item in answer_items[:3]]
            + ([context.compressed_summary] if compact_text(context.compressed_summary) else []),
        ):
            return build_fallback_trace(
                context,
                manifest,
                self.max_citations,
                error="llm_answer_unhelpful",
            )
        citation_ids = {citation.citation_id for citation in citations}
        chunk_ids_by_citation = {
            citation.citation_id: list(citation.chunk_ids) for citation in citations
        }
        compressed_context_ids_by_citation = {
            citation.citation_id: list(citation.compressed_context_ids)
            for citation in citations
        }
        allowed_chunk_ids = {
            chunk_id
            for citation in citations
            for chunk_id in citation.chunk_ids
        }
        allowed_compressed_context_ids = {
            context_id
            for citation in citations
            for context_id in citation.compressed_context_ids
        }
        claims = normalize_answer_claims(
            payload.get("claims"),
            citation_ids=citation_ids,
            chunk_ids_by_citation=chunk_ids_by_citation,
            allowed_chunk_ids=allowed_chunk_ids,
            compressed_context_ids_by_citation=compressed_context_ids_by_citation,
            allowed_compressed_context_ids=allowed_compressed_context_ids,
        )
        if not claims:
            claims = fallback_claims(citations)
        answer = ensure_answer_has_citation(answer, citations)
        return AnswerTrace(
            answer=answer,
            seen_chunk_ids=list(manifest.input_chunk_ids),
            seen_compressed_context_ids=list(manifest.input_compressed_context_ids),
            answer_claims=claims,
            citations=citations,
            prompt_manifest_id=manifest.llm_call_id,
            generation_mode="llm",
        )


def build_answer_messages(
    *,
    question: str,
    items: list[EvidenceItem],
    citations: list[Citation],
    compressed_summary: str | None = None,
    revision_request: str | None = None,
) -> list[dict[str, str]]:
    citation_by_chunk_id = {
        chunk_id: citation.citation_id
        for citation in citations
        for chunk_id in citation.chunk_ids
    }
    system_prompt = "\n".join(
        [
            "You are a Wix Help Center customer support agent.",
            "Answer only from the provided evidence.",
            "Use the same language as the user when clear.",
            "Write a direct, natural customer-support answer. Do not concatenate article titles or raw snippets.",
            "Keep the answer concise: usually 3-5 short sentences or up to 4 numbered steps.",
            "Use only evidence that is relevant to the user's actual question; ignore unrelated evidence even if provided.",
            "For 'how' or 'specific steps' questions, provide concise numbered steps when the evidence supports them.",
            "If evidence is incomplete, say what is known and what is not confirmed.",
            "Cite evidence with citation ids like [src_1] or [ctx_1].",
            "Return valid JSON only with keys: answer, claims.",
            "claims items must have: claim, supporting_chunk_ids, supporting_compressed_context_ids, citation_ids.",
        ]
    )
    evidence_lines = []
    for item in items:
        citation_id = citation_by_chunk_id.get(item.chunk_id, "")
        evidence_lines.append(
            "\n".join(
                [
                    f"Citation: {citation_id}",
                    f"Chunk ID: {item.chunk_id}",
                    f"Title: {item.title or '(untitled)'}",
                    f"Evidence: {preview_text(item.text_preview, limit=700)}",
                ]
            )
        )
    sections = [
        f"Question:\n{compact_text(question)}",
        format_compressed_context_section(compressed_summary, citations),
        "Evidence:\n" + "\n\n".join(evidence_lines),
        (
            "JSON schema:\n"
            '{"answer": string, "claims": [{"claim": string, '
            '"supporting_chunk_ids": string[], '
            '"supporting_compressed_context_ids": string[], '
            '"citation_ids": string[]}]}'
        ),
    ]
    revision = compact_text(revision_request)
    if revision:
        sections.insert(1, f"Revision request:\n{revision}")
    sections = [section for section in sections if section]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def format_compressed_context_section(
    summary: Any,
    citations: list[Citation],
) -> str:
    text = compact_text(summary)
    compressed_citations = [citation for citation in citations if citation.compressed_context_ids]
    if not text or not compressed_citations:
        return ""
    blocks = []
    for citation in compressed_citations:
        context_ids = list(citation.compressed_context_ids)
        id_label = "Compressed Context ID" if len(context_ids) == 1 else "Compressed Context IDs"
        blocks.append(
            "\n".join(
                [
                    f"Citation: {citation.citation_id}",
                    f"{id_label}: {', '.join(context_ids)}",
                    f"Summary: {preview_text(text, limit=900)}",
                ]
            )
        )
    return "Compressed context summary:\n" + "\n\n".join(blocks)


def complete_answer_payload(
    generator: LLMAnswerGenerator,
    messages: list[dict[str, str]],
) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = generator.caller.call(messages)
        return answer_payload_to_dict(payload), ""
    except StructuredLLMError as exc:
        if exc.attempts and all(
            not compact_text(attempt.get("raw_text_preview")) for attempt in exc.attempts
        ):
            return None, f"llm_call_failed: {exc.attempts[-1].get('error', str(exc))}"
        return None, compact_text(str(exc))[:300]


def parse_answer_payload(raw_text: str) -> dict[str, Any]:
    try:
        payload = AnswerPayload(**parse_json_object(raw_text))
    except Exception as exc:
        raise LLMAnswerGeneratorError(f"Answer generator returned invalid JSON: {exc}") from exc
    return answer_payload_to_dict(payload)


def answer_payload_to_dict(payload: AnswerPayload) -> dict[str, Any]:
    return {
        "answer": payload.answer,
        "claims": payload.claims,
    }


def normalize_answer_claims(
    value: Any,
    *,
    citation_ids: set[str],
    chunk_ids_by_citation: dict[str, list[str]],
    allowed_chunk_ids: set[str],
    compressed_context_ids_by_citation: dict[str, list[str]],
    allowed_compressed_context_ids: set[str],
) -> list[AnswerClaim]:
    raw_claims = value if isinstance(value, list) else []
    claims = []
    for index, raw_claim in enumerate(raw_claims, start=1):
        if not isinstance(raw_claim, dict):
            continue
        claim_text = compact_text(raw_claim.get("claim"))
        if not claim_text:
            continue
        raw_citation_ids = unique_texts(raw_claim.get("citation_ids") or [])
        valid_citation_ids = [
            citation_id for citation_id in raw_citation_ids if citation_id in citation_ids
        ]
        raw_chunk_ids = unique_texts(raw_claim.get("supporting_chunk_ids") or [])
        valid_chunk_ids = [
            chunk_id for chunk_id in raw_chunk_ids if chunk_id in allowed_chunk_ids
        ]
        raw_compressed_context_ids = unique_texts(
            raw_claim.get("supporting_compressed_context_ids") or []
        )
        valid_compressed_context_ids = [
            context_id
            for context_id in raw_compressed_context_ids
            if context_id in allowed_compressed_context_ids
        ]
        for citation_id in valid_citation_ids:
            for chunk_id in chunk_ids_by_citation.get(citation_id, []):
                if chunk_id not in valid_chunk_ids:
                    valid_chunk_ids.append(chunk_id)
            for context_id in compressed_context_ids_by_citation.get(citation_id, []):
                if context_id not in valid_compressed_context_ids:
                    valid_compressed_context_ids.append(context_id)
        if not valid_chunk_ids and not valid_compressed_context_ids and not valid_citation_ids:
            continue
        claims.append(
            AnswerClaim(
                claim_id=f"claim_{index}",
                claim=claim_text,
                supporting_chunk_ids=valid_chunk_ids,
                supporting_compressed_context_ids=valid_compressed_context_ids,
                citation_ids=valid_citation_ids,
            )
        )
    return claims


def build_citations(
    items: list[EvidenceItem],
    *,
    compressed_context_ids: list[str] | None = None,
) -> list[Citation]:
    citations = [
        Citation(
            citation_id=f"src_{index}",
            article_id=item.article_id,
            chunk_ids=[item.chunk_id],
            title=item.title,
            url=None,
        )
        for index, item in enumerate(items, start=1)
    ]
    context_ids = unique_texts(compressed_context_ids or [])
    if context_ids:
        citations.append(
            Citation(
                citation_id="ctx_1",
                article_id="compressed_context",
                compressed_context_ids=context_ids,
                title="Compressed evidence",
                url=None,
            )
        )
    return citations


def fallback_answer_payload(question: str, citations: list[Citation]) -> dict[str, Any]:
    citation_text = " ".join(f"[{citation.citation_id}]" for citation in citations[:2])
    return {
        "answer": compact_text(
            f"I found relevant Wix Help Center evidence for this question. {citation_text}"
        ),
        "claims": [],
    }


def build_fallback_trace(
    context: EvidenceContext,
    manifest: PromptManifest,
    max_citations: int,
    *,
    error: str,
) -> AnswerTrace:
    trace = build_extractive_answer(context, manifest, max_citations)
    trace.generation_mode = "extractive_fallback"
    trace.generator_error = compact_text(error)[:300]
    return trace


def select_answer_items(context: EvidenceContext, max_citations: int) -> list[EvidenceItem]:
    active_items = list(context.active_items)
    if not active_items:
        return []
    priority_ids = priority_chunk_ids_from_context(context)
    selected = [item for item in active_items if item.chunk_id in priority_ids]
    if selected:
        return selected[:max_citations]

    scored_items = [
        (score_answer_item(context.question, item), index, item)
        for index, item in enumerate(active_items)
    ]
    relevant = [row for row in scored_items if row[0] > 0]
    if relevant:
        relevant.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in relevant[:max_citations]]
    return active_items[:max_citations]


def priority_chunk_ids_from_context(context: EvidenceContext) -> list[str]:
    output = []
    for chunk_id in context.eval_info.get("checker_seen_chunk_ids", []) or []:
        add_unique(output, chunk_id)
    for facet in context.covered_facets:
        for chunk_id in facet.get("supporting_chunk_ids") or []:
            add_unique(output, chunk_id)
    return output


def score_answer_item(question: str, item: EvidenceItem) -> int:
    haystack = compact_text(f"{item.title or ''} {item.text_preview}").casefold()
    score = 0
    for term in significant_terms(question):
        if term in haystack:
            score += 3
    question_lower = compact_text(question).casefold()
    if "payment" in haystack and ("支付" in question_lower or "付款" in question_lower or "收款" in question_lower):
        score += 3
    if "paypal" in haystack and "paypal" in question_lower:
        score += 5
    if "booking" in haystack and ("booking" in question_lower or "预订" in question_lower or "预约" in question_lower):
        score += 3
    if "store" in haystack and ("store" in question_lower or "商店" in question_lower):
        score += 3
    return score


def add_unique(output: list[str], value: Any) -> None:
    text = compact_text(value)
    if text and text not in output:
        output.append(text)


def fallback_claims(citations: list[Citation]) -> list[AnswerClaim]:
    return [
        AnswerClaim(
            claim_id=f"claim_{index}",
            claim=f"The cited Wix Help Center article is relevant to the user's question.",
            supporting_chunk_ids=list(citation.chunk_ids),
            supporting_compressed_context_ids=list(citation.compressed_context_ids),
            citation_ids=[citation.citation_id],
        )
        for index, citation in enumerate(citations[:2], start=1)
    ]


def ensure_answer_has_citation(answer: str, citations: list[Citation]) -> str:
    if any(f"[{citation.citation_id}]" in answer for citation in citations):
        return answer
    if not citations:
        return answer
    return compact_text(f"{answer} [{citations[0].citation_id}]")


def unique_texts(values: Any) -> list[str]:
    output = []
    raw_values = values if isinstance(values, list) else [values]
    for value in raw_values:
        text = compact_text(value)
        if text and text not in output:
            output.append(text)
    return output
