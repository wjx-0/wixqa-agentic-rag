from __future__ import annotations

import re

from src.agentic.dialogue_state import AnswerClaim, AnswerTrace, Citation
from src.agentic.evidence_context import EvidenceContext, EvidenceItem
from src.agentic.provenance import PromptManifest
from src.utils.text_utils import compact_text


def build_extractive_answer(
    context: EvidenceContext,
    manifest: PromptManifest,
    max_citations: int,
) -> AnswerTrace:
    items = context.active_items[:max_citations]
    raw_citations = [citation_from_item(index, item) for index, item in enumerate(items, start=1)]
    compressed_citation = citation_from_compressed_context(context)
    citations = raw_citations + ([compressed_citation] if compressed_citation else [])
    citation_ids_by_chunk = {
        chunk_id: citation.citation_id
        for citation in raw_citations
        for chunk_id in citation.chunk_ids
    }
    claims = [
        AnswerClaim(
            claim_id=f"claim_{index}",
            claim=evidence_summary_from_item(item),
            supporting_chunk_ids=[item.chunk_id],
            citation_ids=[citation_ids_by_chunk[item.chunk_id]],
        )
        for index, item in enumerate(items, start=1)
        if item.chunk_id in citation_ids_by_chunk
    ]
    if compressed_citation:
        claims.append(
            AnswerClaim(
                claim_id=f"claim_{len(claims) + 1}",
                claim=summary_from_compressed_context(context.compressed_summary),
                supporting_compressed_context_ids=list(
                    compressed_citation.compressed_context_ids
                ),
                citation_ids=[compressed_citation.citation_id],
            )
        )
    answer = build_evidence_answer(
        items,
        raw_citations,
        compressed_summary=context.compressed_summary,
        compressed_citation=compressed_citation,
    )
    return AnswerTrace(
        answer=answer,
        seen_chunk_ids=list(manifest.input_chunk_ids),
        seen_compressed_context_ids=list(manifest.input_compressed_context_ids),
        answer_claims=claims,
        citations=citations,
        prompt_manifest_id=manifest.llm_call_id,
        generation_mode="extractive_fallback",
    )


def citation_from_item(index: int, item: EvidenceItem) -> Citation:
    return Citation(
        citation_id=f"src_{index}",
        article_id=item.article_id,
        chunk_ids=[item.chunk_id],
        title=item.title,
        url=None,
    )


def citation_from_compressed_context(context: EvidenceContext) -> Citation | None:
    context_ids = unique_context_ids(context.compressed_context_ids)
    if not compact_text(context.compressed_summary) or not context_ids:
        return None
    return Citation(
        citation_id="ctx_1",
        article_id="compressed_context",
        compressed_context_ids=context_ids,
        title="Compressed evidence",
        url=None,
    )


def claim_text_from_item(item: EvidenceItem) -> str:
    return evidence_summary_from_item(item)


def build_evidence_answer(
    items: list[EvidenceItem],
    citations: list[Citation],
    *,
    compressed_summary: str | None = None,
    compressed_citation: Citation | None = None,
) -> str:
    parts = []
    if compressed_citation:
        summary = summary_from_compressed_context(compressed_summary)
        if summary:
            parts.append(f"{summary} [{compressed_citation.citation_id}]")
    for item, citation in zip(items, citations):
        summary = evidence_summary_from_item(item)
        if summary:
            parts.append(f"{summary} [{citation.citation_id}]")
        if len(parts) == 3:
            break
    if parts:
        return compact_text(" ".join(parts))
    citation_refs = " ".join(f"[{citation.citation_id}]" for citation in citations)
    return compact_text(
        f"I do not have enough specific evidence to answer beyond the cited Wix sources. {citation_refs}"
    )


def evidence_summary_from_item(item: EvidenceItem) -> str:
    title = compact_text(item.title)
    text = compact_text(item.text_preview)
    if title and text.casefold().startswith(title.casefold()):
        text = text[len(title) :].lstrip(" :-–—")
    sentence = first_meaningful_sentence(text)
    if sentence:
        return sentence
    return title or "The cited Wix Help Center article has relevant evidence."


def summary_from_compressed_context(summary: str | None) -> str:
    sentence = first_meaningful_sentence(compact_text(summary), max_chars=260)
    if sentence:
        return sentence
    return "The compressed Wix Help Center context contains answer-relevant evidence."


def first_meaningful_sentence(text: str, *, max_chars: int = 240) -> str:
    normalized = compact_text(text)
    if not normalized:
        return ""
    candidates = re.split(r"(?<=[.!?])\s+", normalized)
    for candidate in candidates:
        candidate = compact_text(candidate)
        if len(candidate) >= 18:
            return trim_sentence(candidate, max_chars=max_chars)
    return trim_sentence(normalized, max_chars=max_chars)


def trim_sentence(text: str, *, max_chars: int) -> str:
    text = compact_text(text)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated or text[:max_chars].strip()


def unique_context_ids(values: list[str]) -> list[str]:
    output = []
    for value in values:
        text = compact_text(value)
        if text and text not in output:
            output.append(text)
    return output
