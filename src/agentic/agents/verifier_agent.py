from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.agentic.dialogue_state import AnswerTrace, VerifierTrace
from src.agentic.evidence_context import EvidenceContext, EvidenceItem
from src.llm.structured import (
    StructuredLLMCaller,
    parse_json_object,
)
from src.utils.answer_quality import answer_is_unhelpful
from src.utils.text_utils import compact_text, preview_text


DEFAULT_CLAIM_CHECKER_MAX_TOKENS = 768
DEFAULT_CLAIM_CHECKER_TEMPERATURE = 0.0


class VerifierAgent:
    name = "verifier_agent"

    def __init__(self, claim_checker: Any | None = None) -> None:
        self.claim_checker = claim_checker

    def run(
        self,
        *,
        answer: AnswerTrace,
        context: EvidenceContext,
    ) -> VerifierTrace:
        local_trace = verify_local_citations(answer, context)
        if local_trace.status != "ready_to_answer":
            return local_trace
        if self.claim_checker is None:
            return local_trace
        checked = self.claim_checker.verify(answer=answer, context=context)
        if isinstance(checked, VerifierTrace):
            return checked
        return VerifierTrace(**checked)


class ClaimCheckerPayload(BaseModel):
    status: Any
    reason: Any = ""
    unsupported_claims: Any = Field(default_factory=list)
    missing_facets: Any = Field(default_factory=list)
    suggested_queries: Any = Field(default_factory=list)


class LLMClaimChecker:
    def __init__(
        self,
        *,
        client: Any,
        temperature: float = DEFAULT_CLAIM_CHECKER_TEMPERATURE,
        max_tokens: int = DEFAULT_CLAIM_CHECKER_MAX_TOKENS,
        context_preview_chars: int = 700,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("claim checker max_tokens must be positive.")
        if context_preview_chars <= 0:
            raise ValueError("claim checker context_preview_chars must be positive.")
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_preview_chars = context_preview_chars
        self.model = compact_text(getattr(client, "model", "")) or "claim_checker"
        self.caller = StructuredLLMCaller(
            client=client,
            schema=ClaimCheckerPayload,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_min_tokens=1024,
        )

    def verify(self, *, answer: AnswerTrace, context: EvidenceContext) -> VerifierTrace:
        try:
            payload = self.caller.call(
                build_claim_checker_messages(
                    answer=answer,
                    context=context,
                    max_context_chars=self.context_preview_chars,
                )
            )
            return verifier_trace_from_payload(
                payload,
                answer=answer,
                context=context,
                model=self.model,
            )
        except Exception as exc:
            reason = compact_text(str(exc))[:240]
            return VerifierTrace(
                status="unsupported_answer",
                reason=f"Semantic claim verifier failed closed: {reason}",
                verifier_seen_chunk_ids=sorted(item.chunk_id for item in context.active_items),
                checked_claim_ids=[claim.claim_id for claim in answer.answer_claims],
                unsupported_claims=[
                    {
                        "reason": "claim_checker_failed",
                        "error": reason,
                        "checker_model": self.model,
                    }
                ],
            )


def verify_local_citations(answer: AnswerTrace, context: EvidenceContext) -> VerifierTrace:
    allowed_chunk_ids = {item.chunk_id for item in context.active_items}
    allowed_compressed_context_ids = set(context.compressed_context_ids)
    unsupported = []
    for claim in answer.answer_claims:
        missing = [
            chunk_id
            for chunk_id in claim.supporting_chunk_ids
            if chunk_id not in allowed_chunk_ids
        ]
        if missing:
            unsupported.append(
                {
                    "claim_id": claim.claim_id,
                    "reason": "supporting_chunk_ids_not_visible",
                    "chunk_ids": missing,
                }
            )
        missing_compressed_context_ids = [
            context_id
            for context_id in claim.supporting_compressed_context_ids
            if context_id not in allowed_compressed_context_ids
        ]
        if missing_compressed_context_ids:
            unsupported.append(
                {
                    "claim_id": claim.claim_id,
                    "reason": "supporting_compressed_context_ids_not_visible",
                    "compressed_context_ids": missing_compressed_context_ids,
                }
            )
    citation_ids = {citation.citation_id for citation in answer.citations}
    for claim in answer.answer_claims:
        missing_citations = [
            citation_id for citation_id in claim.citation_ids if citation_id not in citation_ids
        ]
        if missing_citations:
            unsupported.append(
                {
                    "claim_id": claim.claim_id,
                    "reason": "citation_id_not_found",
                    "citation_ids": missing_citations,
                }
            )
    for citation in answer.citations:
        missing = [
            chunk_id for chunk_id in citation.chunk_ids if chunk_id not in allowed_chunk_ids
        ]
        if missing:
            unsupported.append(
                {
                    "citation_id": citation.citation_id,
                    "reason": "citation_chunk_ids_not_visible",
                    "chunk_ids": missing,
                }
            )
        missing_compressed_context_ids = [
            context_id
            for context_id in citation.compressed_context_ids
            if context_id not in allowed_compressed_context_ids
        ]
        if missing_compressed_context_ids:
            unsupported.append(
                {
                    "citation_id": citation.citation_id,
                    "reason": "citation_compressed_context_ids_not_visible",
                    "compressed_context_ids": missing_compressed_context_ids,
                }
            )
    if unsupported:
        return VerifierTrace(
            status="unsupported_answer",
            reason="Answer cites evidence outside the visible answer context.",
            verifier_seen_chunk_ids=sorted(allowed_chunk_ids),
            checked_claim_ids=[claim.claim_id for claim in answer.answer_claims],
            unsupported_claims=unsupported,
        )
    if answer_is_unhelpful(answer.answer):
        return VerifierTrace(
            status="unsupported_answer",
            reason="Answer is too generic and does not answer the user's question.",
            verifier_seen_chunk_ids=sorted(allowed_chunk_ids),
            checked_claim_ids=[claim.claim_id for claim in answer.answer_claims],
            unsupported_claims=[
                {
                    "reason": "answer_not_responsive",
                    "answer": compact_text(answer.answer),
                }
            ],
        )
    has_claim_support = any(
        claim.supporting_chunk_ids or claim.supporting_compressed_context_ids
        for claim in answer.answer_claims
    )
    if not answer.answer_claims or not answer.citations or not has_claim_support:
        return VerifierTrace(
            status="insufficient_evidence",
            reason="Answer has no claim support map or citations.",
            verifier_seen_chunk_ids=sorted(allowed_chunk_ids),
            missing_facets=[{"description": "Need cited claim support before answering."}],
            suggested_queries=[],
        )
    return VerifierTrace(
        status="ready_to_answer",
        reason="All answer claims cite visible evidence.",
        verifier_seen_chunk_ids=sorted(allowed_chunk_ids),
        checked_claim_ids=[claim.claim_id for claim in answer.answer_claims],
    )


def build_claim_checker_messages(
    *,
    answer: AnswerTrace,
    context: EvidenceContext,
    max_context_chars: int,
) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        [
            "You are a strict verifier for a Wix Help Center RAG answer.",
            "Check whether each answer claim is directly supported by the listed evidence chunks or compressed context summaries.",
            "Do not use outside knowledge or product intuition.",
            "A citation being present is not enough; the cited evidence text must support the claim.",
            "Return one compact JSON object only.",
            "status must be ready_to_answer, unsupported_answer, or insufficient_evidence.",
            "Use unsupported_answer when a claim is contradicted, invented, or only weakly inferred.",
            "Use insufficient_evidence when more evidence is needed before deciding.",
            (
                "JSON schema: {\"status\":\"ready_to_answer|unsupported_answer|insufficient_evidence\","
                "\"reason\":\"short reason\",\"unsupported_claims\":[{\"claim_id\":\"claim id\","
                "\"reason\":\"why unsupported\",\"supporting_chunk_ids\":[\"chunk id\"],"
                "\"supporting_compressed_context_ids\":[\"context id\"]}],"
                "\"missing_facets\":[{\"description\":\"needed evidence\"}],"
                "\"suggested_queries\":[\"query\"]}"
            ),
        ]
    )
    evidence_by_id = {item.chunk_id: item for item in context.active_items}
    evidence_lines = [
        format_claim_checker_evidence_item(item, max_context_chars=max_context_chars)
        for item in context.active_items
    ]
    claim_lines = []
    for claim in answer.answer_claims:
        cited_items = [
            evidence_by_id[chunk_id]
            for chunk_id in claim.supporting_chunk_ids
            if chunk_id in evidence_by_id
        ]
        claim_lines.append(
            "\n".join(
                [
                    f"Claim ID: {claim.claim_id}",
                    f"Claim: {compact_text(claim.claim)}",
                    "Supporting chunk IDs: " + ", ".join(claim.supporting_chunk_ids),
                    "Supporting compressed context IDs: "
                    + ", ".join(claim.supporting_compressed_context_ids),
                    "Citation IDs: " + ", ".join(claim.citation_ids),
                    "Supporting titles: "
                    + "; ".join(compact_text(item.title) for item in cited_items if compact_text(item.title)),
                ]
            )
        )
    sections = [
        f"Question:\n{compact_text(context.question)}",
        f"Answer:\n{compact_text(answer.answer)}",
        "Claims:\n" + "\n\n".join(claim_lines),
        "Visible evidence chunks:\n" + "\n\n".join(evidence_lines),
    ]
    compressed_context = format_claim_checker_compressed_context(
        context,
        max_context_chars=max_context_chars,
    )
    if compressed_context:
        sections.append("Visible compressed context summaries:\n" + compressed_context)
    user_prompt = "\n\n".join(sections)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def format_claim_checker_evidence_item(item: EvidenceItem, *, max_context_chars: int) -> str:
    return "\n".join(
        [
            f"Chunk ID: {item.chunk_id}",
            f"Title: {item.title or '(untitled)'}",
            f"Evidence: {preview_text(item.text_preview, limit=max_context_chars)}",
        ]
    )


def format_claim_checker_compressed_context(
    context: EvidenceContext,
    *,
    max_context_chars: int,
) -> str:
    summary = compact_text(context.compressed_summary)
    context_ids = [
        context_id
        for value in context.compressed_context_ids
        if (context_id := compact_text(value))
    ]
    if not summary or not context_ids:
        return ""
    id_label = "Compressed Context ID" if len(context_ids) == 1 else "Compressed Context IDs"
    return "\n".join(
        [
            f"{id_label}: {', '.join(context_ids)}",
            f"Summary: {preview_text(summary, limit=max_context_chars)}",
        ]
    )


def parse_claim_checker_response(
    raw_text: str,
    *,
    answer: AnswerTrace,
    context: EvidenceContext,
    model: str,
) -> VerifierTrace:
    try:
        payload = ClaimCheckerPayload(**parse_json_object(raw_text))
    except Exception as exc:
        raise ValueError(f"claim checker returned invalid JSON: {exc}") from exc

    return verifier_trace_from_payload(
        payload,
        answer=answer,
        context=context,
        model=model,
    )


def verifier_trace_from_payload(
    payload: ClaimCheckerPayload,
    *,
    answer: AnswerTrace,
    context: EvidenceContext,
    model: str,
) -> VerifierTrace:
    status = normalize_verifier_status(payload.status)
    unsupported_claims = normalize_unsupported_claims(payload.unsupported_claims)
    missing_facets = normalize_missing_facets(payload.missing_facets)
    suggested_queries = [
        text
        for value in payload.suggested_queries or []
        if (text := compact_text(value))
    ][:3]
    reason = compact_text(payload.reason) or default_claim_checker_reason(status)
    if status == "ready_to_answer" and unsupported_claims:
        status = "unsupported_answer"
    if status == "ready_to_answer" and missing_facets:
        status = "insufficient_evidence"
    if status == "unsupported_answer" and not unsupported_claims:
        unsupported_claims = [
            {
                "reason": "claim_checker_marked_unsupported",
                "checker_model": model,
            }
        ]
    if status == "insufficient_evidence" and not missing_facets:
        missing_facets = [{"description": "Semantic verifier requested more evidence."}]
    return VerifierTrace(
        status=status,
        reason=reason,
        verifier_seen_chunk_ids=sorted(item.chunk_id for item in context.active_items),
        checked_claim_ids=[claim.claim_id for claim in answer.answer_claims],
        unsupported_claims=unsupported_claims,
        missing_facets=missing_facets,
        suggested_queries=suggested_queries,
    )


def normalize_verifier_status(value: Any) -> str:
    status = compact_text(value).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "ready": "ready_to_answer",
        "supported": "ready_to_answer",
        "ok": "ready_to_answer",
        "unsupported": "unsupported_answer",
        "not_supported": "unsupported_answer",
        "needs_more_evidence": "insufficient_evidence",
        "insufficient": "insufficient_evidence",
    }
    status = aliases.get(status, status)
    if status not in {"ready_to_answer", "unsupported_answer", "insufficient_evidence"}:
        raise ValueError(f"unsupported verifier status: {status!r}")
    return status


def normalize_unsupported_claims(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    output = []
    for row in rows:
        if isinstance(row, dict):
            normalized = dict(row)
            normalized["reason"] = compact_text(normalized.get("reason")) or "unsupported claim"
            output.append(normalized)
        elif compact_text(row):
            output.append({"reason": compact_text(row)})
    return output


def normalize_missing_facets(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    output = []
    for row in rows:
        if isinstance(row, dict):
            description = compact_text(row.get("description") or row.get("text") or row.get("reason"))
            output.append({**row, "description": description or "Need more supporting evidence."})
        elif compact_text(row):
            output.append({"description": compact_text(row)})
    return output


def default_claim_checker_reason(status: str) -> str:
    if status == "ready_to_answer":
        return "All claims are directly supported by visible evidence."
    if status == "insufficient_evidence":
        return "More evidence is needed before answering."
    return "At least one answer claim is not supported by visible evidence."
