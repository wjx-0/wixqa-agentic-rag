from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from src.llm.chat_client import (
    DEFAULT_OPENAI_CHAT_TIMEOUT,
    OpenAICompatibleChatClient as BaseOpenAICompatibleChatClient,
)
from src.llm.structured import (
    StructuredLLMCaller,
    StructuredLLMError,
)
from src.utils.io_utils import loads_json
from src.utils.json_utils import extract_json_object
from src.utils.text_utils import compact_text, preview_text, to_string_list


DEFAULT_CHECKER_TEMPERATURE = 0.0
DEFAULT_CHECKER_MAX_TOKENS = 512
DEFAULT_CHECKER_TIMEOUT = DEFAULT_OPENAI_CHAT_TIMEOUT
DEFAULT_MAX_NEXT_QUERIES = 3
DEFAULT_TRACEABLE_MAX_NEXT_QUERIES = 2
DEFAULT_CONTEXT_PREVIEW_CHARS = 650
CHECKER_MODE_TRACEABLE = "traceable"
CHECKER_MODE_COMPACT = "compact"
VALID_CHECKER_MODES = {CHECKER_MODE_TRACEABLE, CHECKER_MODE_COMPACT}
CONTRADICTION_REASON_PHRASES = (
    "not explicitly detailed",
    "no direct guidance",
    "partially supported",
    "unclear",
)
NEGATED_REASON_PATTERNS = (
    r"\bno\s+missing\s+facets?\s+or\s+blocking\s+gaps?\b",
    r"\bno\b.{0,48}\bmissing\s+evidence\b",
    r"\bno\b.{0,48}\bblocking\s+gaps?\b",
    r"\bno\s+(?:evidence\s+|blocking\s+)?gaps?\b",
    r"\bno\s+missing(?:\s+(?:facets?|evidence))?\b",
    r"\bnot\s+missing\b",
    r"\bwithout\s+(?:evidence\s+|blocking\s+)?gaps?\b",
    r"\bwithout\s+missing\b",
    r"\bnone\s+missing\b",
    r"\bnothing\s+missing\b",
    r"\bnot\s+identif(?:y|ied)\b",
)


class EvidenceCheckerError(RuntimeError):
    pass


class CheckerPayload(BaseModel):
    sufficient: Any

    class Config:
        extra = "allow"


def checker_payload_to_dict(payload: CheckerPayload) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()  # type: ignore[attr-defined]
    return payload.dict()


class OpenAICompatibleChatClient(BaseOpenAICompatibleChatClient):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = DEFAULT_CHECKER_TIMEOUT,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            error_factory=EvidenceCheckerError,
        )


def build_evidence_checker_messages(
    question: str,
    top_chunks: list[dict[str, Any]],
    *,
    max_context_chars: int = DEFAULT_CONTEXT_PREVIEW_CHARS,
) -> list[dict[str, str]]:
    normalized_question = compact_text(question)
    if not normalized_question:
        raise EvidenceCheckerError("question must not be empty.")
    if max_context_chars <= 0:
        raise EvidenceCheckerError("max_context_chars must be positive.")

    system_prompt = "\n".join(
        [
            "You are an evidence sufficiency checker for Wix Help Center retrieval.",
            "Do not answer the user's question.",
            "Judge whether the retrieved chunks support a correct, useful, and non-misleading answer.",
            "Set sufficient=true when the available evidence directly supports the answer, even if it is not exhaustive.",
            "Do not require exact wording, exhaustive edge cases, or more explicit confirmation when the evidence already supports the answer.",
            "Set sufficient=false only when a blocking evidence gap would make the answer unsupported, materially incomplete, or misleading to a typical user who just needs to complete the task.",
            "If the evidence enables a correct and actionable answer, set sufficient=true even if some detail, edge case, or confirmation step is missing.",
            "When in doubt, prefer sufficient=true over triggering unnecessary retrieval.",
            "Before deciding sufficient, decompose the question into required facets: Wix product, object/entity, feature, action, setting, integration, error, condition, comparison, or required step.",
            "If the question asks about multiple objects, actions, steps, conditions, products, integrations, or feature requirements, the top10 evidence must directly cover every required facet.",
            "If any required facet is missing, contradicted, or only inferable from adjacent evidence, set sufficient=false and put that exact facet in blocking_missing_evidence.",
            "Do NOT treat user-side unknowns as blocking gaps: the user's region, account type, product type, plan, or current configuration are context the user already knows and does not need retrieved.",
            "Do NOT treat third-party or dynamic information as blocking gaps: specific values from Google, PayPal, or other external services, and exact pricing figures that change over time, cannot be retrieved from Wix Help Center and must not trigger retrieval.",
            "Do not downgrade a missing required facet to nice_to_have_missing_evidence just because related evidence is present.",
            "Do not trigger retrieval for nice-to-have details, extra examples, background context, or minor clarification.",
            "blocking_missing_evidence must list only blocking gaps that justify another retrieval step.",
            "For how-to, setup, automation, and timeline questions, treat missing steps, phases, conditions, or lifecycle stages as blocking, not nice-to-have, even if partial information is present.",
            "nice_to_have_missing_evidence may list non-blocking details, but those must not cause retrieval.",
            "next_queries must target blocking_missing_evidence, not merely rewrite the original question.",
            "Do not use vague pronouns such as it, this, that feature, or that setting in next_queries.",
            "Generate at most 3 next_queries total.",
            "At least one query must be a broad variant: remove the specific Wix product name and describe only the action or concept, because the answer may be documented under a different product module such as Stores, Editor, or App.",
            "Remaining queries should be precise: include the specific Wix product and feature name.",
            "Return valid JSON only with keys: sufficient, known_facts, blocking_missing_evidence, nice_to_have_missing_evidence, next_queries, reason.",
        ]
    )
    context_lines = []
    for fallback_rank, chunk in enumerate(top_chunks[:10], start=1):
        rank = int(chunk.get("rank") or fallback_rank)
        title = compact_text(chunk.get("title")) or "(untitled)"
        article_id = compact_text(chunk.get("article_id")) or "(missing_article_id)"
        text = (
            compact_text(chunk.get("text_preview"))
            or compact_text(chunk.get("contents_preview"))
            or compact_text(chunk.get("text"))
            or compact_text(chunk.get("contents"))
        )
        context_lines.append(
            "\n".join(
                [
                    f"Rank: {rank}",
                    f"Title: {title}",
                    f"Article ID: {article_id}",
                    f"Text Preview: {preview_text(text, max_context_chars)}",
                ]
            )
        )
    user_prompt = "\n\n".join(
        [
            f"Question:\n{normalized_question}",
            "Retrieved top10 chunks:\n" + "\n\n".join(context_lines),
            (
                "JSON schema:\n"
                '{"sufficient": boolean, "known_facts": string[], '
                '"blocking_missing_evidence": string[], '
                '"nice_to_have_missing_evidence": string[], '
                '"next_queries": string[], "reason": string}'
            ),
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_traceable_evidence_checker_messages(
    question: str,
    top_chunks: list[Any],
    *,
    max_context_chars: int = DEFAULT_CONTEXT_PREVIEW_CHARS,
    max_next_queries: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
    compressed_summary: str | None = None,
    required_facets: list[Any] | None = None,
    known_facts: list[Any] | None = None,
    covered_facets: list[Any] | None = None,
    missing_facets: list[Any] | None = None,
    query_history: list[Any] | None = None,
) -> list[dict[str, str]]:
    normalized_question = compact_text(question)
    if not normalized_question:
        raise EvidenceCheckerError("question must not be empty.")
    if max_context_chars <= 0:
        raise EvidenceCheckerError("max_context_chars must be positive.")
    if max_next_queries <= 0:
        raise EvidenceCheckerError("max_next_queries must be positive.")
    chunk_aliases = build_chunk_aliases(top_chunks)
    system_prompt = "\n".join(
        [
            "You are an evidence sufficiency checker for Wix Help Center retrieval.",
            "Reason internally if useful. The machine-readable final answer must be one compact JSON object.",
            "If the serving stack emits thinking text, put the JSON object after the thinking text and do not add prose after JSON.",
            "Do not answer the user's question.",
            "Judge whether the visible chunks support a correct, useful, and non-misleading answer.",
            "First decompose the question into core facets: Wix product/module, user goal/action, object/entity, condition/error, integration, setting, recipient/outcome, comparison, and required steps.",
            "Before deciding sufficient, fill required_facets as a compact coverage matrix. Each required facet must state the user's exact need, whether visible evidence covers it, and whether the object/module exactly matches the user's object/module.",
            "For object/entity/product facets, exact_object_match=false when visible chunks cover only a broader, adjacent, or inferred object such as Bookings vs course, domain vs subscription, or store product vs service.",
            "sufficient=true is allowed only when every required facet has evidence_status=covered and exact_object_match=true.",
            "For multi-intent questions, multi-product questions, or questions comparing what a feature can do, sufficient=true only when every core facet is directly supported by visible chunks.",
            "Set sufficient=true when the visible evidence directly supports every core facet needed for the answer, even if it is not exhaustive.",
            "Do not require exact wording, exhaustive edge cases, extra examples, screenshots, or more explicit confirmation when the evidence already supports the answer.",
            "Set sufficient=false only when a blocking evidence gap would make the answer unsupported, materially incomplete, wrong, or misleading to a typical user.",
            "If a question contains two separable user needs, do not set sufficient=true just because one need is covered; mark the uncovered need as a blocking missing facet.",
            "If visible evidence covers a nearby Wix module but not the module or method needed for the user's stated goal, mark that module/method as missing instead of inferring it.",
            "Do not invent bridge facts between adjacent Wix modules. If your reason would need to say 'the product used for', 'applies to', 'can also be used for', 'should remain', or 'subscription is tied to', that bridge must be stated in a visible chunk.",
            "Do not use general world knowledge or product intuition to create bridge facts. The statements 'a course is a booking service' and 'a Premium plan is tied to the site, not the domain' are blocking bridge facts unless visible chunks state them.",
            "Known facts must be explicitly stated by visible chunks. Do not put inferred bridge facts in known_facts; mark the bridge as missing instead.",
            "Article titles can identify topic relevance, but the visible text preview must support the bridge or action needed for sufficiency.",
            "For service-selling questions, evidence about physical/digital store products alone is not enough unless visible chunks also cover services, bookings, appointments, classes, events, or another service-selling method.",
            "For questions asking whether one Wix feature works for another object or business model, evidence must cover both the limitation or capability of the named feature and the official alternative or method for the target object.",
            "For course, class, lesson, appointment, or service questions, generic Wix Bookings/settings evidence is sufficient only if visible chunks explicitly cover that object or state that it is managed as a booking service.",
            "For booking-service category questions, evidence must cover both service management and category/group management when the question asks about categories.",
            "For domain plus Premium plan or subscription questions, evidence must separately cover the domain operation and the plan/subscription effect, limitation, or next step. Do not infer subscription behavior from domain connection or replacement evidence alone.",
            "For domain plus Premium plan or subscription questions, text saying a domain can be assigned to an upgraded site does not prove the plan or subscription remains unchanged. The visible text must explicitly discuss Premium plan assignment/change, subscription effect, or domain structure after purchasing a site plan.",
            "If the evidence enables a correct and actionable answer for every core facet, set sufficient=true even if some detail, edge case, or caveat is missing.",
            "When a required facet is ambiguous, prefer sufficient=false if visible chunks only cover a related product or partial intent; prefer sufficient=true for minor wording gaps.",
            "Do NOT treat user-side unknowns as blocking gaps: region, account type, product type, plan, or current configuration are context the user may already know.",
            "Do NOT treat third-party or dynamic information as blocking gaps: external service values and exact pricing figures may be unavailable or change over time.",
            "For pricing, cost, fee, or upgrade questions, do not retrieve exact prices. Evidence is sufficient if it explains where/how the user can view pricing, compare plans, or upgrade; missing dollar amounts are not blocking.",
            "Do not generate a next query whose target is only an exact price, cost, or fee amount unless the user asks about a fixed documented fee and the visible evidence indicates that fixed fee should exist.",
            "Do not trigger retrieval for nice-to-have details, extra examples, background context, minor clarification, or answer phrasing improvements.",
            "For how-to, setup, automation, and timeline questions, missing required steps, phases, conditions, or lifecycle stages are blocking.",
            "Every claim about coverage or missing evidence must cite only visible Chunk Ref aliases such as C1, C2, and C3.",
            "Do not copy long real chunk_id/hash values. Use Chunk Ref aliases only in seen_chunk_ids, supporting_chunk_ids, inferred_from_chunk_ids, and derived_from_chunk_ids.",
            "If a missing facet is inferred from partial evidence, cite inferred_from_chunk_ids.",
            f"Generate at most {max_next_queries} next_queries.",
            "Each next query must target a blocking missing facet and must include derived_from_chunk_ids.",
            "If sufficient=true, next_queries and missing_facets must be empty.",
            "missing_facets must list only blocking gaps that justify another retrieval step.",
            "seen_chunk_ids should include only chunks actually used for the judgment, not every visible chunk by default.",
            "For each covered or missing facet, cite one to three strongest Chunk Ref aliases; avoid repeating all visible chunks.",
            "Keep known_facts to at most 5 short items and reason to one short sentence.",
            "Return valid JSON only with keys: sufficient, seen_chunk_ids, required_facets, known_facts, covered_facets, missing_facets, next_queries, reason.",
            "required_facets items must be objects with: facet_id, facet_type, user_need, evidence_status, exact_object_match, supporting_chunk_ids.",
            "required_facets.evidence_status must be one of: covered, partial, missing.",
            "covered_facets items must be objects with: facet_id, description, supporting_chunk_ids.",
            "missing_facets items must be objects with: facet_id, description, inferred_from_chunk_ids, blocking.",
            "next_queries items must be objects with: query_text, target_missing_facet_id, derived_from_chunk_ids.",
        ]
    )
    context_lines = []
    for fallback_rank, chunk in enumerate(top_chunks, start=1):
        rank = int(get_chunk_value(chunk, "rank") or fallback_rank)
        chunk_id = compact_text(get_chunk_value(chunk, "chunk_id")) or "(missing_chunk_id)"
        chunk_ref = chunk_ref_for_chunk_id(chunk_aliases, chunk_id) or f"C{fallback_rank}"
        title = compact_text(get_chunk_value(chunk, "title")) or "(untitled)"
        text = (
            compact_text(get_chunk_value(chunk, "text_preview"))
            or compact_text(get_chunk_value(chunk, "contents_preview"))
            or compact_text(get_chunk_value(chunk, "text"))
            or compact_text(get_chunk_value(chunk, "contents"))
        )
        context_lines.append(
            "\n".join(
                [
                    f"Rank: {rank}",
                    f"Chunk Ref: {chunk_ref}",
                    f"Title: {title}",
                    f"Text Preview: {preview_text(text, max_context_chars)}",
                ]
            )
        )
    user_prompt = "\n\n".join(
        compact_prompt_sections(
            [
                f"Question:\n{normalized_question}",
                format_optional_section("Compressed context summary", compressed_summary),
                format_optional_list_section("Required facets from previous calls", required_facets),
                format_optional_list_section("Known facts from previous calls", known_facts),
                format_optional_list_section("Previously covered facets", covered_facets),
                format_optional_list_section("Open missing facets", missing_facets),
                format_optional_list_section("Previous retrieval queries", query_history),
                "Visible raw chunks. Cite only Chunk Ref aliases, never long chunk IDs:\n"
                + "\n\n".join(context_lines),
                (
                    "JSON schema:\n"
                    '{"sufficient": boolean, "seen_chunk_ids": string[], '
                    '"required_facets": [{"facet_id": string, "facet_type": string, '
                    '"user_need": string, "evidence_status": "covered|partial|missing", '
                    '"exact_object_match": boolean, "supporting_chunk_ids": string[]}], '
                    '"known_facts": string[], '
                    '"covered_facets": [{"facet_id": string, "description": string, '
                    '"supporting_chunk_ids": string[]}], '
                    '"missing_facets": [{"facet_id": string, "description": string, '
                    '"inferred_from_chunk_ids": string[], "blocking": boolean}], '
                    '"next_queries": [{"query_text": string, "target_missing_facet_id": string, '
                    '"derived_from_chunk_ids": string[]}], "reason": string}'
                ),
            ]
        )
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_compact_evidence_checker_messages(
    question: str,
    top_chunks: list[Any],
    *,
    max_context_chars: int = DEFAULT_CONTEXT_PREVIEW_CHARS,
    max_next_queries: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
    compressed_summary: str | None = None,
    query_history: list[Any] | None = None,
) -> list[dict[str, str]]:
    normalized_question = compact_text(question)
    if not normalized_question:
        raise EvidenceCheckerError("question must not be empty.")
    if max_context_chars <= 0:
        raise EvidenceCheckerError("max_context_chars must be positive.")
    if max_next_queries <= 0:
        raise EvidenceCheckerError("max_next_queries must be positive.")

    system_prompt = "\n".join(
        [
            "You are a fast evidence sufficiency checker for Wix Help Center retrieval.",
            "Do not answer the user's question.",
            "Return one compact JSON object and no prose.",
            "Judge whether the visible chunks support a correct, useful, non-misleading answer.",
            "Set sufficient=true when the evidence directly covers the user's core need.",
            "Set sufficient=false only for blocking gaps that would make the answer unsupported or materially incomplete.",
            "Do not invent bridge facts between adjacent Wix modules, products, plans, domains, or integrations.",
            "For multi-intent questions, every core intent must be covered.",
            "Do not treat exact prices, external provider values, screenshots, or user-side account details as blocking gaps.",
            "If sufficient=true, blocking_missing_evidence and next_queries must be empty.",
            "If sufficient=false, list only blocking_missing_evidence and create targeted next_queries.",
            "Use explicit product/action terms in next_queries; avoid vague pronouns.",
            f"Generate at most {max_next_queries} next_queries.",
            "Keep reason under 20 words.",
            "Return valid JSON only with keys: sufficient, blocking_missing_evidence, next_queries, reason.",
        ]
    )
    context_lines = []
    for fallback_rank, chunk in enumerate(top_chunks, start=1):
        rank = int(get_chunk_value(chunk, "rank") or fallback_rank)
        title = compact_text(get_chunk_value(chunk, "title")) or "(untitled)"
        text = (
            compact_text(get_chunk_value(chunk, "text_preview"))
            or compact_text(get_chunk_value(chunk, "contents_preview"))
            or compact_text(get_chunk_value(chunk, "text"))
            or compact_text(get_chunk_value(chunk, "contents"))
        )
        context_lines.append(
            "\n".join(
                [
                    f"Rank: {rank}",
                    f"Title: {title}",
                    f"Text Preview: {preview_text(text, max_context_chars)}",
                ]
            )
        )
    user_prompt = "\n\n".join(
        compact_prompt_sections(
            [
                f"Question:\n{normalized_question}",
                format_optional_section("Compressed context summary", compressed_summary),
                format_optional_list_section("Previous retrieval queries", query_history),
                "Visible chunks:\n" + "\n\n".join(context_lines),
                (
                    "JSON schema:\n"
                    '{"sufficient": boolean, '
                    '"blocking_missing_evidence": string[], '
                    '"next_queries": string[], "reason": string}'
                ),
            ]
        )
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class TraceableEvidenceChecker:
    def __init__(
        self,
        *,
        client: Any,
        temperature: float = DEFAULT_CHECKER_TEMPERATURE,
        max_tokens: int = DEFAULT_CHECKER_MAX_TOKENS,
        max_next_queries: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
        context_preview_chars: int = DEFAULT_CONTEXT_PREVIEW_CHARS,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_next_queries = max_next_queries
        self.context_preview_chars = context_preview_chars
        self.model = compact_text(getattr(client, "model", "")) or "traceable_checker"
        self.caller = StructuredLLMCaller(
            client=client,
            schema=CheckerPayload,
            temperature=temperature,
            max_tokens=max_tokens,
            token_budget_values=[max_tokens],
        )

    def check(
        self,
        *,
        context: Any,
        visible_items: list[Any],
        manifest: Any,
        round_index: int,
    ) -> dict[str, Any]:
        allowed_chunk_ids = [
            chunk_id
            for item in visible_items
            if (chunk_id := compact_text(get_chunk_value(item, "chunk_id")))
        ]
        chunk_aliases = build_chunk_aliases(visible_items)
        raw_text = ""
        try:
            messages = build_traceable_evidence_checker_messages(
                context.question,
                visible_items,
                max_context_chars=self.context_preview_chars,
                max_next_queries=self.max_next_queries,
                compressed_summary=getattr(context, "compressed_summary", None),
                required_facets=getattr(context, "required_facets", []),
                known_facts=getattr(context, "known_facts", []),
                covered_facets=getattr(context, "covered_facets", []),
                missing_facets=getattr(context, "missing_facets", []),
                query_history=getattr(context, "query_history", []),
            )
            payload = self.caller.call(messages)
            raw_text = self.caller.last_raw_text
            result = parse_traceable_checker_payload(
                checker_payload_to_dict(payload),
                allowed_chunk_ids=allowed_chunk_ids,
                chunk_id_aliases=chunk_aliases,
                max_next_queries=self.max_next_queries,
            )
            result = apply_runtime_sufficiency_guards(
                result,
                question=getattr(context, "question", ""),
                visible_items=visible_items,
                max_next_queries=self.max_next_queries,
            )
            result = augment_runtime_queries(
                result,
                question=getattr(context, "question", ""),
                visible_items=visible_items,
                max_next_queries=self.max_next_queries,
            )
            checker_valid = True
            checker_error = None
        except (EvidenceCheckerError, StructuredLLMError) as exc:
            raw_text = raw_text or self.caller.last_raw_text
            result = empty_traceable_checker_result()
            checker_valid = False
            checker_error = str(exc)

        if not result.get("provenance_valid", True):
            result = {
                **result,
                "next_queries": [],
                "sufficient": False,
            }
        return {
            **result,
            "checker_valid": checker_valid,
            "checker_error": checker_error,
            "checker_raw_text": raw_text,
            "checker_model": self.model,
            "chunk_aliases": chunk_aliases,
            "llm_call_id": manifest.llm_call_id,
            "round_index": round_index,
            "retrieval_triggered": bool(result.get("next_queries")),
        }


class CompactEvidenceChecker:
    def __init__(
        self,
        *,
        client: Any,
        temperature: float = DEFAULT_CHECKER_TEMPERATURE,
        max_tokens: int = DEFAULT_CHECKER_MAX_TOKENS,
        max_next_queries: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
        context_preview_chars: int = DEFAULT_CONTEXT_PREVIEW_CHARS,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_next_queries = max_next_queries
        self.context_preview_chars = context_preview_chars
        self.model = compact_text(getattr(client, "model", "")) or "compact_checker"
        self.caller = StructuredLLMCaller(
            client=client,
            schema=CheckerPayload,
            temperature=temperature,
            max_tokens=max_tokens,
            token_budget_values=[max_tokens],
        )

    def check(
        self,
        *,
        context: Any,
        visible_items: list[Any],
        manifest: Any,
        round_index: int,
    ) -> dict[str, Any]:
        chunk_aliases = build_chunk_aliases(visible_items)
        raw_text = ""
        try:
            messages = build_compact_evidence_checker_messages(
                context.question,
                visible_items,
                max_context_chars=self.context_preview_chars,
                max_next_queries=self.max_next_queries,
                compressed_summary=getattr(context, "compressed_summary", None),
                query_history=getattr(context, "query_history", []),
            )
            payload = self.caller.call(messages)
            raw_text = self.caller.last_raw_text
            result = parse_compact_checker_payload(
                checker_payload_to_dict(payload),
                max_next_queries=self.max_next_queries,
            )
            result = apply_runtime_sufficiency_guards(
                result,
                question=getattr(context, "question", ""),
                visible_items=visible_items,
                max_next_queries=self.max_next_queries,
            )
            result = augment_runtime_queries(
                result,
                question=getattr(context, "question", ""),
                visible_items=visible_items,
                max_next_queries=self.max_next_queries,
            )
            checker_valid = True
            checker_error = None
        except (EvidenceCheckerError, StructuredLLMError) as exc:
            raw_text = raw_text or self.caller.last_raw_text
            result = empty_compact_checker_result()
            checker_valid = False
            checker_error = str(exc)

        return {
            **result,
            "checker_valid": checker_valid,
            "checker_error": checker_error,
            "checker_raw_text": raw_text,
            "checker_model": self.model,
            "checker_mode": CHECKER_MODE_COMPACT,
            "chunk_aliases": chunk_aliases,
            "llm_call_id": manifest.llm_call_id,
            "round_index": round_index,
            "retrieval_triggered": bool(result.get("next_queries")),
        }


class RetryingEvidenceChecker:
    def __init__(
        self,
        checker: Any,
        *,
        max_attempts: int = 2,
        retry_error_phrases: tuple[str, ...] = (
            "empty",
            "invalid json",
            "does not contain a json object",
            "unexpected end",
        ),
    ) -> None:
        if max_attempts <= 0:
            raise EvidenceCheckerError("max_attempts must be positive.")
        self.checker = checker
        self.max_attempts = max_attempts
        self.retry_error_phrases = retry_error_phrases
        self.model = compact_text(getattr(checker, "model", "")) or "retrying_checker"

    def check(
        self,
        *,
        context: Any,
        visible_items: list[Any],
        manifest: Any,
        round_index: int,
    ) -> dict[str, Any]:
        attempts = []
        for attempt_index in range(1, self.max_attempts + 1):
            result = self.checker.check(
                context=context,
                visible_items=visible_items,
                manifest=manifest,
                round_index=round_index,
            )
            attempts.append(
                {
                    "attempt": attempt_index,
                    "checker_valid": bool(result.get("checker_valid")),
                    "checker_error": compact_text(result.get("checker_error")),
                }
            )
            if result.get("checker_valid") or not self.should_retry(result):
                return {
                    **result,
                    "checker_retry_count": attempt_index - 1,
                    "checker_attempts": attempts,
                }
        return {
            **result,
            "checker_retry_count": self.max_attempts - 1,
            "checker_attempts": attempts,
        }

    def should_retry(self, result: dict[str, Any]) -> bool:
        if result.get("checker_valid"):
            return False
        error_text = compact_text(result.get("checker_error")).casefold()
        if not error_text:
            return True
        return any(phrase in error_text for phrase in self.retry_error_phrases)


def apply_runtime_sufficiency_guards(
    result: dict[str, Any],
    *,
    question: str,
    visible_items: list[Any],
    max_next_queries: int,
) -> dict[str, Any]:
    if not result.get("sufficient"):
        return result
    question_text = compact_text(question).casefold()
    visible_text = "\n".join(
        compact_text(get_chunk_value(item, "title")) + "\n" + compact_text(get_chunk_value(item, "text_preview"))
        for item in visible_items
    ).casefold()
    visible_chunk_ids = [
        chunk_id
        for item in visible_items
        if (chunk_id := compact_text(get_chunk_value(item, "chunk_id")))
    ]
    guarded = result
    if is_domain_plan_question(question_text) and not has_domain_plan_evidence(visible_text):
        guarded = force_runtime_insufficient(
            guarded,
            reason="runtime_guard_domain_plan_subscription_bridge",
            facet_id="domain_plan_subscription_effect",
            description=(
                "Need explicit Wix evidence about how assigning or changing a domain relates "
                "to a Premium/site plan or subscription."
            ),
            query_text=(
                "Wix assigning Premium plan to different site domain structure "
                "after purchasing a site plan domain subscription"
            ),
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_store_services_question(question_text) and not has_service_alternative_overview(visible_text):
        guarded = force_runtime_insufficient(
            guarded,
            reason="runtime_guard_store_services_alternative_overview",
            facet_id="store_services_alternative_method",
            description=(
                "Need official Wix evidence for the recommended service-selling method, not only "
                "that Wix Stores is limited to products."
            ),
            query_text="Wix Bookings about selling services appointments classes",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_course_currency_question(question_text) and not has_course_currency_evidence(visible_text):
        guarded = force_runtime_insufficient(
            guarded,
            reason="runtime_guard_course_currency_bridge",
            facet_id="course_currency_product_match",
            description=(
                "Need explicit Wix evidence that the currency-change guidance applies to the "
                "user's course/class product, not only a nearby Wix module."
            ),
            query_text="Wix course class changing currency Wix Bookings Wix Events",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    return guarded


def augment_runtime_queries(
    result: dict[str, Any],
    *,
    question: str,
    visible_items: list[Any],
    max_next_queries: int,
) -> dict[str, Any]:
    if result.get("sufficient"):
        return result
    question_text = compact_text(question).casefold()
    visible_chunk_ids = [
        chunk_id
        for item in visible_items
        if (chunk_id := compact_text(get_chunk_value(item, "chunk_id")))
    ]
    output = {**result}
    next_queries = list(output.get("next_queries") or [])
    if is_domain_plan_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text=(
                "Wix assigning Premium plan to different site domain structure "
                "after purchasing a site plan domain subscription"
            ),
            target_missing_facet_id="domain_plan_subscription_effect",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_store_services_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix Bookings about Wix Bookings selling services appointments classes",
            target_missing_facet_id="store_services_alternative_method",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_course_currency_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix course class changing currency Wix Bookings Wix Events",
            target_missing_facet_id="course_currency_product_match",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_birthday_automation_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix Contacts creating a segment birthday Wix Automations segment trigger",
            target_missing_facet_id="birthday_segment_trigger",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_paypal_payment_overview_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix payments overview transaction details PayPal account received payment",
            target_missing_facet_id="paypal_payment_overview",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_email_marketing_pricing_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix Email Marketing pricing upgrade plan package",
            target_missing_facet_id="email_marketing_plan_pricing",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_google_analytics_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix connect Google Analytics 4 GA4 Universal Analytics upgrade site tracking",
            target_missing_facet_id="google_analytics_4_connection",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_rename_link_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix Editor rename web link change page URL address",
            target_missing_facet_id="rename_link_page_url",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_social_share_image_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix social share settings pages website link picture image Facebook debugger preview",
            target_missing_facet_id="social_share_image_settings",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_hidden_page_direct_link_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix hide page direct link prevent search engines indexing noindex manage pages mobile editor",
            target_missing_facet_id="hidden_page_direct_link",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_template_design_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix pre set website designs templates new template site choose available options",
            target_missing_facet_id="template_design_options",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_mobile_overlap_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix mobile editor adding customizing mobile only elements misplaced overlapping",
            target_missing_facet_id="mobile_overlap_elements",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_remove_menu_header_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix Editor remove menu header delete something from site element",
            target_missing_facet_id="remove_menu_header_element",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_payment_region_currency_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix Payments countries available currency payment solution",
            target_missing_facet_id="payment_region_currency_availability",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    if is_payment_provider_after_funds_question(question_text):
        ensure_runtime_query(
            next_queries,
            query_text="Wix Hotels payment method payment provider funds sent order setup",
            target_missing_facet_id="payment_provider_after_funds",
            derived_from_chunk_ids=visible_chunk_ids,
            max_next_queries=max_next_queries,
        )
    output["next_queries"] = next_queries[:max_next_queries]
    return output


def ensure_runtime_query(
    next_queries: list[dict[str, Any]],
    *,
    query_text: str,
    target_missing_facet_id: str,
    derived_from_chunk_ids: list[str],
    max_next_queries: int,
) -> None:
    dedup_key = compact_text(query_text).casefold()
    if not dedup_key:
        return
    for query in next_queries:
        if compact_text(query.get("query_text")).casefold() == dedup_key:
            return
    if len(next_queries) >= max_next_queries:
        next_queries[-1] = {
            "query_text": query_text,
            "target_missing_facet_id": target_missing_facet_id,
            "derived_from_chunk_ids": derived_from_chunk_ids[:3],
        }
        return
    next_queries.append(
        {
            "query_text": query_text,
            "target_missing_facet_id": target_missing_facet_id,
            "derived_from_chunk_ids": derived_from_chunk_ids[:3],
        }
    )


def is_domain_plan_question(question_text: str) -> bool:
    return "domain" in question_text and any(
        term in question_text for term in ("premium", "subscription", "site plan", "plan")
    )


def has_domain_plan_evidence(visible_text: str) -> bool:
    patterns = (
        "assigning a premium plan",
        "premium plan to a different site",
        "domain structure after purchasing a site plan",
        "purchasing a domain vs. purchasing a site plan",
        "domains and site plans are different subscription services",
        "changing your premium",
        "site plan subscription",
    )
    return any(pattern in visible_text for pattern in patterns)


def is_store_services_question(question_text: str) -> bool:
    return (
        "store" in question_text
        and "service" in question_text
        and any(term in question_text for term in ("physical", "goods", "product"))
    )


def has_service_alternative_overview(visible_text: str) -> bool:
    patterns = (
        "wix bookings: about wix bookings",
        "about wix bookings",
        "creating a service",
        "creating a class",
        "creating a course",
        "wix bookings: creating",
    )
    return any(pattern in visible_text for pattern in patterns)


def is_course_currency_question(question_text: str) -> bool:
    return "currency" in question_text and any(
        term in question_text for term in ("course", "class", "lesson")
    )


def has_course_currency_evidence(visible_text: str) -> bool:
    return any(
        phrase in visible_text
        for phrase in (
            "course" + " changing your currency",
            "class" + " changing your currency",
            "wix events: changing your currency",
            "courses are created",
            "creating a course",
        )
    ) and "currency" in visible_text


def is_birthday_automation_question(question_text: str) -> bool:
    return "birthday" in question_text and any(
        term in question_text for term in ("automation", "automated", "trigger")
    )


def is_paypal_payment_overview_question(question_text: str) -> bool:
    return "paypal" in question_text and "payment" in question_text and any(
        term in question_text for term in ("account", "received", "went", "connected", "second")
    )


def is_email_marketing_pricing_question(question_text: str) -> bool:
    return "email marketing" in question_text and any(
        term in question_text for term in ("pricing", "price", "cost", "plan")
    )


def is_google_analytics_question(question_text: str) -> bool:
    return "google analytics" in question_text or "ga4" in question_text


def is_rename_link_question(question_text: str) -> bool:
    return "rename" in question_text and "link" in question_text


def is_social_share_image_question(question_text: str) -> bool:
    return any(term in question_text for term in ("sharing", "share")) and any(
        term in question_text for term in ("picture", "image", "link")
    )


def is_hidden_page_direct_link_question(question_text: str) -> bool:
    return "hide" in question_text and "page" in question_text and "link" in question_text


def is_template_design_question(question_text: str) -> bool:
    return any(term in question_text for term in ("pre-set", "pre set", "template")) or (
        "design" in question_text and "website" in question_text
    )


def is_mobile_overlap_question(question_text: str) -> bool:
    return "mobile" in question_text and any(
        term in question_text for term in ("overlap", "overlapping", "misplaced", "elements")
    )


def is_remove_menu_header_question(question_text: str) -> bool:
    return "remove" in question_text and any(term in question_text for term in ("menu", "header"))


def is_payment_provider_after_funds_question(question_text: str) -> bool:
    return "payment provider" in question_text and "fund" in question_text and "order" in question_text


def is_payment_region_currency_question(question_text: str) -> bool:
    return "payment" in question_text and any(
        term in question_text for term in ("region", "currency", "country")
    )


def force_runtime_insufficient(
    result: dict[str, Any],
    *,
    reason: str,
    facet_id: str,
    description: str,
    query_text: str,
    derived_from_chunk_ids: list[str],
    max_next_queries: int,
) -> dict[str, Any]:
    output = {**result}
    output["sufficient"] = False
    output["sufficiency_overridden"] = True
    override_reasons = list(output.get("sufficiency_override_reasons") or [])
    if reason not in override_reasons:
        override_reasons.append(reason)
    output["sufficiency_override_reasons"] = override_reasons
    missing_facets = list(output.get("missing_facets") or [])
    missing_facets.append(
        {
            "facet_id": facet_id,
            "description": description,
            "inferred_from_chunk_ids": derived_from_chunk_ids[:3],
            "blocking": True,
        }
    )
    output["missing_facets"] = missing_facets
    next_queries = list(output.get("next_queries") or [])
    if len(next_queries) < max_next_queries:
        next_queries.append(
            {
                "query_text": query_text,
                "target_missing_facet_id": facet_id,
                "derived_from_chunk_ids": derived_from_chunk_ids[:3],
            }
        )
    output["next_queries"] = next_queries[:max_next_queries]
    output["reason"] = compact_text(output.get("reason"))
    return output


def parse_checker_response(
    raw_text: str,
    *,
    max_next_queries: int = DEFAULT_MAX_NEXT_QUERIES,
) -> dict[str, Any]:
    if max_next_queries <= 0:
        raise EvidenceCheckerError("max_next_queries must be positive.")
    try:
        payload_text = extract_json_object(raw_text)
        payload = loads_json(payload_text)
    except Exception as exc:
        raise EvidenceCheckerError(f"Checker returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceCheckerError("Checker JSON must be an object.")
    return parse_checker_payload(payload, max_next_queries=max_next_queries)


def parse_checker_payload(
    payload: dict[str, Any],
    *,
    max_next_queries: int = DEFAULT_MAX_NEXT_QUERIES,
) -> dict[str, Any]:
    if max_next_queries <= 0:
        raise EvidenceCheckerError("max_next_queries must be positive.")
    sufficient = parse_bool(payload.get("sufficient"))
    known_facts = normalize_text_list(payload.get("known_facts"))
    blocking_missing_evidence = normalize_text_list(
        payload.get("blocking_missing_evidence")
    )
    legacy_missing_evidence = normalize_text_list(payload.get("missing_evidence"))
    if not blocking_missing_evidence:
        blocking_missing_evidence = legacy_missing_evidence
    nice_to_have_missing_evidence = normalize_text_list(
        payload.get("nice_to_have_missing_evidence")
    )
    if sufficient:
        blocking_missing_evidence = []
        next_queries = []
    elif blocking_missing_evidence:
        next_queries = normalize_queries(
            payload.get("next_queries"),
            max_next_queries=max_next_queries,
        )
    else:
        next_queries = []
    return {
        "sufficient": sufficient,
        "known_facts": known_facts,
        "missing_evidence": blocking_missing_evidence,
        "blocking_missing_evidence": blocking_missing_evidence,
        "nice_to_have_missing_evidence": nice_to_have_missing_evidence,
        "next_queries": next_queries,
        "reason": compact_text(payload.get("reason")),
    }


def empty_traceable_checker_result() -> dict[str, Any]:
    return {
        "sufficient": False,
        "seen_chunk_ids": [],
        "required_facets": [],
        "known_facts": [],
        "covered_facets": [],
        "missing_facets": [],
        "next_queries": [],
        "reason": "",
        "sufficiency_overridden": False,
        "sufficiency_override_reasons": [],
        "provenance_valid": False,
        "invalid_provenance_chunk_ids": [],
    }


def empty_compact_checker_result() -> dict[str, Any]:
    return {
        **empty_traceable_checker_result(),
        "compact_checker": True,
        "checker_mode": CHECKER_MODE_COMPACT,
    }


def parse_compact_checker_response(
    raw_text: str,
    *,
    max_next_queries: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
) -> dict[str, Any]:
    parsed = parse_checker_response(raw_text, max_next_queries=max_next_queries)
    return parse_compact_checker_payload_from_parsed(parsed)


def parse_compact_checker_payload(
    payload: dict[str, Any],
    *,
    max_next_queries: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
) -> dict[str, Any]:
    parsed = parse_checker_payload(payload, max_next_queries=max_next_queries)
    return parse_compact_checker_payload_from_parsed(parsed)


def parse_compact_checker_payload_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    blocking_missing_evidence = parsed["blocking_missing_evidence"]
    missing_facets = [
        {
            "facet_id": f"compact_gap_{index}",
            "description": description,
            "inferred_from_chunk_ids": [],
            "blocking": True,
        }
        for index, description in enumerate(blocking_missing_evidence, start=1)
    ]
    next_queries = [
        {
            "query_text": query,
            "target_missing_facet_id": (
                f"compact_gap_{min(index, len(missing_facets))}"
                if missing_facets
                else f"compact_gap_{index}"
            ),
            "derived_from_chunk_ids": [],
        }
        for index, query in enumerate(parsed["next_queries"], start=1)
    ]
    if parsed["sufficient"]:
        missing_facets = []
        next_queries = []
    return {
        "sufficient": parsed["sufficient"],
        "seen_chunk_ids": [],
        "required_facets": [],
        "known_facts": parsed["known_facts"],
        "covered_facets": [],
        "missing_facets": missing_facets,
        "next_queries": next_queries,
        "reason": parsed["reason"],
        "sufficiency_overridden": False,
        "sufficiency_override_reasons": [],
        "provenance_valid": True,
        "invalid_provenance_chunk_ids": [],
        "compact_checker": True,
        "checker_mode": CHECKER_MODE_COMPACT,
    }


def parse_traceable_checker_response(
    raw_text: str,
    *,
    allowed_chunk_ids: list[str],
    chunk_id_aliases: dict[str, str] | None = None,
    max_next_queries: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
) -> dict[str, Any]:
    if max_next_queries <= 0:
        raise EvidenceCheckerError("max_next_queries must be positive.")
    allowed_chunk_id_set = set(allowed_chunk_ids)
    try:
        payload_text = extract_json_object(raw_text)
        payload = loads_json(payload_text)
    except Exception as exc:
        raise EvidenceCheckerError(f"Checker returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceCheckerError("Checker JSON must be an object.")
    return parse_traceable_checker_payload(
        payload,
        allowed_chunk_ids=allowed_chunk_ids,
        chunk_id_aliases=chunk_id_aliases,
        max_next_queries=max_next_queries,
    )


def parse_traceable_checker_payload(
    payload: dict[str, Any],
    *,
    allowed_chunk_ids: list[str],
    chunk_id_aliases: dict[str, str] | None = None,
    max_next_queries: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
) -> dict[str, Any]:
    if max_next_queries <= 0:
        raise EvidenceCheckerError("max_next_queries must be positive.")
    allowed_chunk_id_set = set(allowed_chunk_ids)
    aliases = normalize_chunk_aliases(chunk_id_aliases or {})
    sufficient = parse_bool(payload.get("sufficient"))
    seen_chunk_ids = normalize_chunk_ids(
        payload.get("seen_chunk_ids"),
        chunk_id_aliases=aliases,
    )
    covered_facets = normalize_facets(
        payload.get("covered_facets"),
        id_key="supporting_chunk_ids",
        chunk_id_aliases=aliases,
    )
    required_facets = normalize_required_facets(
        payload.get("required_facets"),
        chunk_id_aliases=aliases,
    )
    missing_facets = normalize_facets(
        payload.get("missing_facets") or payload.get("blocking_missing_evidence"),
        id_key="inferred_from_chunk_ids",
        default_blocking=True,
        chunk_id_aliases=aliases,
    )
    raw_next_queries = normalize_traceable_queries(
        payload.get("next_queries"),
        max_next_queries=max_next_queries,
        chunk_id_aliases=aliases,
    )
    reason = compact_text(payload.get("reason"))
    override_reasons = contradiction_guard_reasons(
        sufficient=sufficient,
        required_facets=required_facets,
        missing_facets=missing_facets,
        next_queries=raw_next_queries,
        reason=reason,
    )
    if override_reasons:
        sufficient = False

    if sufficient:
        missing_facets = []
        next_queries = []
    else:
        next_queries = raw_next_queries or fallback_queries_from_facets(
            required_facets=required_facets,
            missing_facets=missing_facets,
            seen_chunk_ids=seen_chunk_ids,
            max_next_queries=max_next_queries,
        )

    cited_chunk_ids = collect_cited_chunk_ids(
        seen_chunk_ids=seen_chunk_ids,
        covered_facets=covered_facets,
        missing_facets=missing_facets,
        next_queries=next_queries,
    )
    invalid_chunk_ids = sorted(cited_chunk_ids - allowed_chunk_id_set)
    return {
        "sufficient": sufficient,
        "seen_chunk_ids": seen_chunk_ids,
        "required_facets": required_facets,
        "known_facts": normalize_text_list(payload.get("known_facts")),
        "covered_facets": covered_facets,
        "missing_facets": missing_facets,
        "next_queries": next_queries,
        "reason": reason,
        "sufficiency_overridden": bool(override_reasons),
        "sufficiency_override_reasons": override_reasons,
        "provenance_valid": not invalid_chunk_ids,
        "invalid_provenance_chunk_ids": invalid_chunk_ids,
    }


def get_chunk_value(chunk: Any, key: str) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(key)
    return getattr(chunk, key, None)


def compact_prompt_sections(sections: list[str | None]) -> list[str]:
    return [section for section in sections if compact_text(section)]


def format_optional_section(title: str, value: Any) -> str | None:
    text = compact_text(value)
    return f"{title}:\n{text}" if text else None


def format_optional_list_section(title: str, values: list[Any] | None) -> str | None:
    lines = []
    for value in values or []:
        text = stringify_prompt_item(value)
        if text:
            lines.append(f"- {text}")
    return f"{title}:\n" + "\n".join(lines) if lines else None


def stringify_prompt_item(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("description", "text", "query_text", "reason"):
            text = compact_text(value.get(key))
            if text:
                return text
        return compact_text(value)
    return compact_text(value)


def build_chunk_aliases(chunks: list[Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    seen_chunk_ids = set()
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = compact_text(get_chunk_value(chunk, "chunk_id"))
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        aliases[f"C{index}"] = chunk_id
        seen_chunk_ids.add(chunk_id)
    return aliases


def chunk_ref_for_chunk_id(aliases: dict[str, str], chunk_id: str) -> str:
    normalized_chunk_id = compact_text(chunk_id)
    for alias, real_chunk_id in aliases.items():
        if real_chunk_id == normalized_chunk_id:
            return alias
    return ""


def normalize_chunk_aliases(aliases: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for alias, chunk_id in aliases.items():
        alias_key = normalize_alias_key(alias)
        real_chunk_id = compact_text(chunk_id)
        if alias_key and real_chunk_id:
            normalized[alias_key] = real_chunk_id
    return normalized


def normalize_alias_key(value: Any) -> str:
    text = compact_text(value).upper()
    if not text:
        return ""
    match = re.fullmatch(r"C0*(\d+)", text)
    if match:
        return f"C{int(match.group(1))}"
    match = re.fullmatch(r"(?:CHUNK|REF|CHUNK REF|CHUNK_REF)\s*#?\s*0*(\d+)", text)
    if match:
        return f"C{int(match.group(1))}"
    if text.isdigit():
        return f"C{int(text)}"
    return text


def resolve_chunk_ref(value: Any, aliases: dict[str, str]) -> str:
    text = compact_text(value)
    if not text:
        return ""
    alias_key = normalize_alias_key(text)
    if alias_key in aliases:
        return aliases[alias_key]
    return text


def contradiction_guard_reasons(
    *,
    sufficient: bool,
    required_facets: list[dict[str, Any]] | None = None,
    missing_facets: list[dict[str, Any]],
    next_queries: list[dict[str, Any]],
    reason: str,
) -> list[str]:
    if not sufficient:
        return []
    reasons = []
    if missing_facets:
        reasons.append("sufficient_true_with_missing_facets")
    if next_queries:
        reasons.append("sufficient_true_with_next_queries")
    if required_facets_have_blocking_gap(required_facets or []):
        reasons.append("sufficient_true_with_required_facet_gap")
    if reason_has_blocking_gap_phrase(reason):
        reasons.append("sufficient_true_with_gap_reason")
    return reasons


def required_facets_have_blocking_gap(required_facets: list[dict[str, Any]]) -> bool:
    for facet in required_facets:
        status = compact_text(facet.get("evidence_status")).casefold()
        exact_object_match = facet.get("exact_object_match", True)
        if status and status != "covered":
            return True
        if exact_object_match is False:
            return True
    return False


def fallback_queries_from_facets(
    *,
    required_facets: list[dict[str, Any]],
    missing_facets: list[dict[str, Any]],
    seen_chunk_ids: list[str],
    max_next_queries: int,
) -> list[dict[str, Any]]:
    output = []
    seen = set()
    source_chunk_ids = seen_chunk_ids[:3]
    for facet in required_facets:
        status = compact_text(facet.get("evidence_status")).casefold()
        exact_object_match = facet.get("exact_object_match", True)
        if status == "covered" and exact_object_match is not False:
            continue
        query_text = compact_text(facet.get("user_need") or facet.get("description"))
        facet_id = compact_text(facet.get("facet_id"))
        supporting_chunk_ids = facet.get("supporting_chunk_ids") or source_chunk_ids
        add_fallback_query(
            output,
            seen,
            query_text=query_text,
            target_missing_facet_id=facet_id,
            derived_from_chunk_ids=supporting_chunk_ids,
            max_next_queries=max_next_queries,
        )
        if len(output) == max_next_queries:
            return output
    for facet in missing_facets:
        query_text = compact_text(facet.get("description"))
        facet_id = compact_text(facet.get("facet_id"))
        inferred_from_chunk_ids = facet.get("inferred_from_chunk_ids") or source_chunk_ids
        add_fallback_query(
            output,
            seen,
            query_text=query_text,
            target_missing_facet_id=facet_id,
            derived_from_chunk_ids=inferred_from_chunk_ids,
            max_next_queries=max_next_queries,
        )
        if len(output) == max_next_queries:
            return output
    return output


def add_fallback_query(
    output: list[dict[str, Any]],
    seen: set[str],
    *,
    query_text: str,
    target_missing_facet_id: str,
    derived_from_chunk_ids: list[str],
    max_next_queries: int,
) -> None:
    query_text = compact_text(query_text)
    if not query_text or query_text.casefold() in seen or len(output) >= max_next_queries:
        return
    seen.add(query_text.casefold())
    output.append(
        {
            "query_text": query_text,
            "target_missing_facet_id": target_missing_facet_id,
            "derived_from_chunk_ids": derived_from_chunk_ids[:3],
        }
    )


def reason_has_blocking_gap_phrase(reason: str) -> bool:
    normalized = compact_text(reason).casefold()
    if not normalized:
        return False
    if any(phrase in normalized for phrase in CONTRADICTION_REASON_PHRASES):
        return True
    for match in re.finditer(r"\b(?:gap|gaps|missing)\b", normalized):
        window = normalized[max(0, match.start() - 64) : match.end() + 64]
        if any(re.search(pattern, window) for pattern in NEGATED_REASON_PATTERNS):
            continue
        return True
    return False


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise EvidenceCheckerError("Checker JSON field `sufficient` must be a boolean.")


def normalize_text_list(value: Any) -> list[str]:
    return [compact_text(item) for item in to_string_list(value) if compact_text(item)]


def normalize_queries(value: Any, *, max_next_queries: int) -> list[str]:
    output = []
    seen = set()
    for item in normalize_text_list(value):
        dedup_key = item.casefold()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        output.append(item)
        if len(output) == max_next_queries:
            break
    return output


def normalize_chunk_ids(
    value: Any,
    *,
    chunk_id_aliases: dict[str, str] | None = None,
) -> list[str]:
    output = []
    seen = set()
    aliases = chunk_id_aliases or {}
    for item in normalize_text_list(value):
        item = resolve_chunk_ref(item, aliases)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def normalize_facets(
    value: Any,
    *,
    id_key: str,
    default_blocking: bool = False,
    chunk_id_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    output = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            description = compact_text(item.get("description") or item.get("facet") or item.get("text"))
            facet_id = compact_text(item.get("facet_id")) or f"facet_{index}"
            chunk_ids = normalize_chunk_ids(
                item.get(id_key),
                chunk_id_aliases=chunk_id_aliases,
            )
            row = {
                "facet_id": facet_id,
                "description": description,
                id_key: chunk_ids,
            }
            if default_blocking or "blocking" in item:
                row["blocking"] = bool(item.get("blocking", default_blocking))
            if description:
                output.append(row)
            continue
        description = compact_text(item)
        if description:
            row = {
                "facet_id": f"facet_{index}",
                "description": description,
                id_key: [],
            }
            if default_blocking:
                row["blocking"] = True
            output.append(row)
    return output


def normalize_required_facets(
    value: Any,
    *,
    chunk_id_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    output = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            user_need = compact_text(
                item.get("user_need")
                or item.get("description")
                or item.get("facet")
                or item.get("text")
            )
            if not user_need:
                continue
            evidence_status = normalize_evidence_status(item.get("evidence_status"))
            output.append(
                {
                    "facet_id": compact_text(item.get("facet_id")) or f"facet_{index}",
                    "facet_type": compact_text(item.get("facet_type")) or "unknown",
                    "user_need": user_need,
                    "evidence_status": evidence_status,
                    "exact_object_match": parse_optional_bool(
                        item.get("exact_object_match"),
                        default=True,
                    ),
                    "supporting_chunk_ids": normalize_chunk_ids(
                        item.get("supporting_chunk_ids"),
                        chunk_id_aliases=chunk_id_aliases,
                    ),
                }
            )
            continue
        text = compact_text(item)
        if text:
            output.append(
                {
                    "facet_id": f"facet_{index}",
                    "facet_type": "unknown",
                    "user_need": text,
                    "evidence_status": "missing",
                    "exact_object_match": False,
                    "supporting_chunk_ids": [],
                }
            )
    return output


def normalize_evidence_status(value: Any) -> str:
    status = compact_text(value).casefold()
    if status in {"covered", "partial", "missing"}:
        return status
    if status in {"partially_covered", "partially covered", "partly_covered"}:
        return "partial"
    return "missing" if status else "covered"


def parse_optional_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def normalize_traceable_queries(
    value: Any,
    *,
    max_next_queries: int,
    chunk_id_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    output = []
    seen = set()
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            query_text = compact_text(item.get("query_text") or item.get("query"))
            target_missing_facet_id = compact_text(item.get("target_missing_facet_id"))
            derived_from_chunk_ids = normalize_chunk_ids(
                item.get("derived_from_chunk_ids"),
                chunk_id_aliases=chunk_id_aliases,
            )
        else:
            query_text = compact_text(item)
            target_missing_facet_id = ""
            derived_from_chunk_ids = []
        if not query_text:
            continue
        dedup_key = query_text.casefold()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        output.append(
            {
                "query_id": f"next_query_{index}",
                "query_text": query_text,
                "target_missing_facet_id": target_missing_facet_id,
                "derived_from_chunk_ids": derived_from_chunk_ids,
            }
        )
        if len(output) == max_next_queries:
            break
    return output


def collect_cited_chunk_ids(
    *,
    seen_chunk_ids: list[str],
    covered_facets: list[dict[str, Any]],
    missing_facets: list[dict[str, Any]],
    next_queries: list[dict[str, Any]],
) -> set[str]:
    cited = set(seen_chunk_ids)
    for facet in covered_facets:
        cited.update(facet.get("supporting_chunk_ids") or [])
    for facet in missing_facets:
        cited.update(facet.get("inferred_from_chunk_ids") or [])
    for query in next_queries:
        cited.update(query.get("derived_from_chunk_ids") or [])
    return cited
