from __future__ import annotations

import re
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.utils.io_utils import dumps_json, loads_json
from src.utils.text_utils import compact_text, preview_text, to_string_list


DEFAULT_CHECKER_TEMPERATURE = 0.0
DEFAULT_CHECKER_MAX_TOKENS = 512
DEFAULT_CHECKER_TIMEOUT = 60.0
DEFAULT_MAX_NEXT_QUERIES = 3
DEFAULT_TRACEABLE_MAX_NEXT_QUERIES = 2
DEFAULT_CONTEXT_PREVIEW_CHARS = 650


class EvidenceCheckerError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = DEFAULT_CHECKER_TIMEOUT,
    ) -> None:
        self.base_url = compact_text(base_url).rstrip("/")
        self.model = compact_text(model)
        self.api_key = compact_text(api_key)
        self.timeout = float(timeout)
        if not self.base_url:
            raise EvidenceCheckerError("llm_base_url must not be empty.")
        if not self.model:
            raise EvidenceCheckerError("llm_model must not be empty.")
        if self.timeout <= 0:
            raise EvidenceCheckerError("llm_timeout must be positive.")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_CHECKER_TEMPERATURE,
        max_tokens: int = DEFAULT_CHECKER_MAX_TOKENS,
    ) -> str:
        if max_tokens <= 0:
            raise EvidenceCheckerError("llm_max_tokens must be positive.")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib_request.Request(
            self.completions_url,
            data=dumps_json(payload, indent=False),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_payload = loads_json(response.read())
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EvidenceCheckerError(
                f"LLM endpoint returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except urllib_error.URLError as exc:
            raise EvidenceCheckerError(f"LLM endpoint request failed: {exc}") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EvidenceCheckerError("LLM response is missing choices[0].message.content.") from exc
        return compact_text(content)

    @property
    def completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"


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
    if len(top_chunks) > 10:
        raise EvidenceCheckerError("traceable checker can inspect at most 10 raw chunks.")

    system_prompt = "\n".join(
        [
            "You are an evidence sufficiency checker for Wix Help Center retrieval.",
            "Do not answer the user's question.",
            "Judge whether the visible chunks are sufficient to support a correct answer.",
            "Every claim about coverage or missing evidence must cite only chunk_id values from the visible chunks.",
            "If a missing facet is inferred from partial evidence, cite inferred_from_chunk_ids.",
            f"Generate at most {max_next_queries} next_queries.",
            "Each next query must target a blocking missing facet and must include derived_from_chunk_ids.",
            "If sufficient=true, next_queries and missing_facets must be empty.",
            "Return valid JSON only with keys: sufficient, seen_chunk_ids, known_facts, covered_facets, missing_facets, next_queries, reason.",
            "covered_facets items must be objects with: facet_id, description, supporting_chunk_ids.",
            "missing_facets items must be objects with: facet_id, description, inferred_from_chunk_ids, blocking.",
            "next_queries items must be objects with: query_text, target_missing_facet_id, derived_from_chunk_ids.",
        ]
    )
    context_lines = []
    for fallback_rank, chunk in enumerate(top_chunks, start=1):
        rank = int(get_chunk_value(chunk, "rank") or fallback_rank)
        chunk_id = compact_text(get_chunk_value(chunk, "chunk_id")) or "(missing_chunk_id)"
        snippet_id = compact_text(get_chunk_value(chunk, "snippet_id")) or chunk_id
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
                    f"Chunk ID: {chunk_id}",
                    f"Snippet ID: {snippet_id}",
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
                format_optional_list_section("Known facts from previous calls", known_facts),
                format_optional_list_section("Previously covered facets", covered_facets),
                format_optional_list_section("Open missing facets", missing_facets),
                format_optional_list_section("Previous retrieval queries", query_history),
                "Visible raw chunks:\n" + "\n\n".join(context_lines),
                (
                    "JSON schema:\n"
                    '{"sufficient": boolean, "seen_chunk_ids": string[], '
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

    def check(
        self,
        *,
        context: Any,
        visible_items: list[Any],
        manifest: Any,
        round_index: int,
    ) -> dict[str, Any]:
        allowed_chunk_ids = [item.chunk_id for item in visible_items]
        try:
            messages = build_traceable_evidence_checker_messages(
                context.question,
                visible_items,
                max_context_chars=self.context_preview_chars,
                max_next_queries=self.max_next_queries,
                compressed_summary=context.compressed_summary,
                known_facts=context.known_facts,
                covered_facets=context.covered_facets,
                missing_facets=context.missing_facets,
                query_history=context.query_history,
            )
            raw_text = self.client.complete(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            result = parse_traceable_checker_response(
                raw_text,
                allowed_chunk_ids=allowed_chunk_ids,
                max_next_queries=self.max_next_queries,
            )
            checker_valid = True
            checker_error = None
        except EvidenceCheckerError as exc:
            raw_text = ""
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
            "llm_call_id": manifest.llm_call_id,
            "round_index": round_index,
            "retrieval_triggered": bool(result.get("next_queries")),
        }


def parse_checker_response(
    raw_text: str,
    *,
    max_next_queries: int = DEFAULT_MAX_NEXT_QUERIES,
) -> dict[str, Any]:
    if max_next_queries <= 0:
        raise EvidenceCheckerError("max_next_queries must be positive.")
    payload_text = extract_json_object(raw_text)
    try:
        payload = loads_json(payload_text)
    except Exception as exc:
        raise EvidenceCheckerError(f"Checker returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceCheckerError("Checker JSON must be an object.")

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
        "known_facts": [],
        "covered_facets": [],
        "missing_facets": [],
        "next_queries": [],
        "reason": "",
        "provenance_valid": False,
        "invalid_provenance_chunk_ids": [],
    }


def parse_traceable_checker_response(
    raw_text: str,
    *,
    allowed_chunk_ids: list[str],
    max_next_queries: int = DEFAULT_TRACEABLE_MAX_NEXT_QUERIES,
) -> dict[str, Any]:
    if max_next_queries <= 0:
        raise EvidenceCheckerError("max_next_queries must be positive.")
    allowed_chunk_id_set = set(allowed_chunk_ids)
    payload_text = extract_json_object(raw_text)
    try:
        payload = loads_json(payload_text)
    except Exception as exc:
        raise EvidenceCheckerError(f"Checker returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceCheckerError("Checker JSON must be an object.")

    sufficient = parse_bool(payload.get("sufficient"))
    seen_chunk_ids = normalize_chunk_ids(payload.get("seen_chunk_ids"))
    covered_facets = normalize_facets(
        payload.get("covered_facets"),
        id_key="supporting_chunk_ids",
    )
    missing_facets = normalize_facets(
        payload.get("missing_facets") or payload.get("blocking_missing_evidence"),
        id_key="inferred_from_chunk_ids",
        default_blocking=True,
    )
    if sufficient:
        missing_facets = []
        next_queries = []
    else:
        next_queries = normalize_traceable_queries(
            payload.get("next_queries"),
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
        "known_facts": normalize_text_list(payload.get("known_facts")),
        "covered_facets": covered_facets,
        "missing_facets": missing_facets,
        "next_queries": next_queries,
        "reason": compact_text(payload.get("reason")),
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


def extract_json_object(raw_text: str) -> str:
    text = compact_text(raw_text)
    if not text:
        raise EvidenceCheckerError("Checker response is empty.")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise EvidenceCheckerError("Checker response does not contain a JSON object.")
    return raw_text[start : end + 1].strip()


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


def normalize_chunk_ids(value: Any) -> list[str]:
    output = []
    seen = set()
    for item in normalize_text_list(value):
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
) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    output = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            description = compact_text(item.get("description") or item.get("facet") or item.get("text"))
            facet_id = compact_text(item.get("facet_id")) or f"facet_{index}"
            chunk_ids = normalize_chunk_ids(item.get(id_key))
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


def normalize_traceable_queries(
    value: Any,
    *,
    max_next_queries: int,
) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    output = []
    seen = set()
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            query_text = compact_text(item.get("query_text") or item.get("query"))
            target_missing_facet_id = compact_text(item.get("target_missing_facet_id"))
            derived_from_chunk_ids = normalize_chunk_ids(item.get("derived_from_chunk_ids"))
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
