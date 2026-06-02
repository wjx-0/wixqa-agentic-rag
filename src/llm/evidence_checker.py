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
            "Set sufficient=false only when a blocking evidence gap would make the answer unsupported, materially incomplete, or misleading.",
            "Before deciding sufficient, decompose the question into required facets: Wix product, object/entity, feature, action, setting, integration, error, condition, comparison, or required step.",
            "If the question asks about multiple objects, actions, steps, conditions, products, integrations, or feature requirements, the top10 evidence must directly cover every required facet.",
            "If any required facet is missing, contradicted, or only inferable from adjacent evidence, set sufficient=false and put that exact facet in blocking_missing_evidence.",
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
