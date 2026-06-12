from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.utils.io_utils import loads_json
from src.utils.json_utils import extract_json_object
from src.utils.text_utils import compact_text


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredLLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class StructuredLLMCaller:
    def __init__(
        self,
        *,
        client: Any,
        schema: type[SchemaT],
        temperature: float = 0.0,
        max_tokens: int = 512,
        retry_min_tokens: int = 512,
        token_budget_values: list[int] | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise StructuredLLMError("structured LLM max_tokens must be positive.")
        self.client = client
        self.schema = schema
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_min_tokens = retry_min_tokens
        self.token_budget_values = list(token_budget_values or [])
        self.model = compact_text(getattr(client, "model", "")) or "structured_llm"
        self.last_raw_text = ""
        self.last_attempts: list[dict[str, Any]] = []

    def call(self, messages: list[dict[str, str]]) -> SchemaT:
        attempts = []
        self.last_raw_text = ""
        self.last_attempts = []
        budgets = self.token_budget_values or token_budgets(
            self.max_tokens,
            retry_min_tokens=self.retry_min_tokens,
        )
        for max_tokens in budgets:
            raw_text = ""
            try:
                raw_text = self.client.complete(
                    messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                self.last_raw_text = raw_text
                payload = parse_json_object(raw_text)
                result = validate_schema(self.schema, payload)
                self.last_attempts = attempts
                return result
            except Exception as exc:
                attempts.append(
                    {
                        "max_tokens": max_tokens,
                        "error": compact_text(str(exc))[:300],
                        "raw_text_preview": compact_text(raw_text)[:300],
                    }
                )
                self.last_attempts = attempts
        last_error = attempts[-1]["error"] if attempts else "no attempts executed"
        raise StructuredLLMError(
            f"Structured LLM call failed after {len(attempts)} attempt(s): {last_error}",
            attempts=attempts,
        )


def token_budgets(max_tokens: int, *, retry_min_tokens: int = 512) -> list[int]:
    retry_budget = max(retry_min_tokens, max_tokens * 2)
    budgets = []
    for value in (max_tokens, retry_budget):
        if value not in budgets:
            budgets.append(value)
    return budgets


def parse_json_object(raw_text: str) -> dict[str, Any]:
    try:
        payload = loads_json(extract_json_object(raw_text))
    except Exception as exc:
        raise StructuredLLMError(f"structured LLM returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise StructuredLLMError("structured LLM JSON must be an object.")
    return payload


def validate_schema(schema: type[SchemaT], payload: dict[str, Any]) -> SchemaT:
    try:
        if hasattr(schema, "model_validate"):
            return schema.model_validate(payload)  # type: ignore[attr-defined]
        return schema(**payload)
    except ValidationError as exc:
        raise StructuredLLMError(f"structured LLM schema validation failed: {exc}") from exc
