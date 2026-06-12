from __future__ import annotations

from collections.abc import Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.utils.io_utils import dumps_json, loads_json
from src.utils.text_utils import compact_text


DEFAULT_OPENAI_CHAT_TIMEOUT = 60.0
DEFAULT_OPENAI_CHAT_TEMPERATURE = 0.0
DEFAULT_OPENAI_CHAT_MAX_TOKENS = 512


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = DEFAULT_OPENAI_CHAT_TIMEOUT,
        error_factory: Callable[[str], Exception] | None = None,
    ) -> None:
        self.base_url = compact_text(base_url).rstrip("/")
        self.model = compact_text(model)
        self.api_key = compact_text(api_key)
        self.timeout = float(timeout)
        self.error_factory = error_factory or RuntimeError
        if not self.base_url:
            raise self.error_factory("llm_base_url must not be empty.")
        if not self.model:
            raise self.error_factory("llm_model must not be empty.")
        if self.timeout <= 0:
            raise self.error_factory("llm_timeout must be positive.")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_OPENAI_CHAT_TEMPERATURE,
        max_tokens: int = DEFAULT_OPENAI_CHAT_MAX_TOKENS,
    ) -> str:
        if max_tokens <= 0:
            raise self.error_factory("llm_max_tokens must be positive.")
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
            raise self.error_factory(
                f"LLM endpoint returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except urllib_error.URLError as exc:
            raise self.error_factory(f"LLM endpoint request failed: {exc}") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise self.error_factory(
                "LLM response is missing choices[0].message.content."
            ) from exc
        return compact_text(content)

    @property
    def completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"
