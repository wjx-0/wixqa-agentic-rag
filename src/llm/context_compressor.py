from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.agentic.evidence_context import EvidenceContext, EvidenceItem
from src.llm.structured import StructuredLLMCaller
from src.utils.text_utils import compact_text, preview_text


DEFAULT_CONTEXT_COMPRESSION_MAX_TOKENS = 512
DEFAULT_CONTEXT_COMPRESSION_TEMPERATURE = 0.0
DEFAULT_CONTEXT_COMPRESSION_PREVIEW_CHARS = 500
DEFAULT_COMPRESSED_SUMMARY_MAX_CHARS = 2000


class ContextCompressionPayload(BaseModel):
    summary: str


class LLMContextCompressor:
    uses_api = True

    def __init__(
        self,
        *,
        client: Any,
        temperature: float = DEFAULT_CONTEXT_COMPRESSION_TEMPERATURE,
        max_tokens: int = DEFAULT_CONTEXT_COMPRESSION_MAX_TOKENS,
        preview_chars: int = DEFAULT_CONTEXT_COMPRESSION_PREVIEW_CHARS,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("context compression max_tokens must be positive.")
        if preview_chars <= 0:
            raise ValueError("context compression preview_chars must be positive.")
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.preview_chars = preview_chars
        self.caller = StructuredLLMCaller(
            client=client,
            schema=ContextCompressionPayload,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_min_tokens=max_tokens,
        )

    def summarize(
        self,
        *,
        context: EvidenceContext,
        dropped_items: list[EvidenceItem],
        existing_summary: str | None,
        compact_id: str,
    ) -> str:
        payload = self.caller.call(
            build_context_compression_messages(
                context=context,
                dropped_items=dropped_items,
                existing_summary=existing_summary,
                compact_id=compact_id,
                preview_chars=self.preview_chars,
            )
        )
        return compact_text(payload.summary)[:DEFAULT_COMPRESSED_SUMMARY_MAX_CHARS]


def build_context_compression_messages(
    *,
    context: EvidenceContext,
    dropped_items: list[EvidenceItem],
    existing_summary: str | None,
    compact_id: str,
    preview_chars: int,
) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        [
            "Compress Wix Help Center evidence that is about to leave the raw context window.",
            "Keep only facts that are explicit in the listed chunks and useful for answering the question.",
            "Preserve product names, settings, limitations, steps, errors, and chunk ids.",
            "Do not include gold labels, evaluation metadata, or outside knowledge.",
            "Return one compact JSON object only: {\"summary\":\"...\"}.",
        ]
    )
    evidence_lines = [
        "\n".join(
            [
                f"Chunk ID: {item.chunk_id}",
                f"Title: {item.title or '(untitled)'}",
                f"Evidence: {preview_text(item.text_preview, limit=preview_chars)}",
            ]
        )
        for item in dropped_items
    ]
    user_prompt = "\n\n".join(
        [
            f"Compact ID:\n{compact_id}",
            f"Question:\n{compact_text(context.question)}",
            f"Existing compressed summary:\n{compact_text(existing_summary) or '(none)'}",
            "Dropped raw chunks to compress:\n" + ("\n\n".join(evidence_lines) if evidence_lines else "(none)"),
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
