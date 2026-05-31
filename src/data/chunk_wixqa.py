from __future__ import annotations

from typing import Protocol

from src.data.schema import KBArticle, KBChunk
from src.utils.text_utils import compact_text


DEFAULT_CHUNK_TOKENIZER_NAME = "BAAI/bge-m3"


class ChunkingError(ValueError):
    """Raised when chunking parameters are invalid."""


class ChunkTokenizer(Protocol):
    name: str

    def spans(self, text: str) -> list[tuple[int, int]]:
        ...


class HFChunkTokenizer:
    def __init__(self, *, local_files_only: bool = False):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ChunkingError(
                "Missing dependency `transformers`. Install project dependencies with "
                "`pip install -r requirements.txt` and retry."
            ) from exc

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                DEFAULT_CHUNK_TOKENIZER_NAME,
                use_fast=True,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise ChunkingError(
                f"Could not load tokenizer {DEFAULT_CHUNK_TOKENIZER_NAME!r}: {exc}"
            ) from exc

        if not getattr(self.tokenizer, "is_fast", False):
            raise ChunkingError(
                f"Tokenizer {DEFAULT_CHUNK_TOKENIZER_NAME!r} is not a fast tokenizer; "
                "offset-based chunking requires one."
            )
        self.name = DEFAULT_CHUNK_TOKENIZER_NAME

    def spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") or []
        return [(int(start), int(end)) for start, end in offsets if int(end) > int(start)]


def build_chunk_tokenizer(
    *,
    local_files_only: bool = False,
) -> ChunkTokenizer:
    return HFChunkTokenizer(local_files_only=local_files_only)


def validate_chunk_params(chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
    if chunk_size_tokens <= 0:
        raise ChunkingError("chunk_size_tokens must be a positive integer.")
    if chunk_overlap_tokens < 0:
        raise ChunkingError("chunk_overlap_tokens must be zero or a positive integer.")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ChunkingError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")


def chunk_article(
    article: KBArticle,
    *,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 128,
    tokenizer: ChunkTokenizer | None = None,
) -> list[KBChunk]:
    validate_chunk_params(chunk_size_tokens, chunk_overlap_tokens)
    tokenizer = tokenizer or build_chunk_tokenizer()

    contents = compact_text(article.contents)
    if not contents:
        return []

    spans = tokenizer.spans(contents)
    if not spans:
        return []

    stride = chunk_size_tokens - chunk_overlap_tokens
    chunks: list[KBChunk] = []
    start_token = 0

    while start_token < len(spans):
        end_token = min(start_token + chunk_size_tokens, len(spans))
        start_char = spans[start_token][0]
        end_char = spans[end_token - 1][1]
        chunk_contents = contents[start_char:end_char].strip()
        chunk_index = len(chunks)
        text = f"{article.title or ''}\n{chunk_contents}".strip()

        chunks.append(
            KBChunk(
                chunk_id=f"{article.article_id}_chunk_{chunk_index:04d}",
                article_id=article.article_id,
                chunk_index=chunk_index,
                title=article.title,
                url=article.url,
                text=text,
                contents=chunk_contents,
                start_token=start_token,
                end_token=end_token,
                num_tokens=end_token - start_token,
                metadata={
                    "article_type": article.article_type,
                    "chunk_tokenizer": tokenizer.name,
                    "chunk_size_tokens": chunk_size_tokens,
                    "chunk_overlap_tokens": chunk_overlap_tokens,
                    "source_article_metadata": article.metadata,
                },
            )
        )

        if end_token >= len(spans):
            break
        start_token += stride

    return chunks


def chunk_articles(
    articles: list[KBArticle],
    *,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 128,
    tokenizer: ChunkTokenizer | None = None,
) -> list[KBChunk]:
    validate_chunk_params(chunk_size_tokens, chunk_overlap_tokens)
    tokenizer = tokenizer or build_chunk_tokenizer()
    chunks: list[KBChunk] = []
    for article in articles:
        chunks.extend(
            chunk_article(
                article,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                tokenizer=tokenizer,
            )
        )
    return chunks
