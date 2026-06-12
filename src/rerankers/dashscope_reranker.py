from __future__ import annotations

import math
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.data.schema import KBChunk
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_INSTRUCTION,
)
from src.utils.io_utils import dumps_json, loads_json
from src.utils.text_utils import compact_text, preview_text


DEFAULT_DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
DEFAULT_DASHSCOPE_RERANK_MODEL = "qwen3-rerank"
DEFAULT_DASHSCOPE_RERANK_TIMEOUT = 60.0


class DashScopeRerankerError(RuntimeError):
    pass


class DashScopeReranker:
    def __init__(
        self,
        *,
        api_key: str,
        url: str = DEFAULT_DASHSCOPE_RERANK_URL,
        model: str = DEFAULT_DASHSCOPE_RERANK_MODEL,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
        timeout: float = DEFAULT_DASHSCOPE_RERANK_TIMEOUT,
    ) -> None:
        self.api_key = compact_text(api_key)
        self.url = compact_text(url)
        self.model_name = compact_text(model)
        self.instruction = compact_text(instruction)
        self.timeout = float(timeout)
        if not self.api_key:
            raise DashScopeRerankerError("DashScope reranker api_key must not be empty.")
        if not self.url:
            raise DashScopeRerankerError("DashScope reranker url must not be empty.")
        if not self.model_name:
            raise DashScopeRerankerError("DashScope reranker model must not be empty.")
        if not self.instruction:
            raise DashScopeRerankerError("DashScope reranker instruction must not be empty.")
        if self.timeout <= 0:
            raise DashScopeRerankerError("DashScope reranker timeout must be positive.")

    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        chunk_lookup: dict[str, KBChunk],
        *,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise DashScopeRerankerError("Reranker batch_size must be positive.")
        normalized_question = compact_text(question)
        if not normalized_question:
            raise DashScopeRerankerError("Reranker question must not be empty.")

        enriched_candidates = build_enriched_candidates(candidates, chunk_lookup)
        if not enriched_candidates:
            return []

        scored = []
        for start in range(0, len(enriched_candidates), batch_size):
            batch = enriched_candidates[start : start + batch_size]
            documents = [row["_document_text"] for row in batch]
            response = self._post_rerank(
                {
                    "model": self.model_name,
                    "query": normalized_question,
                    "documents": documents,
                    "top_n": len(documents),
                    "return_documents": False,
                    "instruct": self.instruction,
                }
            )
            scored.extend(score_batch(batch, response))

        ranked = sorted(
            scored,
            key=lambda result: (
                -float(result["rerank_score"]),
                int(result["hybrid_rank"]),
                str(result["chunk_id"]),
            ),
        )
        return [
            {
                **{key: value for key, value in result.items() if key != "_document_text"},
                "rank": rank,
                "score": result["rerank_score"],
            }
            for rank, result in enumerate(ranked, start=1)
        ]

    def _post_rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib_request.Request(
            self.url,
            data=dumps_json(payload, indent=False),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_payload = loads_json(response.read())
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DashScopeRerankerError(
                f"DashScope reranker returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except urllib_error.URLError as exc:
            raise DashScopeRerankerError(f"DashScope reranker request failed: {exc}") from exc
        if not isinstance(response_payload, dict):
            raise DashScopeRerankerError("DashScope reranker response must be a JSON object.")
        return response_payload


def build_enriched_candidates(
    candidates: list[dict[str, Any]],
    chunk_lookup: dict[str, KBChunk],
) -> list[dict[str, Any]]:
    enriched = []
    for fallback_rank, candidate in enumerate(candidates, start=1):
        chunk_id = candidate.get("chunk_id")
        if not chunk_id:
            raise DashScopeRerankerError("Hybrid candidate is missing chunk_id.")
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            raise DashScopeRerankerError(
                f"Hybrid candidate references unknown chunk_id: {chunk_id}."
            )
        enriched.append(
            {
                **candidate,
                "hybrid_rank": int(candidate.get("rank") or fallback_rank),
                "chunk_index": chunk.chunk_index,
                "title": chunk.title,
                "url": chunk.url,
                "text_preview": preview_text(chunk.text),
                "contents_preview": preview_text(chunk.contents),
                "_document_text": chunk.text,
            }
        )
    return enriched


def score_batch(batch: list[dict[str, Any]], response: dict[str, Any]) -> list[dict[str, Any]]:
    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise DashScopeRerankerError("DashScope reranker response is missing results list.")
    if len(raw_results) != len(batch):
        raise DashScopeRerankerError(
            "DashScope reranker did not return a score for every document: "
            f"results={len(raw_results)}, documents={len(batch)}."
        )

    scored_by_index: dict[int, float] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            raise DashScopeRerankerError("DashScope reranker result entries must be objects.")
        index = parse_result_index(item)
        score = parse_result_score(item)
        if index < 0 or index >= len(batch):
            raise DashScopeRerankerError(f"DashScope reranker returned out-of-range index: {index}.")
        if index in scored_by_index:
            raise DashScopeRerankerError(f"DashScope reranker returned duplicate index: {index}.")
        scored_by_index[index] = score

    missing_indexes = sorted(set(range(len(batch))) - set(scored_by_index))
    if missing_indexes:
        raise DashScopeRerankerError(
            f"DashScope reranker response is missing indexes: {missing_indexes}."
        )
    return [
        {
            **row,
            "rerank_score": scored_by_index[index],
            "dashscope_rerank_model": response.get("model"),
        }
        for index, row in enumerate(batch)
    ]


def parse_result_index(item: dict[str, Any]) -> int:
    try:
        return int(item["index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DashScopeRerankerError("DashScope reranker result is missing a valid index.") from exc


def parse_result_score(item: dict[str, Any]) -> float:
    score = item.get("relevance_score", item.get("score"))
    try:
        score = float(score)
    except (TypeError, ValueError) as exc:
        raise DashScopeRerankerError(
            "DashScope reranker result is missing a valid relevance_score."
        ) from exc
    if not math.isfinite(score):
        raise DashScopeRerankerError("DashScope reranker returned a non-finite score.")
    return score
