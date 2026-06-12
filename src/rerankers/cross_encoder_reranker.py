from __future__ import annotations

import math
import os
import threading
from typing import Any

from src.data.schema import KBChunk
from src.utils.text_utils import preview_text


DEFAULT_RERANK_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANK_INSTRUCTION = (
    "Given a Wix Help Center question, retrieve relevant passages that contain "
    "the information needed to answer the question."
)
DEFAULT_INSTRUCTION_NAME = "wixqa_help_center_v1"
DEFAULT_RERANK_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 1024


class CrossEncoderRerankerError(RuntimeError):
    pass


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANK_MODEL_NAME,
        local_files_only: bool = True,
        device: str | None = None,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
        max_length: int = DEFAULT_MAX_LENGTH,
        model: Any | None = None,
    ):
        if not instruction.strip():
            raise CrossEncoderRerankerError("Reranker instruction must not be empty.")
        if max_length <= 0:
            raise CrossEncoderRerankerError("Reranker max_length must be a positive integer.")

        self.model_name = model_name
        self.instruction = instruction
        self.max_length = max_length
        self._lock = threading.RLock()
        self.model = model or self._load_model(
            model_name=model_name,
            local_files_only=local_files_only,
            device=device,
            instruction=instruction,
            max_length=max_length,
        )

    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        chunk_lookup: dict[str, KBChunk],
        *,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise CrossEncoderRerankerError("Reranker batch_size must be a positive integer.")

        enriched_candidates = []
        pairs = []
        for fallback_rank, candidate in enumerate(candidates, start=1):
            chunk_id = candidate.get("chunk_id")
            if not chunk_id:
                raise CrossEncoderRerankerError("Hybrid candidate is missing chunk_id.")
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                raise CrossEncoderRerankerError(
                    f"Hybrid candidate references unknown chunk_id: {chunk_id}."
                )
            pairs.append((question, chunk.text))
            enriched_candidates.append(
                {
                    **candidate,
                    "hybrid_rank": int(candidate.get("rank") or fallback_rank),
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "url": chunk.url,
                    "text_preview": preview_text(chunk.text),
                    "contents_preview": preview_text(chunk.contents),
                }
            )

        if not pairs:
            return []

        with self._lock:
            try:
                scores = self.model.predict(
                    pairs,
                    prompt_name="query",
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise CrossEncoderRerankerError(
                    f"Qwen3 reranker scoring failed for {len(pairs)} candidates: {exc}"
                ) from exc

        if len(scores) != len(enriched_candidates):
            raise CrossEncoderRerankerError(
                "Qwen3 reranker score count does not match candidate count: "
                f"scores={len(scores)}, candidates={len(enriched_candidates)}."
            )

        scored = []
        for candidate, score in zip(enriched_candidates, scores):
            rerank_score = float(score)
            if not math.isfinite(rerank_score):
                raise CrossEncoderRerankerError(
                    f"Qwen3 reranker returned a non-finite score for {candidate['chunk_id']}."
                )
            scored.append({**candidate, "rerank_score": rerank_score})

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
                **result,
                "rank": rank,
                "score": result["rerank_score"],
            }
            for rank, result in enumerate(ranked, start=1)
        ]

    @staticmethod
    def _load_model(
        *,
        model_name: str,
        local_files_only: bool,
        device: str | None,
        instruction: str,
        max_length: int,
    ) -> Any:
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise CrossEncoderRerankerError(
                "Missing dependency `sentence-transformers`. Install project dependencies "
                "with `pip install -r requirements.txt` and retry."
            ) from exc

        try:
            return CrossEncoder(
                model_name,
                local_files_only=local_files_only,
                device=device,
                prompts={"query": instruction},
                default_prompt_name="query",
                max_length=max_length,
            )
        except Exception as exc:
            offline_hint = (
                " Cache Qwen/Qwen3-Reranker-0.6B first or rerun with "
                "`--local_files_only false`."
                if local_files_only
                else ""
            )
            raise CrossEncoderRerankerError(
                f"Failed to load reranker model {model_name!r}.{offline_hint} Original error: {exc}"
            ) from exc
