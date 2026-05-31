from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.data.schema import KBChunk
from src.retrievers.tokenizer import tokenize
from src.utils.text_utils import preview_text


class ChunkBM25Retriever:
    def __init__(self, chunks: list[KBChunk]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k_chunks: int = 100) -> list[dict]:
        if top_k_chunks <= 0 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if self.bm25 is None or not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[: min(top_k_chunks, len(self.chunks))]

        results = []
        for rank, index in enumerate(ranked_indexes, start=1):
            chunk = self.chunks[index]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "article_id": chunk.article_id,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "url": chunk.url,
                    "score": float(scores[index]),
                    "contents_preview": preview_text(chunk.contents, 300),
                }
            )
        return results
