from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.data.schema import KBArticle
from src.retrievers.tokenizer import tokenize
from src.utils.text_utils import preview_text


class BM25Retriever:
    def __init__(self, articles: list[KBArticle]):
        self.articles = articles
        self.corpus_tokens = [tokenize(self._article_text(article)) for article in articles]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if top_k <= 0 or not self.articles:
            return []

        query_tokens = tokenize(query)
        if self.bm25 is None or not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[: min(top_k, len(self.articles))]

        results = []
        for rank, index in enumerate(ranked_indexes, start=1):
            article = self.articles[index]
            results.append(
                {
                    "rank": rank,
                    "article_id": article.article_id,
                    "title": article.title,
                    "url": article.url,
                    "score": float(scores[index]),
                    "contents_preview": preview_text(article.contents, 300),
                }
            )
        return results

    @staticmethod
    def _article_text(article: KBArticle) -> str:
        if article.title:
            return f"{article.title}\n{article.contents}"
        return article.contents
