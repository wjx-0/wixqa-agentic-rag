from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KBArticle(BaseModel):
    article_id: str
    title: Optional[str] = None
    url: Optional[str] = None
    contents: str
    article_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KBChunk(BaseModel):
    chunk_id: str
    article_id: str
    chunk_index: int
    title: Optional[str] = None
    url: Optional[str] = None
    text: str
    contents: str
    start_token: int
    end_token: int
    num_tokens: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QAExample(BaseModel):
    qid: str
    dataset_name: str
    question: str
    answer: Optional[str] = None
    article_ids: List[str] = Field(default_factory=list)
    num_gold_articles: int
    is_multi_article: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
