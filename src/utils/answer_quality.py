from __future__ import annotations

import re
from typing import Any

from src.utils.text_utils import compact_text, contains_cjk


UNHELPFUL_ANSWER_PHRASES = (
    "hello! how can i assist",
    "how can i assist you today",
    "i found relevant wix help center evidence",
    "use the cited sources",
    "relevant to this question",
    "relevant to your question",
    "我找到了一些相关",
    "相关 wix 帮助中心",
)
QUESTION_STOPWORDS = {
    "about",
    "after",
    "again",
    "another",
    "can",
    "could",
    "does",
    "for",
    "from",
    "have",
    "help",
    "how",
    "into",
    "like",
    "need",
    "should",
    "site",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "wix",
    "would",
    "your",
}


def answer_is_unhelpful(
    answer: str,
    *,
    question: str | None = None,
    evidence_texts: list[Any] | None = None,
) -> bool:
    normalized = compact_text(answer).casefold()
    if not normalized:
        return True
    if any(phrase in normalized for phrase in UNHELPFUL_ANSWER_PHRASES):
        return True
    if question is None and evidence_texts is None:
        return False
    question_text = compact_text(question)
    if contains_cjk(answer) or contains_cjk(question_text):
        return looks_like_short_greeting(normalized)
    question_terms = significant_terms(question_text)
    if question_terms:
        overlap = sum(1 for term in question_terms if term in normalized)
        if overlap == 0:
            return True
    evidence_blob = " ".join(compact_text(text) for text in evidence_texts or []).casefold()
    evidence_terms = significant_terms(evidence_blob)
    if evidence_terms and not any(term in normalized for term in evidence_terms[:6]):
        return True
    return False


def looks_like_short_greeting(normalized: str) -> bool:
    compacted = normalized.replace(" ", "")
    return compacted in {"你好", "您好", "hello", "hi", "hey"} or len(compacted) <= 4


def significant_terms(text: str) -> list[str]:
    output = []
    seen = set()
    for token in re.findall(r"[a-z0-9]+", compact_text(text).casefold()):
        if token in QUESTION_STOPWORDS or len(token) <= 2:
            continue
        if token not in seen:
            seen.add(token)
            output.append(token)
    return output
