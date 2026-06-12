from __future__ import annotations

import re
from typing import Any

from src.utils.text_utils import compact_text


GENERIC_OBJECT_TERMS = {
    "account",
    "accounts",
    "booking",
    "bookings",
    "cart",
    "checkout",
    "course",
    "customer",
    "customers",
    "dashboard",
    "domain",
    "domains",
    "editor",
    "email",
    "emails",
    "invoice",
    "invoices",
    "login",
    "order",
    "orders",
    "page",
    "pages",
    "password",
    "payment",
    "payments",
    "plan",
    "plans",
    "product",
    "products",
    "service",
    "services",
    "shipping",
    "site",
    "store",
    "stores",
    "subscription",
    "subscriptions",
    "tax",
    "website",
    "websites",
    "支付",
    "收款",
    "付款",
    "网站",
    "域名",
    "账户",
    "账号",
    "密码",
    "商店",
    "订单",
    "产品",
    "服务",
    "预订",
    "预约",
    "套餐",
}
GENERIC_ACTION_TERMS = {
    "accept",
    "add",
    "book",
    "cancel",
    "change",
    "configure",
    "connect",
    "create",
    "delete",
    "disable",
    "edit",
    "enable",
    "fix",
    "manage",
    "publish",
    "refund",
    "remove",
    "reset",
    "sell",
    "set",
    "setup",
    "troubleshoot",
    "update",
    "upgrade",
    "verify",
    "连接",
    "设置",
    "添加",
    "创建",
    "售卖",
    "销售",
    "收款",
    "绑定",
    "修改",
    "取消",
    "升级",
    "管理",
    "验证",
    "退款",
    "发布",
}
ISSUE_TERMS = {
    "blocked",
    "broken",
    "declined",
    "error",
    "failed",
    "failure",
    "issue",
    "not working",
    "problem",
    "trouble",
    "失败",
    "报错",
    "错误",
    "不能",
    "不了",
    "无法",
}
QUESTION_CUES = {
    "?",
    "？",
    "can",
    "could",
    "does",
    "how",
    "is",
    "should",
    "what",
    "when",
    "where",
    "which",
    "why",
    "怎么",
    "如何",
    "什么",
    "哪里",
    "能不能",
    "可以",
    "是否",
    "吗",
    "呢",
}
FOLLOW_UP_CUES = {
    "and",
    "and if",
    "also",
    "what about",
    "how about",
    "then",
    "that",
    "this",
    "it",
    "same",
    "instead",
    "呢",
    "那",
    "这个",
    "那个",
    "同样",
}
STANDALONE_STARTERS = {
    "can i",
    "could i",
    "do i",
    "does",
    "how can i",
    "how do i",
    "how to",
    "is there",
    "what is",
    "what are",
    "where do i",
    "why does",
    "怎么",
    "如何",
    "在哪里",
    "为什么",
}
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "have",
    "help",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "you",
    "your",
}


def looks_like_follow_up(message: str) -> bool:
    lowered = compact_text(message).casefold()
    return any(contains_term(lowered, cue) for cue in FOLLOW_UP_CUES)


def has_question_cue(message: str) -> bool:
    lowered = compact_text(message).casefold()
    return any(contains_term(lowered, cue) for cue in QUESTION_CUES)


def looks_like_standalone_question(message: str) -> bool:
    lowered = compact_text(message).casefold()
    if not has_question_cue(lowered):
        return False
    return any(lowered.startswith(starter) for starter in STANDALONE_STARTERS)


def has_support_action(message: str) -> bool:
    lowered = compact_text(message).casefold()
    return contains_any_term(lowered, GENERIC_ACTION_TERMS)


def has_support_object(message: str, domain_terms: tuple[str, ...] = ()) -> bool:
    lowered = compact_text(message).casefold()
    return contains_any_term(lowered, GENERIC_OBJECT_TERMS) or contains_any_term(
        lowered,
        domain_terms,
    )


def has_issue_signal(message: str) -> bool:
    lowered = compact_text(message).casefold()
    return contains_any_term(lowered, ISSUE_TERMS)


def support_intent_score(message: str, domain_terms: tuple[str, ...] = ()) -> int:
    text = compact_text(message)
    lowered = text.casefold()
    score = 0
    if has_question_cue(lowered):
        score += 2
    if has_support_object(lowered, domain_terms):
        score += 2
    if has_support_action(lowered):
        score += 2
    if has_issue_signal(lowered):
        score += 2
    if contains_any_term(lowered, domain_terms):
        score += 2
    if re.search(r"\b(?:i|we)\s+(?:want|need|would like|am trying|are trying)\b", lowered):
        score += 1
    if any(token in text for token in ("想", "需要", "怎么弄")):
        score += 1
    return score


def context_relevance_score(
    message: str,
    context_texts: list[str],
    *,
    domain_terms: tuple[str, ...] = (),
) -> float:
    message_terms = significant_terms(message, domain_terms=domain_terms)
    context_terms: set[str] = set()
    for text in context_texts:
        context_terms.update(significant_terms(text, domain_terms=domain_terms))
    if not message_terms or not context_terms:
        return 0.0
    overlap = message_terms & context_terms
    return len(overlap) / max(1, min(len(message_terms), len(context_terms)))


def is_context_related(
    message: str,
    context_texts: list[str],
    *,
    domain_terms: tuple[str, ...] = (),
    min_score: float = 0.25,
) -> bool:
    if context_relevance_score(message, context_texts, domain_terms=domain_terms) >= min_score:
        return True
    return related_by_common_support_object(message, context_texts)


def has_bridge_overlap(message: str, context_texts: list[str]) -> bool:
    return related_by_common_support_object(message, context_texts)


def related_by_common_support_object(message: str, context_texts: list[str]) -> bool:
    message_terms = significant_terms(message)
    if not message_terms:
        return False
    context_text = " ".join(compact_text(text).casefold() for text in context_texts)
    if {"service", "services"} & message_terms and any(
        term in context_text for term in ("product", "products", "store", "stores", "sell")
    ):
        return True
    if {"product", "products"} & message_terms and any(
        term in context_text for term in ("service", "services", "store", "stores", "sell")
    ):
        return True
    if {"payment", "payments", "paypal"} & message_terms and any(
        term in context_text for term in ("payment", "payments", "paypal", "checkout")
    ):
        return True
    return False


def significant_terms(text: str, *, domain_terms: tuple[str, ...] = ()) -> set[str]:
    normalized = compact_text(text).casefold()
    output: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", normalized):
        if token in STOPWORDS or len(token) <= 2:
            continue
        output.add(token)
    for term in GENERIC_OBJECT_TERMS | GENERIC_ACTION_TERMS | set(domain_terms):
        if contains_term(normalized, term):
            output.add(term.casefold())
    for cjk_term in ("支付", "收款", "付款", "网站", "域名", "订单", "产品", "服务", "预订"):
        if cjk_term in normalized:
            output.add(cjk_term)
    return output


def contains_any_term(lowered_text: str, terms: Any) -> bool:
    return any(contains_term(lowered_text, term) for term in terms)


def contains_term(lowered_text: str, term: str) -> bool:
    lowered_term = compact_text(term).casefold()
    if not lowered_term:
        return False
    if any("\u4e00" <= char <= "\u9fff" for char in lowered_term):
        return lowered_term in lowered_text
    if not lowered_term.replace(" ", "").isalnum():
        return lowered_term in lowered_text
    return re.search(rf"\b{re.escape(lowered_term)}\b", lowered_text) is not None
