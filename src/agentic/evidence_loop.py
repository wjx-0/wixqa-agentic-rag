from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from src.agentic.context_budget import (
    ContextBudgetConfig,
    ContextUsageSnapshot,
    SOFT_AUTO_COMPACT_RATIO,
    build_context_usage_snapshot,
)
from src.agentic.evidence_context import EvidenceContext, EvidenceItem, build_evidence_item
from src.agentic.evidence_selection import (
    dedupe_items,
    dedupe_ranked_candidate_rows,
    group_candidate_rows_by_query_id,
    pack_visible_items,
    select_diverse_items,
    select_new_candidate_rows,
)
from src.agentic.provenance import CompactBoundary, GapQueryProvenance, PromptManifest
from src.data.schema import KBChunk
from src.utils.text_utils import compact_text
from src.utils.time_utils import elapsed_ms


ACTION_QUERY_STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "can",
    "center",
    "confirm",
    "for",
    "from",
    "help",
    "how",
    "i",
    "in",
    "is",
    "it",
    "know",
    "learn",
    "like",
    "my",
    "need",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "up",
    "user",
    "users",
    "website",
    "will",
    "with",
    "would",
}
ACTION_QUERY_PRODUCT_ANCHORS = {
    "booking",
    "bookings",
    "site",
    "sites",
    "store",
    "stores",
    "wix",
}
MAX_ACTION_QUERY_TERMS = 40


class EvidenceLoopConfig(BaseModel):
    max_rounds: int = 4
    min_retrieval_rounds: int = 0
    max_queries_per_round: int = 2
    per_query_retrieve_top_k_chunks: int = 20
    max_raw_chunks_per_checker_call: int = 30
    max_new_raw_chunks_per_round: int = 5
    max_new_chunks_per_article: int = 1
    max_visible_chunks_per_article: int = 2
    rerank_batch_size: int = 32
    max_parallel_retrieval_queries: int = 4
    budget: ContextBudgetConfig = Field(default_factory=ContextBudgetConfig)


class EvidenceCompletionLoopResult(BaseModel):
    context: EvidenceContext
    checker_outputs: list[dict[str, Any]] = Field(default_factory=list)
    prompt_manifests: list[PromptManifest] = Field(default_factory=list)
    usage_snapshots: list[ContextUsageSnapshot] = Field(default_factory=list)
    compact_boundaries: list[CompactBoundary] = Field(default_factory=list)
    completed: bool = False
    rounds_completed: int = 0
    retrieval_rounds: int = 0
    second_hop_query_count: int = 0
    total_latency_ms: float = 0.0


class EvidenceLoopError(RuntimeError):
    pass


def run_evidence_completion_loop(
    *,
    context: EvidenceContext,
    checker: Any,
    retriever: Any,
    reranker: Any,
    chunk_lookup: dict[str, KBChunk],
    context_summarizer: Any | None = None,
    config: EvidenceLoopConfig | None = None,
) -> EvidenceCompletionLoopResult:
    config = config or EvidenceLoopConfig()
    validate_loop_config(config)

    loop_started_at = time.monotonic()
    visible_items = list(context.active_items)
    prompt_manifests = list(context.prompt_manifests)
    usage_snapshots: list[ContextUsageSnapshot] = []
    compact_boundaries: list[CompactBoundary] = []
    checker_outputs = []
    completed = False
    retrieval_rounds = 0
    second_hop_query_count = 0

    for round_index in range(config.max_rounds + 1):
        round_started_at = time.monotonic()
        manifest = ensure_round_manifest(
            context=context,
            visible_items=visible_items,
            round_index=round_index,
            prompt_manifests=prompt_manifests,
        )
        snapshot = build_snapshot_for_manifest(
            context=context,
            manifest=manifest,
            visible_items=visible_items,
            config=config,
        )
        if should_compact_window(snapshot, visible_items, config):
            source_visible_items = list(visible_items)
            visible_items = compact_visible_window(
                context=context,
                visible_items=visible_items,
                manifest=manifest,
                config=config,
            )
            compact_boundary = compact_context(
                context=context,
                source_visible_items=source_visible_items,
                visible_items=visible_items,
                phase=manifest.phase,
                before_snapshot=snapshot,
                context_summarizer=context_summarizer,
            )
            compact_boundaries.append(compact_boundary)
            update_manifest_visible_items(manifest, visible_items)
            manifest.input_compressed_context_ids = list(context.compressed_context_ids)
            context.active_items = visible_items
            snapshot = build_snapshot_for_manifest(
                context=context,
                manifest=manifest,
                visible_items=visible_items,
                config=config,
            )
            compact_boundary.after_usage_ratio = snapshot.usage_ratio
        manifest.prompt_token_count = snapshot.projected_input_tokens
        usage_snapshots.append(snapshot)
        if snapshot.budget_status == "emergency_compact":
            raise EvidenceLoopError(
                f"Context remains over emergency budget for qid={context.qid}, "
                f"phase={manifest.phase}, usage_ratio={snapshot.usage_ratio:.4f}."
            )

        checker_started_at = time.monotonic()
        checker_output = call_checker(
            checker,
            context=context,
            visible_items=visible_items,
            manifest=manifest,
            round_index=round_index,
        )
        checker_output["checker_latency_ms"] = elapsed_ms(checker_started_at)
        checker_output["retrieval_latency_ms"] = 0.0
        checker_output["rerank_latency_ms"] = 0.0
        checker_output["retrieval_executed"] = False
        checker_output["retrieval_query_count"] = 0
        checker_outputs.append(checker_output)
        update_context_from_checker_output(context, checker_output)
        force_minimum_retrieval = should_force_minimum_retrieval(
            checker_output=checker_output,
            round_index=round_index,
            config=config,
        )
        checker_output["minimum_retrieval_forced"] = force_minimum_retrieval
        if checker_output.get("sufficient") and not force_minimum_retrieval:
            checker_output["round_latency_ms"] = elapsed_ms(round_started_at)
            completed = True
            break
        if round_index == config.max_rounds:
            checker_output["round_latency_ms"] = elapsed_ms(round_started_at)
            break

        query_objects = normalize_loop_queries(
            checker_output.get("next_queries"),
            max_queries=config.max_queries_per_round,
            round_index=round_index,
        )
        if not query_objects and should_build_audit_queries_without_model_queries(
            checker_output=checker_output,
            force_minimum_retrieval=force_minimum_retrieval,
        ):
            checker_output["fallback_retrieval_query_used"] = True
            query_objects = build_minimum_retrieval_queries(
                context=context,
                checker_output=checker_output,
                max_queries=config.max_queries_per_round,
                round_index=round_index,
            )
        else:
            checker_output["fallback_retrieval_query_used"] = False
        if not query_objects:
            checker_output["round_latency_ms"] = elapsed_ms(round_started_at)
            break

        retrieval_started_at = time.monotonic()
        new_candidate_rows = retrieve_new_candidates(
            query_objects=query_objects,
            context=context,
            retriever=retriever,
            chunk_lookup=chunk_lookup,
            config=config,
        )
        checker_output["retrieval_latency_ms"] = elapsed_ms(retrieval_started_at)
        retrieval_rounds += 1
        second_hop_query_count += len(query_objects)
        checker_output["retrieval_executed"] = True
        checker_output["retrieval_query_count"] = len(query_objects)
        rerank_started_at = time.monotonic()
        selected_new_rows = rerank_new_candidates(
            query_objects=query_objects,
            candidate_rows=new_candidate_rows,
            reranker=reranker,
            chunk_lookup=chunk_lookup,
            config=config,
        )
        checker_output["rerank_latency_ms"] = elapsed_ms(rerank_started_at)
        selected_new_rows = select_new_candidate_rows(selected_new_rows, config=config)
        update_gap_query_selection(context, selected_new_rows)

        next_llm_call_id = build_round_llm_call_id(context.qid, round_index + 1)
        selected_new_items = mark_selected_new_items(
            context=context,
            selected_rows=selected_new_rows,
            chunk_lookup=chunk_lookup,
            llm_call_id=next_llm_call_id,
        )
        visible_items = pack_visible_items(
            previous_visible_items=visible_items,
            selected_new_items=selected_new_items,
            llm_call_id=next_llm_call_id,
            config=config,
            priority_chunk_ids=[
                *context.eval_info.get("selected_second_hop_chunk_ids", []),
                *context.eval_info.get("checker_seen_chunk_ids", []),
            ],
        )
        context.active_items = visible_items
        context.packed_items = visible_items
        context.round_index = round_index + 1
        prompt_manifests.append(
            build_round_prompt_manifest(
                qid=context.qid,
                llm_call_id=next_llm_call_id,
                round_index=round_index + 1,
                visible_items=visible_items,
            )
        )
        checker_output["round_latency_ms"] = elapsed_ms(round_started_at)

    context.prompt_manifests = prompt_manifests
    return EvidenceCompletionLoopResult(
        context=context,
        checker_outputs=checker_outputs,
        prompt_manifests=prompt_manifests,
        usage_snapshots=usage_snapshots,
        compact_boundaries=compact_boundaries,
        completed=completed,
        rounds_completed=context.round_index,
        retrieval_rounds=retrieval_rounds,
        second_hop_query_count=second_hop_query_count,
        total_latency_ms=elapsed_ms(loop_started_at),
    )


def validate_loop_config(config: EvidenceLoopConfig) -> None:
    if config.max_rounds < 0:
        raise EvidenceLoopError("max_rounds must be non-negative.")
    if config.min_retrieval_rounds < 0:
        raise EvidenceLoopError("min_retrieval_rounds must be non-negative.")
    if config.min_retrieval_rounds > config.max_rounds:
        raise EvidenceLoopError("min_retrieval_rounds cannot exceed max_rounds.")
    if config.max_queries_per_round <= 0:
        raise EvidenceLoopError("max_queries_per_round must be positive.")
    if config.per_query_retrieve_top_k_chunks <= 0:
        raise EvidenceLoopError("per_query_retrieve_top_k_chunks must be positive.")
    if config.max_raw_chunks_per_checker_call <= 0:
        raise EvidenceLoopError("max_raw_chunks_per_checker_call must be positive.")
    if config.max_new_raw_chunks_per_round <= 0:
        raise EvidenceLoopError("max_new_raw_chunks_per_round must be positive.")
    if config.max_new_chunks_per_article <= 0:
        raise EvidenceLoopError("max_new_chunks_per_article must be positive.")
    if config.max_visible_chunks_per_article <= 0:
        raise EvidenceLoopError("max_visible_chunks_per_article must be positive.")
    if config.max_parallel_retrieval_queries <= 0:
        raise EvidenceLoopError("max_parallel_retrieval_queries must be positive.")
    if config.max_new_raw_chunks_per_round > config.max_raw_chunks_per_checker_call:
        raise EvidenceLoopError("max_new_raw_chunks_per_round cannot exceed raw chunk window.")


def should_force_minimum_retrieval(
    *,
    checker_output: dict[str, Any],
    round_index: int,
    config: EvidenceLoopConfig,
) -> bool:
    return (
        bool(checker_output.get("sufficient"))
        and round_index < config.min_retrieval_rounds
        and round_index < config.max_rounds
    )


def should_build_audit_queries_without_model_queries(
    *,
    checker_output: dict[str, Any],
    force_minimum_retrieval: bool,
) -> bool:
    if checker_output.get("next_queries"):
        return False
    if force_minimum_retrieval:
        return True
    return (
        not checker_output.get("sufficient")
        and (
            bool(checker_output.get("sufficiency_overridden"))
            or bool(checker_output.get("missing_facets"))
            or bool(checker_output.get("required_facets"))
        )
    )


def build_minimum_retrieval_queries(
    *,
    context: EvidenceContext,
    checker_output: dict[str, Any],
    max_queries: int,
    round_index: int,
) -> list[dict[str, Any]]:
    derived_from_chunk_ids = unique_texts(checker_output.get("seen_chunk_ids") or [])[:3]
    raw_queries: list[dict[str, Any]] = []
    source_text = build_audit_source_text(context.question, checker_output.get("required_facets"))
    special_queries = build_special_audit_queries(source_text)
    if special_queries:
        for query_text in special_queries:
            add_minimum_retrieval_query(
                raw_queries,
                query_text=query_text,
                derived_from_chunk_ids=derived_from_chunk_ids,
            )
    else:
        add_minimum_retrieval_query(
            raw_queries,
            query_text=context.question,
            derived_from_chunk_ids=derived_from_chunk_ids,
        )
        facet_query = build_facet_audit_query(context.question, checker_output.get("required_facets"))
        add_minimum_retrieval_query(
            raw_queries,
            query_text=facet_query,
            derived_from_chunk_ids=derived_from_chunk_ids,
        )
    return normalize_loop_queries(
        raw_queries,
        max_queries=max_queries,
        round_index=round_index,
    )


def add_minimum_retrieval_query(
    raw_queries: list[dict[str, Any]],
    *,
    query_text: str,
    derived_from_chunk_ids: list[str],
) -> None:
    query_text = clean_retrieval_query(query_text)
    if not query_text:
        return
    normalized = query_text.casefold()
    if any(str(row.get("query_text", "")).casefold() == normalized for row in raw_queries):
        return
    raw_queries.append(
        {
            "query_id": f"minimum_retrieval_audit_{len(raw_queries) + 1}",
            "query_text": query_text,
            "target_missing_facet_id": "minimum_retrieval_audit",
            "derived_from_chunk_ids": derived_from_chunk_ids,
        }
    )


def build_facet_audit_query(question: str, required_facets: Any) -> str:
    source_text = build_audit_source_text(question, required_facets)
    special_queries = build_special_audit_queries(source_text)
    if special_queries:
        return special_queries[0]
    terms = extract_action_query_terms(source_text)
    add_bridge_terms(terms, source_text)
    if terms:
        return " ".join(terms[:MAX_ACTION_QUERY_TERMS])
    return f"Wix Help Center {question}"


def build_audit_source_text(question: str, required_facets: Any) -> str:
    source_parts = []
    for facet in as_dict_list(required_facets):
        user_need = str(facet.get("user_need") or "").strip()
        facet_type = str(facet.get("facet_type") or "").strip()
        if user_need:
            source_parts.append(user_need)
        elif facet_type:
            source_parts.append(facet_type)
        if len(source_parts) == 6:
            break
    return " ".join([question, *source_parts])


def build_special_audit_queries(source_text: str) -> list[str]:
    source = source_text.casefold()
    if "payment" in source and any(token in source for token in ("region", "currency", "country")):
        return [
            "payments region currency error troubleshooting accepting payments failures country currency",
            "about payments countries available currency payment solution",
        ]
    if any(token in source for token in ("sharing", "share")) and any(
        token in source for token in ("picture", "image", "link")
    ):
        return ["social share settings pages website link picture image facebook debugger update preview"]
    if "hide" in source and "page" in source and "link" in source:
        return [
            "hide page direct link prevent search engines indexing noindex",
            "manage pages mobile editor hide page mobile version",
        ]
    if "mobile" in source and any(token in source for token in ("overlap", "overlapping", "misplaced", "elements")):
        return ["mobile editor adding customizing mobile only elements misplaced overlapping"]
    if "provider" in source and "payment" in source and ("funds" in source or "order" in source):
        return ["payment method payment provider funds sent order hotels setup"]
    return []


def extract_action_query_terms(text: str) -> list[str]:
    output = []
    seen = set()
    for token in re.findall(r"[A-Za-z0-9]+", text.casefold()):
        if len(token) <= 2:
            continue
        if token in ACTION_QUERY_STOPWORDS or token in ACTION_QUERY_PRODUCT_ANCHORS:
            continue
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def add_bridge_terms(terms: list[str], source_text: str) -> None:
    source = source_text.casefold()
    term_set = set(terms)
    if "email" in source or "emails" in source:
        append_terms(terms, term_set, ["email", "campaign"])
        if any(token in source for token in ("customize", "content", "text")):
            append_terms(terms, term_set, ["elements", "text", "image", "video", "button"])
        if any(token in source for token in ("pricing", "price", "cost", "plan", "upgrade")):
            append_terms(terms, term_set, ["upgrade", "plan", "package"])
    if "google" in source and ("analytics" in source or "ga4" in source):
        append_terms(terms, term_set, ["analytics", "4", "ga4", "universal", "upgrade", "tracking"])
    if "google" in source and (
        "products" in source or "google ads" in source or ("campaign" in source and "product" in source)
    ):
        append_terms(terms, term_set, ["google", "merchant", "products", "approved", "search", "results"])
    if "categor" in source:
        append_terms(terms, term_set, ["categories", "manage", "app"])
    if "payment" in source or "payments" in source:
        if any(token in source for token in ("region", "currency", "error", "issue", "trouble")):
            append_terms(terms, term_set, ["troubleshooting", "accepting", "payments", "failures", "country", "currency"])
            append_terms(terms, term_set, ["about", "countries", "available", "solution"])
        if any(token in source for token in ("paypal", "transaction", "account", "received", "went")):
            append_terms(terms, term_set, ["payments", "overview", "transaction", "details"])
        if "bank" in source or "details" in source:
            append_terms(terms, term_set, ["documentation", "upload", "verify", "account"])
        if "provider" in source and ("funds" in source or "order" in source):
            append_terms(terms, term_set, ["payment", "method", "hotels", "setup"])
    if "rename" in source and "link" in source:
        append_terms(terms, term_set, ["page", "url", "address"])
    if any(token in source for token in ("sharing", "share")) and any(
        token in source for token in ("picture", "image", "link")
    ):
        append_terms(terms, term_set, ["social", "share", "settings", "pages", "image", "facebook", "debugger", "preview"])
    if "hide" in source and "page" in source:
        append_terms(terms, term_set, ["prevent", "search", "engines", "indexing", "noindex", "direct", "link"])
        append_terms(terms, term_set, ["manage", "pages", "mobile", "editor", "version"])
    if any(token in source for token in ("design", "designs", "template", "templates", "pre set", "pre-set")):
        append_terms(terms, term_set, ["templates", "new", "template", "site", "choose", "available", "options"])
    if "gaps" in source and "page" in source:
        append_terms(terms, term_set, ["dynamic", "pages", "troubleshooting", "cms"])
    if "mobile" in source and any(token in source for token in ("overlap", "overlapping", "misplaced", "elements")):
        append_terms(terms, term_set, ["adding", "customizing", "mobile", "only", "elements", "editor"])
    if "domain" in source and "website" in source and any(token in source for token in ("create", "new")):
        append_terms(terms, term_set, ["getting", "started", "website", "builder"])
    if "remove" in source and any(token in source for token in ("menu", "header", "element")):
        append_terms(terms, term_set, ["delete", "something", "from", "site", "element"])


def append_terms(terms: list[str], term_set: set[str], values: list[str]) -> None:
    for value in values:
        if value not in term_set:
            term_set.add(value)
            terms.append(value)


def clean_retrieval_query(value: Any, *, max_chars: int = 280) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated or text[:max_chars].strip()


def ensure_round_manifest(
    *,
    context: EvidenceContext,
    visible_items: list[EvidenceItem],
    round_index: int,
    prompt_manifests: list[PromptManifest],
) -> PromptManifest:
    llm_call_id = build_round_llm_call_id(context.qid, round_index)
    if round_index == 0 and prompt_manifests:
        return prompt_manifests[0]
    for manifest in prompt_manifests:
        if manifest.llm_call_id == llm_call_id:
            return manifest
    manifest = build_round_prompt_manifest(
        qid=context.qid,
        llm_call_id=llm_call_id,
        round_index=round_index,
        visible_items=visible_items,
        compressed_context_ids=context.compressed_context_ids,
    )
    prompt_manifests.append(manifest)
    return manifest


def build_round_prompt_manifest(
    *,
    qid: str,
    llm_call_id: str,
    round_index: int,
    visible_items: list[EvidenceItem],
    compressed_context_ids: list[str] | None = None,
) -> PromptManifest:
    return PromptManifest(
        llm_call_id=llm_call_id,
        phase=f"evidence_gap_round_{round_index}",
        qid=qid,
        input_chunk_ids=[item.chunk_id for item in visible_items],
        input_snippet_ids=[item.snippet_id for item in visible_items],
        input_compressed_context_ids=list(compressed_context_ids or []),
    )


def build_round_llm_call_id(qid: str, round_index: int) -> str:
    return f"{qid}:evidence_gap_round:{round_index}"


def build_snapshot_for_manifest(
    *,
    context: EvidenceContext,
    manifest: PromptManifest,
    visible_items: list[EvidenceItem],
    config: EvidenceLoopConfig,
) -> ContextUsageSnapshot:
    return build_context_usage_snapshot(
        llm_call_id=manifest.llm_call_id,
        phase=manifest.phase,
        qid=context.qid,
        input_texts=[
            context.question,
            context.compressed_summary,
            *serialize_state_items(context.required_facets),
            *serialize_state_items(context.known_facts),
            *serialize_state_items(context.covered_facets),
            *serialize_state_items(context.missing_facets),
            *serialize_state_items(context.query_history),
            *[item.text_preview for item in visible_items],
        ],
        input_chunk_ids=manifest.input_chunk_ids,
        config=config.budget,
    )


def should_compact_window(
    snapshot: ContextUsageSnapshot,
    visible_items: list[EvidenceItem],
    config: EvidenceLoopConfig,
) -> bool:
    return (
        len(visible_items) > config.max_raw_chunks_per_checker_call
        or snapshot.usage_ratio >= SOFT_AUTO_COMPACT_RATIO
    )


def compact_visible_window(
    *,
    context: EvidenceContext,
    visible_items: list[EvidenceItem],
    manifest: PromptManifest,
    config: EvidenceLoopConfig,
) -> list[EvidenceItem]:
    kept_items = trim_visible_window_to_max(
        visible_items,
        max_raw_chunks=config.max_raw_chunks_per_checker_call,
        max_chunks_per_article=config.max_visible_chunks_per_article,
    )
    while len(kept_items) > 1:
        snapshot = build_snapshot_for_manifest(
            context=context,
            manifest=manifest,
            visible_items=kept_items,
            config=config,
        )
        if snapshot.usage_ratio <= config.budget.target_after_compact_ratio:
            break
        next_items = drop_one_visible_item(kept_items)
        if len(next_items) == len(kept_items):
            break
        kept_items = next_items
    for item in kept_items:
        if manifest.llm_call_id not in item.visible_to_llm_call_ids:
            item.visible_to_llm_call_ids.append(manifest.llm_call_id)
    return kept_items


def trim_visible_window_to_max(
    visible_items: list[EvidenceItem],
    *,
    max_raw_chunks: int,
    max_chunks_per_article: int = 2,
) -> list[EvidenceItem]:
    items = dedupe_items(visible_items)
    items = select_diverse_items(
        items,
        max_items=max_raw_chunks,
        max_chunks_per_article=max_chunks_per_article,
    )
    if len(items) <= max_raw_chunks:
        return items
    if max_raw_chunks <= 10:
        return items[-max_raw_chunks:]
    anchored = items[:10]
    newest = items[-(max_raw_chunks - len(anchored)) :]
    return dedupe_items([*anchored, *newest])


def drop_one_visible_item(visible_items: list[EvidenceItem]) -> list[EvidenceItem]:
    if len(visible_items) <= 1:
        return visible_items
    drop_index = min(
        range(len(visible_items)),
        key=lambda index: visible_item_importance(visible_items[index], index=index),
    )
    return [item for index, item in enumerate(visible_items) if index != drop_index]


def visible_item_importance(item: EvidenceItem, *, index: int) -> tuple[float, int]:
    score = 0.0
    if item.second_hop_query_ids:
        score += 50.0
    if item.rerank_score is not None:
        score += min(max(float(item.rerank_score), 0.0), 100.0)
    elif item.score is not None:
        score += min(max(float(item.score), 0.0), 100.0)
    if item.rerank_rank is not None:
        score += max(0.0, 25.0 - float(item.rerank_rank))
    elif item.rank is not None:
        score += max(0.0, 25.0 - float(item.rank))
    if index < 3:
        score += 10.0
    return (score, -index)


def update_manifest_visible_items(manifest: PromptManifest, visible_items: list[EvidenceItem]) -> None:
    manifest.input_chunk_ids = [item.chunk_id for item in visible_items]
    manifest.input_snippet_ids = [item.snippet_id for item in visible_items]


def compact_context(
    *,
    context: EvidenceContext,
    source_visible_items: list[EvidenceItem],
    visible_items: list[EvidenceItem],
    phase: str,
    before_snapshot: ContextUsageSnapshot,
    context_summarizer: Any | None = None,
) -> CompactBoundary:
    compact_id = f"{context.qid}:compact:{len(context.compressed_context_ids) + 1}"
    kept_raw_chunk_ids = [item.chunk_id for item in visible_items]
    source_chunk_ids = [item.chunk_id for item in source_visible_items]
    kept_raw_chunk_id_set = set(kept_raw_chunk_ids)
    dropped_items = [item for item in source_visible_items if item.chunk_id not in kept_raw_chunk_id_set]
    dropped_chunk_ids = [item.chunk_id for item in dropped_items]
    summary, compression_used_api, compression_error = summarize_dropped_context(
        context_summarizer=context_summarizer,
        context=context,
        dropped_items=dropped_items,
        compact_id=compact_id,
    )
    if summary:
        context.compressed_summary = summary
        context.compressed_context_ids.append(compact_id)
    context.known_facts = []
    context.required_facets = context.required_facets[-8:]
    context.covered_facets = []
    context.missing_facets = context.missing_facets[-5:]
    context.query_history = context.query_history[-8:]
    return CompactBoundary(
        compact_id=compact_id,
        qid=context.qid,
        phase=phase,
        source_chunk_ids=source_chunk_ids,
        kept_raw_chunk_ids=kept_raw_chunk_ids,
        compressed_context_ids=[compact_id] if summary else [],
        dropped_chunk_ids=dropped_chunk_ids,
        source_to_summary_map={chunk_id: compact_id for chunk_id in dropped_chunk_ids} if summary else {},
        compression_used_api=compression_used_api,
        compression_fallback_used=bool(dropped_chunk_ids and not summary),
        compression_error=compression_error,
        before_usage_ratio=before_snapshot.usage_ratio,
    )


def summarize_dropped_context(
    *,
    context_summarizer: Any | None,
    context: EvidenceContext,
    dropped_items: list[EvidenceItem],
    compact_id: str,
) -> tuple[str, bool, str | None]:
    if not dropped_items:
        return "", False, None
    if context_summarizer is None:
        return "", False, "context_summarizer_not_configured"
    try:
        if hasattr(context_summarizer, "summarize"):
            summary = context_summarizer.summarize(
                context=context,
                dropped_items=dropped_items,
                existing_summary=context.compressed_summary,
                compact_id=compact_id,
            )
        else:
            summary = context_summarizer(context, dropped_items, compact_id)
        summary_text = compact_text(summary)
        if not summary_text:
            return "", bool(getattr(context_summarizer, "uses_api", False)), "empty_compressed_summary"
        return summary_text, bool(getattr(context_summarizer, "uses_api", False)), None
    except Exception as exc:
        return "", bool(getattr(context_summarizer, "uses_api", False)), compact_text(str(exc))[:240]


def build_deterministic_context_summary(
    context: EvidenceContext,
    *,
    dropped_items: list[EvidenceItem] | None = None,
) -> str:
    lines = []
    if context.compressed_summary:
        lines.append(context.compressed_summary)
    if dropped_items:
        dropped_lines = []
        for item in dropped_items[:20]:
            title = item.title or "(untitled)"
            dropped_lines.append(
                f"{item.chunk_id} | {title}: {item.text_preview[:240]}"
            )
        if dropped_lines:
            lines.append("Compressed raw evidence: " + " ; ".join(dropped_lines))
    if context.known_facts:
        lines.append("Known facts: " + "; ".join(serialize_state_items(context.known_facts)[:20]))
    if context.required_facets:
        lines.append("Required facets: " + "; ".join(serialize_state_items(context.required_facets)[-10:]))
    if context.covered_facets:
        lines.append("Covered facets: " + "; ".join(serialize_state_items(context.covered_facets)[:20]))
    if context.missing_facets:
        lines.append("Open missing facets: " + "; ".join(serialize_state_items(context.missing_facets)[-10:]))
    if context.query_history:
        lines.append("Recent queries: " + "; ".join(serialize_state_items(context.query_history)[-8:]))
    return "\n".join(line for line in lines if line).strip()


def serialize_state_items(items: list[Any]) -> list[str]:
    output = []
    for item in items:
        if isinstance(item, dict):
            text = (
                item.get("description")
                or item.get("text")
                or item.get("query_text")
                or item.get("reason")
                or str(item)
            )
        else:
            text = str(item)
        text = " ".join(str(text).split())
        if text:
            output.append(text)
    return output


def call_checker(
    checker: Any,
    *,
    context: EvidenceContext,
    visible_items: list[EvidenceItem],
    manifest: PromptManifest,
    round_index: int,
) -> dict[str, Any]:
    if hasattr(checker, "check"):
        output = checker.check(
            context=context,
            visible_items=visible_items,
            manifest=manifest,
            round_index=round_index,
        )
    else:
        output = checker(context, visible_items, manifest, round_index)
    if not isinstance(output, dict):
        raise EvidenceLoopError("checker must return a dict.")
    return output


def update_context_from_checker_output(
    context: EvidenceContext,
    checker_output: dict[str, Any],
) -> None:
    seen_chunk_ids = context.eval_info.setdefault("checker_seen_chunk_ids", [])
    for chunk_id in checker_output.get("seen_chunk_ids") or []:
        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.append(chunk_id)
    context.required_facets.extend(as_dict_list(checker_output.get("required_facets")))
    context.known_facts.extend(as_dict_list(checker_output.get("known_facts")))
    context.covered_facets.extend(as_dict_list(checker_output.get("covered_facets")))
    context.missing_facets.extend(as_dict_list(checker_output.get("missing_facets")))


def as_dict_list(value: Any) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    output = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            output.append(item)
        elif item:
            output.append({"id": f"item_{index}", "text": str(item)})
    return output


def normalize_loop_queries(
    value: Any,
    *,
    max_queries: int,
    round_index: int,
) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    output = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            query_text = str(item.get("query_text") or item.get("query") or "").strip()
            source_query_id = str(item.get("query_id") or f"query_{index}")
            target_missing_facet_id = str(item.get("target_missing_facet_id") or "").strip()
            derived_from_chunk_ids = list(item.get("derived_from_chunk_ids") or [])
        else:
            query_text = str(item).strip()
            source_query_id = f"query_{index}"
            target_missing_facet_id = ""
            derived_from_chunk_ids = []
        if not query_text:
            continue
        query_id = f"round_{round_index}_query_{len(output) + 1}"
        output.append(
            {
                "query_id": query_id,
                "source_query_id": source_query_id,
                "query_text": query_text,
                "target_missing_facet_id": target_missing_facet_id,
                "derived_from_chunk_ids": derived_from_chunk_ids,
            }
        )
        if len(output) == max_queries:
            break
    return output


def retrieve_new_candidates(
    *,
    query_objects: list[dict[str, Any]],
    context: EvidenceContext,
    retriever: Any,
    chunk_lookup: dict[str, KBChunk],
    config: EvidenceLoopConfig,
) -> list[dict[str, Any]]:
    item_by_chunk_id = {item.chunk_id: item for item in context.candidate_items}
    active_chunk_ids = {item.chunk_id for item in context.active_items}
    seen_query_chunk_pairs = set()
    new_candidate_rows = []
    rows_by_query_id = retrieve_queries(
        retriever,
        query_objects,
        top_k_chunks=config.per_query_retrieve_top_k_chunks,
        max_workers=config.max_parallel_retrieval_queries,
    )
    for query_object in query_objects:
        context.query_history.append(query_object)
        context.gap_query_provenance.append(
            provenance := GapQueryProvenance(
                query_id=query_object["query_id"],
                target_missing_facet_id=query_object.get("target_missing_facet_id") or None,
                query_text=query_object["query_text"],
                derived_from_chunk_ids=query_object.get("derived_from_chunk_ids") or [],
            )
        )
        rows = rows_by_query_id.get(query_object["query_id"], [])
        provenance.retrieved_chunk_ids = [
            str(row.get("chunk_id"))
            for row in rows[: config.per_query_retrieve_top_k_chunks]
            if row.get("chunk_id")
        ]
        provenance.retrieved_article_ids = unique_texts(
            row.get("article_id")
            for row in rows[: config.per_query_retrieve_top_k_chunks]
            if row.get("article_id")
        )
        for rank, row in enumerate(rows[: config.per_query_retrieve_top_k_chunks], start=1):
            candidate_row = attach_query_metadata(
                {**row, "rank": row.get("rank") or rank},
                query_object,
            )
            chunk_id = candidate_row.get("chunk_id")
            if not chunk_id:
                continue
            query_chunk_pair = (query_object["query_id"], chunk_id)
            if query_chunk_pair in seen_query_chunk_pairs:
                continue
            seen_query_chunk_pairs.add(query_chunk_pair)
            existing_item = item_by_chunk_id.get(chunk_id)
            if existing_item is not None:
                append_query_id_to_item(existing_item, query_object["query_id"])
                if chunk_id in active_chunk_ids:
                    continue
                new_candidate_rows.append(enrich_existing_item_candidate_row(candidate_row, existing_item))
                continue
            item = build_evidence_item(
                candidate_row,
                chunk_lookup,
                source="second_hop_candidate",
                active=False,
            )
            item.second_hop_query_ids.append(query_object["query_id"])
            context.candidate_items.append(item)
            item_by_chunk_id[chunk_id] = item
            new_candidate_rows.append(candidate_row)
        query_candidate_rows = [
            row
            for row in new_candidate_rows
            if row.get("second_hop_query_id") == query_object["query_id"]
        ]
        provenance.candidate_chunk_ids = unique_texts(row.get("chunk_id") for row in query_candidate_rows)
        provenance.candidate_article_ids = unique_texts(row.get("article_id") for row in query_candidate_rows)
    return new_candidate_rows


def attach_query_metadata(row: dict[str, Any], query_object: dict[str, Any]) -> dict[str, Any]:
    query_id = query_object["query_id"]
    query_text = query_object["query_text"]
    query_ids = list(row.get("second_hop_query_ids") or [])
    if query_id not in query_ids:
        query_ids.append(query_id)
    return {
        **row,
        "second_hop_query_id": query_id,
        "second_hop_query_text": query_text,
        "second_hop_query_ids": query_ids,
        "second_hop_retrieval_rank": row.get("rank"),
        "target_missing_facet_id": query_object.get("target_missing_facet_id") or "",
        "derived_from_chunk_ids": query_object.get("derived_from_chunk_ids") or [],
    }


def retrieve_queries(
    retriever: Any,
    query_objects: list[dict[str, Any]],
    *,
    top_k_chunks: int,
    max_workers: int,
) -> dict[str, list[dict[str, Any]]]:
    if not query_objects:
        return {}
    if hasattr(retriever, "search_batch"):
        if search_batch_accepts_top_k_chunks(retriever):
            batches = retriever.search_batch(
                [query_object["query_text"] for query_object in query_objects],
                top_k_chunks=top_k_chunks,
            )
            return map_batch_results_to_query_ids(query_objects, batches)
        if not hasattr(retriever, "search"):
            raise EvidenceLoopError(
                "Retriever search_batch must accept top_k_chunks or be wrapped by an adapter."
            )
    return retrieve_queries_individually(
        retriever,
        query_objects,
        top_k_chunks=top_k_chunks,
        max_workers=max_workers,
    )


def search_batch_accepts_top_k_chunks(retriever: Any) -> bool:
    try:
        parameters = inspect.signature(retriever.search_batch).parameters
    except (TypeError, ValueError):
        return True
    return (
        "top_k_chunks" in parameters
        or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    )


def map_batch_results_to_query_ids(
    query_objects: list[dict[str, Any]],
    batches: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    if len(batches) != len(query_objects):
        raise EvidenceLoopError(
            "Retriever batch result count does not match query count: "
            f"results={len(batches)}, queries={len(query_objects)}."
        )
    return {
        query_object["query_id"]: normalize_retrieval_rows(rows)
        for query_object, rows in zip(query_objects, batches)
    }


def retrieve_queries_individually(
    retriever: Any,
    query_objects: list[dict[str, Any]],
    *,
    top_k_chunks: int,
    max_workers: int,
) -> dict[str, list[dict[str, Any]]]:
    if len(query_objects) == 1 or max_workers <= 1:
        rows_by_query = [
            retrieve_query(
                retriever,
                query_object["query_text"],
                top_k_chunks=top_k_chunks,
            )
            for query_object in query_objects
        ]
    else:
        workers = min(max_workers, len(query_objects))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows_by_query = list(
                executor.map(
                    lambda query_object: retrieve_query(
                        retriever,
                        query_object["query_text"],
                        top_k_chunks=top_k_chunks,
                    ),
                    query_objects,
                )
            )
    return {
        query_object["query_id"]: rows
        for query_object, rows in zip(query_objects, rows_by_query)
    }


def normalize_retrieval_rows(rows: Any) -> list[dict[str, Any]]:
    if isinstance(rows, dict):
        rows = rows.get("hybrid") or rows.get("results") or []
    return list(rows)


def enrich_existing_item_candidate_row(
    row: dict[str, Any],
    item: EvidenceItem,
) -> dict[str, Any]:
    return {
        **row,
        "chunk_id": item.chunk_id,
        "article_id": row.get("article_id") or item.article_id,
        "snippet_id": row.get("snippet_id") or item.snippet_id,
        "title": row.get("title") or item.title,
        "text_preview": row.get("text_preview") or item.text_preview,
        "source": row.get("source") or "second_hop_existing_candidate",
    }


def retrieve_query(retriever: Any, query_text: str, *, top_k_chunks: int) -> list[dict[str, Any]]:
    if hasattr(retriever, "search"):
        rows = retriever.search(query_text, top_k_chunks=top_k_chunks)
    elif hasattr(retriever, "search_batch"):
        batch = retriever.search_batch([query_text], top_k_chunks=top_k_chunks)
        rows = batch[0] if batch else []
    else:
        rows = retriever(query_text, top_k_chunks)
    return normalize_retrieval_rows(rows)


def append_query_id_to_existing_item(
    context: EvidenceContext,
    *,
    chunk_id: str,
    query_id: str,
) -> None:
    for item in context.candidate_items:
        if item.chunk_id == chunk_id:
            append_query_id_to_item(item, query_id)
            return


def append_query_id_to_item(item: EvidenceItem, query_id: str) -> None:
    if query_id not in item.second_hop_query_ids:
        item.second_hop_query_ids.append(query_id)


def rerank_new_candidates(
    *,
    query_objects: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    reranker: Any,
    chunk_lookup: dict[str, KBChunk],
    config: EvidenceLoopConfig,
) -> list[dict[str, Any]]:
    if not candidate_rows:
        return []
    rows_by_query_id = group_candidate_rows_by_query_id(candidate_rows)
    scored_rows = []
    for query_object in query_objects:
        query_id = query_object["query_id"]
        query_text = query_object["query_text"]
        query_rows = rows_by_query_id.get(query_id, [])
        if not query_rows:
            continue
        for row in call_reranker(
            reranker=reranker,
            query_text=query_text,
            candidate_rows=query_rows,
            chunk_lookup=chunk_lookup,
            config=config,
        ):
            scored_rows.append(
                {
                    **row,
                    "second_hop_rerank_query_id": query_id,
                    "second_hop_rerank_query_text": query_text,
                }
            )
    return dedupe_ranked_candidate_rows(scored_rows)


def call_reranker(
    *,
    reranker: Any,
    query_text: str,
    candidate_rows: list[dict[str, Any]],
    chunk_lookup: dict[str, KBChunk],
    config: EvidenceLoopConfig,
) -> list[dict[str, Any]]:
    if hasattr(reranker, "rerank"):
        try:
            return list(
                reranker.rerank(
                    query_text,
                    candidate_rows,
                    chunk_lookup,
                    batch_size=config.rerank_batch_size,
                )
            )
        except TypeError:
            return list(reranker.rerank(query_text, candidate_rows, chunk_lookup))
    return list(reranker(query_text, candidate_rows, chunk_lookup))


def update_gap_query_selection(
    context: EvidenceContext,
    selected_rows: list[dict[str, Any]],
) -> None:
    selected_by_query_id: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        query_id = row.get("second_hop_rerank_query_id") or row.get("second_hop_query_id")
        if query_id:
            selected_by_query_id.setdefault(str(query_id), []).append(row)
    for provenance in context.gap_query_provenance:
        rows = selected_by_query_id.get(provenance.query_id, [])
        if not rows:
            continue
        provenance.selected_chunk_ids = unique_texts(row.get("chunk_id") for row in rows)
        provenance.selected_article_ids = unique_texts(row.get("article_id") for row in rows)


def mark_selected_new_items(
    *,
    context: EvidenceContext,
    selected_rows: list[dict[str, Any]],
    chunk_lookup: dict[str, KBChunk],
    llm_call_id: str,
) -> list[EvidenceItem]:
    selected_items = []
    item_by_chunk_id = {item.chunk_id: item for item in context.candidate_items}
    for row in selected_rows:
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        item = item_by_chunk_id.get(chunk_id)
        if item is None:
            item = build_evidence_item(
                row,
                chunk_lookup,
                source="reranked_new_candidate",
                active=True,
                llm_call_id=llm_call_id,
            )
            context.candidate_items.append(item)
        item.active = True
        item.rerank_rank = int(row["rank"]) if row.get("rank") is not None else item.rerank_rank
        score = row.get("rerank_score", row.get("score"))
        item.rerank_score = float(score) if score is not None else item.rerank_score
        if llm_call_id not in item.visible_to_llm_call_ids:
            item.visible_to_llm_call_ids.append(llm_call_id)
        selected_chunk_ids = context.eval_info.setdefault("selected_second_hop_chunk_ids", [])
        if item.chunk_id not in selected_chunk_ids:
            selected_chunk_ids.append(item.chunk_id)
        selected_items.append(item)
    return selected_items


def unique_texts(values: Any) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
