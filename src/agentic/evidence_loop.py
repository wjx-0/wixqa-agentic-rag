from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.agentic.context_budget import (
    ContextBudgetConfig,
    ContextUsageSnapshot,
    SOFT_AUTO_COMPACT_RATIO,
    build_context_usage_snapshot,
)
from src.agentic.evidence_context import EvidenceContext, EvidenceItem, build_evidence_item
from src.agentic.provenance import CompactBoundary, GapQueryProvenance, PromptManifest
from src.data.schema import KBChunk


class EvidenceLoopConfig(BaseModel):
    max_rounds: int = 4
    max_queries_per_round: int = 2
    per_query_retrieve_top_k_chunks: int = 20
    max_raw_chunks_per_checker_call: int = 10
    max_new_raw_chunks_per_round: int = 5
    rerank_batch_size: int = 32
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


class EvidenceLoopError(RuntimeError):
    pass


def run_evidence_completion_loop(
    *,
    context: EvidenceContext,
    checker: Any,
    retriever: Any,
    reranker: Any,
    chunk_lookup: dict[str, KBChunk],
    config: EvidenceLoopConfig | None = None,
) -> EvidenceCompletionLoopResult:
    config = config or EvidenceLoopConfig()
    validate_loop_config(config)

    visible_items = list(context.active_items[: config.max_raw_chunks_per_checker_call])
    prompt_manifests = list(context.prompt_manifests)
    usage_snapshots: list[ContextUsageSnapshot] = []
    compact_boundaries: list[CompactBoundary] = []
    checker_outputs = []
    completed = False
    retrieval_rounds = 0
    second_hop_query_count = 0

    for round_index in range(config.max_rounds + 1):
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
        if should_compact(snapshot):
            compact_boundary = compact_context(
                context=context,
                visible_items=visible_items,
                phase=manifest.phase,
                before_snapshot=snapshot,
            )
            compact_boundaries.append(compact_boundary)
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

        checker_output = call_checker(
            checker,
            context=context,
            visible_items=visible_items,
            manifest=manifest,
            round_index=round_index,
        )
        checker_output["retrieval_executed"] = False
        checker_output["retrieval_query_count"] = 0
        checker_outputs.append(checker_output)
        update_context_from_checker_output(context, checker_output)
        if checker_output.get("sufficient"):
            completed = True
            break
        if round_index == config.max_rounds:
            break

        query_objects = normalize_loop_queries(
            checker_output.get("next_queries"),
            max_queries=config.max_queries_per_round,
        )
        if not query_objects:
            break

        new_candidate_rows = retrieve_new_candidates(
            query_objects=query_objects,
            context=context,
            retriever=retriever,
            chunk_lookup=chunk_lookup,
            config=config,
        )
        retrieval_rounds += 1
        second_hop_query_count += len(query_objects)
        checker_output["retrieval_executed"] = True
        checker_output["retrieval_query_count"] = len(query_objects)
        selected_new_rows = rerank_new_candidates(
            question=context.question,
            candidate_rows=new_candidate_rows,
            reranker=reranker,
            chunk_lookup=chunk_lookup,
            config=config,
        )[: config.max_new_raw_chunks_per_round]

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
            max_raw_chunks=config.max_raw_chunks_per_checker_call,
            llm_call_id=next_llm_call_id,
        )
        context.active_items = visible_items
        context.round_index = round_index + 1
        prompt_manifests.append(
            build_round_prompt_manifest(
                qid=context.qid,
                llm_call_id=next_llm_call_id,
                round_index=round_index + 1,
                visible_items=visible_items,
            )
        )

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
    )


def validate_loop_config(config: EvidenceLoopConfig) -> None:
    if config.max_rounds < 0:
        raise EvidenceLoopError("max_rounds must be non-negative.")
    if config.max_queries_per_round <= 0:
        raise EvidenceLoopError("max_queries_per_round must be positive.")
    if config.per_query_retrieve_top_k_chunks <= 0:
        raise EvidenceLoopError("per_query_retrieve_top_k_chunks must be positive.")
    if config.max_raw_chunks_per_checker_call <= 0:
        raise EvidenceLoopError("max_raw_chunks_per_checker_call must be positive.")
    if config.max_new_raw_chunks_per_round <= 0:
        raise EvidenceLoopError("max_new_raw_chunks_per_round must be positive.")
    if config.max_new_raw_chunks_per_round > config.max_raw_chunks_per_checker_call:
        raise EvidenceLoopError("max_new_raw_chunks_per_round cannot exceed raw chunk budget.")


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
    )
    prompt_manifests.append(manifest)
    return manifest


def build_round_prompt_manifest(
    *,
    qid: str,
    llm_call_id: str,
    round_index: int,
    visible_items: list[EvidenceItem],
) -> PromptManifest:
    return PromptManifest(
        llm_call_id=llm_call_id,
        phase=f"evidence_gap_round_{round_index}",
        qid=qid,
        input_chunk_ids=[item.chunk_id for item in visible_items],
        input_snippet_ids=[item.snippet_id for item in visible_items],
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
            *serialize_state_items(context.known_facts),
            *serialize_state_items(context.covered_facets),
            *serialize_state_items(context.missing_facets),
            *serialize_state_items(context.query_history),
            *[item.text_preview for item in visible_items],
        ],
        input_chunk_ids=manifest.input_chunk_ids,
        config=config.budget,
    )


def should_compact(snapshot: ContextUsageSnapshot) -> bool:
    return snapshot.usage_ratio >= SOFT_AUTO_COMPACT_RATIO


def compact_context(
    *,
    context: EvidenceContext,
    visible_items: list[EvidenceItem],
    phase: str,
    before_snapshot: ContextUsageSnapshot,
) -> CompactBoundary:
    compact_id = f"{context.qid}:compact:{len(context.compressed_context_ids) + 1}"
    kept_raw_chunk_ids = [item.chunk_id for item in visible_items]
    source_chunk_ids = sorted(
        {
            chunk_id
            for manifest in context.prompt_manifests
            for chunk_id in manifest.input_chunk_ids
        }
    )
    dropped_chunk_ids = [chunk_id for chunk_id in source_chunk_ids if chunk_id not in kept_raw_chunk_ids]
    summary = build_deterministic_context_summary(context)
    if summary:
        context.compressed_summary = summary
        context.compressed_context_ids.append(compact_id)
    context.known_facts = []
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
        source_to_summary_map={chunk_id: compact_id for chunk_id in dropped_chunk_ids},
        before_usage_ratio=before_snapshot.usage_ratio,
    )


def build_deterministic_context_summary(context: EvidenceContext) -> str:
    lines = []
    if context.compressed_summary:
        lines.append(context.compressed_summary)
    if context.known_facts:
        lines.append("Known facts: " + "; ".join(serialize_state_items(context.known_facts)[:20]))
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


def normalize_loop_queries(value: Any, *, max_queries: int) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    output = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            query_text = str(item.get("query_text") or item.get("query") or "").strip()
            query_id = str(item.get("query_id") or f"round_query_{index}")
            target_missing_facet_id = str(item.get("target_missing_facet_id") or "").strip()
            derived_from_chunk_ids = list(item.get("derived_from_chunk_ids") or [])
        else:
            query_text = str(item).strip()
            query_id = f"round_query_{index}"
            target_missing_facet_id = ""
            derived_from_chunk_ids = []
        if not query_text:
            continue
        output.append(
            {
                "query_id": query_id,
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
    existing_chunk_ids = {item.chunk_id for item in context.candidate_items}
    new_candidate_rows = []
    for query_object in query_objects:
        context.query_history.append(query_object)
        context.gap_query_provenance.append(
            GapQueryProvenance(
                query_id=query_object["query_id"],
                target_missing_facet_id=query_object.get("target_missing_facet_id") or None,
                query_text=query_object["query_text"],
                derived_from_chunk_ids=query_object.get("derived_from_chunk_ids") or [],
            )
        )
        rows = retrieve_query(
            retriever,
            query_object["query_text"],
            top_k_chunks=config.per_query_retrieve_top_k_chunks,
        )
        for rank, row in enumerate(rows[: config.per_query_retrieve_top_k_chunks], start=1):
            candidate_row = {**row, "rank": row.get("rank") or rank}
            chunk_id = candidate_row.get("chunk_id")
            if not chunk_id:
                continue
            if chunk_id in existing_chunk_ids:
                append_query_id_to_existing_item(
                    context,
                    chunk_id=chunk_id,
                    query_id=query_object["query_id"],
                )
                continue
            item = build_evidence_item(
                candidate_row,
                chunk_lookup,
                source="second_hop_candidate",
                active=False,
            )
            item.second_hop_query_ids.append(query_object["query_id"])
            context.candidate_items.append(item)
            existing_chunk_ids.add(chunk_id)
            new_candidate_rows.append(candidate_row)
    return new_candidate_rows


def retrieve_query(retriever: Any, query_text: str, *, top_k_chunks: int) -> list[dict[str, Any]]:
    if hasattr(retriever, "search"):
        rows = retriever.search(query_text, top_k_chunks=top_k_chunks)
    elif hasattr(retriever, "search_batch"):
        batch = retriever.search_batch([query_text], top_k_chunks=top_k_chunks)
        rows = batch[0] if batch else []
    else:
        rows = retriever(query_text, top_k_chunks)
    if isinstance(rows, dict):
        rows = rows.get("hybrid") or rows.get("results") or []
    return list(rows)


def append_query_id_to_existing_item(
    context: EvidenceContext,
    *,
    chunk_id: str,
    query_id: str,
) -> None:
    for item in context.candidate_items:
        if item.chunk_id == chunk_id and query_id not in item.second_hop_query_ids:
            item.second_hop_query_ids.append(query_id)
            return


def rerank_new_candidates(
    *,
    question: str,
    candidate_rows: list[dict[str, Any]],
    reranker: Any,
    chunk_lookup: dict[str, KBChunk],
    config: EvidenceLoopConfig,
) -> list[dict[str, Any]]:
    if not candidate_rows:
        return []
    if hasattr(reranker, "rerank"):
        try:
            return list(
                reranker.rerank(
                    question,
                    candidate_rows,
                    chunk_lookup,
                    batch_size=config.rerank_batch_size,
                )
            )
        except TypeError:
            return list(reranker.rerank(question, candidate_rows, chunk_lookup))
    return list(reranker(question, candidate_rows, chunk_lookup))


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
        selected_items.append(item)
    return selected_items


def pack_visible_items(
    *,
    previous_visible_items: list[EvidenceItem],
    selected_new_items: list[EvidenceItem],
    max_raw_chunks: int,
    llm_call_id: str,
) -> list[EvidenceItem]:
    new_items = dedupe_items(selected_new_items)[:max_raw_chunks]
    keep_count = max_raw_chunks - len(new_items)
    kept_previous = dedupe_items(previous_visible_items)[:keep_count]
    visible_items = kept_previous + new_items
    for item in visible_items:
        if llm_call_id not in item.visible_to_llm_call_ids:
            item.visible_to_llm_call_ids.append(llm_call_id)
    return visible_items


def dedupe_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    output = []
    seen = set()
    for item in items:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        output.append(item)
    return output
