from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from src.agentic.agents.dialogue_agent import DialogueAgent
from src.agentic.agents.answer_agent import AnswerAgent
from src.agentic.agents.evidence_agent import EvidenceAgent
from src.agentic.agents.query_agent import (
    LLMConversationMemorySummarizer,
    LLMQueryRewriter,
    QueryAgent,
)
from src.agentic.agents.verifier_agent import LLMClaimChecker, VerifierAgent
from src.agentic.dialogue_orchestrator import DialogueOrchestrator
from src.agentic.dialogue_state import LatencyConfig
from src.agentic.evidence_context import EvidenceContext, build_initial_evidence_context
from src.agentic.evidence_loop import EvidenceCompletionLoopResult
from src.agentic.intent_gate import LLMIntentGate
from src.data.schema import KBChunk
from src.llm.answer_generator import LLMAnswerGenerator
from src.llm.context_compressor import LLMContextCompressor
from src.llm.evidence_checker import (
    CompactEvidenceChecker,
    OpenAICompatibleChatClient,
    RetryingEvidenceChecker,
)
from src.rerankers.cross_encoder_reranker import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_INSTRUCTION,
    CrossEncoderReranker,
)
from src.rerankers.dashscope_reranker import (
    DEFAULT_DASHSCOPE_RERANK_TIMEOUT,
    DashScopeReranker,
)
from src.retrievers.dense_faiss_retriever import DEFAULT_DENSE_MODEL_NAME
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.rrf import DEFAULT_BM25_WEIGHT
from src.utils.env_utils import (
    dashscope_rerank_model_from_env,
    dashscope_rerank_url_from_env,
    deepseek_base_url_from_env,
    load_project_env,
)
from src.utils.io_utils import read_jsonl
from src.utils.text_utils import compact_text


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = "data/processed/wix_kb_chunks.jsonl"
DEFAULT_INDEX_DIR = "indexes/faiss_bge_m3"
DEFAULT_BRANCH_TOP_K_CHUNKS = 50
DEFAULT_INITIAL_RERANK_TOP_K_CHUNKS = 50
DEFAULT_DENSE_WEIGHT = 2.0


class DialogueRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DialogueRuntimeConfig:
    root: Path = DEFAULT_PROJECT_ROOT
    chunks_path: str = DEFAULT_CHUNKS_PATH
    index_dir: str = DEFAULT_INDEX_DIR
    dense_model_name: str = DEFAULT_DENSE_MODEL_NAME
    local_files_only: bool = True
    dense_worker_mode: str = "model_only"
    device: str | None = None
    branch_top_k_chunks: int = DEFAULT_BRANCH_TOP_K_CHUNKS
    initial_rerank_top_k_chunks: int = DEFAULT_INITIAL_RERANK_TOP_K_CHUNKS
    rrf_k: int = 60
    bm25_weight: float = DEFAULT_BM25_WEIGHT
    dense_weight: float = DEFAULT_DENSE_WEIGHT
    dense_query_batch_size: int = 16
    reranker_provider: str = "auto"
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE
    reranker_instruction: str = DEFAULT_RERANK_INSTRUCTION
    llm_temperature: float = 0.0
    checker_max_tokens: int = 1024
    checker_retry_attempts: int = 2
    answer_max_tokens: int = 1600
    answer_temperature: float = 0.1
    enable_api_intent_gate: bool = True
    intent_gate_max_tokens: int = 512
    intent_gate_temperature: float = 0.0
    enable_api_query_rewrite: bool = True
    query_rewrite_max_tokens: int = 512
    query_rewrite_temperature: float = 0.0


class DialogueRuntime:
    def __init__(
        self,
        *,
        orchestrator: DialogueOrchestrator,
        closeables: list[Any] | None = None,
        config: DialogueRuntimeConfig | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.closeables = list(closeables or [])
        self.config = config

    def close(self) -> None:
        for resource in reversed(self.closeables):
            if hasattr(resource, "__exit__"):
                resource.__exit__(None, None, None)
            elif hasattr(resource, "close"):
                resource.close()

    def __enter__(self) -> DialogueRuntime:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def build_real_dialogue_runtime(
    config: DialogueRuntimeConfig | None = None,
) -> DialogueRuntime:
    config = config or DialogueRuntimeConfig()
    load_project_env(config.root)
    chunks = load_chunks(config.root / config.chunks_path)
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    retriever = HybridRetriever(
        chunks=chunks,
        index_dir=config.root / config.index_dir,
        model_name=config.dense_model_name,
        local_files_only=config.local_files_only,
        device=config.device,
        dense_worker_mode=config.dense_worker_mode,
    )
    retriever.__enter__()
    reranker = build_reranker(config)
    llm_client = build_llm_client()
    dialogue_agent = (
        DialogueAgent(
            intent_gate=LLMIntentGate(
                client=llm_client,
                temperature=config.intent_gate_temperature,
                max_tokens=config.intent_gate_max_tokens,
            )
        )
        if config.enable_api_intent_gate
        else DialogueAgent()
    )
    query_agent = (
        QueryAgent(
            rewriter=LLMQueryRewriter(
                client=llm_client,
                temperature=config.query_rewrite_temperature,
                max_tokens=config.query_rewrite_max_tokens,
            ),
            memory_summarizer=LLMConversationMemorySummarizer(
                client=llm_client,
                temperature=config.query_rewrite_temperature,
                max_tokens=config.query_rewrite_max_tokens,
            ),
        )
        if config.enable_api_query_rewrite
        else QueryAgent()
    )
    checker = RetryingEvidenceChecker(
        CompactEvidenceChecker(
            client=llm_client,
            temperature=config.llm_temperature,
            max_tokens=config.checker_max_tokens,
            max_next_queries=1,
            context_preview_chars=350,
        ),
        max_attempts=config.checker_retry_attempts,
    )
    context_builder = LiveContextBuilder(
        retriever=retriever,
        reranker=reranker,
        chunk_lookup=chunk_lookup,
        config=config,
    )
    evidence_agent = EvidenceAgent(
        context_builder=context_builder.build,
        checker=checker,
        retriever=RuntimeHybridRetrieverAdapter(retriever, config),
        reranker=reranker,
        chunk_lookup=chunk_lookup,
        context_summarizer=LLMContextCompressor(
            client=llm_client,
            temperature=config.llm_temperature,
            max_tokens=config.checker_max_tokens,
        ),
    )
    answer_agent = AnswerAgent(
        generator=LLMAnswerGenerator(
            client=llm_client,
            temperature=config.answer_temperature,
            max_tokens=config.answer_max_tokens,
        )
    )
    orchestrator = DialogueOrchestrator(
        dialogue_agent=dialogue_agent,
        query_agent=query_agent,
        evidence_agent=evidence_agent,
        answer_agent=answer_agent,
        verifier_agent=VerifierAgent(
            claim_checker=LLMClaimChecker(
                client=llm_client,
                temperature=config.llm_temperature,
                max_tokens=config.checker_max_tokens,
            )
        ),
        latency=LatencyConfig(
            online_checker_mode="compact",
            online_llm_max_tokens=config.checker_max_tokens,
        ),
    )
    return DialogueRuntime(orchestrator=orchestrator, closeables=[retriever], config=config)


def build_demo_dialogue_runtime() -> DialogueRuntime:
    orchestrator = DialogueOrchestrator(
        evidence_agent=EvidenceAgent(
            context_builder=build_demo_context,
            completion_engine=demo_completion_engine,
        )
    )
    return DialogueRuntime(orchestrator=orchestrator, closeables=[])


class LiveContextBuilder:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        reranker: Any,
        chunk_lookup: dict[str, KBChunk],
        config: DialogueRuntimeConfig,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.chunk_lookup = chunk_lookup
        self.config = config

    def build(self, question: str, state: Any) -> EvidenceContext:
        qid = build_runtime_qid(state)
        batches = self.retriever.search_batch(
            [question],
            branch_top_k_chunks=self.config.branch_top_k_chunks,
            fused_top_k_chunks=self.config.initial_rerank_top_k_chunks,
            rrf_k=self.config.rrf_k,
            bm25_weight=self.config.bm25_weight,
            dense_weight=self.config.dense_weight,
            dense_query_batch_size=self.config.dense_query_batch_size,
        )
        hybrid_rows = list(batches[0]["hybrid"]) if batches else []
        reranked_rows = list(
            self.reranker.rerank(
                question,
                hybrid_rows,
                self.chunk_lookup,
                batch_size=self.config.reranker_batch_size,
            )
        )
        return build_initial_evidence_context(
            candidate_row={
                "qid": qid,
                "dataset_name": "runtime_dialogue",
                "question": question,
                "answer": None,
                "gold_article_ids": [],
                "num_gold_articles": 0,
                "is_multi_article": False,
                "hybrid_candidates": hybrid_rows,
            },
            rerank_trace={
                "qid": qid,
                "top10_reranked_chunks": reranked_rows[:10],
                "case_type": "runtime",
            },
            chunk_lookup=self.chunk_lookup,
        )


class RuntimeHybridRetrieverAdapter:
    def __init__(self, retriever: HybridRetriever, config: DialogueRuntimeConfig) -> None:
        self.retriever = retriever
        self.config = config

    def search(self, query_text: str, *, top_k_chunks: int) -> list[dict[str, Any]]:
        batches = self.search_batch([query_text], top_k_chunks=top_k_chunks)
        return batches[0] if batches else []

    def search_batch(self, queries: list[str], *, top_k_chunks: int) -> list[list[dict[str, Any]]]:
        batches = self.retriever.search_batch(
            queries,
            branch_top_k_chunks=self.config.branch_top_k_chunks,
            fused_top_k_chunks=top_k_chunks,
            rrf_k=self.config.rrf_k,
            bm25_weight=self.config.bm25_weight,
            dense_weight=self.config.dense_weight,
            dense_query_batch_size=self.config.dense_query_batch_size,
        )
        if len(batches) != len(queries):
            raise DialogueRuntimeError(
                "Hybrid retriever returned an unexpected batch count: "
                f"results={len(batches)}, queries={len(queries)}."
            )
        return [list(batch["hybrid"][:top_k_chunks]) for batch in batches]


def build_reranker(config: DialogueRuntimeConfig) -> Any:
    provider = compact_text(config.reranker_provider).casefold() or "auto"
    if provider == "auto":
        provider = "dashscope" if os.environ.get("DASHSCOPE_API_KEY") else "local"
    if provider == "dashscope":
        return DashScopeReranker(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            url=dashscope_rerank_url_from_env(),
            model=dashscope_rerank_model_from_env(),
            instruction=config.reranker_instruction,
            timeout=DEFAULT_DASHSCOPE_RERANK_TIMEOUT,
        )
    if provider == "local":
        return CrossEncoderReranker(
            model_name=os.environ.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B"),
            local_files_only=config.local_files_only,
            device=config.device,
            instruction=config.reranker_instruction,
        )
    raise DialogueRuntimeError("reranker_provider must be one of: auto, dashscope, local.")


def build_llm_client() -> OpenAICompatibleChatClient:
    base_url = first_text(
        os.environ.get("LLM_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        deepseek_base_url_from_env(),
    )
    api_key = first_text(
        os.environ.get("LLM_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("DEEPSEEK_API_KEY"),
    )
    model = first_text(
        os.environ.get("LLM_MODEL"),
        os.environ.get("OPENAI_MODEL"),
        os.environ.get("DEEPSEEK_MODEL"),
    )
    if not base_url:
        raise DialogueRuntimeError("Missing LLM base URL. Set LLM_BASE_URL or DEEPSEEK_BASE_URL.")
    if not model:
        raise DialogueRuntimeError("Missing LLM model. Set LLM_MODEL or DEEPSEEK_MODEL.")
    return OpenAICompatibleChatClient(base_url=base_url, api_key=api_key, model=model)


def load_chunks(path: Path) -> list[KBChunk]:
    if not path.exists():
        raise DialogueRuntimeError(f"Missing chunks file: {path}")
    chunks = [KBChunk(**row) for row in read_jsonl(path) if compact_text(row.get("text"))]
    if not chunks:
        raise DialogueRuntimeError(f"No chunks loaded from: {path}")
    return chunks


def build_runtime_qid(state: Any) -> str:
    user_turn_count = sum(1 for turn in getattr(state, "turns", []) if turn.role == "user")
    conversation_id = compact_text(getattr(state, "conversation_id", "conversation"))
    return f"{conversation_id}:turn_{user_turn_count:04d}"


def first_text(*values: Any) -> str:
    for value in values:
        text = compact_text(value)
        if text:
            return text
    return ""


def build_demo_context(question: str, state: Any) -> EvidenceContext:
    chunks = [
        KBChunk(
            chunk_id="demo_chunk_1",
            article_id="demo_article_1",
            chunk_index=0,
            title="Connecting PayPal as a Payment Provider",
            text="Connecting PayPal as a Payment Provider\nUse the Wix Accept Payments dashboard to connect PayPal.",
            contents="Use the Wix Accept Payments dashboard to connect PayPal.",
            start_token=0,
            end_token=32,
            num_tokens=32,
        ),
        KBChunk(
            chunk_id="demo_chunk_2",
            article_id="demo_article_2",
            chunk_index=0,
            title="Troubleshooting Payment Provider Availability",
            text="Troubleshooting Payment Provider Availability\nPayment providers can depend on business location and account setup.",
            contents="Payment providers can depend on business location and account setup.",
            start_token=0,
            end_token=34,
            num_tokens=34,
        ),
    ]
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    rows = [
        {
            "chunk_id": chunk.chunk_id,
            "article_id": chunk.article_id,
            "title": chunk.title,
            "rank": index,
            "score": float(10 - index),
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    return build_initial_evidence_context(
        candidate_row={
            "qid": build_runtime_qid(state),
            "dataset_name": "dialogue_demo",
            "question": question,
            "answer": None,
            "gold_article_ids": [],
            "num_gold_articles": 0,
            "is_multi_article": False,
            "hybrid_candidates": rows,
        },
        rerank_trace={
            "qid": build_runtime_qid(state),
            "top10_reranked_chunks": rows,
            "case_type": "demo",
        },
        chunk_lookup=chunk_lookup,
    )


def demo_completion_engine(context: EvidenceContext, config: Any) -> EvidenceCompletionLoopResult:
    return EvidenceCompletionLoopResult(
        context=context,
        completed=True,
        rounds_completed=context.round_index,
        retrieval_rounds=0,
        second_hop_query_count=0,
        total_latency_ms=0.0,
    )
