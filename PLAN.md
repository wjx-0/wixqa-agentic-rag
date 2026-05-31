# WixQA Agentic RAG 项目开发计划

## 项目名称

**WixQA Agentic RAG**

中文名称：

**面向企业客服知识库的证据补全式 Agentic RAG 系统**

英文定位：

**Evidence-Completion Agentic RAG for Enterprise Customer Support Knowledge Base**

---

## 1. 项目总体目标

本项目基于 WixQA 构建一个企业客服知识库场景下的 Agentic RAG 系统。

WixQA 提供 Wix Help Center 知识库文章和对应 QA 数据。项目目标不是简单搭建一个 RAG demo，而是围绕企业客服知识库中的多文档、多步骤问答问题，构建一个可评测、可扩展、可观测的 Agentic RAG 检索增强系统。

传统 RAG 通常采用：

```text
Question
  -> Retrieve
  -> Rerank
  -> Top-k Context
  -> Answer
```

这种固定流程在多文档问题中容易出现：

1. 只召回部分支持文章；
2. top-k 中缺少关键 supporting article；
3. reranker 偏向单文档相关性，忽略证据集合完整性；
4. 检索结果不足时生成模型容易幻觉；
5. 无法判断是否需要继续检索。

本项目希望将流程升级为：

```text
Question
  -> Hybrid Retrieval
  -> Reranker
  -> Evidence Sufficiency Check
  -> Gap-aware Query Generation
  -> Second-hop Retrieval
  -> Coverage-aware Selection
  -> Citation-aware Answer / Verifier
```

核心优化目标是：

```text
提高最终上下文中完整支持文章集合的召回能力。
```

---

## 2. 数据集使用方案

本项目使用 WixQA 的四个子集，分工如下：

| 子集                    | 用途                           | 是否作为最终主测试 |
| --------------------- | ---------------------------- | --------- |
| `wix_kb_corpus`       | 企业客服知识库，用于建立 BM25 / Dense 索引 | 否         |
| `wixqa_synthetic`     | pipeline sanity check、弱监督、调参 | 否         |
| `wixqa_expertwritten` | 真实用户问题 + 专家答案，作为主测试集         | 是         |
| `wixqa_simulated`     | 辅助测试集 / 泛化测试                 | 是，次要      |

使用原则：

```text
wix_kb_corpus       -> 被检索的知识库
wixqa_synthetic     -> 快速检查流程是否跑通
wixqa_expertwritten -> 主测试集，重点报告结果
wixqa_simulated     -> 辅助测试集，验证泛化能力
```

重点关注：

```text
len(article_ids) >= 2
```

的 multi-article 样本，因为这些样本更能体现 Agentic RAG 的价值。

---

## 3. 核心评测指标

本项目当前采用 chunk-ranked retrieval evaluation。

每个 QA 样本有 gold `article_ids`。WixQA 没有 gold chunk IDs，因此系统保留 top-k chunks 原始排名，并使用每个 chunk 的 `article_id` 与 gold `article_ids` 对比。

### 3.1 chunk_hit@k

top-k chunks 中是否存在来自任意 gold article 的 chunk：

```text
chunk_hit@k = 1 if retrieved_top_k_chunks contains at least one gold article_id else 0
```

### 3.2 chunk_full_article_hit@k

top-k chunks 是否覆盖全部 gold articles：

```text
chunk_full_article_hit@k = 1 if all gold article_ids are covered by retrieved_top_k_chunks else 0
```

这是本项目最核心的指标。

### 3.3 chunk_article_recall@k

top-k chunks 覆盖了多少 gold articles：

```text
chunk_article_recall@k = number of covered gold articles / number of gold articles
```

### 3.4 chunk_gold_rate@k

top-k chunks 中来自 gold articles 的比例：

```text
chunk_gold_rate@k = number of gold-article chunks / k
```

### 3.5 MRR

第一个来自 gold article 的 chunk 出现在检索结果中的 reciprocal rank。

### 3.6 候选多样性

```text
unique_articles@k_chunks = top-k chunks 覆盖的不同 article_id 数量
duplicate_article_ratio@k_chunks = 1 - unique_articles / 实际返回 chunk 数量
```

### 3.7 Agentic RAG 额外指标

后续 Agentic RAG 阶段还需要记录：

```text
avg_llm_calls
avg_retrieval_rounds
avg_generated_queries
avg_rerank_calls
avg_candidate_articles
```

---

## 4. 项目阶段规划

---

# Phase 1: 数据接入与统计分析

## 目标

完成 WixQA 数据集接入，将 Hugging Face 原始数据转换为项目统一 JSONL 格式，并输出数据统计报告。

当前阶段不实现 retrieval、reranker、LLM、Agentic RAG。

## 输入

```text
Wix/WixQA
```

包含：

```text
wix_kb_corpus
wixqa_expertwritten
wixqa_simulated
wixqa_synthetic
```

## 输出

```text
data/processed/wix_kb_corpus.jsonl
data/processed/wixqa_expertwritten.jsonl
data/processed/wixqa_simulated.jsonl
data/processed/wixqa_synthetic.jsonl

data/stats/wixqa_data_stats.json
data/stats/wixqa_data_stats.md

outputs/data_inspection/
```

## 统一 KBArticle 格式

```json
{
  "article_id": "string",
  "title": "string | null",
  "url": "string | null",
  "contents": "string",
  "article_type": "string | null",
  "metadata": {}
}
```

## 统一 QAExample 格式

```json
{
  "qid": "string",
  "dataset_name": "string",
  "question": "string",
  "answer": "string | null",
  "article_ids": ["string"],
  "num_gold_articles": 1,
  "is_multi_article": false,
  "metadata": {}
}
```

## 验收标准

能够运行：

```bash
python scripts/inspect_wixqa.py
python scripts/prepare_wixqa.py
python -m src.evaluation.data_stats --processed_dir data/processed --output_dir data/stats
```

并生成：

```text
data/processed/wix_kb_corpus.jsonl
data/processed/wixqa_expertwritten.jsonl
data/processed/wixqa_simulated.jsonl
data/processed/wixqa_synthetic.jsonl
data/stats/wixqa_data_stats.json
data/stats/wixqa_data_stats.md
```

需要重点检查：

```text
KB article 总数
QA 样本数量
multi-article 样本数量
article_ids 为空样本数量
gold article_id 在 KB 中的覆盖率
```

---

# Phase 2: BM25 Chunk-level Retrieval Baseline

## 目标

实现最小可用的 BM25 chunk-level retrieval baseline。

当前阶段只做 BM25，不做 Dense、RRF、Reranker、LLM、Agentic RAG。

## Pipeline

```text
Question
  -> BM25 over Wix KB chunks
  -> Top-k chunks
  -> Compare with gold article_ids
  -> Compute retrieval metrics
```

## 检索粒度

当前阶段使用 chunk-level retrieval。

即：

```text
使用 BAAI/bge-m3 tokenizer 生成的 chunk 作为检索单元。
```

检索文本：

```text
chunk.text
```

## 需要实现

```text
src/retrievers/tokenizer.py
src/retrievers/chunk_bm25_retriever.py
src/evaluation/run_chunk_bm25_eval.py
scripts/run_chunk_bm25_baseline.py
```

## 输出

```text
outputs/chunk_bm25_baseline/wixqa_expertwritten_metrics.json
outputs/chunk_bm25_baseline/wixqa_expertwritten_metrics.md
outputs/chunk_bm25_baseline/wixqa_expertwritten_retrieval_traces.jsonl

outputs/chunk_bm25_baseline/wixqa_expertwritten_cases_A_top10_chunks_full.jsonl
outputs/chunk_bm25_baseline/wixqa_expertwritten_cases_B_top100_chunks_full_not_top10_chunks.jsonl
outputs/chunk_bm25_baseline/wixqa_expertwritten_cases_C_top100_chunks_not_full.jsonl
```

## Case 分类

```text
A_top10_chunks_full:
top10 chunks 已经覆盖全部 gold article_ids

B_top100_chunks_full_not_top10_chunks:
top100 chunks 覆盖全部 gold article_ids，但 top10 chunks 没有

C_top100_chunks_not_full:
top100 chunks 仍然没有覆盖全部 gold article_ids
```

## 运行命令

```bash
python scripts/run_chunk_bm25_baseline.py --dataset wixqa_expertwritten --top_k_chunks 100
python scripts/run_chunk_bm25_baseline.py --dataset wixqa_simulated --top_k_chunks 100
python scripts/run_chunk_bm25_baseline.py --dataset wixqa_synthetic --top_k_chunks 100
```

## 验收标准

能够得到 BM25 baseline 指标：

```text
chunk_hit@5 / @10 / @20 / @100
chunk_full_article_hit@5 / @10 / @20 / @100
chunk_article_recall@5 / @10 / @20 / @100
chunk_gold_rate@5 / @10 / @20 / @100
mrr
```

并单独统计：

```text
single_chunk_full_article_hit@10
multi_chunk_full_article_hit@10
single_chunk_article_recall@10
multi_chunk_article_recall@10
```

---

# Phase 3: Dense Retrieval Baseline

## 目标

实现 Dense Retrieval baseline，并与 BM25 进行对比。

当前阶段不做 Agentic RAG。

## Pipeline

```text
Question
  -> Dense Embedding
  -> FAISS Vector Search over BAAI/bge-m3 chunk embeddings
  -> Top-k chunks
  -> Chunk-level Evaluation with gold article_ids
```

## 推荐模型

复用 chunk tokenizer 对应的 embedding 模型：

```text
BAAI/bge-m3
```

## 需要实现

```text
src/retrievers/faiss_store.py
scripts/build_faiss_index.py
scripts/run_dense_faiss_baseline.py
```

## 输出

```text
outputs/dense_faiss_baseline/
```

## 验收标准

能够得到 Dense baseline 指标，并和 BM25 对比。

需要输出表格：

| Method | chunk_full_article_hit@5 | chunk_full_article_hit@10 | chunk_article_recall@10 | MRR |
| ------ | -----------------------: | ------------------------: | ----------------------: | --: |
| BM25   |                        x |                         x |                       x |   x |
| Dense  |                        x |                         x |                       x |   x |

---

# Phase 4: Hybrid Retrieval with RRF Fusion

## 目标

实现 Chunk BM25 + Dense FAISS 的融合检索，建立强 chunk-level retrieval baseline。

## Pipeline

```text
Question batch
  -> Chunk BM25 top-k chunks
  -> Long-lived Dense worker top-k chunks
  -> RRF Fusion
  -> Top-k chunks
  -> Chunk-level Evaluation with gold article_ids
```

## RRF 公式

```text
score(chunk) = sum(weight_i / (k + rank_i(chunk)))
```

默认：

```text
k = 60
按 chunk_id 融合
BM25 与 Dense 权重默认均为 1.0，可通过 CLI 显式调整
快速实验：50 + 50 -> 50
标准诊断：100 + 100 -> 100
```

## 需要实现

```text
src/retrievers/hybrid_retriever.py
src/retrievers/rrf.py
src/retrievers/dense_worker.py
src/evaluation/run_hybrid_rrf_eval.py
scripts/run_hybrid_rrf_baseline.py
```

## 输出

```text
outputs/hybrid_rrf_baseline/
└── hybrid_rrf_b{branch}_f{fused}_k{rrf_k}_bw{bm25_weight}_dw{dense_weight}_{dataset}/
```

## 验收标准

和 BM25 / Dense 进行对比：

| Method             | chunk_full_article_hit@10 | chunk_recall@10 | unique_articles@k_chunks | duplicate_article_ratio@k_chunks | MRR |
| ------------------ | ------------------------: | --------------: | -----------------------: | -------------------------------: | --: |
| Chunk BM25         |                         x |               x |                        x |                                x |   x |
| Dense FAISS        |                         x |               x |                        x |                                x |   x |
| BM25 + Dense + RRF |                         x |               x |                        x |                                x |   x |

Dense worker 启动一次并接收 query batch。默认 `full` 模式在 worker 内加载 SentenceTransformer 与 FAISS；若本机库冲突，显式切换 `model_only` 兼容模式。

互补分析记录 gold article 在三种方法中的首次 chunk rank、单路独有命中、Hybrid top10 救回以及 top50 / top100 保留情况。

---

# Phase 5: Cross-Encoder Reranker

## 目标

在 Hybrid Retrieval 的候选池上加入 Cross-Encoder Reranker，建立强基线。

## Pipeline

```text
Question
  -> BM25 + Dense + RRF
  -> Candidate chunks top50 / top100
  -> Cross-Encoder Reranker
  -> Final top-k chunks
  -> Chunk-level Evaluation with gold article_ids
```

## 推荐 reranker

```text
BAAI/bge-reranker-base
BAAI/bge-reranker-large
cross-encoder/ms-marco-MiniLM-L-6-v2
```

先用小模型，保证能快速跑通。

## 需要实现

```text
src/rerankers/cross_encoder_reranker.py
scripts/run_rerank_baseline.py
```

## 输出

```text
outputs/rerank_baseline/
```

## 重点诊断

需要比较：

```text
before_rerank_article_full_hit@50
after_rerank_article_full_hit@10
after_rerank_article_full_hit@20
```

如果：

```text
before_rerank@50 高，但 after_rerank@10 低
```

说明 reranker 把部分关键 evidence 排掉了，后续需要 coverage-aware selection。

## 验收标准

形成强 baseline 表格：

| Method                | full_hit@5 | full_hit@10 | full_hit@20 | recall@10 | MRR |
| --------------------- | ---------: | ----------: | ----------: | --------: | --: |
| BM25                  |          x |           x |           x |         x |   x |
| Dense                 |          x |           x |           x |         x |   x |
| Hybrid RRF            |          x |           x |           x |         x |   x |
| Hybrid RRF + Reranker |          x |           x |           x |         x |   x |

---

# Phase 6: Error Analysis & Trace Logging

## 目标

对强 baseline 的失败案例进行系统分析，为 Agentic RAG 做准备。

## 需要输出

```text
outputs/error_analysis/
  summary.json
  case_traces.jsonl
  cases_A_top10_full.jsonl
  cases_B_top50_full_not_top10.jsonl
  cases_C_top50_not_full.jsonl
```

## 每条 trace 包含

```json
{
  "qid": "string",
  "dataset_name": "wixqa_expertwritten",
  "question": "string",
  "answer": "string",
  "gold_article_ids": [],
  "num_gold_articles": 2,
  "is_multi_article": true,

  "top10_article_ids": [],
  "top50_article_ids": [],
  "top10_titles": [],
  "top50_titles": [],

  "article_full_hit@10": 0,
  "article_recall@10": 0.5,

  "missing_articles_at_10": [],
  "missing_articles_at_50": [],

  "case_type": "B_top50_full_not_top10"
}
```

## 重点分析

分别统计：

```text
all questions
single-article questions
multi-article questions
expertwritten
simulated
synthetic
```

尤其关注：

```text
multi-article article_full_hit@10
```

因为 Agentic RAG 主要应该提升多文档问题。

---

# Phase 7: Rule-based Second-hop Retrieval

## 目标

先不调用 LLM，实现无 LLM 的 second-hop retrieval，用来验证“补检索”是否有效。

## 核心思路

如果第一轮 top-k 只召回部分文章，则从已有 top articles 中抽取关键词、标题、URL path、功能名、相关短语，构造 second-hop queries，再次检索。

## Pipeline

```text
Question
  -> First-stage Hybrid + Reranker
  -> Extract bridge / support terms from top articles
  -> Build second-hop queries
  -> Retrieve again
  -> Merge evidence pool
  -> Rerank
  -> Evaluation
```

## 可抽取信息

```text
article title
url path tokens
headings
capitalized phrases
question keywords
top article related terms
```

## 输出

```text
outputs/rule_second_hop/
```

## 验收标准

对比：

| Method                      | full_hit@10 | recall@10 | avg_retrieval_rounds |
| --------------------------- | ----------: | --------: | -------------------: |
| Hybrid + Reranker           |           x |         x |                  1.0 |
| + Rule Second-hop Retrieval |           x |         x |                  2.0 |

如果 full_hit@10 在 multi-article subset 上有提升，则进入下一阶段。

---

# Phase 8: Coverage-aware Evidence Selection

## 目标

最终 top10 不再简单取 reranker 前 10，而是优化证据集合覆盖率。

## 问题

普通 reranker 学习：

```text
score(question, article)
```

但多文档问题需要：

```text
which set of articles together supports the answer
```

## 简单策略

从不同来源保留一定数量：

```text
original query results: top n
second-hop query results: top n
global reranker results: fill remaining slots
```

## Pipeline

```text
Candidate Pool
  -> Group by source_stage
  -> Select high-confidence original articles
  -> Select high-confidence second-hop articles
  -> Fill by global reranker
  -> Final top-k
```

## 输出

```text
outputs/coverage_selection/
```

## 验收标准

比较：

| Method                     | full_hit@5 | full_hit@10 | recall@10 |
| -------------------------- | ---------: | ----------: | --------: |
| Hybrid + Reranker          |          x |           x |         x |
| + Second-hop Retrieval     |          x |           x |         x |
| + Coverage-aware Selection |          x |           x |         x |

如果 top10 full_hit 提升，说明 evidence selection 有效。

---

# Phase 9: LLM Evidence Sufficiency Checker

## 目标

引入真正的 Agentic RAG 能力：让 LLM 判断当前证据是否足够、缺什么、下一步应该检索什么。

注意：LLM 不直接回答问题，只负责证据诊断和 query generation。

## 输入

```text
question
top retrieved articles
```

## 输出 JSON

```json
{
  "sufficient": false,
  "known_facts": [],
  "missing_evidence": [],
  "next_queries": []
}
```

## Prompt 约束

```text
1. Do not answer the question.
2. Only judge whether retrieved articles contain enough evidence.
3. next_queries must contain concrete entities or product/function names.
4. Do not use pronouns like "it", "this", "that feature".
5. Generate at most 3 next_queries.
6. Return valid JSON only.
```

## 输出

```text
outputs/llm_evidence_checker/
```

## 验收标准

能够记录：

```text
checker_sufficient
missing_evidence
next_queries
llm_calls
```

并验证 LLM 生成的 next_queries 能带来新的 relevant articles。

---

# Phase 10: Bounded Agentic RAG Loop

## 目标

实现完整的 Evidence-Completion Agentic RAG 检索循环。

## Pipeline

```text
Question
  -> Query Router
  -> Hybrid Retrieval
  -> Reranker
  -> Evidence Sufficiency Checker
  -> if insufficient:
        Gap Query Generation
        Second-hop Retrieval
        Evidence Pool Merge
        Rerank
  -> Coverage-aware Selection
  -> Final top-k articles
```

## 控制条件

为了企业化和可控成本，需要设置：

```text
max_rounds = 2
max_queries_per_round = 3
top_k_per_query = 20
max_candidate_articles = 120
stop if checker says sufficient
stop if no new articles are retrieved
```

## Trace Logging

每条样本记录：

```json
{
  "qid": "string",
  "question": "string",
  "route": "single_article | multi_article | unknown",
  "rounds": [
    {
      "round_id": 0,
      "queries": [],
      "retrieved_article_ids": [],
      "checker_result": {}
    }
  ],
  "final_article_ids": [],
  "gold_article_ids": [],
  "article_full_hit@10": 1,
  "article_recall@10": 1.0,
  "avg_llm_calls": 1,
  "avg_retrieval_rounds": 2
}
```

## 输出

```text
outputs/agentic_rag/
```

## 验收标准

对比强 baseline：

| Method            | full_hit@10 | recall@10 | avg_llm_calls | avg_retrieval_rounds |
| ----------------- | ----------: | --------: | ------------: | -------------------: |
| Hybrid + Reranker |           x |         x |             0 |                    1 |
| Agentic RAG       |           x |         x |             x |                    x |

重点看：

```text
multi-article subset full_hit@10
```

---

# Phase 11: Citation-aware Answer Generation

## 目标

基于最终检索到的 articles 生成带引用的客服答案。

## 输入

```text
question
final top-k articles
```

## 输出格式

```text
Answer:
1. ...
2. ...
3. ...

Sources:
[1] article title - url
[2] article title - url
```

## 要求

```text
1. 答案必须基于检索到的 articles。
2. 每个关键步骤尽量有引用。
3. 如果证据不足，明确说明无法确认。
4. 不编造 Wix 文档中没有的信息。
```

## 输出

```text
outputs/answer_generation/
```

## 初始评测

第一版不追求复杂自动评测，先保存：

```text
question
gold answer
generated answer
used article_ids
gold article_ids
citation urls
```

后续可加入 LLM-as-judge。

---

# Phase 12: Verifier / Abstention

## 目标

企业 RAG 不能只会回答，还要能判断证据不足时拒答。

## Verifier 判断

```text
1. 最终 articles 是否足够回答问题？
2. 生成答案是否被引用 articles 支持？
3. 是否缺少关键步骤？
4. 是否应该拒答或提示人工确认？
```

## 输出

```json
{
  "status": "ready_to_answer | insufficient_evidence | unsupported_answer",
  "reason": "string",
  "missing_evidence": [],
  "suggested_queries": []
}
```

## 企业化价值

```text
减少幻觉
提升引用可信度
支持人工审核
适合客服/技术支持场景
```

---

# Phase 13: Optional - Pairwise Evidence Reranker

## 目标

将单文档 reranking 升级为证据组合完整性排序。

普通 reranker：

```text
score(question, article)
```

Pairwise evidence reranker：

```text
score(question, article_a, article_b)
```

用于判断两个或多个 articles 是否共同支持答案。

## 数据构造

正样本：

```text
(question, gold_article_a, gold_article_b) -> 1
```

负样本：

```text
(question, gold_article_a, hard_negative) -> 0
(question, hard_negative_a, hard_negative_b) -> 0
```

## 适用条件

这一阶段不是 MVP 必需项，适合作为高级算法增强模块。

---

## 5. 推荐完成顺序

## MVP 版本

目标：尽快完成可写简历版本。

```text
Phase 1: 数据接入与统计分析
Phase 2: BM25 Chunk-level Retrieval Baseline
Phase 3: Dense Retrieval Baseline
Phase 4: Hybrid Retrieval with RRF
Phase 5: Cross-Encoder Reranker
Phase 6: Error Analysis & Trace Logging
Phase 7: Rule-based Second-hop Retrieval
Phase 8: Coverage-aware Selection
Phase 10: Bounded Agentic RAG Loop
```

预计时间：

```text
4 - 7 天
```

---

## 完整企业化版本

```text
MVP
+ Phase 9: LLM Evidence Sufficiency Checker
+ Phase 11: Citation-aware Answer Generation
+ Phase 12: Verifier / Abstention
```

预计时间：

```text
7 - 10 天
```

---

## 高级算法版本

```text
完整企业化版本
+ Phase 13: Pairwise Evidence Reranker
```

预计时间：

```text
2 - 3 周
```

---

## 6. 最终简历目标

完成后，简历可以写：

```text
构建面向企业客服知识库的 Evidence-Completion Agentic RAG 系统，基于 WixQA 实现 BM25、Dense Retrieval、RRF Fusion、Cross-Encoder Reranker 等强检索基线，并针对多文档问题中支持文章召回不完整的问题，引入 Evidence Sufficiency Checker、Gap-aware Query Generation、Bounded Second-hop Retrieval 和 Coverage-aware Selection，以 article_full_hit@k、article_recall@k、avg_llm_calls 和 agent trace 评估检索完整性与系统成本。
```

如果最终加入答案生成，可以补充：

```text
进一步实现 Citation-aware Answer Generation 和 Verifier / Abstention 机制，使系统在证据不足时能够拒答或提示人工确认，降低企业客服场景下的幻觉风险。
```

---

## 7. 当前开发状态

## Implemented

```text
Phase 1: Data ingestion and dataset statistics
Phase 2: BM25 Chunk-level Retrieval Baseline
Phase 3: Dense Retrieval Baseline
```

## In Progress

```text
Phase 4: Hybrid Retrieval with RRF
```

## Planned

```text
Phase 5: Cross-Encoder Reranker
Phase 6: Error Analysis & Trace Logging
Phase 7: Rule-based Second-hop Retrieval
Phase 8: Coverage-aware Selection
Phase 9: LLM Evidence Sufficiency Checker
Phase 10: Bounded Agentic RAG Loop
Phase 11: Citation-aware Answer Generation
Phase 12: Verifier / Abstention
Phase 13: Pairwise Evidence Reranker
```
