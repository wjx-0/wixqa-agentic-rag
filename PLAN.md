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

# Phase 5: Qwen3 Chunk Reranker Baseline

## 目标

在 Hybrid Retrieval 的候选池上加入本地 Qwen3 Reranker，建立强基线。

## Pipeline

```text
Saved Hybrid candidates top50 / top100
  -> Qwen/Qwen3-Reranker-0.6B
  -> Final top-k chunks
  -> Chunk-level Evaluation with gold article_ids
```

Reranker 读取 Hybrid 运行目录中的 `candidates.jsonl`，不会重新执行 BM25、Dense 编码或 FAISS 搜索。

## 默认配置

```text
model_name = Qwen/Qwen3-Reranker-0.6B
local_files_only = true
rerank_batch_size = 8
max_length = 1024
instruction_name = wixqa_help_center_v1
```

默认 instruction：

```text
Given a Wix Help Center question, retrieve relevant passages that contain the information needed to answer the question.
```

## 需要实现

```text
src/rerankers/cross_encoder_reranker.py
src/evaluation/run_rerank_eval.py
scripts/run_rerank_baseline.py
```

Hybrid baseline 每次运行额外保存：

```text
candidates.jsonl
comparison.json
```

## 输出

```text
outputs/rerank_baseline/
└── <hybrid_run_name>/
    └── qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024/
```

## 重点诊断

需要比较：

```text
source_candidate_full_article_hit
chunk_full_article_hit@10
chunk_full_article_hit@20
rerank_top10_rescued_gold_article_ids
rerank_top10_dropped_gold_article_ids
gold_article_first_chunk_rank_before_rerank
gold_article_first_chunk_rank_after_rerank
```

如果：

```text
source candidate full hit 高，但 chunk_full_article_hit@10 低
```

说明 reranker 把部分关键 evidence 排掉了，需要保留为排序诊断信号。

## 验收标准

形成强 baseline 表格：

| Method                       | chunk_full_article_hit@5 | chunk_full_article_hit@10 | chunk_full_article_hit@20 | chunk_article_recall@10 | MRR |
| ---------------------------- | -----------------------: | ------------------------: | ------------------------: | ----------------------: | --: |
| Chunk BM25                   |                        x |                         x |                         x |                       x |   x |
| Dense FAISS                  |                        x |                         x |                         x |                       x |   x |
| Hybrid RRF                   |                        x |                         x |                         x |                       x |   x |
| Hybrid RRF + Qwen3 Reranker  |                        x |                         x |                         x |                       x |   x |

---

# Phase 6: Error Analysis & Trace Logging

## 目标

对强 baseline 的失败案例进行系统分析，为 Agentic RAG 做准备。

## 需要输出

```text
outputs/error_analysis/
  summary.json
  summary.md
  case_traces_top50_chunks.jsonl
  cases_A_top10_chunks_full.jsonl
  cases_B_top50_chunks_full_not_top10_chunks.jsonl
  cases_C_top50_chunks_not_full.jsonl
  multi_cases_B_top50_chunks_full_not_top10_chunks.jsonl
  multi_cases_C_top50_chunks_not_full.jsonl
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

  "top10_chunks_article_ids": [],
  "top50_chunks_article_ids": [],
  "top10_chunks_titles": [],
  "top50_chunks_titles": [],

  "article_full_hit@10_chunks": 0,
  "article_recall@10_chunks": 0.5,
  "article_full_hit@50_chunks": 0,
  "article_recall@50_chunks": 0.5,

  "unique_articles@10_chunks": 8,
  "duplicate_article_ratio@10_chunks": 0.2,
  "unique_articles@50_chunks": 35,
  "duplicate_article_ratio@50_chunks": 0.3,

  "missing_articles_at_10_chunks": [],
  "missing_articles_at_50_chunks": [],
  "gold_article_first_chunk_rank": {},

  "case_type": "B_top50_chunks_full_not_top10_chunks"
}
```

其中 before / after rerank 诊断字段为 optional，源 trace 中存在时保留，缺失时不报错。

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
multi-article article_full_hit@10_chunks
```

因为 Agentic RAG 主要应该提升多文档问题。

根据 error analysis，强 baseline 的失败可以拆成两类：

```text
B_top50_chunks_full_not_top10_chunks
  -> 候选池已有完整证据
  -> 问题主要在最终上下文选择 / 覆盖控制

C_top50_chunks_not_full
  -> 候选池本身缺少证据
  -> 问题主要在缺口诊断 / 补检索
```

Phase 6 之后保留两条路线：

```text
Route A / Previous Branch
  -> Rule second-hop
  -> LLM checker gap query
  -> merged-pool rerank
  -> 作为对照实验和经验保留

Route B / New Main Branch
  -> Evidence Context Construction
  -> Evidence Gap Diagnosis
  -> Iterative Evidence Completion Loop
  -> Shared Context Budget Manager / Auto-Compaction
  -> Citation-aware Answer / Verifier
  -> Multi-turn User Dialogue
```

新的主线不再把目标简化为“扩大 merged pool 后重新 rerank”，而是维护一个显式的 `EvidenceContext`：记录当前上下文已经支持哪些事实、缺哪些证据面、每轮检索补到了什么、什么时候足够回答，以及回答前如何压缩成可引用的 answer-ready context。

---

# Route A / Previous Branch - Phase 7A: Rule-based Second-hop Retrieval

> 该路线已经实现，用作旧方案对照和失败经验保留；不再作为后续主线继续扩展。

## 目标

先不调用 LLM，实现无 LLM 的 second-hop retrieval，用来验证“补检索”是否有效。

```text
C_top50_chunks_not_full
```

因为这些样本的现有 top50 chunks 候选池本身缺少 gold articles，仅依靠已有排序无法解决。

本阶段不增加独立的 Coverage Selection。补检索后的候选池沿用现有 reranker 排序，先验证新增候选是否能补齐缺失 supporting articles。

`C_top50_chunks_not_full` 只作为离线评测切片，不作为运行时触发条件。Phase 7 v1 默认对全部样本执行一次 rule-based second-hop；Phase 8 再引入 LLM checker 判断何时需要继续检索。

query 构造和检索逻辑不得读取 `gold_article_ids`、`missing_articles_at_50_chunks` 或 `case_type`。gold labels 仅用于最终指标统计。

## 核心思路

固定使用 baseline reranker top10 chunks 中最先出现的 3 个不同 `article_id`，分别拼接文章标题构造 second-hop queries：

```text
seed source                   = baseline reranker top10 chunks
seed dedup                    = article_id first-seen order
max seed articles             = 3
query template                = "{question} Related Wix Help Center topic: {title}"
second-hop retriever          = Hybrid BM25 + Dense RRF
branch_top_k_chunks           = 100
second_hop_fused_top_k_chunks = 20 per query
rrf_k                         = 60
bm25_weight                   = 1
dense_weight                  = 2
merged candidate upper bound  = 50 + 3 * 20 = 110 chunks
```

合并时按 `chunk_id` 去重，首轮 top50 优先，再按 query 顺序和 second-hop rank 追加。最终使用原始问题重新运行同一 Qwen3 reranker。

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

## Rescue 定义

```text
G   = gold_article_ids
P50 = 首轮 Hybrid top50 chunks 覆盖的 article_ids
PM  = second-hop 合并候选池覆盖的 article_ids
B10 = baseline Qwen3 reranker top10 chunks 覆盖的 article_ids
F10 = 合并候选重新 rerank 后 top10 chunks 覆盖的 article_ids

source_C = G 不是 P50 的子集
source_A = G 是 B10 的子集

C_pool_rescued  = source_C and G 是 PM 的子集
C_top10_rescued = source_C and G 是 F10 的子集
A_dropped       = source_A and G 不是 F10 的子集
```

补回部分文章但仍未覆盖全部 gold articles 时，不计入 rescued。`C_top10_rescued = true` 必然意味着 `C_pool_rescued = true`。

## 输出

```text
outputs/rule_second_hop/
  <hybrid_run_name>/
    <rerank_run_name>/
      title_expand_s3_h20/
        run_config.json
        metrics.json
        metrics.md
        comparison.md
        second_hop_traces.jsonl
        cases_C_pool_rescued_by_second_hop.jsonl
        cases_C_top10_rescued_by_second_hop.jsonl
        multi_cases_C_top10_rescued_by_second_hop.jsonl
        cases_A_dropped_by_second_hop.jsonl
```

## 验收标准

对比：

| Method                      | chunk_full_article_hit@10 | chunk_article_recall@10 | multi_chunk_full_article_hit@10 | C_pool_rescued_count | C_top10_rescued_count | A_dropped_count |
| --------------------------- | ------------------------: | ----------------------: | ------------------------------: | -------------------: | --------------------: | --------------: |
| Top50 Hybrid + Reranker     |                         x |                       x |                               x |                    - |                     - |               - |
| Top100 Hybrid + Reranker    |                         x |                       x |                               x |                    - |                     - |               - |
| + Rule Second-hop Retrieval |                         x |                       x |                               x |                    x |                     x |               x |

先补跑 `b100_f100_k60_bw1_dw2` 公平 top100 对照，只改变 fused cutoff。主指标为最终 `multi_chunk_full_article_hit@10`；同时报告 pool rescue 和 top10 rescue，区分 retrieval 与 reranker 问题。

---

# Route A / Previous Branch - Phase 8A: LLM Evidence Sufficiency Checker + Pool-level Gap Query Eval

> 该路线只做 pool-level gap query eval，不维护显式上下文状态；保留为 Route B 的对照基线。

## 目标

在 Phase 7 标题扩展规则收益有限后，引入 LLM checker 判断 baseline reranker top10 证据是否充足，并生成面向缺失证据的 gap-aware queries。

Phase 8 v1 是 **pool-level eval**：只评估 LLM queries 是否能把缺失 gold articles 补进合并候选池；本阶段不重新 rerank merged pool，不报告最终 top10 context quality。

## 默认运行

```bash
python scripts/run_llm_evidence_checker.py \
  --device cuda \
  --llm_base_url <openai-compatible-url> \
  --llm_api_key <key> \
  --llm_model <server-qwen3-8b-model-name>
```

默认使用当前 top50 Hybrid + Qwen3 reranker baseline、top100 control、Phase 7 rule second-hop 结果、FAISS index 和 chunks。

## Checker 输入输出

输入：

```text
question
baseline reranker top10 chunks
```

只给 LLM `rank/title/article_id/text_preview`，不提供 `gold_article_ids`、`case_type`、`missing_articles`。

输出 JSON：

```json
{
  "sufficient": false,
  "known_facts": [],
  "blocking_missing_evidence": [],
  "nice_to_have_missing_evidence": [],
  "next_queries": [],
  "reason": ""
}
```

Prompt 强约束：

```text
1. Do not answer the user's question.
2. sufficient=true means the chunks support a correct, useful, non-misleading answer.
3. Do not require exhaustive edge cases, exact wording, or more explicit confirmation.
4. sufficient=false only when a blocking evidence gap would make the answer unsupported, materially incomplete, or misleading.
5. blocking_missing_evidence must contain only retrieval-worthy blocking gaps.
6. nice_to_have_missing_evidence must not trigger retrieval.
7. next_queries must target blocking_missing_evidence, not merely rewrite the original question.
8. next_queries must contain concrete Wix product, feature, action, setting, integration, error, or entity names.
9. Do not use vague pronouns like it, this, that feature, or that setting.
10. Generate at most 3 next_queries.
11. Return valid JSON only.
```

## Pool-level Eval

```text
checker insufficient + blocking_missing_evidence + next_queries
  -> Hybrid BM25 + Dense RRF per query
  -> second_hop_fused_top_k_chunks = 20
  -> merge with first-hop Hybrid top50 by chunk_id
  -> evaluate merged pool article coverage
```

固定参数：

```text
max_next_queries = 3
branch_top_k_chunks = 100
second_hop_fused_top_k_chunks = 20
rrf_k = 60
bm25_weight = 1
dense_weight = 2
```

输出：

```text
outputs/llm_evidence_checker/
  hybrid_rrf_b100_f50_k60_bw1_dw2_wixqa_expertwritten/
    qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024/
      qwen3_8b_s3_h20_pool_eval/
        run_config.json
        metrics.json
        metrics.md
        comparison.md
        checker_traces.jsonl
        cases_invalid_checker_json.jsonl
        cases_checker_insufficient.jsonl
        cases_C_pool_rescued_by_llm_checker.jsonl
        multi_cases_C_pool_rescued_by_llm_checker.jsonl
        cases_source_C_checker_sufficient_miss.jsonl
```

核心指标：

```text
checker_valid_count
invalid_json_count
checker_sufficient_count
checker_insufficient_count
source_A_sufficient_rate
source_A_insufficient_count
source_A_unnecessary_retrieval_count
avg_queries_for_source_A
source_C_insufficient_rate
source_C_checker_sufficient_count
LLM_C_pool_rescued_count
multi_LLM_C_pool_rescued_count
pool_new_gold_articles_count
multi_pool_new_gold_articles_count
avg_llm_calls
avg_generated_queries
avg_second_hop_queries
avg_merged_candidates
```

判断标准：

```text
LLM_C_pool_rescued_count > Phase 7 C_pool_rescued_count = 3
multi_LLM_C_pool_rescued_count > Phase 7 multi_C_pool_rescued_count = 1
source_C_checker_sufficient_count 越低越好
source_A_sufficient_rate 越高越好
source_A_unnecessary_retrieval_count 越低越好
```

---

# Route A / Previous Branch - Phase 9A: LLM Gap-query Merged Pool Rerank Loop

> 该路线验证“补进 pool 的证据是否能被 reranker 推入 top10”。当前结果说明，纯 merged-pool rerank 对 multi-article evidence coverage 的提升有限，因此后续转向 Route B。

## 目标

复用 Phase 8 的 LLM gap-query merged pool，不重新调用 LLM，也不重新执行 second-hop retrieval，只把合并候选池交给 Qwen3 reranker 重新排序，验证补进 pool 的证据是否能进入最终上下文。

主指标仍然是 final top10；额外输出 final top20 作为诊断，判断证据是否只是排在 11-20 位。

## Pipeline

```text
Phase 8 checker_traces.jsonl
  -> first-hop top50 + second_hop_results 重建 merged pool
  -> Qwen3 reranker 使用原始 question 重排 merged pool
  -> final top10 主指标
  -> final top20 诊断指标
```

Phase 9 v1 明确不做：

```text
no new LLM calls
no new BM25 / Dense / FAISS retrieval
no independent Coverage Selection
no answer generation
```

## Metrics

严格定义：

```text
G   = gold_article_ids
P50 = first-hop top50 chunks 覆盖的 article_ids
PM  = Phase 8 merged pool 覆盖的 article_ids
B10 = baseline reranker top10 chunks 覆盖的 article_ids
F10 = Phase 9 merged-pool rerank 后 top10 chunks 覆盖的 article_ids
F20 = Phase 9 merged-pool rerank 后 top20 chunks 覆盖的 article_ids

source_C = G 不是 P50 的子集
source_A = G 是 B10 的子集

LLM_C_pool_rescued  = source_C and G 是 PM 的子集
LLM_C_top10_rescued = source_C and G 是 F10 的子集
LLM_C_top20_rescued = source_C and G 是 F20 的子集

A_dropped@10 = source_A and G 不是 F10 的子集
A_dropped@20 = source_A and G 不是 F20 的子集

pool_rescued_but_not_top10
  = LLM_C_pool_rescued and not LLM_C_top10_rescued

pool_rescued_but_top20_only
  = LLM_C_pool_rescued and LLM_C_top20_rescued and not LLM_C_top10_rescued
```

逻辑约束：

```text
LLM_C_top10_rescued => LLM_C_top20_rescued => LLM_C_pool_rescued
```

## 输出

```text
outputs/agentic_rag/
  hybrid_rrf_b100_f50_k60_bw1_dw2_wixqa_expertwritten/
    qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024/
      qwen3_8b_s3_h20_pool_eval/
        rerank_merged_pool/
          run_config.json
          metrics.json
          metrics.md
          comparison.md
          agentic_rerank_traces.jsonl
          cases_LLM_C_top10_rescued_by_rerank.jsonl
          cases_LLM_C_top20_rescued_by_rerank.jsonl
          multi_cases_LLM_C_top10_rescued_by_rerank.jsonl
          multi_cases_LLM_C_top20_rescued_by_rerank.jsonl
          cases_LLM_C_pool_rescued_but_not_top10.jsonl
          cases_LLM_C_pool_rescued_but_top20_only.jsonl
          cases_A_dropped_by_agentic_rerank.jsonl
```

运行：

```bash
python scripts/run_agentic_rerank_loop.py \
  --device cuda \
  --dense_worker_mode model_only
```

`--dense_worker_mode` 仅为服务器命令兼容保留；Phase 9 不加载 Dense retriever。

## 验收标准

```text
全部 200 条样本生成 agentic_rerank_traces.jsonl
LLM_C_pool_rescued_count 复现 Phase 8 的 7
top10 为主指标，top20 仅用于诊断
```

重点看：

```text
multi_chunk_full_article_hit@10
LLM_C_top10_rescued_count
multi_LLM_C_top10_rescued_count
A_dropped@10_count

LLM_C_top20_rescued_count
multi_LLM_C_top20_rescued_count
pool_rescued_but_top20_only_count
```

判断：

```text
若 top20 rescue 明显高于 top10 rescue，下一步优先优化 reranker / selection。
若 top10 和 top20 rescue 都低，下一步继续优化 gap queries 或 candidate precision。
若 multi full@10 提升且 LLM_C_top10_rescued_count > A_dropped@10_count，保留该流程作为 Route A 对照增强，但不覆盖 Route B 主线。
```

---

# Route B / New Main Branch - Evidence Context Loop

## 路线定位

Route B 从 Phase 6 重新分叉，目标是把系统从：

```text
retrieve / rerank / merged pool
```

升级为：

```text
retrieve
  -> build evidence context
  -> diagnose missing facets
  -> generate targeted queries
  -> retrieve more evidence
  -> update evidence context
  -> pack / compress answer-ready context
  -> answer / verify
  -> continue multi-turn dialogue
```

核心变化：

```text
旧路线关注 candidate_pool 是否补齐 gold articles。
新路线关注 EvidenceContext 是否足以支持完整答案。
```

`gold_article_ids`、`case_type`、`missing_articles_at_*` 仍然只用于离线评测，不进入运行时 checker、query generation、packer 或 compressor。

## 可追溯性硬约束

Route B 必须做成可追溯答案系统。系统不记录或暴露模型的隐藏 chain-of-thought，但必须记录结构化 provenance：

```text
1. 每一次 LLM 调用都要有 PromptManifest。
2. PromptManifest 必须记录模型实际看过的 chunk_id、article_id、snippet_id、compressed_context_id。
3. 每个 missing_facet 必须记录它是基于哪些已见 chunk 判断出来的。
4. 每条 next_query 必须记录 target_missing_facet 和 derived_from_chunk_ids。
5. 每次 context compression 必须记录压缩前后的 source_chunk_ids 映射。
6. 最终 answer generation 必须记录大模型实际看过哪些 chunk_id / compressed summaries。
7. 每个答案 claim 尽量记录 supporting_chunk_ids 和 citation ids。
8. verifier 必须记录它检查了哪些 answer claims 和哪些 evidence ids。
```

新增通用追踪对象：

```text
PromptManifest:
  llm_call_id
  phase                         # gap_diagnosis | query_generation | compression | answer | verifier
  qid / conversation_id / turn_id
  input_chunk_ids
  input_article_ids
  input_snippet_ids
  input_compressed_context_ids
  input_query_history_ids
  output_object_id
  prompt_token_count
  output_token_count
  schema_version

GapQueryProvenance:
  query_id
  query_text
  target_missing_facet
  checker_call_id
  derived_from_chunk_ids
  derived_from_compressed_context_ids
  blocking_missing_evidence

AnswerProvenance:
  answer_call_id
  seen_chunk_ids
  seen_compressed_context_ids
  answer_claims
  claim_support_map              # claim_id -> supporting_chunk_ids / citation_ids
```

这些 provenance 字段是运行时 trace 的一部分，不是只在评测阶段补出来。

---

# Route B - Phase 7B: Evidence Context Construction

## 目标

先不重新调用 LLM，也不重新检索。基于已有 Hybrid / Reranker / Error Analysis 输出中的 chunk-level traces，以 `chunk_id` 为基本证据追踪单位构建显式 `EvidenceContext`，把“当前上下文”从隐式 top10 chunks 改成可追踪状态对象。

## 输入

```text
outputs/rerank_baseline/.../rerank_traces.jsonl
outputs/hybrid_rrf_baseline/.../candidates.jsonl
outputs/error_analysis/case_traces_top50_chunks.jsonl
data/processed/wix_kb_chunks.jsonl
```

其中 Reranker 输出提供初始 active chunks，Hybrid 输出提供候选 chunks，`wix_kb_chunks.jsonl` 提供 chunk 文本和元信息；Phase 6 的 case_type 和 gold labels 只用于离线评测与诊断，不进入运行时 prompt。

## 新增模块

```text
src/agentic/
  __init__.py
  evidence_context.py
  provenance.py
  context_budget.py
  context_packer.py
  context_compressor.py
  dialogue_state.py
```

第一版数据结构：

```python
EvidenceItem:
  chunk_id
  article_id
  snippet_id
  title
  text_preview
  source                 # first_hop | second_hop | packed
  first_hop_rank
  rerank_rank
  rerank_score
  second_hop_query_ids
  token_span
  content_hash
  active
  visible_to_llm_call_ids

EvidenceContext:
  qid
  question
  round_index
  candidate_items
  active_items
  packed_items
  compressed_summary
  query_history
  known_facts
  required_facets
  covered_facets
  missing_facets
  prompt_manifests
  gap_query_provenance
  answer_provenance
```

`gold_article_ids` 可以保存在 trace 的 `eval` 子对象中，但不得进入 context prompt。

## 输出

```text
outputs/evidence_context/
  <hybrid_run_name>/
    <rerank_run_name>/
      initial_context/
        run_config.json
        metrics.json
        metrics.md
        evidence_context_traces.jsonl
        cases_B_context_not_full.jsonl
        cases_C_context_not_full.jsonl
        multi_cases_context_not_full.jsonl
```

## 验收标准

```text
1. 能从现有 artifacts 重建 200 条 EvidenceContext。
2. active_items 默认等于 baseline reranker top10。
3. context metrics 与 Phase 5 reranker top10 指标一致。
4. 每条 trace 能同时看到 active context、candidate pool、case type、article diversity。
5. 每条 trace 都包含 initial PromptManifest，记录 baseline active_items 的 chunk_id。
```

---

# Route B - Phase 8B: Evidence Gap Diagnosis

## 目标

将 checker 从“证据是否 sufficient + missing_evidence”升级为“上下文覆盖了哪些 required facets、缺哪些 facets”。

## Checker 输入

```text
question
current EvidenceContext.active_items
optional query_history
optional previous missing_facets
```

不得输入：

```text
gold_article_ids
case_type
missing_articles_at_10_chunks
missing_articles_at_50_chunks
gold answer
```

## Checker 输出 JSON

```json
{
  "sufficient": false,
  "seen_chunk_ids": [],
  "required_facets": [
    {
      "facet_id": "facet_1",
      "facet": "string",
      "why_required": "string"
    }
  ],
  "covered_facets": [
    {
      "facet_id": "facet_1",
      "facet": "string",
      "supporting_chunk_ids": [],
      "supporting_fact": "string"
    }
  ],
  "missing_facets": [
    {
      "facet_id": "facet_2",
      "facet": "string",
      "inferred_from_chunk_ids": [],
      "why_missing": "string"
    }
  ],
  "known_facts": [
    {
      "fact": "string",
      "source_chunk_ids": []
    }
  ],
  "blocking_missing_evidence": [
    {
      "facet_id": "facet_2",
      "missing_evidence": "string",
      "inferred_from_chunk_ids": []
    }
  ],
  "next_queries": [
    {
      "query_id": "gap_query_1",
      "query_text": "string",
      "target_missing_facet_id": "facet_2",
      "derived_from_chunk_ids": []
    }
  ],
  "reason": ""
}
```

字段含义：

```text
required_facets
  问题要被完整回答时必须覆盖的证据面，例如对象、动作、条件、步骤、限制、适用产品。

covered_facets
  当前 active context 已经直接支持的证据面。

missing_facets
  当前 active context 未覆盖且会影响答案完整性的证据面。

next_queries
  只针对 missing_facets / blocking_missing_evidence，不做单纯 paraphrase。
```

`seen_chunk_ids` 必须等于本次 checker PromptManifest 中的 `input_chunk_ids`。如果 checker 输出的 `supporting_chunk_ids`、`inferred_from_chunk_ids` 或 `derived_from_chunk_ids` 引用了未出现在 `seen_chunk_ids` 中的 chunk，视为 invalid trace。

## 输出

```text
outputs/evidence_gap_diagnosis/
  <context_run_name>/
    qwen3_8b_facet_checker/
      run_config.json
      metrics.json
      metrics.md
      gap_diagnosis_traces.jsonl
      prompt_manifests.jsonl
      gap_query_provenance.jsonl
      cases_checker_invalid_json.jsonl
      cases_invalid_provenance_refs.jsonl
      cases_missing_facets.jsonl
      cases_source_A_marked_insufficient.jsonl
      cases_source_C_marked_sufficient.jsonl
```

## 验收标准

```text
checker_valid_count 接近 200
source_A_sufficient_rate 尽量高
source_C_sufficient_count 尽量低
avg_missing_facets 可解释
next_queries 与 missing_facets 对齐
checker_seen_chunk_manifest_rate 接近 1.0
gap_query_provenance_rate 接近 1.0
invalid_provenance_ref_count = 0
```

本阶段仍然不追求最终 top10 提升，目标是把“缺口”从自然语言 missing evidence 升级成可追踪 facets。

---

# Route B - Phase 9B: Iterative Evidence Completion Loop

## 目标

实现真正的上下文循环，而不是一次性 second-hop。

```text
for round in 0..max_rounds:
  build ContextUsageSnapshot
  pack / auto-compact context if projected usage exceeds thresholds
  diagnose EvidenceContext
  if sufficient:
    stop and go to answer generation
  build ContextUsageSnapshot for query generation
  pack / auto-compact query-generation context if needed
  generate next_queries
  retrieve each query
  rerank new candidates with Qwen/Qwen3-Reranker-0.6B
  select top5 new chunks for next LLM-visible context
  update EvidenceContext
```

## 默认配置

```text
max_rounds = 4
max_queries_per_round = 2
second_hop_top_k_chunks = 20 per query, stored in candidate_items
branch_top_k_chunks = 100
rrf_k = 60
bm25_weight = 1
dense_weight = 2
max_raw_chunks_per_checker_call = 10
new_candidate_reranker = Qwen/Qwen3-Reranker-0.6B
max_new_raw_chunks_per_round = 5
```

检索和给模型看的数量分开：

```text
retrieve:
  每条 gap query 检索 fused top20 chunks，全部写入 EvidenceContext.candidate_items。

rerank new candidates:
  合并本轮最多 2 条 query 的新增候选，按 chunk_id 去重。
  使用 Qwen/Qwen3-Reranker-0.6B 对新增候选重新排序。
  选择 top5 new chunks 进入下一次 LLM 可见上下文。

shortlist for LLM:
  每轮最多给 LLM 新看 5 个 raw chunks。
  checker / query-generation 每次最多看 10 个 raw chunks。
  旧 evidence 可以通过 compressed_summary / known_facts 累积，因此模型可以越跑掌握越多上下文，但每次 raw chunks 仍受限。
```

这样做的原因：

```text
top20 per query 用于保证召回，不直接增加 LLM 成本。
0.6B reranker 用于低成本筛选新增证据。
每轮 top5 new chunks 控制 checker 成本。
每次 raw chunks <= 10，但 compressed context 可累积历史证据。
```

## Context Update 原则

```text
1. 所有候选按 chunk_id 去重。
2. 保留 first_hop_rank、second_hop_query_ids、second_hop_ranks。
3. query_history 防止重复查询。
4. 每轮记录 new_chunk_ids、new_article_ids、new_gold_article_ids 仅用于 eval。
5. active_items 随 EvidenceContext 更新；Shared Layer 10B 负责每次 LLM call 前的 packing / auto-compaction。
6. 每轮记录 checker_call_id、checker_seen_chunk_ids、gap_query_provenance、retrieved_chunk_ids、new_candidate_rerank_top5_chunk_ids、active_chunk_ids_after_update。
7. 每次 checker / query-generation LLM call 前都必须经过 Context Budget Manager。
8. LLM 可见的 active_items 是 packer 产物，不等于 candidate_items；新检索 top20 只进候选池，未经 shortlist 不直接给 LLM 看。
```

每轮 trace 至少包含：

```json
{
  "round_index": 1,
  "checker_call_id": "llm_call_...",
  "checker_seen_chunk_ids": [],
  "missing_facets": [],
  "gap_queries": [],
  "gap_query_provenance": [],
  "retrieved_chunk_ids_by_query": {},
  "new_candidate_reranker": "Qwen/Qwen3-Reranker-0.6B",
  "new_candidate_rerank_top5_chunk_ids": [],
  "new_active_chunk_ids": [],
  "active_chunk_ids_after_update": [],
  "stopped_because_sufficient": false
}
```

## 输出

```text
outputs/evidence_context_loop/
  <context_run_name>/
    qwen3_8b_facet_loop_s3_h20_r3/
      run_config.json
      metrics.json
      metrics.md
      context_loop_traces.jsonl
      round_traces.jsonl
      prompt_manifests.jsonl
      gap_query_provenance.jsonl
      cases_context_pool_rescued.jsonl
      multi_cases_context_pool_rescued.jsonl
      cases_stopped_sufficient.jsonl
      cases_max_rounds_not_sufficient.jsonl
```

## 核心指标

```text
avg_llm_calls
avg_retrieval_rounds
avg_generated_queries
avg_candidate_items
context_pool_full_article_hit_rate
multi_context_pool_full_article_hit_rate
C_context_pool_rescued_count
multi_C_context_pool_rescued_count
source_A_unnecessary_loop_count
round_prompt_manifest_rate
gap_query_provenance_rate
retrieval_provenance_rate
```

验收标准：

```text
1. C_context_pool_rescued_count 明显优于 Route A Phase 7A/8A。
2. multi_C_context_pool_rescued_count 有提升。
3. source_A_unnecessary_loop_count 可控。
4. 每轮 trace 能解释为什么继续检索、检索了什么、上下文更新了什么。
5. 每条 gap query 都能回溯到 checker_call_id、target_missing_facet 和 derived_from_chunk_ids。
```

---

# Route B - Shared Layer 10B: Context Budget Manager, Packing & Auto-Compaction

## 目标

Context packing / compression 不是最后回答前才做，而是每次 LLM 调用前都运行的共享层。

这一层的职责不是“刷 top10”，而是：

```text
1. 估算本次 LLM 调用的 projected context usage。
2. 在达到阈值时自动压缩旧 evidence / dialogue memory。
3. 为 checker / query generation / answer / verifier 分别打包 prompt packet。
4. 保留关键证据和 citation provenance。
5. 写入 PromptManifest 和 compact_boundary trace。
```

Packing 每次 LLM 调用前都执行；compression 只在上下文预算接近阈值时自动触发，或由调试命令手动触发。

适用阶段：

```text
Phase 8B checker 前
Phase 9B 每轮 query generation / checker 前
Phase 11B answer / verifier 前
Phase 12B multi-turn dialogue 每轮前
```

## Packer 输入

```text
question
EvidenceContext.candidate_items
EvidenceContext.active_items
known_facts
required_facets / covered_facets / missing_facets
query_history
citation provenance
token_budget
```

## LLM 可见 Chunk 策略

第一次 checker 调用：

```text
只给模型看 baseline reranker top10 chunks。
PromptManifest.input_chunk_ids = baseline_top10_chunk_ids。
```

后续 checker / query-generation 调用：

```text
raw_chunk_budget = 10
max_new_raw_chunks_per_round = 5
new_chunk_selector = Qwen/Qwen3-Reranker-0.6B top5 over this round's retrieved candidates
max_chunks_per_article_in_prompt = 2
```

选择顺序：

```text
1. 保留已覆盖关键 covered_facets 的旧 chunks。
2. 保留最近一轮经 0.6B reranker 选出的 top5 new chunks。
3. 对同一 article 的重复 chunks 降权，最多保留 2 个。
4. 未进入 raw prompt 的旧证据进入 compressed_summary / known_facts，但必须保留 source_chunk_ids。
5. 每次 LLM 可见 raw chunks 总数不得超过 10，除非 answer 阶段明确提高预算并写入 run_config。
```

Answer / verifier 阶段默认仍使用：

```text
max_raw_chunks_per_answer_call = 10
```

如果后续发现 answer 质量需要更多证据，再单独做 ablation，而不是默认放宽。

## Context Budget 计算

每次 LLM 调用前先构建 `ContextUsageSnapshot`：

```text
model_context_window_tokens
reserved_output_tokens
safety_margin_tokens
usable_input_budget
static_instruction_tokens
task_prompt_tokens
dialogue_memory_tokens
evidence_tokens
compressed_context_tokens
query_history_tokens
tool_or_schema_tokens
projected_input_tokens
projected_total_tokens = projected_input_tokens + reserved_output_tokens
usage_ratio = projected_total_tokens / model_context_window_tokens
```

默认预算：

```text
reserved_output_ratio = 0.12
safety_margin_ratio = 0.03
usable_input_budget = model_context_window_tokens
  - reserved_output_tokens
  - safety_margin_tokens
```

所有阈值都基于 `projected_total_tokens / model_context_window_tokens`，而不是只看已经累计的 history tokens。

## Auto-Compaction 阈值策略

参考 Claude Code：

```text
Claude Code 会在接近上下文限制时自动 compaction；
官方成本文档提到 auto-compaction 会摘要 conversation history；
Claude Code SDK 的 /compact 会产生 compact_boundary，并记录 pre-compaction tokens 和 trigger。
```

本项目采用更保守的 RAG 版本，因为长 evidence context 会影响 checker / answer 的精度：

| usage_ratio | 状态 | 动作 |
| ---: | --- | --- |
| `< 0.70` | healthy | 只做 packing，不压缩 |
| `0.70 - 0.80` | watch | 去重、裁剪 inactive duplicates，记录 warning |
| `>= 0.80` | soft_auto_compact | 自动压缩旧轮次 evidence、低价值重复 chunks、旧 dialogue turns |
| `>= 0.90` | hard_auto_compact | 本次 LLM call 前必须压缩到目标比例以下 |
| `>= 0.95` | emergency_compact | 参考 Claude Code auto-compact 临界思路；若压缩后仍超限，则拆分子调用或拒绝本次 LLM call |

压缩目标：

```text
target_after_compact_ratio = 0.60
min_compaction_savings_ratio = 0.15
max_raw_recent_turns_to_keep = 2
max_raw_recent_rounds_to_keep = 1
```

也就是说，触发自动压缩后，不是刚好压到 79%，而是尽量压回 60% 左右，避免下一轮马上再次压缩。

## Manual Compact

保留一个手动压缩入口，类似 Claude Code `/compact Focus on ...`：

```text
manual_compact(focus_instructions)
```

示例：

```text
Focus on missing facets, source_chunk_ids, citation manifest, and user constraints.
```

手动压缩和自动压缩都要写相同的 `CompactBoundary` trace。

## CompactBoundary Trace

每次压缩都记录：

```text
compact_boundary_id
trigger                       # manual | soft_auto | hard_auto | emergency
phase                         # checker | query_generation | answer | verifier | dialogue
pre_compaction_tokens
post_compaction_tokens
pre_usage_ratio
post_usage_ratio
target_after_compact_ratio
focus_instructions
source_chunk_ids
kept_raw_chunk_ids
compressed_context_ids
dropped_chunk_ids
source_to_summary_map
provenance_retention_check
created_at_round
created_at_turn
```

## Claude Code 式上下文压缩参考

Claude Code 的上下文管理思路是：会把长会话压缩成结构化摘要以释放上下文空间，`/compact` 还可以带 focus instructions；官方文档也强调，压缩后项目级持久规则会重新注入，而历史对话会被摘要替代。

本项目借鉴这个思路，但应用到 RAG evidence context：

```text
raw evidence chunks
  -> structured evidence summary
  -> preserve citations / article_ids / urls / chunk_ids
  -> keep recent or decisive raw snippets
  -> drop duplicates and low-value repetition
```

压缩后的 `ContextCompression` 至少包含：

```text
compressed_context_id
question_intent
known_facts
covered_facets
remaining_missing_facets
evidence_summary_by_article
citation_manifest              # article_id, title, url, chunk_ids
raw_snippets_to_keep
source_chunk_ids
source_snippet_ids
summary_support_map            # summary sentence -> source_chunk_ids
query_history
compression_reason
token_budget_before_after
```

压缩原则：

```text
1. 不压缩掉 citation provenance。
2. 不压缩掉 checker 判定 sufficient 所依赖的关键事实。
3. 对同一 article 的重复 chunks 做摘要合并。
4. 最近一轮新增证据优先保留 raw snippet。
5. 如果压缩后 verifier 判断支持不足，回退到未压缩 context 或重新检索。
6. 每次 compression 都写 trace，方便审计丢了什么。
7. compressed summary 的每句话都要能回溯到 source_chunk_ids。
```

## Packing / Compression 策略

```text
如果 active context 在 token budget 内：
  只做去重和 citation manifest，不做 LLM compression。

如果 active context 超出 token budget：
  先按 article / facet 聚合，再做结构化摘要压缩。

如果多轮对话积累了旧上下文：
  保留当前 turn 相关 raw snippets，把旧 turn evidence 压成 session memory summary。
```

## 输出

```text
outputs/evidence_context_packing/
  <loop_run_name>/
    context_budget_v1/
      run_config.json
      metrics.json
      metrics.md
      context_usage_snapshots.jsonl
      compact_boundaries.jsonl
      answer_ready_context_traces.jsonl
      context_compression_traces.jsonl
      prompt_manifests.jsonl
      compression_source_maps.jsonl
      cases_compression_used.jsonl
      cases_verifier_failed_after_compression.jsonl
      cases_context_packed_without_compression.jsonl
```

## 核心指标

```text
answer_ready_context_sufficient_rate
avg_context_tokens_before
avg_context_tokens_after
avg_usage_ratio_before
avg_usage_ratio_after
auto_compaction_count
manual_compaction_count
emergency_compaction_count
compression_token_reduction_ratio
citation_retention_rate
facet_retention_rate
verifier_supported_context_rate
compression_verifier_failure_count
compression_source_map_rate
summary_sentence_support_rate
compact_boundary_trace_rate
```

离线诊断可额外报告：

```text
packed_context_full_article_hit@10
multi_packed_context_full_article_hit@10
packed_context_unique_articles@10
```

验收标准：

```text
1. sufficient 的 EvidenceContext 能被整理成 answer-ready context。
2. compression 后 verifier 仍能确认上下文支持答案。
3. citation_retention_rate 接近 1.0。
4. token 数明显下降时，facet_retention_rate 不明显下降。
5. 每个 compressed_context_id 都能列出 source_chunk_ids。
6. hard_auto_compact 触发后，post_usage_ratio 应低于 target_after_compact_ratio 或明确记录无法压缩原因。
```

---

# Route B - Phase 11B: Citation-aware Answer Generation / Verifier

## 目标

当 checker 判断 EvidenceContext sufficient，并经过 Phase 10B 打包 / 压缩后，直接生成带引用答案。答案生成不再依赖固定 top10，而依赖 answer-ready context。

## Pipeline

```text
question
answer-ready EvidenceContext
  -> build answer PromptManifest with seen_chunk_ids / seen_compressed_context_ids
  -> citation-aware answer
  -> verifier checks answer support
  -> answer / abstain / return to evidence loop
```

## 输出

```text
outputs/evidence_answering/
  <packing_run_name>/
    answer_generation/
      run_config.json
      answer_traces.jsonl
      verifier_traces.jsonl
      prompt_manifests.jsonl
      answer_provenance.jsonl
      metrics.md
```

## Answer 输出 JSON

```json
{
  "answer": "string",
  "seen_chunk_ids": [],
  "seen_compressed_context_ids": [],
  "answer_claims": [
    {
      "claim_id": "claim_1",
      "claim": "string",
      "supporting_chunk_ids": [],
      "supporting_compressed_context_ids": [],
      "citation_ids": []
    }
  ],
  "citations": [
    {
      "citation_id": "src_1",
      "article_id": "string",
      "chunk_ids": [],
      "title": "string",
      "url": "string"
    }
  ]
}
```

`seen_chunk_ids` 必须来自 answer PromptManifest。答案里出现的 citation / claim support 不得引用模型没有看过的 chunk。

## Verifier 输出

```json
{
  "status": "ready_to_answer | insufficient_evidence | unsupported_answer",
  "reason": "string",
  "verifier_seen_chunk_ids": [],
  "checked_claim_ids": [],
  "unsupported_claims": [],
  "missing_facets": [],
  "suggested_queries": []
}
```

Verifier 决策：

```text
ready_to_answer
  -> 返回带引用答案。

insufficient_evidence
  -> 回到 Phase 9B，使用 missing_facets / suggested_queries 继续补证据。

unsupported_answer
  -> 重新生成答案；若仍失败，则拒答或请求人工确认。
```

验收标准：

```text
answer_seen_chunk_manifest_rate 接近 1.0
claim_support_map_rate 尽量高
invalid_answer_support_ref_count = 0
verifier_seen_chunk_manifest_rate 接近 1.0
```

---

# Route B - Phase 12B: Multi-turn User Dialogue

## 目标

实现用户多轮对话，使系统不仅能回答单个 WixQA 问题，还能在连续追问中复用、更新、压缩上下文。

多轮对话的核心不是每轮重新 RAG，而是维护 `DialogueState`：

```text
DialogueState:
  conversation_id
  user_turns
  assistant_turns
  current_user_intent
  active_question
  rewritten_standalone_question
  evidence_context
  context_memory_summary
  answer_history
  citation_history
  prompt_manifest_history
  evidence_visibility_history
  unresolved_user_constraints
  pending_clarification_questions
```

## 多轮流程

```text
user message
  -> classify turn type
  -> rewrite follow-up into standalone question if needed
  -> decide answer from existing EvidenceContext or retrieve more
  -> update EvidenceContext
  -> build ContextUsageSnapshot
  -> auto-compact when projected usage crosses thresholds
  -> pack / compress context if budget pressure is high
  -> answer / ask clarification / abstain
  -> write turn-level PromptManifest and AnswerProvenance
```

Turn type：

```text
new_question
follow_up_question
clarification_answer
correction_or_constraint
topic_shift
```

## 上下文压缩策略

参考 Claude Code 的 session / compact 思路，多轮对话中区分：

```text
short-term active context
  当前 turn 和最近 turn 的 raw evidence、用户约束、未解决问题。

long-term compact memory
  已确认事实、已回答问题、保留引用、用户偏好和仍有效的约束。
```

当上下文接近预算上限时：

```text
1. 保留最近 N 轮原文。
2. 将更早轮次压缩成 conversation_summary。
3. citation_history 不丢失，只压缩展示文本。
4. 未解决的 missing_facets / pending questions 不丢失。
5. topic_shift 时启动新的 EvidenceContext，但保留用户明确约束。
6. 每轮都记录当前回答实际复用了哪些 previous chunk_ids / compressed_context_ids。
```

## 输出

```text
outputs/dialogue_agent/
  session_eval/
    run_config.json
    dialogue_traces.jsonl
    context_memory_traces.jsonl
    compression_traces.jsonl
    prompt_manifests.jsonl
    answer_provenance.jsonl
    evidence_visibility_traces.jsonl
    metrics.json
    metrics.md
```

## 核心指标

```text
avg_turns
avg_retrieval_rounds_per_turn
memory_reuse_rate
follow_up_rewrite_valid_rate
clarification_request_count
context_compression_count
citation_continuity_rate
unsupported_answer_count
turn_prompt_manifest_rate
turn_answer_seen_chunk_rate
memory_source_retention_rate
```

第一版可以先构造少量 multi-turn synthetic cases：

```text
turn1: ask a Wix task question
turn2: ask "what about for services instead of products?"
turn3: add a constraint such as "I use Wix Bookings"
turn4: ask for final steps
```

---

# Legacy Common Phase: Citation-aware Answer Generation

> 保留原计划中的答案生成目标作为历史说明；新主线以 Route B Phase 11B 为准，答案输入来自 answer-ready EvidenceContext，而不是固定 top-k articles。

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

# Legacy Common Phase: Verifier / Abstention

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

# Advanced Optional Phase: Pairwise Evidence Reranker

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

## 当前推荐 MVP 版本

目标：从 Phase 6 之后切到 Evidence Context Loop，先证明“上下文状态 + 缺口诊断 + 上下文打包压缩”比继续调 merged-pool rerank 更有效。

```text
Phase 1: 数据接入与统计分析
Phase 2: BM25 Chunk-level Retrieval Baseline
Phase 3: Dense Retrieval Baseline
Phase 4: Hybrid Retrieval with RRF
Phase 5: Qwen3 Chunk Reranker Baseline
Phase 6: Error Analysis & Trace Logging

Route B Phase 7B: Evidence Context Construction
Route B Phase 8B: Evidence Gap Diagnosis
Route B Phase 9B: Iterative Evidence Completion Loop
Route B Shared Layer 10B: Context Budget Manager / Auto-Compaction
Route B Phase 11B: Citation-aware Answer / Verifier
Route B Phase 12B: Multi-turn User Dialogue
```

第一阶段优先级：

```text
1. 先实现 Phase 7B，复用现有 reranker artifacts，不重新调用 LLM。
2. 再实现 Phase 8B / 9B，让 checker 围绕 EvidenceContext 做缺口诊断和补证据循环。
3. 然后实现 Shared Layer 10B，让每次 LLM call 前都能按阈值自动 packing / compaction。
4. 最后接 Phase 11B 答案生成和 Phase 12B 多轮对话。
```

---

## Route A 对照分支

```text
Route A Phase 7A: Rule-based Second-hop Retrieval
Route A Phase 8A: LLM Evidence Checker Pool-level Eval
Route A Phase 9A: LLM Gap-query Merged Pool Rerank
```

用途：

```text
保留旧路线结果，用于说明为什么需要 EvidenceContext：
1. rule title expansion 只能补回少量 C 类 pool evidence；
2. LLM gap query 可以改善 pool coverage，但不能保证进入 top10；
3. 纯 reranker 仍然偏向单 chunk 相关性，不能稳定优化证据集合完整性。
```

---

## 完整企业化版本

```text
Route B MVP
+ Route B Phase 11B: Citation-aware Answer / Verifier
+ Route B Phase 12B: Multi-turn User Dialogue
```

---

## 高级算法版本

```text
完整企业化版本
+ Advanced Optional Phase: Pairwise Evidence Reranker
```

可选增强：

```text
1. facet-aware chunk grounding
2. article-level context packing
3. pairwise / setwise evidence reranker
4. dialogue-level memory and personalization
```

---

## 6. 最终简历目标

完成后，简历可以写：

```text
构建面向企业客服知识库的可追溯 Evidence-Completion Agentic RAG 系统，基于 WixQA 实现 BM25、Dense Retrieval、RRF Fusion、Cross-Encoder Reranker 等强检索基线，并针对多文档问题中支持文章召回不完整的问题，设计 EvidenceContext 状态管理、facet-level Evidence Gap Diagnosis、Iterative Evidence Completion Loop、Context Budget Manager / Auto-Compaction 和多轮 DialogueState；通过 PromptManifest、GapQueryProvenance 和 AnswerProvenance 记录 checker / query generation / answer / verifier 实际看过的 chunk_id，以 evidence sufficiency、article_recall@k、multi-article coverage、context usage ratio、compression token reduction、claim_support_map_rate 和 agent trace 评估检索完整性、答案可追溯性与系统成本。
```

如果最终加入答案生成，可以补充：

```text
进一步实现 Citation-aware Answer Generation、Verifier / Abstention 和多轮用户对话机制，使系统在证据不足时能够继续补证据、追问澄清或拒答，降低企业客服场景下的幻觉风险。
```

---

## 7. 当前开发状态

## Implemented

```text
Phase 1: Data ingestion and dataset statistics
Phase 2: BM25 Chunk-level Retrieval Baseline
Phase 3: Dense Retrieval Baseline
Phase 4: Hybrid Retrieval with RRF
Phase 5: Qwen3 Chunk Reranker Baseline
Phase 6: Error Analysis & Trace Logging
Route A Phase 7A: Rule-based Second-hop Retrieval implementation
Route A Phase 8A: LLM Evidence Checker pool-level eval implementation
Route A Phase 9A: LLM Gap-query Merged Pool Rerank Loop implementation
```

## Next

```text
Route B Phase 7B: Evidence Context Construction
Route B Phase 8B: Evidence Gap Diagnosis
Route B Phase 9B: Iterative Evidence Completion Loop
Route B Shared Layer 10B: Context Budget Manager / Auto-Compaction
```

## Planned

```text
Route B Phase 11B: Citation-aware Answer / Verifier
Route B Phase 12B: Multi-turn User Dialogue
Advanced Optional Phase: Pairwise Evidence Reranker
```
