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

根据 error analysis，优先处理候选池本身缺少证据的 C 类：

```text
B_top50_chunks_full_not_top10_chunks
  -> 候选池已有完整证据
  -> 暂不单独增加 selection 阶段

C_top50_chunks_not_full
  -> 候选池本身缺少证据
  -> 使用 Phase 7 Rule-based Second-hop Retrieval
```

后续先实现 Phase 7 Rule-based Second-hop Retrieval，验证补检索能否提升 multi-article coverage。

---

# Phase 7: Rule-based Second-hop Retrieval

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

# Phase 8: LLM Evidence Sufficiency Checker + Pool-level Gap Query Eval

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
  "missing_evidence": [],
  "next_queries": [],
  "reason": ""
}
```

Prompt 强约束：

```text
1. Do not answer the user's question.
2. Only judge whether retrieved chunks contain enough evidence.
3. missing_evidence must describe the concrete missing evidence.
4. next_queries must target missing_evidence, not merely rewrite the original question.
5. next_queries must contain concrete Wix product, feature, action, setting, integration, error, or entity names.
6. Do not use vague pronouns like it, this, that feature, or that setting.
7. Generate at most 3 next_queries.
8. Return valid JSON only.
```

## Pool-level Eval

```text
checker insufficient + next_queries
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
```

---

# Phase 9: LLM Gap-query Merged Pool Rerank Loop

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
若 multi full@10 提升且 LLM_C_top10_rescued_count > A_dropped@10_count，保留该流程作为 Agentic RAG 主线。
```

---

# Phase 10: Citation-aware Answer Generation

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

# Phase 11: Verifier / Abstention

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

# Phase 12: Optional - Pairwise Evidence Reranker

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
Phase 5: Qwen3 Chunk Reranker Baseline
Phase 6: Error Analysis & Trace Logging
Phase 7: Rule-based Second-hop Retrieval
Phase 8: LLM Evidence Sufficiency Checker
Phase 9: LLM Gap-query Merged Pool Rerank Loop
```

预计时间：

```text
4 - 7 天
```

---

## 完整企业化版本

```text
MVP
+ Phase 10: Citation-aware Answer Generation
+ Phase 11: Verifier / Abstention
```

预计时间：

```text
7 - 10 天
```

---

## 高级算法版本

```text
完整企业化版本
+ Phase 12: Pairwise Evidence Reranker
```

预计时间：

```text
2 - 3 周
```

---

## 6. 最终简历目标

完成后，简历可以写：

```text
构建面向企业客服知识库的 Evidence-Completion Agentic RAG 系统，基于 WixQA 实现 BM25、Dense Retrieval、RRF Fusion、Cross-Encoder Reranker 等强检索基线，并针对多文档问题中支持文章召回不完整的问题，引入 Evidence Sufficiency Checker、Gap-aware Query Generation 和 Bounded Second-hop Retrieval，以 article_full_hit@k、article_recall@k、avg_llm_calls 和 agent trace 评估检索完整性与系统成本。
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
Phase 4: Hybrid Retrieval with RRF
Phase 5: Qwen3 Chunk Reranker Baseline
Phase 6: Error Analysis & Trace Logging
Phase 7: Rule-based Second-hop Retrieval implementation
Phase 8: LLM Evidence Checker pool-level eval implementation
Phase 9: LLM Gap-query Merged Pool Rerank Loop implementation
```

## Next

```text
Phase 9: Server run over Phase 8 merged pools
```

## Planned

```text
Phase 10: Citation-aware Answer Generation
Phase 11: Verifier / Abstention
Phase 12: Pairwise Evidence Reranker
```
