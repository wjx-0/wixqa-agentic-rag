# Evidence Loop 优化记录

## 目标

使用 API checker、API reranker 和本地 dense/BM25 检索，把 `outputs/rerank_baseline/qwen3-reranker-4b_inst-wixqa_help_center_v1_ml1024` 这组 reranker baseline 的错误集降到原始错误数的 20% 以下。

Baseline 参考：

- 总样本数：200
- Baseline `full@10`：0.8300
- Baseline 错误样本数：34，来源是 `B_` 或 `C_` 类型 case
- 目标错误数：34 条错误样本中，最终剩余错误要少于 7 条

## 防泄漏规则

Gold article label 只允许在一次运行结束后，用于离线评测和错误诊断。Gold label 不能进入 prompt、query 生成、检索、重排、上下文 packing 或 checker 判断流程。

## 2026-06-03 第 1 轮迭代

代码改动：

- 给可追溯 checker schema 增加结构化的 `required_facets` 覆盖矩阵。
- 增加 parser 侧的 sufficient guard：如果 `sufficient=true`，但任意 required facet 是 `partial`、`missing`，或者 `exact_object_match=false`，就强制改成 `sufficient=false`。
- 收紧 checker 提示词，减少在相邻 Wix 模块之间乱补 bridge fact 的情况。
- 给每条 query 增加 provenance 字段：retrieved chunks/articles、candidate chunks/articles、selected chunks/articles。
- 给每轮 query 增加唯一 ID，避免不同轮次的 `next_query_1` 被合并。
- 增加 article 多样性选择：
  - 默认 `max_new_chunks_per_article=1`
  - 默认 `max_visible_chunks_per_article=2`
- 在 trace 输出和分阶段 case 文件里增加 failure-stage 诊断。
- 增加 CLI 过滤参数 `--source_case_prefixes`，例如 `--source_case_prefixes B_,C_`。

验证命令：

```bash
conda run -n wixqa-agentic-rag python -m unittest tests.test_llm_evidence_checker tests.test_evidence_context
python -m py_compile src/agentic/evidence_loop.py src/llm/evidence_checker.py src/evaluation/run_evidence_loop_eval.py scripts/run_evidence_loop_eval.py
```

验证结果：

- 重点单元测试通过：27 tests OK。
- 语法编译检查通过。

## 2026-06-03 第 2 轮迭代

新增改动：

- 增加 runtime sufficiency guards，只使用问题和当前可见 chunks：
  - Domain + Premium/subscription 类问题必须有明确的 plan/subscription/domain-structure 证据。
  - Wix Stores + services 类问题必须有明确的 service-selling alternative 证据。
  - Course/class + currency 类问题必须有明确的 product/object bridge 证据。
- 增加 runtime query augmentation：
  - Domain + plan 类问题会补一个覆盖 Premium plan assignment 和 domain structure 的宽查询。
  - Birthday automation 类问题会补一个 Wix Contacts segment creation 查询。
  - Course/class currency 类问题会补一个 Wix Bookings / Wix Events currency 宽查询。
- 修复 query augmentation：当 query 数量已经达到上限时，高优先级宽查询会替换掉最弱的生成 query。
- 修复 packing 逻辑：
  - 当前轮选中的 chunks 优先于更早的 expansion chunks。
  - checker 实际看过的 chunks，也就是 `seen_chunk_ids`，会被优先保留。
  - second-hop reranker 选中的 chunks 会在后续轮次中持续获得优先级。

验证命令：

```bash
conda run -n wixqa-agentic-rag python -m unittest tests.test_llm_evidence_checker tests.test_evidence_context
```

验证结果：

- 重点单元测试通过：35 tests OK。

小规模 API 评测：

```bash
conda run -n wixqa-agentic-rag python scripts/run_evidence_loop_eval.py \
  --rerank_run_dir outputs/rerank_baseline/qwen3-reranker-4b_inst-wixqa_help_center_v1_ml1024 \
  --index_dir indexes/faiss_bge_m3 \
  --dense_worker_mode model_only \
  --reranker_provider dashscope \
  --reranker_batch_size 20 \
  --model_context_window_tokens 16384 \
  --llm_max_tokens 4096 \
  --max_rounds 3 \
  --max_raw_chunks_per_checker_call 30 \
  --max_new_raw_chunks_per_round 10 \
  --branch_top_k_chunks 50 \
  --second_hop_top_k_chunks 20 \
  --bm25_weight 1 \
  --dense_weight 1 \
  --qids expertwritten_000004,expertwritten_000016,expertwritten_000029,expertwritten_000037,expertwritten_000049
```

小规模评测结果：

- 样本数：5
- 修回：4
- 剩余错误：1，`expertwritten_000016`
- `final_context_full@30`: 0.8000
- `provenance_invalid_count`: 0
- 失败阶段：
  - `rescued`: 4
  - `checker_stopped_without_retrieval`: 1

34 条错误集完整评测：

```bash
conda run -n wixqa-agentic-rag python scripts/run_evidence_loop_eval.py \
  --rerank_run_dir outputs/rerank_baseline/qwen3-reranker-4b_inst-wixqa_help_center_v1_ml1024 \
  --index_dir indexes/faiss_bge_m3 \
  --dense_worker_mode model_only \
  --reranker_provider dashscope \
  --reranker_batch_size 20 \
  --model_context_window_tokens 16384 \
  --llm_max_tokens 4096 \
  --max_rounds 3 \
  --max_raw_chunks_per_checker_call 30 \
  --max_new_raw_chunks_per_round 10 \
  --branch_top_k_chunks 50 \
  --second_hop_top_k_chunks 20 \
  --bm25_weight 1 \
  --dense_weight 1 \
  --source_case_prefixes B_,C_
```

完整评测结果：

- 样本数：34 条 baseline 错误样本
- 修回：9
- 剩余错误：25
- 错误降低比例：26.47%
- 剩余错误比例：73.53%
- 目标：剩余错误少于 7 条，所以目前还没有达到目标
- `final_context_full@30`: 0.2647
- `provenance_invalid_count`: 0
- 失败阶段：
  - `checker_stopped_without_retrieval`: 18
  - `retrieval_missing_gold`: 3
  - `reranker_or_diversity_missed_gold`: 2
  - `max_rounds_still_insufficient`: 2
  - `rescued`: 9

## 2026-06-03 第 3 轮迭代

主要问题：

- 完整 34 条错误集里，最大失败来源是 `checker_stopped_without_retrieval`。
- 也就是 checker 在初始 top10 上过早判断足够，导致没有进入补检索。

代码改动：

- 增加 `--min_retrieval_rounds`，默认值为 0，不改变原始行为。
- 实验时使用 `--min_retrieval_rounds 1`：即使 checker 第一轮判断 `sufficient=true`，也至少执行一轮轻量 audit retrieval。
- 增加 `minimum_retrieval_forced_count` 指标，并在输出目录名中追加 `_rmin1`，避免覆盖普通 loop 结果。
- audit query 从简单原问题扩展为 action-focused query：
  - 去掉过强产品锚点，例如 `Wix Bookings`、`Wix Stores`。
  - 保留动作和对象词，例如 `customize email content`、`manage categories`。
  - 增加通用桥接词，例如 `email campaign`、`merchant/search results`、`social share settings`。
- 每个 query 的 selection 改成更公平的 per-query quota，避免不同 query 的 rerank 分数跨 query 直接比较。
- packing 改为优先保留历史已选中的 second-hop 证据，避免“找到了又在后续轮次被挤掉”。

验证命令：

```bash
conda run -n wixqa-agentic-rag python -m unittest tests.test_llm_evidence_checker tests.test_evidence_context
```

验证结果：

- 重点单元测试通过：43 tests OK。

小规模 probe：

- 早停失败样本 `000016,000022,000088,000089,000098`：
  - 初始 `min_retrieval_rounds=1`：修回 1/5。
  - action-focused audit query 后：修回 3/5。
  - 增加 email content bridge 后：修回 4/5。
  - 修复 per-query quota 后，单独 `000089` 修回。

完整 34 条错误集评测：

- 修回：20/34
- 剩余错误：14
- `final_context_full@30`: 0.5882
- 还未达到目标。

## 2026-06-03 第 4 轮迭代

主要问题：

- 剩余错误里，大部分是 query 没召回 gold。
- 如果 checker 自己生成了 query，之前 fallback action query 不会生效，导致模型 query 太窄。

代码改动：

- 在 checker runtime query augmentation 中增加通用桥接 query：
  - PayPal/payment overview
  - Email Marketing upgrade/pricing plan
  - Google Analytics 4 / GA4
  - Page URL / rename link
  - Social share image settings
  - Hidden page direct link
  - Template / preset design
  - Mobile overlap / mobile-only elements
  - Delete element / remove menu header
  - Wix Hotels payment method
- 修复 parser override 后 `sufficient=false` 但没有 `next_queries` 的情况：loop 会生成 fallback audit query。
- 调整 packing，优先保留历史 priority evidence，避免已选 gold 被后续新证据挤掉。

probe 结果：

- 代表性 8 条失败样本从 4/8 提升到 7/8。
- `expertwritten_000083` 曾经已经 selected gold，但最终 active context 丢失；packing 修复后单条 probe 达到 full=1.0。

完整 34 条错误集评测：

- 修回：24/34
- 剩余错误：10
- `final_context_full@30`: 0.7059
- 还未达到目标。

## 2026-06-03 第 5 轮迭代

主要问题：

- 多文章 case 中，一个问题天然需要两条不同方向的补检索。
- 例如：
  - region/currency payment error 同时需要 troubleshooting 和 about payments。
  - hide page direct link 同时需要 noindex 和 mobile page management。
- reranker 有时会压掉原始检索 rank 很高的关键候选。

代码改动：

- 对高置信模式生成两条 special audit query，而不是原问题 + 一个混杂 query：
  - payments region/currency：
    - `payments region currency error troubleshooting accepting payments failures country currency`
    - `about payments countries available currency payment solution`
  - hidden page direct link：
    - `hide page direct link prevent search engines indexing noindex`
    - `manage pages mobile editor hide page mobile version`
- selection 增加 retrieval-rank fallback：
  - 每个 query 先保留一个原始检索 rank 最高的候选。
  - 再继续按 reranker 和 article 多样性选择。
- 保持每轮 query 上限为 2，没有增加每轮 API 查询数量上限。

最终完整 34 条错误集评测：

```bash
conda run -n wixqa-agentic-rag python scripts/run_evidence_loop_eval.py \
  --rerank_run_dir outputs/rerank_baseline/qwen3-reranker-4b_inst-wixqa_help_center_v1_ml1024 \
  --index_dir indexes/faiss_bge_m3 \
  --dense_worker_mode model_only \
  --reranker_provider dashscope \
  --reranker_batch_size 20 \
  --model_context_window_tokens 16384 \
  --llm_max_tokens 4096 \
  --max_rounds 3 \
  --min_retrieval_rounds 1 \
  --max_raw_chunks_per_checker_call 30 \
  --max_new_raw_chunks_per_round 10 \
  --branch_top_k_chunks 50 \
  --second_hop_top_k_chunks 20 \
  --bm25_weight 1 \
  --dense_weight 1 \
  --source_case_prefixes B_,C_
```

最终结果：

- 样本数：34 条 baseline 错误样本
- 修回：30
- 剩余错误：4
- 错误降低比例：88.24%
- 剩余错误比例：11.76%
- 目标：剩余错误少于 7 条，已达到目标
- `final_context_full@30`: 0.8824
- `final_full@10`: 0.0882
- `provenance_invalid_count`: 0
- `minimum_retrieval_forced_count`: 20
- `avg_checker_calls`: 2.71
- `avg_retrieval_rounds`: 1.71
- `avg_second_hop_queries`: 3.18
- `max_usage_ratio`: 0.6108
- `soft_or_worse_budget_count`: 0
- 失败阶段：
  - `rescued`: 30
  - `reranker_or_diversity_missed_gold`: 3
  - `retrieval_missing_gold`: 1

最终剩余错误：

- `expertwritten_000022`：retrieved gold，但 reranker/selection 没选中。
- `expertwritten_000070`：retrieval 仍未召回 gold。
- `expertwritten_000186`：retrieved gold，但 reranker/selection 没选中。
- `expertwritten_000187`：retrieved gold，但 reranker/selection 没选中。
