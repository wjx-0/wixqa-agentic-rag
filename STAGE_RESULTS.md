# Retrieval And Reranker Stage Results

本文档记录当前 WixQA `wixqa_expertwritten` 主测试集上的阶段性结果、每一步采用的优化方法，以及相对上一阶段的数据提升。

## Evaluation Setup

当前主线评测口径：

```text
dataset = wixqa_expertwritten
records = 200
evaluation unit = chunk
gold signal = chunk.article_id in gold_article_ids
```

WixQA 没有 gold chunk IDs，因此所有 chunk-level 指标仍然用 chunk 所属文章是否命中 gold article 判断。

核心指标：

```text
chunk_full_article_hit@k = top-k chunks 覆盖全部 gold article_ids 的比例
chunk_article_recall@k   = top-k chunks 覆盖 gold article_ids 的平均召回率
chunk_hit@k              = top-k chunks 至少命中一个 gold article_id 的比例
MRR                      = 第一个命中 gold article_id 的倒数排名均值
```

> 注意：当前 Hybrid 对比表中，BM25 / Dense 的 MRR 来自各自分支完整 top100，Hybrid / Reranker 的 MRR 来自最终 top50。阶段对比时优先看 `full@10`、`recall@10` 和候选池 cutoff 下的 `full@k`。

## Overall Progress

主线结果使用同一组 Hybrid 候选配置：

```text
BM25 top100 + Dense top100
-> RRF top50
rrf_k = 60
bm25_weight = 1
dense_weight = 2
-> Qwen3 Reranker top50
```

| Stage | Method | full@10 | recall@10 | hit@10 | full@50 | recall@50 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Chunk BM25 | 0.5150 | 0.5717 | 0.6250 | 0.7100 | 0.7525 | 0.3768 |
| 3 | Dense FAISS | 0.6750 | 0.7375 | 0.7950 | 0.8800 | 0.9133 | 0.5051 |
| 4 | Hybrid RRF | 0.7050 | 0.7625 | 0.8200 | 0.8950 | 0.9250 | 0.5096 |
| 5 | Hybrid RRF + Qwen3 Reranker | 0.8000 | 0.8442 | 0.8900 | 0.8950 | 0.9250 | 0.5194 |

从 Chunk BM25 到 Qwen3 Reranker，主线提升：

| Metric | BM25 | Reranker | Absolute Gain |
| --- | ---: | ---: | ---: |
| full@10 | 0.5150 | 0.8000 | +0.2850 |
| recall@10 | 0.5717 | 0.8442 | +0.2725 |
| hit@10 | 0.6250 | 0.8900 | +0.2650 |
| full@50 | 0.7100 | 0.8950 | +0.1850 |
| recall@50 | 0.7525 | 0.9250 | +0.1725 |
| MRR | 0.3768 | 0.5194 | +0.1426 |

## Stage 1: Data And Chunk Preparation

优化方法：

- 将 WixQA 原始数据统一转换为 `KBArticle`、`QAExample`、`KBChunk` 三类结构。
- 使用 `BAAI/bge-m3` tokenizer 切分知识库文章，保证 BM25、Dense、Hybrid、Reranker 复用同一套 chunks。
- chunk 参数保持固定：

```text
chunk_size_tokens = 512
chunk_overlap_tokens = 128
chunk text = title + "\n" + contents
```

作用：

- 避免后续 Hybrid 和 Reranker 使用不同 chunk 边界。
- 为 Dense FAISS 和 Qwen3 Reranker 提供同一份 `chunk.text`。

## Stage 2: Chunk BM25 Baseline

优化方法：

- 使用同一套 `BAAI/bge-m3` tokenizer chunks。
- BM25 只做 lexical chunk retrieval，不做 article 聚合。
- 直接评测 top100 chunks。

结果：

| Metric | Value |
| --- | ---: |
| hit@10 | 0.6250 |
| full@10 | 0.5150 |
| recall@10 | 0.5717 |
| full@50 | 0.7100 |
| recall@50 | 0.7525 |
| full@100 | 0.7750 |
| recall@100 | 0.8125 |
| MRR | 0.3768 |

观察：

- BM25 对关键词匹配强的问题有效，但语义改写和多文章问题较弱。
- 多文章问题 `full@10 = 0.3077`，说明第一阶段召回完整证据比较困难。

## Stage 3: Dense FAISS Retrieval

优化方法：

- 使用 `BAAI/bge-m3` 编码 `chunk.text`。
- embedding normalize 后使用 `FAISS IndexFlatIP`。
- 仍然直接输出 top100 chunks，不聚合 article ranking。

结果：

| Metric | BM25 | Dense FAISS | Gain |
| --- | ---: | ---: | ---: |
| hit@10 | 0.6250 | 0.7950 | +0.1700 |
| full@10 | 0.5150 | 0.6750 | +0.1600 |
| recall@10 | 0.5717 | 0.7375 | +0.1658 |
| full@50 | 0.7100 | 0.8800 | +0.1700 |
| recall@50 | 0.7525 | 0.9133 | +0.1608 |
| full@100 | 0.7750 | 0.9250 | +0.1500 |
| recall@100 | 0.8125 | 0.9475 | +0.1350 |
| MRR | 0.3768 | 0.5051 | +0.1283 |

观察：

- Dense 是当前最大的一次召回提升。
- 语义匹配能力显著强于 BM25。
- 多文章问题仍然没有完全解决：Dense `multi full@10 = 0.3846`。

## Stage 4: Hybrid RRF Fusion

优化方法：

- BM25 和 Dense 各召回 top100 chunks。
- 按 `chunk_id` 做 RRF 融合。
- 不做 article 聚合，最终仍输出 chunks。
- 增加分支权重：

```text
rrf_score = bm25_weight / (rrf_k + bm25_rank)
          + dense_weight / (rrf_k + dense_rank)
```

主线配置：

```text
branch_top_k_chunks = 100
fused_top_k_chunks = 50
rrf_k = 60
bm25_weight = 1
dense_weight = 2
```

结果：

| Metric | Dense FAISS | Hybrid RRF | Gain |
| --- | ---: | ---: | ---: |
| hit@10 | 0.7950 | 0.8200 | +0.0250 |
| full@10 | 0.6750 | 0.7050 | +0.0300 |
| recall@10 | 0.7375 | 0.7625 | +0.0250 |
| full@50 | 0.8800 | 0.8950 | +0.0150 |
| recall@50 | 0.9133 | 0.9250 | +0.0117 |
| MRR | 0.5051 | 0.5096 | +0.0045 |

观察：

- Hybrid 的提升比 Dense 相对 BM25 的提升小，但方向正确。
- top50 候选池完整覆盖从 Dense 的 `0.8800` 提到 `0.8950`，说明 BM25 分支补到了一部分 Dense 没召到的证据。
- RRF 对 top10 排序也有帮助，但提升有限，后续需要 reranker。

### Hybrid Tuning Notes

已尝试的 expertwritten top50 配置：

| Config | full@10 | recall@10 | full@50 | recall@50 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| k60, bw1, dw2 | 0.7050 | 0.7625 | 0.8950 | 0.9250 | 0.5096 |
| k60, bw1, dw2.5 | 0.7050 | 0.7642 | 0.8900 | 0.9233 | 0.5168 |
| k70, bw1, dw2.5 | 0.7050 | 0.7600 | 0.8900 | 0.9233 | 0.5158 |
| k80, bw1, dw2 | 0.6900 | 0.7500 | 0.8950 | 0.9250 | 0.5069 |
| k80, bw1, dw3 | 0.7100 | 0.7667 | 0.8900 | 0.9233 | 0.5045 |

结论：

- `k60, bw1, dw2` 是当前 reranker 主线使用的候选池，top50 覆盖强，top10 也稳定。
- `k80, bw1, dw3` 的 top10 略高，但 top50 full 低一点。
- 权重和 `rrf_k` 调整有收益，但不如 Dense 和 Reranker 阶段明显。

## Stage 5: Qwen3 Reranker

优化方法：

- 复用 Hybrid 的 `candidates.jsonl`，不重新运行 BM25、Dense 或 FAISS。
- 使用 `Qwen/Qwen3-Reranker-0.6B`。
- 每个候选输入为：

```text
(question, chunk.text)
```

- 按 `rerank_score` 对 Hybrid top50 chunks 重新排序。
- 排序只改变顺序，不新增候选。

服务器运行配置：

```text
source_hybrid_run = hybrid_rrf_b100_f50_k60_bw1_dw2_wixqa_expertwritten
candidate_top_k_chunks = 50
model_name = Qwen/Qwen3-Reranker-0.6B
device = cuda
rerank_batch_size = 32
max_length = 1024
instruction_name = wixqa_help_center_v1
```

结果：

| Metric | Hybrid RRF | Qwen3 Reranker | Gain |
| --- | ---: | ---: | ---: |
| hit@10 | 0.8200 | 0.8900 | +0.0700 |
| full@10 | 0.7050 | 0.8000 | +0.0950 |
| recall@10 | 0.7625 | 0.8442 | +0.0817 |
| full@50 | 0.8950 | 0.8950 | +0.0000 |
| recall@50 | 0.9250 | 0.9250 | +0.0000 |
| MRR | 0.5096 | 0.5194 | +0.0098 |

诊断：

| Diagnostic | Value |
| --- | ---: |
| source_candidate_full_article_hit | 0.8950 |
| top50 full count | 179 / 200 |
| top10 full count after rerank | 160 / 200 |
| rerank_top10_rescued_gold_articles | 32 |
| rerank_top10_dropped_gold_articles | 8 |
| A_top10_chunks_full | 160 |
| B_top50_chunks_full_not_top10_chunks | 19 |
| C_top50_chunks_not_full | 21 |

观察：

- `full@50` 和 `recall@50` 不变是正确现象，因为 reranker 只重排 Hybrid top50，不会新增候选。
- `full@10` 从 `0.7050` 提升到 `0.8000`，说明 Qwen3 成功把候选池里的相关证据推到了前排。
- rescued 多于 dropped：`32` vs `8`，整体排序收益明确。
- 仍有 `21` 个问题在 Hybrid top50 中没有完整 gold article，reranker 无法凭空补召回。

### Single vs Multi Article

| Group | Records | full@10 | recall@10 | full@50 | recall@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| single | 148 | 0.8851 | 0.8851 | 0.9527 | 0.9527 |
| multi | 52 | 0.5577 | 0.7276 | 0.7308 | 0.8462 |

结论：

- 单文章问题已经比较强。
- 多文章问题仍然是主要瓶颈。单 chunk reranker 倾向判断单个 chunk 的相关性，不天然保证证据集合覆盖完整。

## Current Conclusion

当前最清晰的阶段收益链条是：

```text
BM25 解决 lexical baseline
Dense FAISS 带来最大召回提升
Hybrid RRF 小幅补充 Dense 遗漏证据
Qwen3 Reranker 显著提升 top10 排序质量
```

当前最好的一组已跑完整 reranker 的主线结果：

```text
Hybrid source: b100_f50_k60_bw1_dw2
Reranker: Qwen/Qwen3-Reranker-0.6B
full@10 = 0.8000
recall@10 = 0.8442
full@50 = 0.8950
recall@50 = 0.9250
```

## Recommended Next Steps

1. 对 `k80, bw1, dw2` 和 `k80, bw1, dw3` 的 Hybrid 候选池也跑 reranker，对比 top10 是否继续提升。
2. 跑 top100 reranker 诊断，确认扩大候选池是否能减少 `C_top50_chunks_not_full` 类型问题。
3. 针对多文章问题实现 coverage-aware selection，避免 reranker 只把同一篇文章的多个 chunks 排到前面。
4. 回头修复 `REVIEW_NOTES.md` 中记录的工程问题：动态 cutoff、MRR cutoff 口径、Hybrid artifacts 一致性校验。

