# Review Notes

## Deferred Retrieval Fixes

记录日期：2026-05-31

以下问题已经在 Hybrid RRF baseline 代码评审中确认。它们暂不影响当前使用固定实验配置得到的结果，因此推迟到后续集中修复。

### 1. 动态 top-k cutoff 指标

当前公共 chunk 指标预定义：

```text
1, 3, 5, 10, 20, 30, 50, 100
```

但 CLI 接受任意正整数。当 `--top_k_chunks` 或 `--fused_top_k_chunks` 使用 `75` 等未预定义值时，检索完成后可能因为缺少 `unique_articles@75_chunks` 等指标而报错。

当前实验使用 `50` 或 `100`，不受影响。

后续修复：

```text
将用户指定的最终 cutoff 动态加入 ks。
Chunk BM25、Dense FAISS 和 Hybrid RRF 共用同一套逻辑。
补充非预定义 cutoff 的测试，例如 75。
```

### 2. Hybrid 对比报告中的 MRR cutoff

当运行：

```text
branch_top_k_chunks = 100
fused_top_k_chunks = 50
```

对比报告中的 BM25 和 Dense MRR 会遍历各自 top100 分支结果，而 Hybrid MRR 只会遍历最终 top50 融合结果。三个方法的 MRR cutoff 口径不完全一致。

当前 top-k 命中率、完整命中率和 recall 指标不受影响。比较 MRR 时需要注意该差异。

后续修复：

```text
在对比报告中统一计算 mrr@fused_cutoff。
保留完整分支排名用于互补性分析。
明确区分完整分支 MRR 与共同 cutoff MRR。
```

### 3. Hybrid artifacts 一致性校验

Hybrid 当前会读取：

```text
data/processed/wix_kb_chunks.jsonl
indexes/faiss_bge_m3/faiss.index
indexes/faiss_bge_m3/chunk_metadata.jsonl
indexes/faiss_bge_m3/index_config.json
```

但运行前尚未强制校验：

```text
CLI model_name 与 index_config.model_name 一致
BM25 chunks 与 FAISS metadata 的 chunk_id 顺序一致
chunk 文本和数量与构建索引时一致
embedding 配置与索引配置一致
```

当前本机 artifacts 已人工验证一致：

```text
model_name = BAAI/bge-m3
chunks = 11106
FAISS metadata = 11106
chunk_id sequence = identical
embedding_dim = 1024
```

因此当前已有实验结果不受影响。

后续修复：

```text
在 Dense 和 Hybrid 启动检索前增加 fail-fast 校验。
索引配置中记录可验证的 chunk manifest 或 checksum。
补充模型不一致、chunk 顺序不一致和 metadata 数量不一致测试。
```

