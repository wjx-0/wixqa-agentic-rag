# WixQA Agentic RAG

面向企业客服知识库，后续实现证据补全式 Agentic RAG。

当前阶段：**数据接入、数据统计、BM25 baseline、Dense FAISS baseline**。

当前仓库已经完成从 Hugging Face 加载 WixQA 基准数据集、转换为项目统一 JSONL 格式、生成数据统计结果，并实现三类 retrieval baseline：

```text
article-level BM25
chunk-level BM25
dense FAISS over BAAI/bge-m3 chunks
```

本阶段仍不实现 BM25 + Dense 融合、RRF、Reranker、LLM 调用、答案生成或 Agent 循环。

## 数据集

数据集：[Wix/WixQA](https://huggingface.co/datasets/Wix/WixQA)

更详细的数据集说明见 [DATASET.md](DATASET.md)。

当前观察到的 Hugging Face 数据配置：

```text
wix_kb_corpus       -> 企业客服知识库 corpus
wixqa_expertwritten -> 主测试集
wixqa_simulated     -> 辅助测试 / 泛化测试集
wixqa_synthetic     -> 流程冒烟检查 / 弱监督数据
```

脚本会在运行时调用 `get_dataset_config_names("Wix/WixQA")`，并将实际发现的配置到项目标准名称的映射保存到：

```text
outputs/data_inspection/loaded_config_mapping.json
```

## 项目结构

```text
wixqa-agentic-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── stats/
├── src/
│   ├── data/
│   │   ├── chunk_wixqa.py
│   │   ├── load_wixqa.py
│   │   ├── preprocess_wixqa.py
│   │   └── schema.py
│   ├── retrievers/
│   │   ├── bm25_retriever.py
│   │   ├── chunk_bm25_retriever.py
│   │   ├── dense_faiss_retriever.py
│   │   ├── faiss_store.py
│   │   └── tokenizer.py
│   ├── utils/
│   │   ├── io_utils.py
│   │   └── text_utils.py
│   └── evaluation/
│       ├── data_stats.py
│       ├── retrieval_metrics.py
│       ├── run_bm25_eval.py
│       ├── run_chunk_bm25_eval.py
│       └── run_dense_faiss_eval.py
├── scripts/
│   ├── build_faiss_index.py
│   ├── compare_chunk_methods.py
│   ├── download_wixqa.py
│   ├── prepare_wixqa.py
│   ├── prepare_chunks.py
│   ├── inspect_wixqa.py
│   ├── run_bm25_baseline.py
│   ├── run_chunk_bm25_baseline.py
│   └── run_dense_faiss_baseline.py
├── indexes/
│   └── faiss_bge_m3/
└── outputs/
    ├── data_inspection/
    ├── bm25_baseline/
    ├── chunk_bm25_baseline/
    └── dense_faiss_baseline/
```

## 安装

```bash
pip install -r requirements.txt
```

如果使用项目约定的 conda 环境：

```bash
conda activate wixqa-agentic-rag
pip install -r requirements.txt
```

## 运行

查看数据配置、数据划分、字段名和样例：

```bash
python scripts/inspect_wixqa.py
```

下载 / 缓存 WixQA，并完整导出原始 JSONL 与样例信息：

```bash
python scripts/download_wixqa.py
```

将 WixQA 转换为项目统一格式：

```bash
python scripts/prepare_wixqa.py
```

生成数据统计：

```bash
python -m src.evaluation.data_stats --processed_dir data/processed --output_dir data/stats
```

运行 Phase 2 的 BM25 article-level retrieval baseline：

```bash
python scripts/run_bm25_baseline.py --dataset wixqa_expertwritten --top_k 50
python scripts/run_bm25_baseline.py --dataset wixqa_simulated --top_k 50
python scripts/run_bm25_baseline.py --dataset wixqa_synthetic --top_k 50
```

生成基于 `BAAI/bge-m3` tokenizer 的 KB chunks：

```bash
python scripts/prepare_chunks.py \
  --chunk_size_tokens 512 \
  --chunk_overlap_tokens 128
```

运行 BM25 chunk-level retrieval baseline：

```bash
python scripts/run_chunk_bm25_baseline.py --dataset wixqa_expertwritten
python scripts/run_chunk_bm25_baseline.py --dataset wixqa_simulated
python scripts/run_chunk_bm25_baseline.py --dataset wixqa_synthetic
```

chunk-level baseline 默认：

```text
top_k_chunks = 100
top_k_articles = 30
```

即先检索 top 100 chunks，再按 `article_id` 聚合为 top 30 articles，最后按 article-level gold labels 评测。

生成 article-level BM25 与 chunk-level BM25 对比报告：

```bash
python scripts/compare_chunk_methods.py \
  --article_bm25_dir outputs/bm25_baseline \
  --chunk_bm25_dir outputs/chunk_bm25_baseline \
  --output_path outputs/chunk_bm25_baseline/compare_chunk_methods.md
```

构建 Dense FAISS index：

```bash
python scripts/build_faiss_index.py \
  --processed_dir data/processed \
  --chunks_path data/processed/wix_kb_chunks.jsonl \
  --index_dir indexes/faiss_bge_m3 \
  --model_name BAAI/bge-m3 \
  --local_files_only true \
  --batch_size 16
```

运行 Dense FAISS retrieval baseline：

```bash
python scripts/run_dense_faiss_baseline.py \
  --processed_dir data/processed \
  --index_dir indexes/faiss_bge_m3 \
  --dataset wixqa_expertwritten \
  --model_name BAAI/bge-m3 \
  --local_files_only true \
  --top_k_chunks 100 \
  --top_k_articles 30
```

如果 Hugging Face 暂时不可访问，请在网络恢复后重试相同命令。`prepare_chunks.py` 会加载 `BAAI/bge-m3` tokenizer；离线复现时可先缓存 tokenizer，再使用 `--local_files_only`。

## 输出文件

处理后的 JSONL：

```text
data/processed/wix_kb_corpus.jsonl
data/processed/wixqa_expertwritten.jsonl
data/processed/wixqa_simulated.jsonl
data/processed/wixqa_synthetic.jsonl
data/processed/wix_kb_chunks.jsonl
```

统计结果：

```text
data/stats/wixqa_data_stats.json
data/stats/wixqa_data_stats.md
```

便于人工检查的样例：

```text
outputs/data_inspection/kb_samples.json
outputs/data_inspection/expertwritten_samples.json
outputs/data_inspection/simulated_samples.json
outputs/data_inspection/synthetic_samples.json
outputs/data_inspection/multi_article_samples.md
outputs/data_inspection/chunk_samples.json
outputs/data_inspection/chunk_stats.md
```

原始完整数据与样例信息：

```text
data/raw/wixqa_raw_summary.json
data/raw/<config>_<split>.jsonl
data/raw/samples/<config>_samples.json
```

BM25 baseline 输出：

```text
outputs/bm25_baseline/<dataset>_metrics.json
outputs/bm25_baseline/<dataset>_metrics.md
outputs/bm25_baseline/<dataset>_retrieval_traces.jsonl
outputs/bm25_baseline/<dataset>_cases_A_top10_full.jsonl
outputs/bm25_baseline/<dataset>_cases_B_top50_full_not_top10.jsonl
outputs/bm25_baseline/<dataset>_cases_C_top50_not_full.jsonl
```

Chunk BM25 baseline 输出：

```text
outputs/chunk_bm25_baseline/<dataset>_metrics.json
outputs/chunk_bm25_baseline/<dataset>_metrics.md
outputs/chunk_bm25_baseline/<dataset>_retrieval_traces.jsonl
outputs/chunk_bm25_baseline/<dataset>_cases_A_top10_full.jsonl
outputs/chunk_bm25_baseline/<dataset>_cases_B_top30_full_not_top10.jsonl
outputs/chunk_bm25_baseline/<dataset>_cases_C_top30_not_full.jsonl
outputs/chunk_bm25_baseline/compare_chunk_methods.md
```

Dense FAISS index 输出：

```text
indexes/faiss_bge_m3/faiss.index
indexes/faiss_bge_m3/chunk_metadata.jsonl
indexes/faiss_bge_m3/index_config.json
```

Dense FAISS baseline 输出：

```text
outputs/dense_faiss_baseline/dense_bge-m3_<dataset>_metrics.json
outputs/dense_faiss_baseline/dense_bge-m3_<dataset>_metrics.md
outputs/dense_faiss_baseline/dense_bge-m3_<dataset>_retrieval_traces.jsonl
outputs/dense_faiss_baseline/dense_bge-m3_<dataset>_cases_A_top10_full.jsonl
outputs/dense_faiss_baseline/dense_bge-m3_<dataset>_cases_B_top30_full_not_top10.jsonl
outputs/dense_faiss_baseline/dense_bge-m3_<dataset>_cases_C_top30_not_full.jsonl
```

## 数据格式

`KBArticle`:

```text
article_id, title, url, contents, article_type, metadata
```

`QAExample`:

```text
qid, dataset_name, question, answer, article_ids, num_gold_articles, is_multi_article, metadata
```

`KBChunk`:

```text
chunk_id, article_id, chunk_index, title, url, text, contents, start_token, end_token, num_tokens, metadata
```

其中：

```text
contents = 当前 chunk 的正文片段
text = title + "\n" + contents
metadata.chunk_tokenizer = BAAI/bge-m3
```

如果原始数据没有 `qid`，`prepare_wixqa.py` 会生成稳定 ID，例如 `expertwritten_000001`。

## Phase 2-3: Retrieval Baselines

当前实现 BM25 与 Dense FAISS 检索基线。

Article-level BM25：

```text
每篇 Wix KB article 作为一个检索单元
检索文本使用 title + contents
```

Chunk-level BM25：

```text
使用 BAAI/bge-m3 tokenizer 将 KB article 切成 512-token chunks
chunk overlap = 128 tokens
BM25 检索 chunk
按 article_id 聚合回 article 排名
```

注意：chunk 边界使用 `BAAI/bge-m3` tokenizer，是为了后续 Dense / Hybrid Retrieval 复用同一套 chunks。BM25 打分本身仍然是 lexical BM25。

Dense FAISS：

```text
复用 data/processed/wix_kb_chunks.jsonl
使用 BAAI/bge-m3 编码 chunk.text
normalize embeddings = true
FAISS index type = IndexFlatIP
检索 top 100 chunks
按 article_id 聚合为 top 30 articles
```

评测仍然是 article-level，因为 WixQA 的 gold labels 是 `article_ids`。

当前指标包括：

```text
article_hit@k
article_full_hit@k
article_recall@k
article_precision@k
MRR
```

Chunk-level BM25 与 Dense FAISS 的 case 文件按 top30 口径划分：

```text
A_top10_full              -> top10 已包含全部 gold articles
B_top30_full_not_top10    -> top30 包含全部 gold articles，但 top10 未包含全部
C_top30_not_full          -> top30 仍未包含全部 gold articles
```

本阶段不实现 BM25 + Dense 融合、RRF、Reranker、LLM 或 Agentic RAG；这些能力会在后续阶段加入。

## 后续路线图

```text
阶段 1：数据接入与统计
阶段 2：BM25 Article-level Retrieval Baseline
阶段 3：BM25 Chunk-level Retrieval Baseline
阶段 4：Dense FAISS Retrieval Baseline
阶段 5：Hybrid Retrieval with RRF Fusion
阶段 6：Cross-Encoder Reranker
阶段 7：Error Analysis & Trace Logging
阶段 8：Rule-based Second-hop Retrieval
阶段 9：证据充分性检查器与有边界的 Agentic RAG 循环
阶段 10：带引用的答案生成
```
