# WixQA Agentic RAG

面向企业客服知识库，后续实现证据补全式 Agentic RAG。

当前阶段：**LLM Evidence Sufficiency Checker pool-level eval**。

当前仓库已经完成数据接入、数据统计、三类 chunk-level retrieval baseline、Qwen3 重排基线、错误分析、无 LLM 的规则补检索和 LLM checker 评测代码：

```text
chunk-level BM25
dense FAISS over BAAI/bge-m3 chunks
hybrid BM25 + Dense with RRF
Qwen/Qwen3-Reranker-0.6B over saved Hybrid chunks
rule-based title-expanded second-hop retrieval
LLM evidence sufficiency checker + gap-query pool eval
```

Phase 8 只做 pool-level eval：LLM 生成 gap queries 后检查合并候选池覆盖，不做最终 merged-pool rerank、答案生成或 Agent 循环。

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
│   │   ├── chunk_bm25_retriever.py
│   │   ├── dense_faiss_retriever.py
│   │   ├── dense_worker.py
│   │   ├── dense_worker_client.py
│   │   ├── faiss_store.py
│   │   ├── hybrid_retriever.py
│   │   ├── rrf.py
│   │   ├── rule_second_hop.py
│   │   └── tokenizer.py
│   ├── rerankers/
│   │   └── cross_encoder_reranker.py
│   ├── llm/
│   │   └── evidence_checker.py
│   ├── utils/
│   │   ├── io_utils.py
│   │   └── text_utils.py
│   └── evaluation/
│       ├── data_stats.py
│       ├── chunk_eval.py
│       ├── eval_utils.py
│       ├── run_chunk_bm25_eval.py
│       ├── run_dense_faiss_eval.py
│       ├── run_hybrid_rrf_eval.py
│       ├── run_rerank_eval.py
│       ├── run_error_analysis.py
│       ├── run_rule_second_hop_eval.py
│       └── run_llm_evidence_checker_eval.py
├── scripts/
│   ├── build_faiss_index.py
│   ├── download_wixqa.py
│   ├── prepare_wixqa.py
│   ├── prepare_chunks.py
│   ├── inspect_wixqa.py
│   ├── run_chunk_bm25_baseline.py
│   ├── run_dense_faiss_baseline.py
│   ├── run_hybrid_rrf_baseline.py
│   ├── run_rerank_baseline.py
│   ├── run_error_analysis.py
│   ├── run_rule_second_hop.py
│   └── run_llm_evidence_checker.py
├── indexes/
│   └── faiss_bge_m3/
└── outputs/
    ├── data_inspection/
    ├── chunk_bm25_baseline/
    ├── dense_faiss_baseline/
    ├── hybrid_rrf_baseline/
    ├── rerank_baseline/
    ├── error_analysis/
    ├── rule_second_hop/
    └── llm_evidence_checker/
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
```

即直接检索 top 100 chunks，并用这些 chunks 的 `article_id` 对 gold `article_ids` 判断命中。

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
  --top_k_chunks 100
```

如果 Hugging Face 暂时不可访问，请在网络恢复后重试相同命令。`prepare_chunks.py` 会加载 `BAAI/bge-m3` tokenizer；离线复现时可先缓存 tokenizer，再使用 `--local_files_only`。

运行 Hybrid RRF 快速实验：

```bash
python scripts/run_hybrid_rrf_baseline.py \
  --processed_dir data/processed \
  --chunks_path data/processed/wix_kb_chunks.jsonl \
  --index_dir indexes/faiss_bge_m3 \
  --dataset wixqa_expertwritten \
  --model_name BAAI/bge-m3 \
  --local_files_only true \
  --branch_top_k_chunks 50 \
  --fused_top_k_chunks 50 \
  --rrf_k 60 \
  --bm25_weight 1.0 \
  --dense_weight 2.0 \
  --dense_worker_mode full \
  --dense_query_batch_size 16
```

标准诊断配置将两个 top-k 参数改为 `100`。如果本机在同一进程加载 SentenceTransformer 与 FAISS 时崩溃，显式使用 `--dense_worker_mode model_only`。

生成当前推荐的 Reranker 主实验候选池：

```bash
python scripts/run_hybrid_rrf_baseline.py \
  --processed_dir data/processed \
  --chunks_path data/processed/wix_kb_chunks.jsonl \
  --index_dir indexes/faiss_bge_m3 \
  --dataset wixqa_expertwritten \
  --model_name BAAI/bge-m3 \
  --local_files_only true \
  --branch_top_k_chunks 100 \
  --fused_top_k_chunks 50 \
  --rrf_k 70 \
  --bm25_weight 1.0 \
  --dense_weight 2.5 \
  --dense_worker_mode model_only \
  --dense_query_batch_size 16
```

使用本地 Qwen3 模型重排保存的 Hybrid top50 chunks：

```bash
python scripts/run_rerank_baseline.py \
  --hybrid_run_dir outputs/hybrid_rrf_baseline/hybrid_rrf_b100_f50_k70_bw1_dw2p5_wixqa_expertwritten \
  --chunks_path data/processed/wix_kb_chunks.jsonl \
  --model_name Qwen/Qwen3-Reranker-0.6B \
  --local_files_only true \
  --rerank_batch_size 8 \
  --max_length 1024
```

Reranker 只读取保存的 Hybrid 候选池与 chunks，不会重新执行 BM25、Dense 编码或 FAISS 搜索。服务器 GPU 可额外传入 `--device cuda`。

生成 Phase 6 错误分析：

```bash
python scripts/run_error_analysis.py
```

运行 Phase 7 title-expanded second-hop retrieval：

```bash
python scripts/run_rule_second_hop.py --device cuda
```

Phase 7 默认读取主线 top50 baseline 与公平的 top100 control。需要先生成 `hybrid_rrf_b100_f100_k60_bw1_dw2_wixqa_expertwritten` 及其 Qwen3 reranker artifact。

```bash
python scripts/run_hybrid_rrf_baseline.py \
  --dataset wixqa_expertwritten \
  --branch_top_k_chunks 100 \
  --fused_top_k_chunks 100 \
  --rrf_k 60 \
  --bm25_weight 1 \
  --dense_weight 2 \
  --device cuda

python scripts/run_rerank_baseline.py \
  --hybrid_run_dir outputs/hybrid_rrf_baseline/hybrid_rrf_b100_f100_k60_bw1_dw2_wixqa_expertwritten \
  --rerank_batch_size 32 \
  --device cuda
```

运行 Phase 8 LLM evidence checker pool-level eval：

```bash
python scripts/run_llm_evidence_checker.py \
  --device cuda \
  --llm_base_url <openai-compatible-url> \
  --llm_api_key <key> \
  --llm_model <server-qwen3-8b-model-name>
```

Phase 8 默认读取主线 top50 baseline、公平 top100 control、Phase 7 rule second-hop 结果、FAISS index 和 chunks。该阶段只检查 LLM gap queries 是否补齐 merged pool，不重新 rerank merged pool。

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

Chunk BM25 baseline 输出：

```text
outputs/chunk_bm25_baseline/<dataset>_metrics.json
outputs/chunk_bm25_baseline/<dataset>_metrics.md
outputs/chunk_bm25_baseline/<dataset>_retrieval_traces.jsonl
outputs/chunk_bm25_baseline/<dataset>_cases_A_top10_chunks_full.jsonl
outputs/chunk_bm25_baseline/<dataset>_cases_B_top100_chunks_full_not_top10_chunks.jsonl
outputs/chunk_bm25_baseline/<dataset>_cases_C_top100_chunks_not_full.jsonl
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
outputs/dense_faiss_baseline/dense_bge-m3_<dataset>_cases_A_top10_chunks_full.jsonl
outputs/dense_faiss_baseline/dense_bge-m3_<dataset>_cases_B_top100_chunks_full_not_top10_chunks.jsonl
outputs/dense_faiss_baseline/dense_bge-m3_<dataset>_cases_C_top100_chunks_not_full.jsonl
```

Hybrid RRF baseline 输出：

```text
outputs/hybrid_rrf_baseline/
└── hybrid_rrf_b50_f50_k60_bw1_dw2_<dataset>/
    ├── metrics.json
    ├── metrics.md
    ├── retrieval_traces.jsonl
    ├── comparison.md
    ├── comparison.json
    ├── candidates.jsonl
    ├── complementarity_analysis.jsonl
    └── cases_*.jsonl
```

每次运行使用独立子目录。标准诊断配置使用 `hybrid_rrf_b100_f100_k60_bw1_dw2_<dataset>/`。`bw` 和 `dw` 分别记录 BM25 与 Dense 权重，避免不同实验互相覆盖。

Qwen3 Reranker baseline 输出：

```text
outputs/rerank_baseline/
└── <hybrid_run_name>/
    └── qwen3-reranker-0p6b_inst-wixqa_help_center_v1_ml1024/
        ├── run_config.json
        ├── metrics.json
        ├── metrics.md
        ├── comparison.md
        ├── rerank_traces.jsonl
        └── cases_*.jsonl
```

Error analysis 输出：

```text
outputs/error_analysis/
├── summary.json
├── summary.md
├── case_traces_top50_chunks.jsonl
├── cases_A_top10_chunks_full.jsonl
├── cases_B_top50_chunks_full_not_top10_chunks.jsonl
├── cases_C_top50_chunks_not_full.jsonl
├── multi_cases_B_top50_chunks_full_not_top10_chunks.jsonl
└── multi_cases_C_top50_chunks_not_full.jsonl
```

Rule-based Second-hop Retrieval 输出：

```text
outputs/rule_second_hop/
└── <hybrid_run_name>/
    └── <reranker_run_name>/
        └── title_expand_s3_h20/
            ├── run_config.json
            ├── metrics.json
            ├── metrics.md
            ├── comparison.md
            ├── second_hop_traces.jsonl
            └── cases_*.jsonl
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

## Phase 2-5: Retrieval And Reranker Baselines

当前实现 Chunk BM25、Dense FAISS 与 Hybrid RRF 检索基线。

Chunk-level BM25：

```text
使用 BAAI/bge-m3 tokenizer 将 KB article 切成 512-token chunks
chunk overlap = 128 tokens
BM25 检索 chunk
直接按 top 100 chunks 评测
```

注意：chunk 边界使用 `BAAI/bge-m3` tokenizer，是为了后续 Dense / Hybrid Retrieval 复用同一套 chunks。BM25 打分本身仍然是 lexical BM25。

Dense FAISS：

```text
复用 data/processed/wix_kb_chunks.jsonl
使用 BAAI/bge-m3 编码 chunk.text
normalize embeddings = true
FAISS index type = IndexFlatIP
检索 top 100 chunks
直接用 top 100 chunks 的 article_id 对 gold article_ids 评测
```

Dense FAISS 不再聚合成 article ranking；指标按 chunk rank 计算，只用每个 chunk 的 `article_id` 和 gold `article_ids` 对齐判断命中。

Hybrid RRF：

```text
Chunk BM25 top50 + Dense FAISS top50 -> RRF -> Hybrid top50 chunks
标准诊断配置：top100 + top100 -> top100
按 chunk_id 融合
rrf_score(chunk) = sum(weight_i / (60 + rank_i(chunk)))
不聚合为 article ranking
```

两个分支权重均默认为 `1.0`，即标准等权 RRF。运行时可使用 `--bm25_weight 1.0 --dense_weight 2.0` 提高 Dense 分支权重。

Hybrid 使用常驻 Dense worker。默认 `full` 模式在 worker 内加载 SentenceTransformer 与 FAISS index；`model_only` 兼容模式只在 worker 内编码 query，由主进程执行 FAISS 搜索。

Qwen3 Reranker：

```text
复用 Hybrid candidates.jsonl，不重复执行检索
使用 Qwen/Qwen3-Reranker-0.6B
输入为 (question, chunk.text)
按 rerank_score 降序重排 chunks
仍然使用 chunk.article_id 对 gold article_ids 评测
```

默认使用 Wix Help Center 专用 instruction：

```text
Given a Wix Help Center question, retrieve relevant passages that contain the information needed to answer the question.
```

三类 baseline 指标包括：

```text
chunk_hit@k
chunk_full_article_hit@k
chunk_article_recall@k
chunk_gold_rate@k
unique_articles@k_chunks
duplicate_article_ratio@k_chunks
MRR
```

`unique_articles@k_chunks` 表示 top-k chunks 覆盖的不同文章数量。`duplicate_article_ratio@k_chunks` 表示 top-k chunks 中来自重复文章的比例。

Case 文件根据最终 cutoff 动态划分：

```text
A_top10_chunks_full                         -> top10 chunks 已覆盖全部 gold article_ids
B_top{cutoff}_chunks_full_not_top10_chunks  -> top cutoff chunks 覆盖全部 gold article_ids，但 top10 chunks 未覆盖全部
C_top{cutoff}_chunks_not_full               -> top cutoff chunks 仍未覆盖全部 gold article_ids
```

Rule-based Second-hop Retrieval：

```text
baseline reranker top10 chunks
  -> 按 article_id 去重，选取最多 3 个标题
  -> 构造 question + title 扩展查询
  -> 每条查询执行 Hybrid top20
  -> 与首轮 Hybrid top50 按 chunk_id 合并
  -> 使用原问题重新运行 Qwen3 reranker
```

Phase 7 对全部样本执行一次补检索。A/B/C 分类和 gold labels 只用于离线评测，不得用于触发 query、构造 query 或合并候选。`C_pool_rescued` 表示原 top50 缺证据但合并池已覆盖全部 gold articles；`C_top10_rescued` 表示重新 rerank 后 top10 已覆盖全部 gold articles。

LLM Evidence Sufficiency Checker pool-level eval：

```text
baseline reranker top10 chunks
  -> LLM 判断证据是否充足
  -> insufficient 时生成最多 3 条 missing-evidence-oriented queries
  -> 每条 query 执行 Hybrid top20
  -> 与首轮 Hybrid top50 按 chunk_id 合并
  -> 只评估 merged pool 是否覆盖缺失 gold articles
```

Phase 8 明确不是最终 rerank loop：`comparison.md` 中的 Phase 8 行只报告 pool rescue，不报告 final top10 context quality。Gold labels 只用于离线评测，不进入 prompt 或 query generation。

## 后续路线图

```text
阶段 1：数据接入与统计
阶段 2：BM25 Chunk-level Retrieval Baseline
阶段 3：Dense FAISS Retrieval Baseline
阶段 4：Hybrid Retrieval with RRF Fusion
阶段 5：Qwen3 Chunk Reranker Baseline
阶段 6：Error Analysis & Trace Logging
阶段 7：Rule-based Second-hop Retrieval
阶段 8：LLM Evidence Sufficiency Checker
阶段 9：Bounded Agentic RAG Loop
阶段 10：Citation-aware Answer Generation
阶段 11：Verifier / Abstention
阶段 12：Optional Pairwise Evidence Reranker
```
