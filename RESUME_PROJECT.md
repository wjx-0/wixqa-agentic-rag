# 简历项目经历：WixQA Agentic RAG

## 推荐写法

**WixQA Agentic RAG 企业客服知识库问答系统**  
基于 WixQA 数据集构建面向企业客服场景的多阶段 RAG 问答系统，覆盖数据处理、BM25/Dense/Hybrid 检索、Qwen3 重排、LLM 证据充分性检查、多轮 gap-query 补检索、答案生成与可视化对话 Demo。

- 设计并实现统一的知识库文章、问答样本、chunk schema 与评测链路，使用 `BAAI/bge-m3` tokenizer 切分知识库，保证 BM25、FAISS Dense、Hybrid RRF 与 Reranker 复用同一套 chunk 边界。
- 搭建 chunk-level 检索基线：BM25、`BAAI/bge-m3` + FAISS、BM25 + Dense RRF 融合；在 WixQA expertwritten 200 条主测试集上，Hybrid + Qwen3 Reranker 将 `full@10` 从 BM25 的 0.5150 提升至 0.8000，`hit@10` 从 0.6250 提升至 0.8900。
- 引入 `Qwen/Qwen3-Reranker-0.6B` 对 Hybrid top50 候选重排，将 `full@10` 从 0.7050 提升至 0.8000，并通过 rescued/dropped 诊断定位多文章问题的证据覆盖瓶颈。
- 实现 LLM evidence checker 与 gap-query second-hop retrieval，对证据不足样本生成缺失证据导向查询；pool-level 评测中 merged pool `full_article_hit_rate` 达到 0.9300，多文章 merged pool 覆盖率达到 0.8077。
- 优化多轮 Evidence Completion Loop，加入 provenance 追踪、防 gold label 泄漏、上下文预算控制、最小 audit retrieval、per-query quota 与 article diversity selection；针对 34 条 reranker baseline 错误样本，最终修回 30 条，错误降低 88.24%，`provenance_invalid_count=0`。
- 构建多 Agent 对话流水线与 Web Console，串联 Dialogue Agent、Query Agent、Evidence Agent、Answer Agent、Verifier Agent，并通过 NDJSON 事件流展示检索、重排、证据校验、答案生成与耗时诊断。

## 精简版

**WixQA Agentic RAG 企业客服知识库问答系统**  
使用 Python、FAISS、BM25、BAAI/bge-m3、Qwen3 Reranker 和 LLM evidence checker 构建多阶段 RAG 系统，实现数据处理、混合检索、重排、证据补全、多 Agent 对话和 Web Demo。Hybrid + Qwen3 Reranker 在 WixQA expertwritten 200 条测试集上将 `full@10` 从 0.5150 提升至 0.8000，`hit@10` 从 0.6250 提升至 0.8900；进一步通过 gap-query evidence loop 在 34 条 baseline 错误样本中修回 30 条，错误降低 88.24%。

## 面试展开思路

### 项目背景

企业客服知识库问答不能只追求语义相似度，还要保证回答有完整证据支撑。WixQA 中存在不少多文章、多证据问题，单轮检索和单 chunk reranker 容易只找到部分相关内容，导致答案缺关键条件或边界。

### 技术方案

主链路分为四层：

1. 数据与 chunk 统一：将 WixQA corpus、QA 与 chunk 标准化，所有实验复用相同 chunk 边界。
2. 召回层：BM25 负责关键词匹配，`BAAI/bge-m3` + FAISS 负责语义召回，再通过 RRF 做 Hybrid 融合。
3. 排序层：用 Qwen3 Reranker 对 Hybrid 候选进行 cross-encoder 重排，提升 top10 证据质量。
4. Agentic 证据补全层：LLM checker 判断当前证据是否足够，不足时生成 gap queries，执行二跳检索、重排、证据选择和上下文 packing。

### 难点与解决

- 多文章问题：用 `full_article_hit@k` 和 single/multi 分组诊断，发现多文章 full@10 明显低于单文章；后续引入 gap-query 和 article diversity selection 补完整证据。
- 防止评测泄漏：gold article label 只在离线评测和错误分析使用，不进入 prompt、query generation、检索、重排或 checker 判断。
- 上下文预算：实现 prompt manifest、usage snapshot、compact boundary 和 visible window packing，避免多轮补检索后上下文失控。
- 错误定位：将失败阶段拆成 checker 过早停止、retrieval missing gold、reranker/selection missed gold 等，指导后续迭代。

## 技术关键词

Python, RAG, Agentic RAG, WixQA, BM25, FAISS, BAAI/bge-m3, Hybrid Retrieval, Reciprocal Rank Fusion, Qwen3 Reranker, Cross-Encoder Reranking, LLM Evidence Checker, Gap Query, Second-hop Retrieval, Context Budget, Provenance, Evaluation Pipeline, Web Console, NDJSON Streaming

## 可以写在项目亮点里的指标

- BM25 baseline：`full@10=0.5150`, `hit@10=0.6250`, `recall@10=0.5717`
- Dense FAISS：`full@10=0.6750`, `hit@10=0.7950`, `recall@10=0.7375`
- Hybrid RRF：`full@10=0.7050`, `hit@10=0.8200`, `recall@10=0.7625`
- Hybrid + Qwen3 Reranker：`full@10=0.8000`, `hit@10=0.8900`, `recall@10=0.8442`
- LLM gap-query pool：`merged_pool_full_article_hit_rate=0.9300`, `multi_merged_pool_full_article_hit_rate=0.8077`
- Evidence loop 错误集优化：34 条 baseline 错误样本修回 30 条，错误降低 88.24%，`provenance_invalid_count=0`

## 一句话介绍

这是一个面向企业客服知识库的 Agentic RAG 系统，不只是做向量检索，而是完整实现了“混合召回、语义重排、证据充分性判断、缺失证据补检索、上下文预算管理、答案校验和可视化对话”的端到端问答流程。
