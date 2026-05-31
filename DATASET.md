# WixQA 数据集说明

本项目使用 Hugging Face 上的 `Wix/WixQA` 作为企业客服知识库 RAG 基准数据集。

当前阶段只做数据接入、格式统一和数据统计，不实现检索、重排、LLM 调用或 Agent 流程。

## 下载方式

使用项目约定的 conda 环境运行：

```bash
conda run -n wixqa-agentic-rag python scripts/download_wixqa.py
```

下载脚本会缓存数据集，并在 `data/raw/` 下保存完整原始 JSONL、少量样例和摘要。

## 本次下载结果

数据集：`Wix/WixQA`

| 配置 | 划分 | 样本数 | 字段 |
| --- | --- | ---: | --- |
| `wixqa_expertwritten` | `train` | 200 | `question`, `answer`, `article_ids` |
| `wixqa_simulated` | `train` | 200 | `question`, `answer`, `article_ids` |
| `wixqa_synthetic` | `train` | 6221 | `question`, `answer`, `article_ids` |
| `wix_kb_corpus` | `train` | 6221 | `id`, `url`, `contents`, `title`, `html_content`, `article_type` |

## 子集用途

`wix_kb_corpus` 是企业客服知识库 corpus，后续用于建立检索索引。

`wixqa_expertwritten` 是真实用户问题和专家撰写答案，后续作为主测试集。

`wixqa_simulated` 是模拟 / 专家验证 QA，后续作为辅助测试集或泛化测试集。

`wixqa_synthetic` 是合成 QA，后续用于流程冒烟检查、调参或弱监督。

## 字段说明

知识库文章字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 原始文章 ID，后续映射为 `article_id` |
| `url` | Wix 支持文章 URL |
| `contents` | 清洗后的正文内容 |
| `title` | 文章标题 |
| `html_content` | 原始 HTML 内容 |
| `article_type` | 文章类型 |

QA 样本字段：

| 字段 | 说明 |
| --- | --- |
| `question` | 用户问题 |
| `answer` | 标准答案 |
| `article_ids` | 支撑答案的知识库文章 ID 列表 |

## 原始样例文件

完整原始数据会写入以下 JSONL 文件，每行是一条 Hugging Face 原始记录：

```text
data/raw/wixqa_expertwritten_train.jsonl
data/raw/wixqa_simulated_train.jsonl
data/raw/wixqa_synthetic_train.jsonl
data/raw/wix_kb_corpus_train.jsonl
```

样例和摘要会写入以下文件：

```text
data/raw/wixqa_raw_summary.json
data/raw/samples/wix_kb_corpus_samples.json
data/raw/samples/wixqa_expertwritten_samples.json
data/raw/samples/wixqa_simulated_samples.json
data/raw/samples/wixqa_synthetic_samples.json
```

其中 `data/raw/samples/*_samples.json` 只保存少量样例，用于快速人工检查字段；完整数据以 `data/raw/<config>_<split>.jsonl` 命名保存。`data/raw/` 已在 `.gitignore` 中忽略，不应提交到 git。

## 后续处理

将原始数据转换为项目统一 JSONL 格式：

```bash
conda run -n wixqa-agentic-rag python scripts/prepare_wixqa.py
```

生成数据统计：

```bash
conda run -n wixqa-agentic-rag python -m src.evaluation.data_stats --processed_dir data/processed --output_dir data/stats
```
