# AGENTS.md

## 环境

使用名为 `wixqa-agentic-rag` 的 conda 环境。

```bash
conda activate wixqa-agentic-rag
pip install -r requirements.txt
```

非交互式运行时，优先使用：

```bash
conda run -n wixqa-agentic-rag python scripts/inspect_wixqa.py
conda run -n wixqa-agentic-rag python scripts/prepare_wixqa.py
conda run -n wixqa-agentic-rag python -m src.evaluation.data_stats --processed_dir data/processed --output_dir data/stats
```


## 代码风格

- 代码保持简洁、易读、模块化。
- 字段映射优先清晰显式，遇到问题给出明确警告，避免过度“聪明”的推断。
- 除非当前阶段明确需要，否则不要引入大型框架。
- 所有脚本都应能从项目根目录运行。
- 不要提交 `data/raw/`、`data/processed/`、`data/stats/` 或 `outputs/` 下的生成数据。
