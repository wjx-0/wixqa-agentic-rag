# WixQA Dialogue Web Console

启动真实多轮客服窗口：

```bash
conda run -n wixqa-agentic-rag python scripts/run_dialogue_web.py --mode real --port 8765
```

打开：

```text
http://127.0.0.1:8765
```

真实模式会加载本地向量库，并使用 `.env` 中配置的 LLM / reranker API。
默认 fast path：

- compact checker
- `checker_max_tokens=1024`
- `checker_retry_attempts=2`
- 1 轮补证据
- 每轮 1 个 gap query
- 二跳 top3

快速验证 UI 和多轮事件流，不调用外部 API：

```bash
conda run -n wixqa-agentic-rag python scripts/run_dialogue_web.py --mode demo --port 8765
```

常用真实模式参数：

```bash
conda run -n wixqa-agentic-rag python scripts/run_dialogue_web.py \
  --mode real \
  --port 8765 \
  --reranker_provider dashscope \
  --dense_worker_mode model_only \
  --checker_max_tokens 1024 \
  --checker_retry_attempts 2
```

前端只调用 `/api/chat/stream`，后端返回 NDJSON `AgentEvent` 流。
每轮会显示：

- status
- Dialogue Agent
- Query Agent
- Evidence Agent
- Answer Agent
- Verifier Agent
- final answer

后台过程中的 `checker / retrieval / rerank / answer / verifier` 耗时会写入
`payload.stage_latency_ms`，页面右侧会展示详细 payload 和证据引用。
