from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
DEFAULT_DASHSCOPE_RERANK_MODEL = "qwen3-rerank"
DEFAULT_ENV_FILENAMES = (".env", ".env copy", ".env.local")


def load_project_env(root: str | Path, *, override: bool = False) -> list[Path]:
    loaded = []
    protected_keys = set(os.environ) if not override else set()
    for filename in DEFAULT_ENV_FILENAMES:
        path = Path(root) / filename
        if load_env_file(path, override=override, protected_keys=protected_keys):
            loaded.append(path)
    return loaded


def load_env_file(
    path: str | Path,
    *,
    override: bool = False,
    protected_keys: set[str] | None = None,
) -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False
    protected_keys = protected_keys or set()
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in protected_keys:
            os.environ[key] = value
    return True


def parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def deepseek_base_url_from_env() -> str:
    if os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_MODEL"):
        return os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    return os.environ.get("DEEPSEEK_BASE_URL", "")


def dashscope_rerank_url_from_env() -> str:
    return os.environ.get("DASHSCOPE_RERANK_URL", DEFAULT_DASHSCOPE_RERANK_URL)


def dashscope_rerank_model_from_env() -> str:
    return os.environ.get("DASHSCOPE_RERANK_MODEL", DEFAULT_DASHSCOPE_RERANK_MODEL)
