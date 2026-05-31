from __future__ import annotations

import re
from typing import Any

from src.utils.io_utils import loads_json


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return str(value).strip()


def compact_text(value: Any) -> str:
    return WHITESPACE_RE.sub(" ", clean_text(value)).strip()


def preview_text(value: Any, limit: int = 300) -> str:
    text = compact_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def stable_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:06d}"


def to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                return to_string_list(loads_json(text))
            except Exception:
                pass
        return [text]
    if isinstance(value, (list, tuple, set)):
        output: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (list, tuple, set)):
                output.extend(to_string_list(item))
                continue
            text = clean_text(item)
            if text:
                output.append(text)
        return output
    return [clean_text(value)] if clean_text(value) else []

