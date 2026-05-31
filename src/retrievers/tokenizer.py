from __future__ import annotations

import re

from src.utils.text_utils import compact_text


TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    normalized = compact_text(text).lower()
    return [token for token in TOKEN_RE.findall(normalized) if len(token) >= 2]
