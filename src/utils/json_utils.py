from __future__ import annotations

import re

from src.utils.text_utils import compact_text


def extract_json_object(raw_text: str) -> str:
    text = compact_text(raw_text)
    if not text:
        raise ValueError("response is empty.")
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        return fenced.group(1).strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response does not contain a JSON object.")
    return raw_text[start : end + 1].strip()
