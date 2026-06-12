from __future__ import annotations

from typing import Any


def average(values: Any) -> float:
    items = list(values)
    return sum(float(value) for value in items) / len(items) if items else 0.0
