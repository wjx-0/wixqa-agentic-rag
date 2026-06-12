from __future__ import annotations

import time


def elapsed_ms(started_at: float) -> float:
    return (time.monotonic() - started_at) * 1000
