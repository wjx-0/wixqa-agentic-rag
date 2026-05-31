from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import orjson
except ImportError:  # pragma: no cover - dependency is listed, fallback helps error paths.
    orjson = None


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Cannot convert {type(value)!r} to a JSON object")


def dumps_json(value: Any, *, indent: bool = False) -> bytes:
    if orjson is not None:
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(value, option=option)
    text = json.dumps(value, ensure_ascii=False, indent=2 if indent else None)
    return text.encode("utf-8")


def loads_json(data: str | bytes) -> Any:
    if orjson is not None:
        return orjson.loads(data)
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def write_json(path: str | Path, value: Any, *, indent: bool = True) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_bytes(dumps_json(value, indent=indent) + b"\n")


def read_json(path: str | Path) -> Any:
    return loads_json(Path(path).read_bytes())


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    target = Path(path)
    ensure_dir(target.parent)
    count = 0
    with target.open("wb") as file:
        for row in rows:
            payload = model_to_dict(row) if not isinstance(row, dict) else row
            file.write(dumps_json(payload, indent=False))
            file.write(b"\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("rb") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield loads_json(line)

