from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DATASET_NAME = "Wix/WixQA"
CANONICAL_CONFIGS = (
    "wix_kb_corpus",
    "wixqa_expertwritten",
    "wixqa_simulated",
    "wixqa_synthetic",
)


class WixQALoadError(RuntimeError):
    """Raised when WixQA cannot be discovered or loaded."""


@dataclass(frozen=True)
class DatasetSplitInfo:
    name: str
    num_rows: int | None
    features: list[str]


def _datasets_module() -> Any:
    try:
        import datasets
    except ImportError as exc:
        raise WixQALoadError(
            "Missing dependency `datasets`. Install project dependencies with "
            "`pip install -r requirements.txt` and retry."
        ) from exc
    return datasets


def _format_load_error(action: str, dataset_name: str, exc: Exception) -> str:
    return (
        f"Could not {action} Hugging Face dataset {dataset_name!r}. "
        "Check network access and Hugging Face availability, then retry. "
        f"Original error: {exc}"
    )


def _known_wixqa_configs(dataset_name: str) -> list[str] | None:
    if dataset_name == DATASET_NAME:
        return list(CANONICAL_CONFIGS)
    return None


def discover_configs(dataset_name: str = DATASET_NAME) -> list[str]:
    datasets = _datasets_module()
    try:
        configs = list(datasets.get_dataset_config_names(dataset_name))
        if configs == ["default"]:
            fallback = _known_wixqa_configs(dataset_name)
            if fallback is not None:
                return fallback
        return configs
    except WixQALoadError:
        raise
    except Exception as exc:
        fallback = _known_wixqa_configs(dataset_name)
        if fallback is not None:
            return fallback
        raise WixQALoadError(_format_load_error("discover configs for", dataset_name, exc)) from exc


def load_config(config_name: str, dataset_name: str = DATASET_NAME) -> dict[str, Any]:
    datasets = _datasets_module()
    try:
        loaded = datasets.load_dataset(dataset_name, config_name)
    except Exception as exc:
        raise WixQALoadError(
            _format_load_error(f"load config {config_name!r} from", dataset_name, exc)
        ) from exc

    if isinstance(loaded, datasets.DatasetDict):
        return dict(loaded.items())
    return {"train": loaded}


def split_info(split_name: str, dataset: Any) -> DatasetSplitInfo:
    features = list(getattr(dataset, "features", {}).keys())
    num_rows = getattr(dataset, "num_rows", None)
    if num_rows is None:
        try:
            num_rows = len(dataset)
        except TypeError:
            num_rows = None
    return DatasetSplitInfo(name=split_name, num_rows=num_rows, features=features)


def infer_wixqa_config_kind(config_name: str) -> str | None:
    normalized = config_name.lower().replace("-", "_")
    if "kb" in normalized and "corpus" in normalized:
        return "wix_kb_corpus"
    if "expert" in normalized:
        return "wixqa_expertwritten"
    if "simulated" in normalized or "simulation" in normalized:
        return "wixqa_simulated"
    if "synthetic" in normalized:
        return "wixqa_synthetic"
    return None


def map_actual_configs(config_names: list[str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for config_name in config_names:
        kind = infer_wixqa_config_kind(config_name)
        if kind and kind not in mapped:
            mapped[kind] = config_name
    return mapped
