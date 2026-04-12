from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


class DuplicateKeySafeLoader(yaml.SafeLoader):
    """YAML loader that raises on duplicate keys instead of silently overriding."""


def _construct_mapping(loader: DuplicateKeySafeLoader, node, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = getattr(key_node.start_mark, "line", 0) + 1
            raise ValueError(f"Duplicate YAML key '{key}' at line {line}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_config_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=DuplicateKeySafeLoader)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping at top level")
    validate_config(cfg, config_path=path)
    return cfg


def validate_config(cfg: Dict[str, Any], config_path: str | None = None) -> None:
    required_sections = ["retrieval", "generation", "training", "paths"]
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    generation = cfg["generation"]
    model_name = generation.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("generation.model_name must be a non-empty string")

    adapter_path = generation.get("adapter_path")
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Configured adapter_path does not exist: {adapter_path}"
            )

        adapter_config_path = adapter_dir / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(
                f"adapter_path exists but adapter_config.json is missing: {adapter_config_path}"
            )

        with open(adapter_config_path, "r") as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg.get("base_model_name_or_path")
        if base_model and str(base_model).strip() != str(model_name).strip():
            where = f" in {config_path}" if config_path else ""
            raise ValueError(
                "Adapter/model mismatch"
                f"{where}: generation.model_name='{model_name}' but adapter expects '{base_model}'"
            )

    training = cfg["training"]
    method = training.get("method", "dpo")
    if method not in {"dpo", "rejection", "metric_loss"}:
        raise ValueError(f"training.method must be one of dpo|rejection|metric_loss, got '{method}'")

    retrieval = cfg["retrieval"]
    if not retrieval.get("index_path"):
        raise ValueError("retrieval.index_path is required")
    if not retrieval.get("corpus_path"):
        raise ValueError("retrieval.corpus_path is required")