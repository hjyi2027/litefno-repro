from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(base: dict, updates: dict) -> dict:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_config = config.pop("base_config", None)
    if base_config:
        base_path = (path.parent / base_config).resolve()
        base = load_config(base_path)
        config = deep_merge(base, config)
    return config


def apply_overrides(config: dict, overrides: Iterable[str]) -> dict:
    updated = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value: {item}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = updated
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return updated


def resolve_path(config: dict, *keys: str) -> Path:
    target: Any = config
    for key in keys:
        target = target[key]
    return Path(target).expanduser().resolve()
