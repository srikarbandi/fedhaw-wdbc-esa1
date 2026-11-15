"""YAML configuration loading with nested overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(overrides: dict | None = None) -> dict:
    """default.yaml, then configs/selected.yaml (tuned values, if present), then overrides."""
    cfg = load_yaml(CONFIG_DIR / "default.yaml")
    sel = CONFIG_DIR / "selected.yaml"
    if sel.exists():
        cfg = deep_update(cfg, load_yaml(sel).get("overrides", {}))
    return deep_update(cfg, overrides or {})


def load_experiments() -> dict:
    return load_yaml(CONFIG_DIR / "experiments.yaml")


def set_by_path(cfg: dict, dotted: str, value) -> dict:
    """set_by_path(cfg, 'training.lr', 0.05) -> new dict."""
    keys = dotted.split(".")
    upd: dict = {}
    cur = upd
    for k in keys[:-1]:
        cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value
    return deep_update(cfg, upd)


def save_yaml(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False)
