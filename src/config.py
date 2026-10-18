from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(_resolve(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_experiment_config(path: str | Path = CONFIG_DIR / "experiment.yaml") -> dict[str, Any]:
    cfg = load_yaml(path)
    cfg["data"]["data_dir"] = str(_resolve(cfg["data"]["data_dir"]))
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str(_resolve(value))
    return cfg


def load_actions_config(path: str | Path = CONFIG_DIR / "actions.yaml") -> dict[str, Any]:
    return load_yaml(path)
