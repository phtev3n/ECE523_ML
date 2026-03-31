from __future__ import annotations
import yaml
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
