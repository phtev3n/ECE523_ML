from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and all parents) if it does not already exist.

    Returns the Path object so callers can chain: ``p = ensure_dir(...) / "file.json"``.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    """Load and return a JSON file as a Python object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    """Serialise data to a JSON file with 2-space indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
