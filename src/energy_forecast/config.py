"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/config.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file (relative to the project root)."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)
