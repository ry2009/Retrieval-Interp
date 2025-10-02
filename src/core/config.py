"""Configuration utilities."""
from __future__ import annotations

import yaml


def load_config(path):
    """Load YAML config from `path` and return as dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
