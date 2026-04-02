"""
Configuration loading and management.

Supports layered YAML configs: base.yaml is always loaded first,
then the experiment config is deep-merged on top.

Usage:
    cfg = load_config("configs/experiments/baseline.yaml")
    print(cfg.model.name)
    print(cfg.prefilling.k_values)
"""

import copy
import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml

# Root of the repo (one level above this file)
REPO_ROOT = Path(__file__).parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


# ---------------------------------------------------------------------------
# Dot-access config wrapper
# ---------------------------------------------------------------------------

class Config:
    """
    Recursive dict wrapper that allows dot-access to config values.

    Example:
        cfg = Config({"model": {"name": "Llama"}})
        cfg.model.name  # "Llama"
    """

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        """Recursively convert back to a plain dict."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge *override* into *base* (non-destructive copy).

    Dict values are merged recursively; all other values are overwritten.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    experiment_path: Optional[Union[str, Path]] = None,
    base_path: Union[str, Path] = BASE_CONFIG_PATH,
    overrides: Optional[dict] = None,
) -> Config:
    """
    Load configuration.

    Args:
        experiment_path: Path to experiment-specific YAML (merged on top of base).
        base_path: Path to base YAML (defaults to configs/base.yaml).
        overrides: Optional dict of additional overrides applied last.

    Returns:
        Config object with dot-accessible fields.
    """
    base = _load_yaml(base_path) if Path(base_path).exists() else {}

    if experiment_path is not None:
        experiment = _load_yaml(experiment_path)
        merged = _deep_merge(base, experiment)
    else:
        merged = copy.deepcopy(base)

    if overrides:
        merged = _deep_merge(merged, overrides)

    # Resolve environment variables for model paths
    local_path = os.environ.get("LOCAL_MODEL_PATH")
    if local_path and merged.get("model", {}).get("local_path") is None:
        merged.setdefault("model", {})["local_path"] = local_path

    return Config(merged)


def save_config_snapshot(cfg: Config, output_path: Union[str, Path]) -> None:
    """Save a config snapshot alongside experiment outputs for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)


def config_to_json(cfg: Config) -> str:
    """Serialize config to JSON string (for embedding in output metadata)."""
    return json.dumps(cfg.to_dict(), indent=2, default=str)
