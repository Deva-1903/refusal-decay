"""
Tests for config loading and merging (no model required).
"""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config, _deep_merge, save_config_snapshot, config_to_json


# ---------------------------------------------------------------------------
# Tests for Config wrapper
# ---------------------------------------------------------------------------

def test_config_dot_access():
    cfg = Config({"model": {"name": "llama"}, "seed": 42})
    assert cfg.model.name == "llama"
    assert cfg.seed == 42


def test_config_to_dict():
    data = {"model": {"name": "llama"}, "prefilling": {"k_values": [0, 1, 3]}}
    cfg = Config(data)
    d = cfg.to_dict()
    assert d["model"]["name"] == "llama"
    assert d["prefilling"]["k_values"] == [0, 1, 3]


def test_config_get_default():
    cfg = Config({"a": 1})
    assert cfg.get("a") == 1
    assert cfg.get("missing", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Tests for deep merge
# ---------------------------------------------------------------------------

def test_deep_merge_override_scalar():
    base = {"a": 1, "b": 2}
    override = {"b": 99}
    result = _deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"] == 99


def test_deep_merge_nested():
    base = {"model": {"name": "3b", "dtype": "bfloat16"}}
    override = {"model": {"name": "8b"}}
    result = _deep_merge(base, override)
    assert result["model"]["name"] == "8b"
    assert result["model"]["dtype"] == "bfloat16"  # preserved


def test_deep_merge_non_destructive():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    result = _deep_merge(base, override)
    assert result["a"]["x"] == 1
    assert result["a"]["y"] == 2
    # base should be unchanged
    assert "y" not in base["a"]


def test_deep_merge_override_list():
    base = {"k_values": [0, 1, 3]}
    override = {"k_values": [0, 5, 10]}
    result = _deep_merge(base, override)
    assert result["k_values"] == [0, 5, 10]


# ---------------------------------------------------------------------------
# Tests for load_config with YAML files
# ---------------------------------------------------------------------------

def test_load_config_from_yaml():
    base_data = {"model": {"name": "3b", "dtype": "bfloat16"}, "seed": 42}
    exp_data = {"model": {"name": "8b"}, "seed": 99}

    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "base.yaml"
        exp_path = Path(tmp) / "exp.yaml"

        with open(base_path, "w") as f:
            yaml.dump(base_data, f)
        with open(exp_path, "w") as f:
            yaml.dump(exp_data, f)

        cfg = load_config(exp_path, base_path=base_path)

    assert cfg.model.name == "8b"
    assert cfg.model.dtype == "bfloat16"  # from base
    assert cfg.seed == 99


def test_load_config_no_experiment():
    base_data = {"model": {"name": "llama"}}
    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base_data, f)
        cfg = load_config(base_path=base_path)
    assert cfg.model.name == "llama"


def test_load_config_with_overrides():
    base_data = {"a": 1, "b": 2}
    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base_data, f)
        cfg = load_config(base_path=base_path, overrides={"b": 99, "c": 3})
    assert cfg.a == 1
    assert cfg.b == 99
    assert cfg.c == 3


def test_save_config_snapshot():
    cfg = Config({"model": {"name": "test"}, "seed": 42})
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = Path(tmp) / "sub" / "config.yaml"
        save_config_snapshot(cfg, snap_path)
        assert snap_path.exists()
        with open(snap_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["model"]["name"] == "test"


def test_config_to_json():
    cfg = Config({"model": {"name": "test"}})
    j = config_to_json(cfg)
    import json
    d = json.loads(j)
    assert d["model"]["name"] == "test"
