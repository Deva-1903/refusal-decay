"""
Tests for refusal direction math (no model required — uses random tensors).
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probing.direction import save_direction, load_direction


# ---------------------------------------------------------------------------
# Projection math
# ---------------------------------------------------------------------------

def test_projection_onto_unit_vector():
    """Dot product with a unit vector returns the scalar projection."""
    direction = torch.tensor([1.0, 0.0, 0.0])  # unit vector along x
    activation = torch.tensor([3.0, 4.0, 5.0])
    proj = torch.dot(activation, direction).item()
    assert abs(proj - 3.0) < 1e-5


def test_direction_is_unit_norm():
    """After normalization, direction should have L2 norm = 1."""
    raw = torch.randn(128)
    direction = F.normalize(raw, dim=0)
    assert abs(direction.norm().item() - 1.0) < 1e-5


def test_difference_in_means_shape():
    """DiM direction should have same shape as input activations."""
    hidden_size = 64
    n_prompts = 10
    harmful_acts = torch.randn(n_prompts, hidden_size)
    benign_acts = torch.randn(n_prompts, hidden_size)

    direction = F.normalize(harmful_acts.mean(0) - benign_acts.mean(0), dim=0)
    assert direction.shape == (hidden_size,)


def test_projection_sign():
    """
    If harmful mean is strictly positive and benign mean is zero,
    the direction points along harmful, so harmful activations
    project positively.
    """
    hidden_size = 4
    harmful_acts = torch.ones(5, hidden_size)  # all ones
    benign_acts = torch.zeros(5, hidden_size)  # all zeros

    direction = F.normalize(harmful_acts.mean(0) - benign_acts.mean(0), dim=0)

    for act in harmful_acts:
        proj = torch.dot(act, direction).item()
        assert proj > 0, "Harmful activations should project positively onto refusal direction."


def test_orthogonal_component_zero_projection():
    """The component orthogonal to the direction should have zero projection."""
    direction = F.normalize(torch.tensor([1.0, 0.0, 0.0]), dim=0)
    activation = torch.tensor([3.0, 4.0, 5.0])

    # Decompose into parallel and orthogonal
    proj_scalar = torch.dot(activation, direction)
    parallel = proj_scalar * direction
    orthogonal = activation - parallel

    proj_ortho = torch.dot(orthogonal, direction).item()
    assert abs(proj_ortho) < 1e-5


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_directions_roundtrip(tmp_path):
    hidden_size = 64
    directions = {
        8: torch.randn(hidden_size),
        16: torch.randn(hidden_size),
    }
    # Normalize
    directions = {k: F.normalize(v, dim=0) for k, v in directions.items()}

    save_path = tmp_path / "test_direction.pt"
    save_direction(directions, save_path, metadata={"test": True})

    loaded = load_direction(save_path)
    assert set(loaded.keys()) == {8, 16}
    assert loaded[8].shape == (hidden_size,)

    for layer in [8, 16]:
        assert torch.allclose(loaded[layer], directions[layer], atol=1e-5)


def test_load_direction_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_direction(tmp_path / "nonexistent.pt")


def test_direction_metadata_preserved(tmp_path):
    """Metadata is saved and recoverable."""
    directions = {0: F.normalize(torch.randn(32), dim=0)}
    meta = {"model_name": "test-model", "n_harmful": 50}
    save_path = tmp_path / "dir.pt"
    save_direction(directions, save_path, metadata=meta)

    payload = torch.load(save_path, map_location="cpu", weights_only=True)
    assert payload["metadata"]["model_name"] == "test-model"
    assert payload["metadata"]["n_harmful"] == 50
