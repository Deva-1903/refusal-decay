"""
All-position directional interventions (Arditi et al. 2024 style).

Unlike the single-generated-token-position edits in
``report7_interventions``, these interventions modify the residual stream at
**every token position** on **every** forward pass (prefill and each
generation step), across a chosen set of layers. This is the standard
directional-ablation / directional-addition recipe and is strong enough to
move behavior in the clean setting, which makes it the positive control the
single-position experiments lacked.

Two operations:
  - ablate:  h <- h - (h . d_hat) d_hat        (project the direction out)
  - add:     h <- h + alpha * d_hat            (push along the direction)

The direction is L2-normalized per layer before use. The same module is used
for the clean-setting positive control (E1) and for the attacked-condition
contrast (E2); only the prefill length k and the prompt set change.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data.schema import Prompt
from src.generation.prefilling import build_prefilled_input

logger = logging.getLogger(__name__)


def _model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _apply(hidden: torch.Tensor, direction: torch.Tensor, mode: str, alpha: float) -> torch.Tensor:
    """Apply the directional op to the full hidden state (all positions)."""
    d = direction.to(hidden.device, dtype=hidden.dtype)
    d = F.normalize(d, dim=0)
    if mode == "ablate":
        # project the direction out at every position: h - (h.d) d
        proj = torch.matmul(hidden, d).unsqueeze(-1)  # (batch, seq, 1)
        return hidden - proj * d
    if mode == "add":
        return hidden + alpha * d
    raise ValueError(f"Unknown directional mode: {mode!r} (expected 'ablate' or 'add').")


@contextmanager
def directional_hooks(
    model: PreTrainedModel,
    layer_directions: dict[int, torch.Tensor],
    mode: str,
    alpha: float = 1.0,
):
    """
    Register forward hooks that apply ``mode`` using each layer's direction.

    Args:
        model: Llama/Qwen-style causal LM with ``model.model.layers``.
        layer_directions: {layer_idx: direction tensor}. The hook at layer L
            uses layer_directions[L]. To ablate one direction everywhere, pass
            the same vector for every layer (see ``broadcast_direction``).
        mode: "ablate" or "add".
        alpha: scale for the "add" mode (ignored for "ablate").
    """
    handles = []
    transformer_layers = model.model.layers

    def make_hook(direction: torch.Tensor):
        def hook_fn(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            hidden = _apply(hidden, direction, mode, alpha)
            if is_tuple:
                return (hidden,) + tuple(output[1:])
            return hidden
        return hook_fn

    try:
        for layer_idx, direction in layer_directions.items():
            handle = transformer_layers[layer_idx].register_forward_hook(make_hook(direction))
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            handle.remove()


def broadcast_direction(direction: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    """Use a single direction at every layer in ``layers`` (Arditi-style)."""
    return {int(layer): direction for layer in layers}


def per_layer_directions(directions: dict[int, torch.Tensor], layers: list[int]) -> dict[int, torch.Tensor]:
    """Use each layer's own extracted direction; skip layers without one."""
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        layer = int(layer)
        if layer in directions:
            out[layer] = directions[layer]
        else:
            logger.warning("No direction for layer %d; skipping it in the intervention.", layer)
    return out


@torch.no_grad()
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    max_new_tokens: int,
) -> str:
    """Greedy generation (no intervention). Returns decoded continuation only."""
    input_ids = build_prefilled_input(tokenizer, prompt.text, prefix_text, k)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(_model_device(model))
    attention_mask = torch.ones_like(input_ids)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_with_directional_intervention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    layer_directions: dict[int, torch.Tensor],
    mode: str,
    alpha: float,
    max_new_tokens: int,
) -> str:
    """Greedy generation with all-position directional hooks active throughout."""
    with directional_hooks(model, layer_directions, mode=mode, alpha=alpha):
        return generate_text(model, tokenizer, prompt, k, prefix_text, max_new_tokens)
