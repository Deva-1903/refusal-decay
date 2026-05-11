"""
Focused Report 7 interventions.

These helpers intentionally mirror the simple hook-based Report 6 patching
style while adding the two stronger causal tests needed for Report 7:
cross-condition source patching and direct additive direction intervention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data.schema import Prompt
from src.generation.prefilling import build_prefilled_input

logger = logging.getLogger(__name__)


def get_model_input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def build_input_ids(
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    prefix_text: str,
    k: int,
    model: PreTrainedModel,
) -> torch.Tensor:
    input_ids = build_prefilled_input(
        tokenizer=tokenizer,
        prompt_text=prompt.text,
        prefix_text=prefix_text,
        k=k,
    )
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(get_model_input_device(model))


def generate_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    input_ids = build_input_ids(tokenizer, prompt, prefix_text, k, model)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = input_ids.shape[1]
    generated_ids = output_ids[0, input_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids


def extract_direction_component(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    layer: int,
    position: int,
    direction: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Extract scalar projection and direction component at an input position."""
    input_ids = build_input_ids(tokenizer, prompt, prefix_text, k, model)
    attention_mask = torch.ones_like(input_ids)
    captured: list[tuple[float, torch.Tensor] | None] = [None]

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        act = hidden[0, position, :].detach()
        layer_direction = direction.to(act.device, dtype=act.dtype)
        projection = torch.dot(act, layer_direction)
        captured[0] = (projection.item(), (projection * layer_direction).detach().cpu())

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    if captured[0] is None:
        raise RuntimeError(f"Source hook did not fire for layer {layer}")
    return captured[0]


@dataclass
class InterventionResult:
    text: str
    token_ids: list[int]
    applied: bool
    projection_before: float | None
    projection_after: float | None


def _run_generation_with_step_intervention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    layer: int,
    target_position: int,
    direction: torch.Tensor,
    max_new_tokens: int,
    edit_fn,
) -> InterventionResult:
    """
    Run generation with a one-time residual edit at a generated-token position.

    target_position is zero-indexed over generated tokens, so 0 means the first
    autoregressive step after the prefill pass.
    """
    if target_position < 0:
        raise ValueError(f"target_position must be >= 0 for generated-token interventions, got {target_position}")

    input_ids = build_input_ids(tokenizer, prompt, prefix_text, k, model)
    attention_mask = torch.ones_like(input_ids)
    step_counter = [0]
    applied = [False]
    projection_before: list[float | None] = [None]
    projection_after: list[float | None] = [None]

    def hook_fn(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if hidden.shape[1] != 1:
            return output

        current_step = step_counter[0]
        step_counter[0] += 1

        if current_step == target_position and not applied[0]:
            hidden = hidden.clone()
            # Match dtype as well — bf16 models would otherwise fail in torch.dot.
            layer_direction = direction.to(hidden.device, dtype=hidden.dtype)
            projection_before[0] = torch.dot(hidden[0, 0, :], layer_direction).item()
            hidden = edit_fn(hidden, layer_direction)
            projection_after[0] = torch.dot(hidden[0, 0, :], layer_direction).item()
            applied[0] = True

        if is_tuple:
            return (hidden,) + output[1:]
        return hidden

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    input_len = input_ids.shape[1]
    generated_ids = output_ids[0, input_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return InterventionResult(
        text=generated_text,
        token_ids=generated_ids,
        applied=applied[0],
        projection_before=projection_before[0],
        projection_after=projection_after[0],
    )


def generate_with_component_patch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    target_k: int,
    prefix_text: str,
    layer: int,
    target_position: int,
    direction: torch.Tensor,
    source_component: torch.Tensor,
    mode: str,
    max_new_tokens: int,
) -> InterventionResult:
    def edit_fn(hidden: torch.Tensor, layer_direction: torch.Tensor) -> torch.Tensor:
        layer_source = source_component.to(hidden.device)
        if mode == "replace":
            existing_projection = torch.dot(hidden[0, 0, :], layer_direction)
            hidden[0, 0, :] -= existing_projection * layer_direction
            hidden[0, 0, :] += layer_source
        elif mode == "add":
            hidden[0, 0, :] += layer_source
        else:
            raise ValueError(f"Unknown patch mode: {mode}")
        return hidden

    return _run_generation_with_step_intervention(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        k=target_k,
        prefix_text=prefix_text,
        layer=layer,
        target_position=target_position,
        direction=direction,
        max_new_tokens=max_new_tokens,
        edit_fn=edit_fn,
    )


def make_direction_for_type(
    base_direction: torch.Tensor,
    direction_type: str,
    seed: int,
    layer: int,
) -> torch.Tensor:
    # Compute internally in float32 (needed for stable orthogonalization),
    # but return in the input's original dtype so downstream hooks on a
    # bfloat16 model don't hit dtype-mismatch errors in torch.dot.
    target_dtype = base_direction.dtype
    base = F.normalize(base_direction.float().cpu(), dim=0)
    if direction_type == "refusal":
        return base.to(target_dtype)
    if direction_type == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + layer * 1009)
        random_vec = torch.randn(base.shape, generator=generator)
        return F.normalize(random_vec, dim=0).to(target_dtype)
    if direction_type == "orthogonal":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + layer * 1009 + 17)
        random_vec = torch.randn(base.shape, generator=generator)
        random_vec = random_vec - torch.dot(random_vec, base) * base
        return F.normalize(random_vec, dim=0).to(target_dtype)
    raise ValueError(f"Unsupported direction_type: {direction_type}")


def generate_with_additive_direction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    layer: int,
    target_position: int,
    direction: torch.Tensor,
    alpha: float,
    max_new_tokens: int,
) -> InterventionResult:
    def edit_fn(hidden: torch.Tensor, layer_direction: torch.Tensor) -> torch.Tensor:
        hidden[0, 0, :] += alpha * layer_direction
        return hidden

    return _run_generation_with_step_intervention(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        k=k,
        prefix_text=prefix_text,
        layer=layer,
        target_position=target_position,
        direction=direction,
        max_new_tokens=max_new_tokens,
        edit_fn=edit_fn,
    )

