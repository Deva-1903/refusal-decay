"""
Pilot patching experiment.

Research hypothesis:
  Under a prefilling attack, the refusal-direction projection declines at later
  token positions (tracing experiment). The patching experiment tests whether
  this decline is *causally* relevant: if we restore the early-token refusal-
  direction component at a later position, does the model become more likely
  to refuse?

  Expected direction of effect:
    BASELINE (no patch): compliance (because prefilling suppressed refusal)
    PATCHED: refusal rate increases, or projection increases

  A result where patching INCREASES refusal (compliance → refusal) would
  support the hypothesis that the positional decay of the refusal direction
  causally mediates compliance under attack.
  Absence of effect would suggest the refusal decision is already committed
  by the time the patched position is processed.

Method:
  For a given layer L and source token position ts:
  1. Run a normal forward pass on the prompt.
  2. Extract the refusal-direction component r̂ at (layer=L, position=ts):
       component = (h_{ts} · r̂) * r̂  where r̂ is the unit refusal direction
  3. Run generation on the SAME prompt with a forward hook that, at each
     autoregressive step when the sequence position is tt, replaces (or adds)
     the refusal-direction component in the residual stream with the source value.
  4. Record the generated text before and after patching.

This is a causal intervention: if restoring the refusal component at position tt
increases refusal behavior, that position's refusal signal has causal relevance.

Note: "source" and "target" here are token positions within the same prompt.
We patch within a single prompt's generation to isolate positional effects.

TODO: Extend to cross-prompt patching (source prompt → target prompt) in
future experiments.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import Config
from src.data.schema import Prompt
from src.generation.prefilling import build_prefilled_input
from src.utils.io_utils import ensure_dir, save_jsonl

logger = logging.getLogger(__name__)


def _build_prefilled_input_tensor(
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
    prefix_text: str,
    k: int,
) -> torch.Tensor:
    input_ids = build_prefilled_input(tokenizer, prompt_text, prefix_text, k)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids


# ---------------------------------------------------------------------------
# Source activation extraction
# ---------------------------------------------------------------------------

def _extract_source_component(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    layer: int,
    source_position: int,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Extract the scalar projection and the direction component at (layer, source_position)
    during a normal forward pass.

    Returns:
        direction_component: Tensor of shape (hidden_size,) = projection * direction
    """
    input_ids = _build_prefilled_input_tensor(tokenizer, prompt.text, prefix_text, k)
    attention_mask = torch.ones_like(input_ids)

    captured = [None]

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # source_position is relative to the input sequence
        # -1 = last input token (or last forced prefix token)
        act = hidden[0, source_position, :].detach()
        layer_direction = direction.to(act.device)
        proj = torch.dot(act, layer_direction)
        captured[0] = (proj.item(), (proj * layer_direction).detach().cpu())

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    if captured[0] is None:
        raise RuntimeError(f"Hook did not fire for layer {layer}")

    scalar_proj, component = captured[0]
    logger.debug(
        "Source component at layer=%d pos=%d: projection=%.4f",
        layer, source_position, scalar_proj,
    )
    return component  # (hidden_size,)


# ---------------------------------------------------------------------------
# Patched generation
# ---------------------------------------------------------------------------

def _generate_with_patch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    layer: int,
    target_generation_pos: int,
    source_component: torch.Tensor,
    direction: torch.Tensor,
    mode: str,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    """
    Generate text with a hook that patches the residual stream at
    (layer, target_generation_pos) during autoregressive generation.

    Tracks autoregressive step count to know when we're at position tt.

    Args:
        target_generation_pos: 0-indexed position in the *generated* sequence
            at which to apply the patch (not counting input tokens).
        mode: "replace" — remove existing direction component, insert source.
              "add"     — add source component on top of existing.

    Returns:
        (generated_text, generated_token_ids)
    """
    input_ids = _build_prefilled_input_tensor(tokenizer, prompt.text, prefix_text, k)
    attention_mask = torch.ones_like(input_ids)

    # Step counter: 0 = first autoregressive step (first generated token), etc.
    step_counter = [0]
    # Flag: only patch once
    patched = [False]

    def hook_fn(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output

        # During generate(), after prefill, each step has seq_len=1
        if hidden.shape[1] != 1:
            # This is the prefill step — skip
            return output

        current_step = step_counter[0]
        step_counter[0] += 1

        if current_step == target_generation_pos and not patched[0]:
            hidden = hidden.clone()
            layer_direction = direction.to(hidden.device)
            layer_source_component = source_component.to(hidden.device)
            if mode == "replace":
                existing_proj = torch.dot(hidden[0, 0, :], layer_direction)
                hidden[0, 0, :] -= existing_proj * layer_direction
                hidden[0, 0, :] += layer_source_component
            elif mode == "add":
                hidden[0, 0, :] += layer_source_component
            else:
                raise ValueError(f"Unknown patch mode: {mode}")
            patched[0] = True
            logger.debug("Patch applied at step=%d (layer=%d)", current_step, layer)

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
    return generated_text, generated_ids


# ---------------------------------------------------------------------------
# Baseline (unpatched) generation
# ---------------------------------------------------------------------------

def _generate_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    """Normal generation without any patching."""
    input_ids = _build_prefilled_input_tensor(tokenizer, prompt.text, prefix_text, k)
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
    return tokenizer.decode(generated_ids, skip_special_tokens=True), generated_ids


# ---------------------------------------------------------------------------
# Full patching experiment
# ---------------------------------------------------------------------------

def patch_direction_component(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    layer: int,
    source_position: int,
    target_position: int,
    direction: torch.Tensor,
    cfg: Config,
) -> dict:
    """
    Run one complete patch experiment for a single prompt.

    Extracts the refusal-direction component from (layer, source_position) and
    injects it at (layer, target_position) during generation.

    Returns:
        Dict with baseline and patched generation results.
    """
    prefix_text: str = getattr(cfg.prefilling, "prefix_text", "Sure, here is how you can do that:")
    mode: str = getattr(cfg.patching, "mode", "replace")
    max_new_tokens: int = getattr(cfg.generation, "max_new_tokens", 256)
    k_values = getattr(cfg.prefilling, "k_values", [0])
    if len(k_values) != 1:
        raise ValueError(f"Patching pilot expects exactly one prefilling k, got {k_values}")
    k: int = k_values[0]

    # Step 1: baseline
    baseline_text, baseline_ids = _generate_baseline(
        model, tokenizer, prompt, k, prefix_text, max_new_tokens
    )

    # Step 2: extract source component from the prompt's own forward pass
    source_component = _extract_source_component(
        model, tokenizer, prompt, k, prefix_text,
        layer, source_position, direction
    )

    # Step 3: patched generation
    patched_text, patched_ids = _generate_with_patch(
        model, tokenizer, prompt, k, prefix_text,
        layer, target_position, source_component, direction, mode, max_new_tokens
    )

    return {
        "prompt_id": prompt.prompt_id,
        "label": prompt.label,
        "category": prompt.category,
        "layer": layer,
        "source_position": source_position,
        "target_position": target_position,
        "mode": mode,
        "baseline_text": baseline_text,
        "patched_text": patched_text,
        "baseline_n_tokens": len(baseline_ids),
        "patched_n_tokens": len(patched_ids),
    }


def run_patching_experiment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[Prompt],
    directions: dict[int, torch.Tensor],
    cfg: Config,
    output_dir: str | Path,
    resume: bool = True,
) -> list[dict]:
    """
    Run the full patching experiment grid.

    Grid: prompts × layers × target_positions.
    Source position is fixed (cfg.patching.source_position).

    Args:
        model, tokenizer: Loaded model.
        prompts: Prompts to experiment on.
        directions: Per-layer refusal directions.
        cfg: Config object.
        output_dir: Where to save results.
        resume: Skip already-completed files.

    Returns:
        List of result dicts.
    """
    output_dir = ensure_dir(output_dir)
    layers: list[int] = getattr(cfg.patching, "layers", [8, 16, 24])
    target_positions: list[int] = getattr(cfg.patching, "target_positions", [5, 10, 15])
    source_position: int = getattr(cfg.patching, "source_position", 0)

    all_results: list[dict] = []

    for layer in layers:
        if layer not in directions:
            logger.warning("No direction for layer %d, skipping.", layer)
            continue
        direction = directions[layer]

        for target_pos in target_positions:
            out_path = output_dir / f"patch_layer{layer}_ts{source_position}_tt{target_pos}.jsonl"

            if resume and out_path.exists():
                logger.info("Skipping (exists): %s", out_path)
                from src.utils.io_utils import load_jsonl
                all_results.extend(load_jsonl(out_path))
                continue

            logger.info(
                "Patching layer=%d source_pos=%d target_pos=%d over %d prompts ...",
                layer, source_position, target_pos, len(prompts),
            )
            run_results: list[dict] = []

            for prompt in prompts:
                try:
                    result = patch_direction_component(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        layer=layer,
                        source_position=source_position,
                        target_position=target_pos,
                        direction=direction,
                        cfg=cfg,
                    )
                    run_results.append(result)
                except Exception as e:
                    logger.error(
                        "Error patching prompt %s (layer=%d, tt=%d): %s",
                        prompt.prompt_id, layer, target_pos, e,
                    )
                    run_results.append({
                        "prompt_id": prompt.prompt_id,
                        "layer": layer,
                        "target_position": target_pos,
                        "error": str(e),
                    })

            save_jsonl(run_results, out_path)
            all_results.extend(run_results)
            logger.info("Saved %d patching results to %s", len(run_results), out_path)

    return all_results
