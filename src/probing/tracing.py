"""
Tracing: project residual-stream activations onto the refusal direction
at every generated token position.

For each prompt × k-value combination:
  1. Run generate() with forward hooks active.
  2. At each autoregressive step, record the hidden state at each configured layer.
  3. Project each hidden state onto the refusal direction.
  4. Store: (prompt_id, k, layer, token_position, projection_value).

Output: tidy long-format DataFrame saved as Parquet or CSV.

NOTE: During generation, the model processes one new token per step.
We collect the last-position hidden state at each step (= the state of the
current token being generated). For the initial forward pass (prefill phase),
we capture the full sequence; we record the last position which corresponds
to the last input token (or the last forced prefix token).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import Config
from src.data.schema import Prompt
from src.generation.prefilling import build_prefilled_input
from src.probing.direction import ActivationCollector
from src.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core tracing function
# ---------------------------------------------------------------------------

def trace_single_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Prompt,
    k: int,
    prefix_text: str,
    layers: list[int],
    directions: dict[int, torch.Tensor],
    max_new_tokens: int = 64,
) -> list[dict]:
    """
    Generate from one prompt with prefix k and record per-layer, per-step
    projections onto the refusal direction.

    Args:
        model: Causal LM.
        tokenizer: Tokenizer.
        prompt: Prompt object.
        k: Number of forced prefix tokens.
        prefix_text: Full prefix string.
        layers: Layer indices to trace.
        directions: {layer_idx: direction_tensor} from extract_refusal_direction.
        max_new_tokens: Maximum tokens to generate (keep small for tracing).

    Returns:
        List of dicts, one per (layer, token_position) combination.
    """
    input_ids = build_prefilled_input(
        tokenizer=tokenizer,
        prompt_text=prompt.text,
        prefix_text=prefix_text,
        k=k,
    ).to(model.device)

    # Collect activations step by step via hooks.
    # During generate(), the model does:
    #   - 1 full forward pass on the input (prefill)
    #   - N autoregressive steps (each a single-token forward pass)
    # We tag each hook call with a step counter.

    step_records: list[dict] = []  # flat records to return

    # We need to count hook invocations to track token position.
    step_counter = [0]
    collected: dict[int, Optional[torch.Tensor]] = {l: None for l in layers}
    step_data: list[tuple[int, dict[int, torch.Tensor]]] = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: (batch, seq_len, hidden_size)
            # During generation, seq_len=1 after the prefill step
            last_hidden = hidden[0, -1, :].detach().cpu()  # (hidden_size,)
            collected[layer_idx] = last_hidden
        return hook_fn

    # Track when all layers for a step have been collected.
    # We use a separate counter hook on the last layer to flush.
    def flush_hook(module, input, output):
        step_idx = step_counter[0]
        step_data.append((step_idx, {l: v.clone() for l, v in collected.items() if v is not None}))
        step_counter[0] += 1

    hooks = []
    transformer_layers = model.model.layers
    for layer_idx in layers:
        h = transformer_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    # Flush on the last configured layer
    flush_h = transformer_layers[layers[-1]].register_forward_hook(flush_hook)
    hooks.append(flush_h)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in hooks:
            h.remove()

    # Decode generated tokens for reference
    input_len = input_ids.shape[1]
    generated_ids = output_ids[0, input_len:].tolist()
    generated_tokens = [tokenizer.decode([t], skip_special_tokens=True) for t in generated_ids]

    # Build flat records
    for step_idx, layer_acts in step_data:
        # step_idx=0 is the prefill step (position = last input token / last prefix token)
        # step_idx>0 are autoregressive steps (position = step_idx-th generated token)
        is_prefill = step_idx == 0
        gen_pos = step_idx - 1  # index into generated tokens (-1 = prefill, 0 = first gen token, ...)
        token_text = generated_tokens[gen_pos] if (0 <= gen_pos < len(generated_tokens)) else "[PREFILL]"

        for layer_idx, act in layer_acts.items():
            if layer_idx not in directions:
                continue
            direction = directions[layer_idx].to(act.device)
            projection = torch.dot(act, direction).item()

            step_records.append({
                "prompt_id": prompt.prompt_id,
                "label": prompt.label,
                "category": prompt.category,
                "prefix_k": k,
                "step": step_idx,
                "is_prefill": is_prefill,
                "gen_token_pos": gen_pos,
                "token_text": token_text,
                "layer": layer_idx,
                "projection": projection,
            })

    return step_records


# ---------------------------------------------------------------------------
# Full tracing sweep
# ---------------------------------------------------------------------------

def trace_projections(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[Prompt],
    k_values: list[int],
    directions: dict[int, torch.Tensor],
    cfg: Config,
    output_dir: str | Path,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run projection tracing across all prompts × k values.

    Args:
        model, tokenizer: Loaded model.
        prompts: Prompts to trace.
        k_values: Prefix lengths to sweep.
        directions: Per-layer refusal directions.
        cfg: Config object.
        output_dir: Where to save trace outputs.
        resume: Skip if output file already exists.

    Returns:
        Combined DataFrame with all trace records.
    """
    output_dir = ensure_dir(output_dir)
    prefix_text: str = getattr(cfg.prefilling, "prefix_text", "Sure, here is how you can do that:")
    layers: list[int] = list(directions.keys())
    # Keep generation short for tracing to limit memory/time
    max_new_tokens: int = min(getattr(cfg.generation, "max_new_tokens", 256), 64)
    save_format: str = getattr(cfg.tracing, "save_format", "parquet")

    all_records: list[dict] = []

    for k in k_values:
        out_name = f"traces_k{k:02d}.{save_format}"
        out_path = output_dir / out_name

        if resume and out_path.exists():
            logger.info("Loading cached traces for k=%d from %s", k, out_path)
            df_cached = pd.read_parquet(out_path) if save_format == "parquet" else pd.read_csv(out_path)
            all_records.extend(df_cached.to_dict(orient="records"))
            continue

        logger.info("Tracing k=%d over %d prompts ...", k, len(prompts))
        k_records: list[dict] = []

        for i, prompt in enumerate(prompts):
            try:
                records = trace_single_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    k=k,
                    prefix_text=prefix_text,
                    layers=layers,
                    directions=directions,
                    max_new_tokens=max_new_tokens,
                )
                k_records.extend(records)
            except Exception as e:
                logger.error("Error tracing prompt %s (k=%d): %s", prompt.prompt_id, k, e)

            if (i + 1) % 5 == 0:
                logger.info("  %d/%d prompts traced", i + 1, len(prompts))

        df_k = pd.DataFrame(k_records)
        if save_format == "parquet":
            df_k.to_parquet(out_path, index=False)
        else:
            df_k.to_csv(out_path, index=False)
        logger.info("Saved %d trace records to %s", len(k_records), out_path)
        all_records.extend(k_records)

    if not all_records:
        logger.warning("No trace records collected.")
        return pd.DataFrame()

    return pd.DataFrame(all_records)
