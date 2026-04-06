"""
Model loading and generation pipeline.

Supports:
  - Normal generation (k=0, no prefix)
  - Prefilling generation (k>0, forced prefix tokens)

All outputs are saved as JSONL with a config snapshot for reproducibility.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config, save_config_snapshot
from src.data.schema import Prompt
from src.utils.io_utils import ensure_dir, save_jsonl, output_exists
from .prefilling import build_prefilled_input

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: Config) -> tuple:
    """
    Load model and tokenizer from HuggingFace (or local path).

    Reads HF_TOKEN from environment if set. Uses LOCAL_MODEL_PATH override
    if configured.

    Args:
        cfg: Loaded Config object.

    Returns:
        (model, tokenizer) tuple.
    """
    model_name: str = cfg.model.name
    local_path: Optional[str] = getattr(cfg.model, "local_path", None)
    model_id = local_path if local_path else model_name

    hf_token = os.environ.get("HF_TOKEN")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype_str = getattr(cfg.model, "dtype", "bfloat16")
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    logger.info("Loading tokenizer from %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s (dtype=%s)", model_id, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=getattr(cfg.model, "device_map", "auto"),
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded. Device map: %s", model.hf_device_map if hasattr(model, "hf_device_map") else "single device")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single prompt generation
# ---------------------------------------------------------------------------

def generate_one(
    model,
    tokenizer,
    prompt: Prompt,
    k: int,
    cfg: Config,
) -> dict:
    """
    Generate a response for a single prompt with optional prefilling.

    Args:
        model: Loaded causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Prompt object.
        k: Number of prefix tokens to force (0 = no prefilling).
        cfg: Config object.

    Returns:
        Dict with prompt, generated text, tokens, and metadata.
    """
    prefix_text: str = getattr(cfg.prefilling, "prefix_text", "Sure, here is how you can do that:")
    max_new_tokens: int = getattr(cfg.generation, "max_new_tokens", 256)
    do_sample: bool = getattr(cfg.generation, "do_sample", False)

    # Build input with optional prefix
    input_ids = build_prefilled_input(
        tokenizer=tokenizer,
        prompt_text=prompt.text,
        prefix_text=prefix_text,
        k=k,
    )
    input_length = input_ids.shape[1]
    logger.debug("generate_one: prompt=%s k=%d input_length=%d", prompt.prompt_id, k, input_length)

    # Build generate kwargs — only pass sampling params when do_sample=True
    attention_mask = torch.ones_like(input_ids)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = getattr(cfg.generation, "temperature", 1.0)
        gen_kwargs["top_p"] = getattr(cfg.generation, "top_p", 1.0)

    logger.debug("about to call model.generate")
    logger.debug(
        "input_ids metadata: device=%s dtype=%s shape=%s",
        input_ids.device,
        input_ids.dtype,
        tuple(input_ids.shape),
    )
    with torch.no_grad():
        try:
            output_ids = model.generate(**gen_kwargs)
            logger.debug("model.generate returned successfully")
        except Exception:
            logger.exception("model.generate failed")
            raise

    # Decode only the newly generated tokens (exclude the input)
    generated_ids = output_ids[0, input_length:]
    logger.debug("generate_one: prompt=%s generated %d tokens", prompt.prompt_id, len(generated_ids))
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "prompt_id": prompt.prompt_id,
        "label": prompt.label,
        "category": prompt.category,
        "source": prompt.source,
        "prompt_text": prompt.text,
        "prefix_k": k,
        "prefix_text_used": tokenizer.decode(
            input_ids[0, input_length - k:] if k > 0 else [],
            skip_special_tokens=True,
        ),
        "generated_text": generated_text,
        "n_generated_tokens": len(generated_ids),
        "input_length": input_length,
    }


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_responses(
    model,
    tokenizer,
    prompts: list[Prompt],
    k_values: list[int],
    cfg: Config,
    output_dir: str | Path,
    resume: bool = True,
) -> list[dict]:
    """
    Run generation for all prompts x all k values.

    Args:
        model, tokenizer: Loaded from load_model_and_tokenizer.
        prompts: List of Prompt objects.
        k_values: List of prefix lengths to sweep over.
        cfg: Config object.
        output_dir: Where to save JSONL outputs.
        resume: If True, skip prompts whose output file already exists.

    Returns:
        List of all result dicts.
    """
    output_dir = ensure_dir(output_dir)
    all_results: list[dict] = []

    seed: int = getattr(cfg.generation, "seed", 42)
    torch.manual_seed(seed)

    for k in k_values:
        out_path = output_dir / f"k{k:02d}.jsonl"

        if resume and output_exists(out_path):
            logger.info("Skipping k=%d, output already exists at %s", k, out_path)
            from src.utils.io_utils import load_jsonl
            all_results.extend(load_jsonl(out_path))
            continue

        logger.info("Generating for k=%d over %d prompts ...", k, len(prompts))
        k_results: list[dict] = []
        t0 = time.time()

        for i, prompt in enumerate(prompts):
            try:
                result = generate_one(model, tokenizer, prompt, k, cfg)
                k_results.append(result)
            except Exception as e:
                logger.exception("Error on prompt %s (k=%d)", prompt.prompt_id, k)
                k_results.append({
                    "prompt_id": prompt.prompt_id,
                    "prefix_k": k,
                    "error": f"{type(e).__name__}: {e!r}",
                })

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                logger.info("  %d/%d prompts done (%.1fs elapsed)", i + 1, len(prompts), elapsed)

        logger.debug("about to write %d output rows to %s", len(k_results), out_path)
        save_jsonl(k_results, out_path)
        logger.debug("finished writing %d output rows to %s", len(k_results), out_path)
        all_results.extend(k_results)
        logger.info("Saved %d results to %s", len(k_results), out_path)

    return all_results
