"""
Prefilling attack implementation.

A prefilling attack forces the model to begin its response with k tokens
of a compliant-sounding prefix (e.g., "Sure, here is how you can do that:").
This is done by appending the prefix tokens to the encoded chat prompt before
handing control to the model's generate() method.

The key insight: when the model sees those tokens already in its context window
as "prior output", it conditions its continuation on them and is more likely to
produce a compliant (non-refusing) response.

Reference: Qi et al. (2023), Wei et al. (2024).
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def build_prefilled_input(
    tokenizer,
    prompt_text: str,
    prefix_text: str,
    k: int,
    system_prompt: Optional[str] = None,
) -> torch.Tensor:
    """
    Build input_ids for a chat prompt with k forced prefix tokens.

    Constructs the full input as:
        [chat_template(user_prompt, add_generation_prompt=True)] + [prefix_tokens[:k]]

    The model will then generate starting from token k+1 of the prefix.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        prompt_text: The user's prompt text.
        prefix_text: The full compliant prefix string to tokenize and truncate.
        k: Number of prefix tokens to inject. k=0 means no prefilling.
        system_prompt: Optional system message (None = no system turn).

    Returns:
        input_ids tensor of shape (1, seq_len).
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    # Tokenize chat with the generation prompt suffix (e.g., <|start_header_id|>assistant<|end_header_id|>)
    chat_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )  # shape: (1, n_chat_tokens)

    if k == 0:
        return chat_ids

    # Tokenize the full prefix, suppress the BOS token that some tokenizers add
    prefix_ids = tokenizer.encode(
        prefix_text,
        add_special_tokens=False,
    )

    if len(prefix_ids) < k:
        logger.warning(
            "Prefix '%s' tokenizes to only %d tokens, but k=%d requested. "
            "Using all %d tokens.",
            prefix_text,
            len(prefix_ids),
            k,
            len(prefix_ids),
        )
        k = len(prefix_ids)

    prefix_ids_truncated = torch.tensor([prefix_ids[:k]])  # shape: (1, k)
    combined = torch.cat([chat_ids, prefix_ids_truncated], dim=1)

    logger.debug(
        "Built prefilled input: chat=%d tokens + prefix=%d tokens = %d total",
        chat_ids.shape[1],
        k,
        combined.shape[1],
    )
    return combined


def decode_prefix_used(tokenizer, input_ids: torch.Tensor, chat_length: int) -> str:
    """
    Decode only the prefix tokens that were injected (beyond the chat template).

    Args:
        tokenizer: HuggingFace tokenizer.
        input_ids: Full input tensor (1, seq_len).
        chat_length: Number of tokens in the base chat (without prefix).

    Returns:
        Decoded string of the injected prefix tokens.
    """
    prefix_ids = input_ids[0, chat_length:]
    if prefix_ids.numel() == 0:
        return ""
    return tokenizer.decode(prefix_ids, skip_special_tokens=True)
