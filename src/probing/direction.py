"""
Refusal direction extraction via difference-in-means (DiM).

Method:
  1. Run the model on N harmful prompts and N benign prompts.
  2. Collect the residual-stream activation (hidden state) at the end of
     each prompt's last token, at each configured transformer layer.
  3. Compute: direction[layer] = mean(harmful_acts[layer]) - mean(benign_acts[layer])
  4. L2-normalize each direction vector.
  5. Save directions to disk as a dict of {layer_idx: tensor}.

This is the standard "representation engineering" / difference-in-means probe.
It gives a direction in the model's representation space along which harmful
and harmless content are most separated at each layer.

Reference: Zou et al. (2023) "Representation Engineering"; Arditi et al. (2024).
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data.schema import Prompt
from src.generation.prefilling import build_prefilled_input

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation collection via forward hooks
# ---------------------------------------------------------------------------

class ActivationCollector:
    """
    Context manager that registers forward hooks on specified transformer layers
    and collects the residual-stream output (post-layer hidden state) after
    a forward pass.

    Works with Llama-style models where layers are at model.model.layers[i].
    Each layer hook captures output[0] which is the hidden state tensor
    of shape (batch, seq_len, hidden_size).
    """

    def __init__(self, model: PreTrainedModel, layers: list[int]) -> None:
        self.model = model
        self.layers = layers
        self._hooks: list = []
        # {layer_idx: list of tensors collected across calls}
        self.activations: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # Llama layer output: (hidden_states, past_key_value, ...)
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden shape: (batch, seq_len, hidden_size)
            self.activations[layer_idx].append(hidden.detach().cpu())
        return hook_fn

    def __enter__(self):
        transformer_layers = self.model.model.layers
        for layer_idx in self.layers:
            handle = transformer_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def clear(self) -> None:
        """Reset collected activations between runs."""
        for l in self.layers:
            self.activations[l].clear()

    def get_last_token(self, layer_idx: int, call_index: int = -1) -> torch.Tensor:
        """
        Return the last-token activation from the most recent (or specified) call.

        Args:
            layer_idx: Which layer's activations to retrieve.
            call_index: Index into the list of collected activations.

        Returns:
            Tensor of shape (hidden_size,).
        """
        hidden = self.activations[layer_idx][call_index]  # (batch, seq_len, hidden)
        return hidden[0, -1, :]  # last token, first batch element

    def get_position(
        self, layer_idx: int, position: int, call_index: int = -1
    ) -> torch.Tensor:
        """
        Return the activation at a specific token position.

        Args:
            layer_idx: Layer index.
            position: Token position (-1 for last).
            call_index: Index into collected activations.

        Returns:
            Tensor of shape (hidden_size,).
        """
        hidden = self.activations[layer_idx][call_index]  # (batch, seq_len, hidden)
        return hidden[0, position, :]


# ---------------------------------------------------------------------------
# Refusal direction extraction
# ---------------------------------------------------------------------------

def _collect_last_token_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[Prompt],
    layers: list[int],
    direction_position: int = -1,
) -> dict[int, torch.Tensor]:
    """
    Run forward passes on all prompts and collect activations.

    Args:
        model: Loaded causal LM.
        tokenizer: Corresponding tokenizer.
        prompts: List of Prompt objects.
        layers: List of layer indices.
        direction_position: Token position to extract (-1 = last token).

    Returns:
        Dict mapping layer_idx → tensor of shape (n_prompts, hidden_size).
    """
    all_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    with ActivationCollector(model, layers) as collector:
        for prompt in prompts:
            collector.clear()
            messages = [{"role": "user", "content": prompt.text}]
            tok_out = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # apply_chat_template may return a BatchEncoding or a raw Tensor
            logger.debug("tokenizer output type: %s", type(tok_out))
            if isinstance(tok_out, torch.Tensor):
                input_ids = tok_out
                attention_mask = torch.ones_like(input_ids)
            else:  # BatchEncoding / dict-like
                input_ids = tok_out["input_ids"]
                attention_mask = tok_out.get("attention_mask", torch.ones_like(input_ids))

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            logger.debug("input_ids shape: %s, attention_mask present: True", input_ids.shape)

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for layer_idx in layers:
                act = collector.get_position(layer_idx, direction_position)
                all_acts[layer_idx].append(act)

    return {
        layer_idx: torch.stack(acts, dim=0)  # (n_prompts, hidden_size)
        for layer_idx, acts in all_acts.items()
    }


def extract_refusal_direction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    harmful_prompts: list[Prompt],
    benign_prompts: list[Prompt],
    layers: list[int],
    direction_position: int = -1,
    held_out_n: Optional[int] = None,
) -> dict[int, torch.Tensor]:
    """
    Compute the refusal direction per layer via difference-in-means.

    Direction[layer] = normalize(mean(harmful_acts) - mean(benign_acts))

    Args:
        model: Loaded causal LM.
        tokenizer: Corresponding tokenizer.
        harmful_prompts: List of Prompt(label="harmful") objects.
        benign_prompts: List of Prompt(label="benign") objects.
        layers: Layer indices to compute directions for.
        direction_position: Token position to use (-1 = last token).
        held_out_n: If set, use only the first N prompts from each set.

    Returns:
        Dict mapping layer_idx → normalized direction tensor of shape (hidden_size,).
    """
    if held_out_n is not None:
        harmful_prompts = harmful_prompts[:held_out_n]
        benign_prompts = benign_prompts[:held_out_n]

    n = min(len(harmful_prompts), len(benign_prompts))
    harmful_prompts = harmful_prompts[:n]
    benign_prompts = benign_prompts[:n]

    logger.info(
        "Extracting refusal direction from %d harmful + %d benign prompts across %d layers",
        n, n, len(layers),
    )

    logger.info("Collecting harmful prompt activations ...")
    harmful_acts = _collect_last_token_activations(
        model, tokenizer, harmful_prompts, layers, direction_position
    )

    logger.info("Collecting benign prompt activations ...")
    benign_acts = _collect_last_token_activations(
        model, tokenizer, benign_prompts, layers, direction_position
    )

    directions: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        harm_mean = harmful_acts[layer_idx].mean(dim=0)  # (hidden_size,)
        beni_mean = benign_acts[layer_idx].mean(dim=0)
        diff = harm_mean - beni_mean
        direction = F.normalize(diff, dim=0)
        directions[layer_idx] = direction
        logger.debug(
            "Layer %d: direction norm before normalization = %.4f",
            layer_idx,
            diff.norm().item(),
        )

    logger.info("Refusal directions computed for layers: %s", layers)
    return directions


# ---------------------------------------------------------------------------
# Save / load directions
# ---------------------------------------------------------------------------

def save_direction(
    directions: dict[int, torch.Tensor],
    path: str | Path,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save computed refusal directions to disk.

    Saved as a dict with keys "directions" and optional "metadata".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "directions": {str(k): v for k, v in directions.items()},
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    logger.info("Saved refusal directions to %s", path)


def load_direction(path: str | Path) -> dict[int, torch.Tensor]:
    """
    Load previously saved refusal directions.

    Returns:
        Dict mapping layer_idx (int) → direction tensor.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Refusal direction file not found: {path}\n"
            "Run scripts/extract_refusal_direction.py first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    directions = {int(k): v for k, v in payload["directions"].items()}
    logger.info("Loaded refusal directions for layers: %s", sorted(directions.keys()))
    return directions
