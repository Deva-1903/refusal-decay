"""
Data schema for prompts used throughout the pipeline.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class Prompt:
    """
    Canonical prompt representation used throughout the pipeline.

    Fields
    ------
    prompt_id : str
        Unique identifier, e.g. "h001" or "b001".
    text : str
        The raw prompt text sent to the model.
    label : str
        "harmful" or "benign".
    category : str
        Topic category (e.g. "weapons", "cybercrime", "cooking").
    source : str
        Where the prompt came from (e.g. "fake_example", "advbench").
    """

    prompt_id: str
    text: str
    label: str                     # "harmful" | "benign"
    category: str = "unknown"
    source: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Prompt":
        return cls(
            prompt_id=d["prompt_id"],
            text=d["text"],
            label=d.get("label", "unknown"),
            category=d.get("category", "unknown"),
            source=d.get("source", "unknown"),
        )
