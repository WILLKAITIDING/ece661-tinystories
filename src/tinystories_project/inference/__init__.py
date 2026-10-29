"""Checkpoint loading and text generation.

Loading (`loader`) is separate from generation (`generate`) so any model
satisfying the `TextGenerator` protocol in `interface` works with every
evaluation here, whether or not it supports Hugging Face `.generate()`.
"""

from tinystories_project.inference.generate import HFTextGenerator, generate_stories
from tinystories_project.inference.interface import (
    GeneratedText,
    GenerationParams,
    ModelMetadata,
    TextGenerator,
)
from tinystories_project.inference.loader import ModelBundle, load_causal_lm

__all__ = [
    "GeneratedText", "GenerationParams", "HFTextGenerator", "ModelBundle",
    "ModelMetadata", "TextGenerator", "generate_stories", "load_causal_lm",
]
