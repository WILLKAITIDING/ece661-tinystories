"""Checkpoint loading and text generation.

Loading (`loader`) and generation (`generate`) are kept separate on purpose:
any model that satisfies the `TextGenerator` protocol in `interface` works with
every evaluation in this repo, whether or not it supports `.generate()`.
"""

from tinystories_project.inference.generate import (
    HFTextGenerator,
    generate_stories,
)
from tinystories_project.inference.interface import (
    GeneratedText,
    GenerationParams,
    ModelMetadata,
    TextGenerator,
)
from tinystories_project.inference.loader import (
    GPT_NEO_TOKENIZER_ID,
    CheckpointLoadError,
    ModelBundle,
    count_parameters,
    load_causal_lm,
    prepare_tokenizer_for_generation,
    verify_tokenizer_matches,
)

__all__ = [
    "GPT_NEO_TOKENIZER_ID",
    "CheckpointLoadError",
    "GeneratedText",
    "GenerationParams",
    "HFTextGenerator",
    "ModelBundle",
    "ModelMetadata",
    "TextGenerator",
    "count_parameters",
    "generate_stories",
    "load_causal_lm",
    "prepare_tokenizer_for_generation",
    "verify_tokenizer_matches",
]
