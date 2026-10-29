"""The shared generation contract for every model in this project.

This module defines *what a generator must do*, not *how any particular model
does it*. Everything downstream (qualitative figures, ROUGE, the Gemini judge,
the Phase 2 architecture table) consumes `GeneratedText` and never touches a
model object directly.

Three concrete types matter:

* `GenerationParams`  - the sampling settings, recorded verbatim in every output
                        record so an experiment can be replayed exactly.
* `ModelMetadata`     - identity + size of whatever produced the text. This is
                        what makes the Phase 2 architecture comparison possible.
* `TextGenerator`     - the protocol a model wrapper must satisfy.

HOW TO PLUG IN A CUSTOM MODEL (Rachel / Luel)
---------------------------------------------
Do NOT assume `model.generate()` exists. A minGRU, Mamba, or hand-rolled
Transformer only has to provide a class satisfying `TextGenerator`:

    class MyGenerator:
        @property
        def metadata(self) -> ModelMetadata: ...

        def generate(
            self,
            prompts: Sequence[str],
            params: GenerationParams,
            seed: int | None = None,
        ) -> list[GeneratedText]: ...

Required behaviour of `generate`:
  1. Return exactly `len(prompts) * params.num_return_sequences` items.
  2. `completion` holds ONLY newly generated text - never echo the prompt.
  3. Set `prompt_index` to the index of the prompt within `prompts`, and
     `sample_index` to which of the N samples this is.
  4. Seed the RNG from `seed` if given, and record it on each item.
  5. Set `finish_reason` to "eos" if generation stopped at the end-of-sequence
     token, or "length" if it hit `max_new_tokens`.

Satisfy that and every evaluation script in this repo works on your model with
no changes. `HFTextGenerator` in `generate.py` is the reference implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

FinishReason = str  # "eos" | "length" | "unknown"


@dataclass(frozen=True)
class GenerationParams:
    """Sampling configuration. Immutable so it cannot drift mid-experiment."""

    max_new_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None
    do_sample: bool = True
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}. "
                "For greedy decoding set do_sample=False instead of temperature=0."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {self.top_k}")
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}"
            )
        if self.num_return_sequences < 1:
            raise ValueError(
                f"num_return_sequences must be >= 1, got {self.num_return_sequences}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationParams":
        """Build from a dict, ignoring unknown keys so configs can carry extras."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class ModelMetadata:
    """Identity and shape of the thing that generated text.

    `n_parameters_total` and `n_parameters_non_embedding` are both recorded
    because they differ enormously for tiny models: TinyStories-1M is ~3.75M
    parameters in total but only ~0.40M outside its 50257-token embedding
    table. The Phase 2 "parameter-matched" comparison is only meaningful if
    everyone agrees which number they are matching.
    """

    model_id: str
    architecture: str
    tokenizer_id: str | None = None
    revision: str | None = None
    source: str = "huggingface"  # "huggingface" | "local" | "custom"
    n_parameters_total: int | None = None
    n_parameters_non_embedding: int | None = None
    n_parameters_embedding: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    n_layers: int | None = None
    hidden_size: int | None = None
    dtype: str | None = None
    device: str | None = None
    framework: str = "pytorch"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class GeneratedText:
    """One completion for one prompt. The atom every evaluation consumes."""

    prompt: str
    completion: str
    prompt_index: int = 0
    sample_index: int = 0
    seed: int | None = None
    n_prompt_tokens: int | None = None
    n_new_tokens: int | None = None
    finish_reason: FinishReason = "unknown"

    @property
    def full_text(self) -> str:
        """Prompt and continuation joined, as a reader would see the story."""
        return self.prompt + self.completion

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["full_text"] = self.full_text
        return data


@runtime_checkable
class TextGenerator(Protocol):
    """What every model wrapper must provide. See the module docstring."""

    @property
    def metadata(self) -> ModelMetadata: ...

    def generate(
        self,
        prompts: Sequence[str],
        params: GenerationParams,
        seed: int | None = None,
    ) -> list[GeneratedText]: ...
