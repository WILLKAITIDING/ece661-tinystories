"""Text generation.

`HFTextGenerator` is the reference implementation of the `TextGenerator`
protocol for any Hugging Face causal LM. `generate_stories` is the ergonomic
entry point used by scripts, notebooks, and every later evaluation task.

Reproducibility, stated precisely
---------------------------------
`model.generate()` takes no `torch.Generator`, so sampling is made reproducible
by seeding global RNGs right before each batch, using a seed derived from the
base seed and the batch index (`seeding.derive_seed`).

Consequence, and it is a real limitation: results are identical across runs
given the same model, params, prompt ORDER, batch_size, device, and library
versions. Changing `batch_size` re-groups prompts into different batches and
therefore changes the sampled text (it is still valid sampling, just not
bit-identical to the previous run). Record `batch_size` with your results.
Greedy decoding (`do_sample=False`) is unaffected by any of this.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from tinystories_project.inference.interface import (
    GeneratedText,
    GenerationParams,
    ModelMetadata,
    TextGenerator,
)
from tinystories_project.inference.loader import (
    ModelBundle,
    prepare_tokenizer_for_generation,
)
from tinystories_project.seeding import derive_seed, seed_everything

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 8


class HFTextGenerator:
    """Wraps a Hugging Face causal LM + tokenizer as a `TextGenerator`."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        metadata: ModelMetadata | None = None,
        device: torch.device | str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model = model
        self.tokenizer = prepare_tokenizer_for_generation(tokenizer)
        self.batch_size = batch_size
        self._device = torch.device(device) if device is not None else _infer_device(model)
        self._metadata = metadata or ModelMetadata(
            model_id=getattr(getattr(model, "config", None), "_name_or_path", "unknown"),
            architecture=type(model).__name__,
            source="custom",
        )

    @classmethod
    def from_bundle(
        cls, bundle: ModelBundle, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> "HFTextGenerator":
        """Preferred constructor: carries the loader's verified metadata along."""
        return cls(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            metadata=bundle.metadata,
            device=bundle.device,
            batch_size=batch_size,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def generate(
        self,
        prompts: Sequence[str],
        params: GenerationParams | None = None,
        seed: int | None = None,
    ) -> list[GeneratedText]:
        """Generate completions for `prompts`.

        Returns `len(prompts) * params.num_return_sequences` items, ordered by
        prompt and then by sample index.
        """
        params = params or GenerationParams()
        if isinstance(prompts, str):
            raise TypeError(
                "prompts must be a sequence of strings, not a single string. "
                "Pass [prompt] instead."
            )
        prompts = list(prompts)
        if not prompts:
            return []

        results: list[GeneratedText] = []
        for batch_index, start in enumerate(range(0, len(prompts), self.batch_size)):
            batch = prompts[start : start + self.batch_size]
            batch_seed = (
                derive_seed(seed, "batch", batch_index) if seed is not None else None
            )
            if batch_seed is not None:
                seed_everything(batch_seed)
            results.extend(
                self._generate_batch(
                    batch, params, offset=start, seed=batch_seed
                )
            )
        return results

    def _generate_batch(
        self,
        prompts: list[str],
        params: GenerationParams,
        offset: int,
        seed: int | None,
    ) -> list[GeneratedText]:
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(self._device)
        prompt_len = encoded["input_ids"].shape[1]

        self._warn_if_context_exceeded(prompt_len, params.max_new_tokens)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.do_sample,
            "num_return_sequences": params.num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        # Only pass sampling knobs when sampling, so greedy runs stay truly
        # greedy and transformers does not warn about ignored arguments.
        if params.do_sample:
            gen_kwargs["temperature"] = params.temperature
            gen_kwargs["top_p"] = params.top_p
            if params.top_k is not None:
                gen_kwargs["top_k"] = params.top_k
        if params.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = params.repetition_penalty

        with torch.no_grad():
            output = self.model.generate(**encoded, **gen_kwargs)

        # Left padding means every sequence's new tokens start at prompt_len.
        new_tokens = output[:, prompt_len:]

        # True (unpadded) prompt lengths, for honest token accounting.
        attn = encoded.get("attention_mask")
        true_prompt_lens = (
            attn.sum(dim=1).tolist() if attn is not None else [prompt_len] * len(prompts)
        )

        results: list[GeneratedText] = []
        n = params.num_return_sequences
        for row, token_ids in enumerate(new_tokens):
            prompt_pos = row // n
            sample_index = row % n
            completion, n_new, finish_reason = self._decode_completion(token_ids)
            results.append(
                GeneratedText(
                    prompt=prompts[prompt_pos],
                    completion=completion,
                    prompt_index=offset + prompt_pos,
                    sample_index=sample_index,
                    seed=seed,
                    n_prompt_tokens=int(true_prompt_lens[prompt_pos]),
                    n_new_tokens=n_new,
                    finish_reason=finish_reason,
                )
            )
        return results

    def _decode_completion(self, token_ids: torch.Tensor) -> tuple[str, int, str]:
        """Decode new tokens, trimming at EOS and reporting why we stopped."""
        ids = token_ids.tolist()
        eos_id = self.tokenizer.eos_token_id
        finish_reason = "length"
        if eos_id is not None and eos_id in ids:
            ids = ids[: ids.index(eos_id)]
            finish_reason = "eos"
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text, len(ids), finish_reason

    def _warn_if_context_exceeded(self, prompt_len: int, max_new_tokens: int) -> None:
        limit = getattr(self._metadata, "max_position_embeddings", None)
        if limit and prompt_len + max_new_tokens > limit:
            logger.warning(
                "prompt (%d tokens) + max_new_tokens (%d) exceeds the model's "
                "context window of %d; output will be truncated by the model.",
                prompt_len,
                max_new_tokens,
                limit,
            )


def generate_stories(
    generator: TextGenerator | ModelBundle,
    prompts: Sequence[str],
    *,
    params: GenerationParams | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    do_sample: bool | None = None,
    num_return_sequences: int | None = None,
    seed: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[GeneratedText]:
    """Generate stories from any `TextGenerator` (or a `ModelBundle`).

    Accepts either a full `GenerationParams` or individual keyword overrides,
    so both of these work:

        generate_stories(bundle, prompts, max_new_tokens=200, temperature=1.0)
        generate_stories(my_custom_gen, prompts, params=cfg.generation, seed=42)

    Passing a `ModelBundle` wraps it in an `HFTextGenerator` automatically; a
    custom model just needs to satisfy the `TextGenerator` protocol.
    """
    if isinstance(generator, ModelBundle):
        generator = HFTextGenerator.from_bundle(generator, batch_size=batch_size)

    overrides = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if params is None:
        params = GenerationParams(**overrides)
    elif overrides:
        params = GenerationParams.from_dict({**params.to_dict(), **overrides})

    return generator.generate(prompts, params, seed=seed)


def _infer_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")
