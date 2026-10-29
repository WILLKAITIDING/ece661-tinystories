"""Checkpoint loading.

Deliberately separate from `generate.py`: loading is about getting weights onto
a device, generation is about sampling tokens. Keeping them apart means Rachel
and Luel can load their own checkpoints however they like and still reuse the
whole evaluation stack.

Verified facts about the official TinyStories checkpoints (checked against the
Hugging Face API on 2026-10-28, not assumed):

  * All 14 `roneneldan/TinyStories-*` repos are `GPTNeoForCausalLM`, public and
    ungated, fine-tuned from `EleutherAI/gpt-neo-125M`.
  * `gpt_neo` is a NATIVE Transformers architecture, so `trust_remote_code` is
    NOT required and is never set by this module.
  * Weights ship as `pytorch_model.bin` only (no safetensors).
  * The tokenizer is a GPT-2 BPE with vocab 50257 and `pad_token = null`, so a
    pad token MUST be assigned before batched generation. The bundled files
    were verified identical to `EleutherAI/gpt-neo-125M`.
  * Loading under transformers >= 5 reports `attn.attention.bias` and
    `attn.attention.masked_bias` as UNEXPECTED. Those are obsolete causal-mask
    buffers from transformers 4.28, not learned weights. Ignoring them is
    correct and does not change outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tinystories_project.runtime import describe_device, resolve_device
from tinystories_project.inference.interface import ModelMetadata

logger = logging.getLogger(__name__)

# The checkpoints' own README recommends this tokenizer. The repos also ship
# their own BPE files, verified identical to this one (vocab 50257, same ids).
# Pass tokenizer_id=GPT_NEO_TOKENIZER_ID if a checkpoint ever ships none.
GPT_NEO_TOKENIZER_ID = "EleutherAI/gpt-neo-125M"

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class CheckpointLoadError(RuntimeError):
    """Raised when a model or tokenizer cannot be loaded, with a usable hint."""


@dataclass
class ModelBundle:
    """A loaded model, its tokenizer, and the metadata describing both."""

    model: Any
    tokenizer: Any
    metadata: ModelMetadata
    device: torch.device

    def __repr__(self) -> str:  # keep reprs short in notebooks
        m = self.metadata
        return (
            f"ModelBundle({m.model_id!r}, {m.architecture}, "
            f"{m.n_parameters_total:,} params, device={self.device})"
        )


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Split a parameter count into embedding and non-embedding parts.

    Tiny models are dominated by their embedding table, so the totals alone are
    misleading. Tied input/output embeddings are counted once.
    """
    seen: set[int] = set()
    embedding = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    embedding += p.numel()

    total = 0
    counted: set[int] = set()
    for p in model.parameters():
        if id(p) not in counted:
            counted.add(id(p))
            total += p.numel()

    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": total - embedding,
    }


def load_causal_lm(
    model_id: str,
    *,
    tokenizer_id: str | None = None,
    device: str = "auto",
    dtype: str = "float32",
    revision: str | None = None,
    cache_dir: str | None = None,
) -> ModelBundle:
    """Load a causal LM + tokenizer and put the model in eval mode.

    Args:
        model_id: Hub repo id or a local directory path.
        tokenizer_id: Defaults to `model_id`. Override when a checkpoint ships
            no tokenizer (then use `GPT_NEO_TOKENIZER_ID`), or for Luel's
            Amharic SentencePiece tokenizers paired with English weights.
        device: "auto" | "cpu" | "cuda" | "mps". Never hardcoded to cuda.
        dtype: float32 by default. The TinyStories weights are float32 and these
            models are small, so reduced precision buys nothing and would make
            results harder to compare across machines.
        revision: Pin a specific commit SHA for strict reproducibility.
        cache_dir: Override the Hugging Face cache location.

    Raises:
        CheckpointLoadError: on any load failure, with the cause attached.
    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")

    resolved_device = resolve_device(device)
    tokenizer_id = tokenizer_id or model_id
    hub_kwargs: dict[str, Any] = {}
    if revision:
        hub_kwargs["revision"] = revision
    if cache_dir:
        hub_kwargs["cache_dir"] = cache_dir

    logger.info("Loading model %s onto %s", model_id, describe_device(resolved_device))

    try:
        config = AutoConfig.from_pretrained(model_id, **hub_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=DTYPES[dtype], **hub_kwargs
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise CheckpointLoadError(
            f"Could not load model {model_id!r}. Check the repo id is correct, "
            f"that you are online, and that the repo is not gated. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **hub_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointLoadError(
            f"Could not load tokenizer {tokenizer_id!r}. If this checkpoint "
            f"ships no tokenizer files, pass tokenizer_id={GPT_NEO_TOKENIZER_ID!r}. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    tokenizer = prepare_tokenizer_for_generation(tokenizer)

    try:
        model.to(resolved_device)
    except (RuntimeError, NotImplementedError) as exc:
        logger.warning(
            "Could not move model to %s (%s). Falling back to CPU.",
            resolved_device,
            exc,
        )
        resolved_device = torch.device("cpu")
        model.to(resolved_device)

    model.eval()  # disable dropout; we never train in this module

    # Keep model and tokenizer consistent about padding.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    counts = count_parameters(model)
    metadata = ModelMetadata(
        model_id=model_id,
        architecture=type(model).__name__,
        tokenizer_id=tokenizer_id,
        revision=revision,
        source="local" if _looks_like_path(model_id) else "huggingface",
        n_parameters_total=counts["total"],
        n_parameters_non_embedding=counts["non_embedding"],
        n_parameters_embedding=counts["embedding"],
        vocab_size=getattr(config, "vocab_size", None),
        max_position_embeddings=getattr(config, "max_position_embeddings", None),
        n_layers=_first_attr(config, "num_layers", "n_layer", "num_hidden_layers"),
        hidden_size=_first_attr(config, "hidden_size", "n_embd"),
        dtype=dtype,
        device=describe_device(resolved_device),
        framework="pytorch",
        extra={"model_type": getattr(config, "model_type", None)},
    )

    logger.info(
        "Loaded %s: %s params total (%s non-embedding)",
        model_id,
        f"{counts['total']:,}",
        f"{counts['non_embedding']:,}",
    )
    return ModelBundle(
        model=model, tokenizer=tokenizer, metadata=metadata, device=resolved_device
    )


def prepare_tokenizer_for_generation(tokenizer: Any) -> Any:
    """Make a tokenizer safe for *batched* decoder-only generation.

    Two corrections, both required for correctness rather than convenience:

    1. The TinyStories tokenizer has `pad_token = None`. Batching without a pad
       token raises; we reuse the EOS token, which is standard for GPT-2/Neo.
    2. Decoder-only models must be LEFT-padded. With the default right padding
       the model continues from pad tokens and emits garbage for every sequence
       shorter than the longest in the batch. This is silent corruption, so we
       always force it.
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise CheckpointLoadError(
                "Tokenizer has neither a pad token nor an eos token; cannot batch."
            )
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("pad_token was None; set to eos_token %r", tokenizer.eos_token)

    tokenizer.padding_side = "left"
    return tokenizer


def _looks_like_path(model_id: str) -> bool:
    return model_id.startswith((".", "/", "~")) or model_id.count("/") > 1


def _first_attr(obj: Any, *names: str) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None
