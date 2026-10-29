"""Shared fixtures.

Unit tests must stay fast and offline, so they build a *tiny randomly
initialised* GPT-Neo (vocab 64, hidden 16, 2 layers - a few thousand
parameters) instead of downloading a checkpoint. The text it produces is
gibberish, which is fine: these tests check plumbing, not language quality.

Real-checkpoint tests live in tests/test_integration_real_model.py and are
marked `integration` so they are skipped by default.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM

from tinystories_project.inference.interface import GenerationParams, ModelMetadata


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """A real GPT-2 BPE tokenizer (cached locally by HF after first use)."""
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:  # offline and not cached
        pytest.skip(f"gpt2 tokenizer unavailable offline: {exc}")
    return tok


@pytest.fixture(scope="session")
def tiny_model(tiny_tokenizer):
    """A randomly initialised GPT-Neo small enough to run instantly on CPU."""
    config = GPTNeoConfig(
        vocab_size=len(tiny_tokenizer),
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=128,
        attention_types=[[["global", "local"], 1]],
        window_size=16,
        bos_token_id=tiny_tokenizer.eos_token_id,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    model = GPTNeoForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def tiny_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id="test/tiny-gpt-neo",
        architecture="GPTNeoForCausalLM",
        tokenizer_id="gpt2",
        n_parameters_total=1234,
        n_parameters_non_embedding=234,
        n_parameters_embedding=1000,
        vocab_size=64,
        max_position_embeddings=128,
        n_layers=2,
        hidden_size=16,
        dtype="float32",
        device="cpu",
    )


@pytest.fixture
def short_params() -> GenerationParams:
    """Few tokens so tests finish fast."""
    return GenerationParams(max_new_tokens=8, temperature=1.0, top_p=0.95)


@pytest.fixture
def prompts() -> list[str]:
    # Deliberately different lengths, to exercise left-padding.
    return [
        "Once upon a time, there was a little girl named Lily.",
        "The cat sat.",
        "Tom and Jane were playing in the park when they found a tiny blue key.",
    ]
