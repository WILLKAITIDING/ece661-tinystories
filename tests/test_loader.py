"""Loader behaviour that does not require downloading a checkpoint."""

from __future__ import annotations

import pytest
import torch

from tinystories_project.devices import (
    available_devices,
    describe_device,
    resolve_device,
)
from tinystories_project.inference.loader import (
    CheckpointLoadError,
    count_parameters,
    load_causal_lm,
    prepare_tokenizer_for_generation,
)
from tinystories_project.seeding import MAX_SEED, derive_seed, seed_everything


# --- device handling -------------------------------------------------------

def test_cpu_is_always_available():
    assert "cpu" in available_devices()


def test_auto_never_raises():
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_explicit_cpu_is_honoured():
    assert resolve_device("cpu").type == "cpu"


def test_unavailable_device_falls_back_to_cpu():
    """A config written on a GPU box must still run on a laptop."""
    missing = next(d for d in ("cuda", "mps") if d not in available_devices())
    assert resolve_device(missing).type == "cpu"


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu-v5")


def test_describe_device_is_readable():
    assert describe_device(torch.device("cpu")) == "cpu"


# --- parameter counting ----------------------------------------------------

def test_counts_split_embedding_and_non_embedding(tiny_model):
    counts = count_parameters(tiny_model)
    assert counts["total"] == counts["embedding"] + counts["non_embedding"]
    assert counts["embedding"] > 0 and counts["non_embedding"] > 0


def test_total_matches_torch(tiny_model):
    expected = sum(p.numel() for p in tiny_model.parameters())
    assert count_parameters(tiny_model)["total"] == expected


def test_tied_embeddings_are_not_double_counted(tiny_model):
    """GPT-Neo ties lm_head to wte; counting it twice would inflate sizes."""
    counts = count_parameters(tiny_model)
    naive = sum(p.numel() for p in tiny_model.parameters())
    assert counts["total"] == naive


# --- tokenizer preparation -------------------------------------------------

def test_pad_token_is_assigned_when_missing(tiny_tokenizer):
    tok = prepare_tokenizer_for_generation(tiny_tokenizer)
    assert tok.pad_token is not None
    assert tok.pad_token == tok.eos_token


def test_padding_side_is_forced_left(tiny_tokenizer):
    """Right padding silently corrupts decoder-only batched generation."""
    tiny_tokenizer.padding_side = "right"
    assert prepare_tokenizer_for_generation(tiny_tokenizer).padding_side == "left"


# --- error handling --------------------------------------------------------

def test_bad_dtype_raises_before_any_download():
    with pytest.raises(ValueError, match="dtype"):
        load_causal_lm("roneneldan/TinyStories-1M", dtype="int4")


def test_nonexistent_checkpoint_raises_checkpoint_load_error():
    with pytest.raises(CheckpointLoadError, match="Could not load model"):
        load_causal_lm("this-org-does-not-exist/nor-does-this-model-xyz")


# --- seeding ---------------------------------------------------------------

def test_seed_everything_returns_seed():
    assert seed_everything(42) == 42


def test_seed_everything_makes_torch_deterministic():
    seed_everything(7)
    a = torch.randn(5)
    seed_everything(7)
    assert torch.equal(a, torch.randn(5))


@pytest.mark.parametrize("bad", [-1, MAX_SEED + 1])
def test_out_of_range_seeds_raise(bad):
    with pytest.raises(ValueError, match="seed"):
        seed_everything(bad)


@pytest.mark.parametrize("bad", ["42", 4.2, True])
def test_non_int_seeds_raise(bad):
    with pytest.raises(TypeError, match="int"):
        seed_everything(bad)


def test_derive_seed_is_deterministic_and_in_range():
    a = derive_seed(42, "batch", 3)
    assert a == derive_seed(42, "batch", 3)
    assert 0 <= a <= MAX_SEED


def test_derive_seed_separates_different_inputs():
    assert derive_seed(42, "batch", 0) != derive_seed(42, "batch", 1)
    assert derive_seed(1, "batch", 0) != derive_seed(2, "batch", 0)


def test_derive_seed_is_stable_across_processes():
    """Hard-coded so a change to the derivation is caught, not silently shipped."""
    assert derive_seed(42, "batch", 0) == 1793512846
