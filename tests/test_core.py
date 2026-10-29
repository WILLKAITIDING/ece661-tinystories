"""Config, generation params, device/seed runtime, and offline loader checks."""

from __future__ import annotations

import pytest
import torch
import yaml

from tinystories_project.config import (
    Task1Config,
    find_project_root,
    load_task1_config,
    resolve_path,
)
from tinystories_project.inference.interface import GenerationParams
from tinystories_project.inference.loader import (
    CheckpointLoadError,
    count_parameters,
    load_causal_lm,
    prepare_tokenizer_for_generation,
)
from tinystories_project.runtime import (
    MAX_SEED,
    available_devices,
    batch_seed,
    describe_device,
    resolve_device,
    seed_everything,
)

MINIMAL = {
    "run_name": "unit",
    "model": {"model_id": "roneneldan/TinyStories-1M"},
    "prompts": ["Once upon a time"],
}


def write_cfg(tmp_path, data):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- config ----------------------------------------------------------------

def test_loads_minimal_config(tmp_path):
    cfg = load_task1_config(write_cfg(tmp_path, MINIMAL))
    assert cfg.model.model_id == "roneneldan/TinyStories-1M"
    assert cfg.seed == 42, "default seed must be stable across runs"
    assert cfg.model.device == "auto", "device must default to auto, never cuda"


def test_generation_section_is_parsed_with_defaults_kept(tmp_path):
    data = {**MINIMAL, "generation": {"max_new_tokens": 50, "temperature": 0.7}}
    cfg = load_task1_config(write_cfg(tmp_path, data))
    assert (cfg.generation.max_new_tokens, cfg.generation.temperature) == (50, 0.7)
    assert cfg.generation.top_p == 0.95


def test_unknown_generation_keys_are_ignored(tmp_path):
    """Configs may carry keys meant for later tasks without breaking Task 1."""
    data = {**MINIMAL, "generation": {"max_new_tokens": 10, "future_knob": 3}}
    assert load_task1_config(write_cfg(tmp_path, data)).generation.max_new_tokens == 10


@pytest.mark.parametrize(
    "data, match",
    [
        ({"prompts": ["x"]}, "model"),
        ({"model": {}, "prompts": ["x"]}, "model_id"),
        ({**MINIMAL, "prompts": []}, "at least one prompt"),
        ({**MINIMAL, "prompts": ["ok", "  "]}, "non-empty"),
        ({**MINIMAL, "prompts": ["same", "same"]}, "unique"),
    ],
)
def test_invalid_configs_raise(tmp_path, data, match):
    with pytest.raises(ValueError, match=match):
        load_task1_config(write_cfg(tmp_path, data))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_task1_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_task1_config(path)


def test_config_roundtrips_to_dict(tmp_path):
    cfg = load_task1_config(write_cfg(tmp_path, MINIMAL))
    assert Task1Config.from_dict(cfg.to_dict()).generation == cfg.generation


def test_paths_resolve_against_project_root(tmp_path):
    assert (find_project_root() / "pyproject.toml").is_file()
    assert resolve_path("results/task1").is_absolute()
    assert resolve_path(tmp_path) == tmp_path  # absolute paths left alone


def test_shipped_task1_config_is_valid():
    """The config we actually ship must load and name a verified checkpoint."""
    cfg = load_task1_config(find_project_root() / "configs" / "task1.yaml")
    assert cfg.model.model_id.startswith("roneneldan/TinyStories")
    assert cfg.model.device == "auto"
    assert cfg.prompts


# --- generation params -----------------------------------------------------

def test_param_defaults_match_documented_values():
    p = GenerationParams()
    assert (p.max_new_tokens, p.temperature, p.top_p) == (200, 1.0, 0.95)
    assert p.do_sample is True


def test_params_are_immutable():
    with pytest.raises(Exception):
        GenerationParams().temperature = 2.0  # frozen dataclass


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"temperature": -1.0}, "temperature"),
        ({"top_p": 1.5}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
        ({"num_return_sequences": 0}, "num_return_sequences"),
    ],
)
def test_invalid_params_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GenerationParams(**kwargs)


def test_temperature_zero_error_suggests_greedy():
    """A common mistake deserves an actionable message."""
    with pytest.raises(ValueError, match="do_sample=False"):
        GenerationParams(temperature=0.0)


def test_params_roundtrip_and_ignore_unknown_keys():
    p = GenerationParams(max_new_tokens=42, top_k=50)
    assert GenerationParams.from_dict(p.to_dict()) == p
    assert GenerationParams.from_dict({"max_new_tokens": 7, "nope": True}).max_new_tokens == 7


# --- devices ---------------------------------------------------------------

def test_cpu_is_always_available_and_auto_never_raises():
    assert "cpu" in available_devices()
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    assert resolve_device("cpu").type == "cpu"
    assert describe_device(torch.device("cpu")) == "cpu"


def test_unavailable_device_falls_back_to_cpu():
    """A config written on a GPU box must still run on a laptop."""
    missing = next(d for d in ("cuda", "mps") if d not in available_devices())
    assert resolve_device(missing).type == "cpu"


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu-v5")


# --- seeding ---------------------------------------------------------------

def test_seed_everything_makes_torch_deterministic():
    assert seed_everything(7) == 7
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


def test_batch_seed_is_deterministic_and_distinct_per_batch():
    assert batch_seed(42, 3) == batch_seed(42, 3)
    assert batch_seed(42, 0) != batch_seed(42, 1)
    assert 0 <= batch_seed(MAX_SEED, 5) <= MAX_SEED, "must wrap, not overflow"


# --- loader (no network) ---------------------------------------------------

def test_parameter_counts_split_and_sum_correctly(tiny_model):
    counts = count_parameters(tiny_model)
    assert counts["total"] == counts["embedding"] + counts["non_embedding"]
    assert counts["embedding"] > 0 and counts["non_embedding"] > 0
    # GPT-Neo ties lm_head to wte; counting it twice would inflate sizes.
    assert counts["total"] == sum(p.numel() for p in tiny_model.parameters())


def test_pad_token_is_assigned_when_missing(tiny_tokenizer):
    tok = prepare_tokenizer_for_generation(tiny_tokenizer)
    assert tok.pad_token is not None and tok.pad_token == tok.eos_token


def test_padding_side_is_forced_left(tiny_tokenizer):
    """Right padding silently corrupts decoder-only batched generation."""
    tiny_tokenizer.padding_side = "right"
    assert prepare_tokenizer_for_generation(tiny_tokenizer).padding_side == "left"


def test_bad_dtype_raises_before_any_download():
    with pytest.raises(ValueError, match="dtype"):
        load_causal_lm("roneneldan/TinyStories-1M", dtype="int4")


def test_nonexistent_checkpoint_raises_checkpoint_load_error():
    with pytest.raises(CheckpointLoadError, match="Could not load model"):
        load_causal_lm("this-org-does-not-exist/nor-does-this-model-xyz")
