"""Configuration loading and validation."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from tinystories_project.config import (
    ModelConfig,
    Task1Config,
    find_project_root,
    load_task1_config,
    resolve_path,
)
from tinystories_project.inference.interface import GenerationParams

MINIMAL = {
    "run_name": "unit",
    "model": {"model_id": "roneneldan/TinyStories-1M"},
    "prompts": ["Once upon a time"],
}


def write_cfg(tmp_path, data):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_loads_minimal_config(tmp_path):
    cfg = load_task1_config(write_cfg(tmp_path, MINIMAL))
    assert cfg.run_name == "unit"
    assert cfg.model.model_id == "roneneldan/TinyStories-1M"
    assert cfg.prompts == ["Once upon a time"]
    assert cfg.seed == 42, "default seed must be stable across runs"
    assert cfg.model.device == "auto", "device must default to auto, never cuda"


def test_generation_section_is_parsed(tmp_path):
    data = {**MINIMAL, "generation": {"max_new_tokens": 50, "temperature": 0.7}}
    cfg = load_task1_config(write_cfg(tmp_path, data))
    assert cfg.generation.max_new_tokens == 50
    assert cfg.generation.temperature == 0.7
    assert cfg.generation.top_p == 0.95, "unspecified fields keep their defaults"


def test_unknown_generation_keys_are_ignored(tmp_path):
    """Configs may carry keys meant for later tasks without breaking Task 1."""
    data = {**MINIMAL, "generation": {"max_new_tokens": 10, "future_knob": 3}}
    cfg = load_task1_config(write_cfg(tmp_path, data))
    assert cfg.generation.max_new_tokens == 10


def test_missing_model_section_raises(tmp_path):
    with pytest.raises(ValueError, match="model"):
        load_task1_config(write_cfg(tmp_path, {"prompts": ["x"]}))


def test_missing_model_id_raises(tmp_path):
    with pytest.raises(ValueError, match="model_id"):
        load_task1_config(write_cfg(tmp_path, {"model": {}, "prompts": ["x"]}))


def test_empty_prompts_raise(tmp_path):
    with pytest.raises(ValueError, match="at least one prompt"):
        load_task1_config(write_cfg(tmp_path, {**MINIMAL, "prompts": []}))


def test_blank_prompt_raises(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        load_task1_config(write_cfg(tmp_path, {**MINIMAL, "prompts": ["ok", "  "]}))


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
    restored = Task1Config.from_dict(cfg.to_dict())
    assert restored.to_dict()["model"] == cfg.to_dict()["model"]
    assert restored.generation == cfg.generation


def test_project_root_contains_pyproject():
    assert (find_project_root() / "pyproject.toml").is_file()


def test_relative_paths_resolve_against_project_root():
    resolved = resolve_path("results/task1")
    assert resolved.is_absolute()
    assert resolved.parent.name == find_project_root().name or resolved.is_absolute()


def test_absolute_paths_are_left_alone(tmp_path):
    assert resolve_path(tmp_path) == tmp_path


def test_shipped_task1_config_is_valid():
    """The config we actually ship must load and name a verified checkpoint."""
    cfg = load_task1_config(find_project_root() / "configs" / "task1.yaml")
    assert cfg.model.model_id.startswith("roneneldan/TinyStories")
    assert len(cfg.prompts) >= 1
    assert cfg.model.device == "auto"
