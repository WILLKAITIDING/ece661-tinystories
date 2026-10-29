"""YAML experiment configuration.

Experiments are described by files in `configs/`, never by numbers typed into a
notebook. A run records the config it used, so "which settings produced this
table?" always has an answer.

Paths in a config are resolved relative to the PROJECT ROOT (the directory
holding pyproject.toml), never to the caller's working directory, so the same
config works from a script, a notebook, or a cluster job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tinystories_project.inference.interface import GenerationParams

PROJECT_MARKER = "pyproject.toml"


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk upward from `start` until a directory contains pyproject.toml."""
    here = Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / PROJECT_MARKER).is_file():
            return candidate
    # Installed as a site-package with no repo around it: fall back to CWD.
    return Path.cwd()


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Make a possibly-relative config path absolute against the project root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (root or find_project_root()) / p


@dataclass
class ModelConfig:
    """Which checkpoint to load and how."""

    model_id: str
    tokenizer_id: str | None = None
    revision: str | None = None
    device: str = "auto"
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OutputConfig:
    """Where generated records land."""

    dir: str = "results/task1"
    filename: str | None = None  # auto-named from run_name + timestamp if None
    write_manifest: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Task1Config:
    """A complete, replayable Task 1 run."""

    run_name: str
    model: ModelConfig
    generation: GenerationParams = field(default_factory=GenerationParams)
    prompts: list[str] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    batch_size: int = 8
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("Config must define at least one prompt.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if any(not isinstance(p, str) or not p.strip() for p in self.prompts):
            raise ValueError("Every prompt must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "model": self.model.to_dict(),
            "generation": self.generation.to_dict(),
            "output": self.output.to_dict(),
            "prompts": list(self.prompts),
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: str | None = None) -> "Task1Config":
        if "model" not in data:
            raise ValueError("Config is missing the required 'model' section.")
        model_data = data["model"]
        if "model_id" not in model_data:
            raise ValueError("Config section 'model' is missing 'model_id'.")

        return cls(
            run_name=data.get("run_name", "task1"),
            model=ModelConfig(**model_data),
            generation=GenerationParams.from_dict(data.get("generation", {}) or {}),
            prompts=list(data.get("prompts", []) or []),
            output=OutputConfig(**(data.get("output", {}) or {})),
            seed=int(data.get("seed", 42)),
            batch_size=int(data.get("batch_size", 8)),
            source_path=source_path,
        )


def load_task1_config(path: str | Path) -> Task1Config:
    """Load and validate a Task 1 YAML config."""
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} must contain a YAML mapping at the top level.")
    return Task1Config.from_dict(data, source_path=str(cfg_path))
