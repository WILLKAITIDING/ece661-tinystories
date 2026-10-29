"""Evaluation prompt sets.

A prompt set is a JSON file holding prompts in one shared shape, so the same
loader serves the paper's Figure 9-11 prompts, our 20 original prompts, and
later Luel's translated Amharic prompts. Nothing here is English-specific or
TinyStories-specific: `category` is the only controlled vocabulary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# The three capability types defined in Section 4.2 of the TinyStories paper.
CATEGORIES = ("factual", "reasoning", "context")
ORIGINS = ("paper", "team32")


@dataclass(frozen=True)
class EvalPrompt:
    """One prompt, plus the criterion a human will grade it against."""

    id: str
    text: str
    category: str
    origin: str
    capability: str = ""
    expected: str = ""
    figure: int | None = None
    source_ref: str | None = None
    criterion_author: str = "team32"
    # Lower-case substrings that a conservative automatic screen accepts. These
    # are a mechanical rendering of `expected`, written before any model was
    # run. Empty means no automatic check is possible and a human must judge.
    auto_accept: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("prompt id must not be empty")
        if not self.text.strip():
            raise ValueError(f"{self.id}: prompt text must not be empty")
        if self.category not in CATEGORIES:
            raise ValueError(
                f"{self.id}: category {self.category!r} not one of {CATEGORIES}"
            )
        if self.origin not in ORIGINS:
            raise ValueError(f"{self.id}: origin {self.origin!r} not one of {ORIGINS}")


@dataclass
class PromptSet:
    """A named collection of prompts with its provenance."""

    name: str
    prompts: list[EvalPrompt]
    description: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    path: str | None = None

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError(f"{self.name}: prompt set is empty")
        seen: set[str] = set()
        for p in self.prompts:
            if p.id in seen:
                raise ValueError(f"{self.name}: duplicate prompt id {p.id!r}")
            seen.add(p.id)
        # Identical text would produce identical record_ids, which would break
        # resume and silently merge two prompts into one.
        texts: dict[str, str] = {}
        for p in self.prompts:
            if p.text in texts:
                raise ValueError(
                    f"{self.name}: prompts {texts[p.text]!r} and {p.id!r} have "
                    "identical text"
                )
            texts[p.text] = p.id

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> Iterator[EvalPrompt]:
        return iter(self.prompts)

    def __getitem__(self, prompt_id: str) -> EvalPrompt:
        for p in self.prompts:
            if p.id == prompt_id:
                return p
        raise KeyError(prompt_id)

    @property
    def texts(self) -> list[str]:
        return [p.text for p in self.prompts]

    def by_category(self, category: str) -> list[EvalPrompt]:
        return [p for p in self.prompts if p.category == category]

    def counts(self) -> dict[str, int]:
        return {c: len(self.by_category(c)) for c in CATEGORIES}


def load_prompt_set(path: str | Path) -> PromptSet:
    """Load and validate a prompt set from JSON."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Prompt set not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "prompts" not in data:
        raise ValueError(f"{p}: file has no 'prompts' key")

    known = set(EvalPrompt.__dataclass_fields__)
    prompts = []
    for entry in data["prompts"]:
        fields = {k: v for k, v in entry.items() if k in known}
        fields["auto_accept"] = tuple(fields.get("auto_accept") or ())
        prompts.append(EvalPrompt(**fields))
    return PromptSet(
        name=data.get("name", p.stem),
        prompts=prompts,
        description=data.get("description", ""),
        source=data.get("source", {}),
        notes=data.get("notes", []),
        path=str(p),
    )


def load_prompt_sets(paths: list[str | Path]) -> list[PromptSet]:
    return [load_prompt_set(p) for p in paths]
