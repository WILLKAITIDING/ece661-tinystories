"""Human-reviewed labels for generated completions.

Task 2 labels every protocol-A completion with the same three-way scheme the
TinyStories paper uses for Figures 9-11 (green / yellow / red). Rachel and Luel
annotate their own models with the same file format, so everything here is
model- and language-agnostic: it joins to generation records by `record_id`.

PROVENANCE MATTERS. The Task 2 labels were drafted by an AI assistant and then
reviewed by a human, who overrode some and accepted the rest. That is NOT the
same as independent human annotation, and `AnnotationSet.provenance()` reports
the split so a report can state it accurately.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# Mirrors the paper's colour coding; `ambiguous` is for completions a human
# genuinely cannot judge, and is excluded from strict accuracy denominators.
LABELS = ("correct", "partial", "incorrect", "ambiguous")
SCORED_LABELS = ("correct", "partial", "incorrect")

# Rows whose note matches this were changed by the human reviewer.
_OVERRIDE_RE = re.compile(r"\boverrid", re.I)


@dataclass(frozen=True)
class Annotation:
    """One label for one generation record."""

    record_id: str
    label: str
    annotator: str = ""
    notes: str = ""
    model: str = ""
    prompt_id: str = ""
    category: str = ""
    origin: str = ""

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(
                f"{self.record_id}: label {self.label!r} not one of {LABELS}"
            )
        if not self.record_id:
            raise ValueError("annotation is missing a record_id")

    @property
    def is_override(self) -> bool:
        """True if a human changed an AI draft label for this row."""
        return bool(_OVERRIDE_RE.search(self.notes))


@dataclass
class AnnotationSet:
    """All annotations for one evaluation run."""

    annotations: list[Annotation]
    path: str | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for a in self.annotations:
            if a.record_id in seen:
                raise ValueError(f"duplicate record_id {a.record_id!r}")
            seen.add(a.record_id)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self) -> Iterator[Annotation]:
        return iter(self.annotations)

    @property
    def by_record(self) -> dict[str, Annotation]:
        return {a.record_id: a for a in self.annotations}

    def counts(self) -> dict[str, int]:
        c = Counter(a.label for a in self.annotations)
        return {label: c.get(label, 0) for label in LABELS if c.get(label, 0)}

    def provenance(self) -> dict[str, Any]:
        """Who produced these labels, and how many a human changed."""
        overrides = [a for a in self.annotations if a.is_override]
        return {
            "n_total": len(self.annotations),
            "n_human_overrides": len(overrides),
            "n_accepted_ai_drafts": len(self.annotations) - len(overrides),
            "annotators": sorted({a.annotator for a in self.annotations if a.annotator}),
            "override_record_ids": sorted(a.record_id for a in overrides),
        }

    def validate_against(self, record_ids: Iterable[str]) -> None:
        """Fail loudly if the labels do not line up with the generation run."""
        expected, got = set(record_ids), set(self.by_record)
        if expected != got:
            missing, extra = sorted(expected - got)[:5], sorted(got - expected)[:5]
            raise ValueError(
                f"annotations do not match the generation records: "
                f"{len(expected - got)} unlabelled (e.g. {missing}), "
                f"{len(got - expected)} unknown (e.g. {extra})"
            )


def load_annotations(path: str | Path) -> AnnotationSet:
    """Load annotations from the CSV exported from the annotation template."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Annotation file not found: {p}")
    known = set(Annotation.__dataclass_fields__)
    rows = []
    with p.open(newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            label = (row.get("label") or "").strip().lower()
            if not label:
                raise ValueError(f"{p}:{i} has an empty label; annotation is incomplete")
            fields = {k: (v or "").strip() for k, v in row.items() if k in known}
            fields["label"] = label
            rows.append(Annotation(**fields))
    return AnnotationSet(annotations=rows, path=str(p))


def accuracy(labels: Sequence[str]) -> dict[str, Any]:
    """Strict accuracy and partial rate over a group of labels.

    `strict` counts only `correct`. `lenient` gives half credit for `partial`,
    which is a convenience for ranking, not a claim about the paper's scheme.
    `ambiguous` rows are excluded from the denominator.
    """
    scored = [l for l in labels if l in SCORED_LABELS]
    n = len(scored)
    if n == 0:
        return {"n": 0, "correct": 0, "partial": 0, "incorrect": 0,
                "strict": None, "lenient": None, "partial_rate": None}
    c = Counter(scored)
    return {
        "n": n,
        "correct": c["correct"],
        "partial": c["partial"],
        "incorrect": c["incorrect"],
        "strict": c["correct"] / n,
        "lenient": (c["correct"] + 0.5 * c["partial"]) / n,
        "partial_rate": c["partial"] / n,
    }
