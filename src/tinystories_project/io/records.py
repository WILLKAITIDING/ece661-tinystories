"""The on-disk record schema shared by every task in the project.

ONE JSONL LINE = ONE COMPLETION. Not one line per prompt, because the proposal
calls for 10 completions per prompt for the Gemini judge, and ROUGE needs each
completion separately. Keeping the atom small means Tasks 2-4 and the Phase 2
architecture table all read the same file format without reshaping anything.

Schema version 1.0
------------------
    schema_version : str   this document's version
    record_id      : str   deterministic; stable across reruns, so an
                           interrupted judge pass can resume by skipping ids
                           it has already scored
    run_id         : str   groups all records written by one invocation
    timestamp_utc  : str   ISO-8601, UTC
    prompt_id      : str   stable id for the prompt (for cross-model joins)
    prompt         : str   the input text
    completion     : str   ONLY the newly generated text
    full_text      : str   prompt + completion, denormalised for readability
    prompt_index   : int   position in the prompt list
    sample_index   : int   which of the N completions for this prompt
    seed           : int?  the seed actually used for this batch
    n_prompt_tokens: int?
    n_new_tokens   : int?
    finish_reason  : str   "eos" | "length" | "unknown"
    params         : obj   the GenerationParams used, verbatim
    model          : obj   the ModelMetadata, verbatim
    extra          : obj   free-form; later tasks add scores here

Stability contract: fields may be ADDED in a minor version. Existing field
names and meanings do not change without bumping the major version, because
Task 3 and Task 4 read files written by Task 1.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from tinystories_project.inference.interface import (
    GeneratedText,
    GenerationParams,
    ModelMetadata,
)

SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_prompt_id(prompt: str) -> str:
    """Stable short id for a prompt, so the same prompt joins across models."""
    return "p_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def make_record_id(
    model_id: str, prompt: str, sample_index: int, params: GenerationParams
) -> str:
    """Deterministic id for one completion slot.

    Depends on the model, prompt, sample index, and settings - but NOT on the
    generated text - so a resumed run can tell which slots are already filled.
    """
    payload = json.dumps(
        {
            "model_id": model_id,
            "prompt": prompt,
            "sample_index": sample_index,
            "params": params.to_dict(),
        },
        sort_keys=True,
    )
    return "r_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class GenerationRecord:
    """One serializable completion. See the module docstring for the schema."""

    record_id: str
    run_id: str
    timestamp_utc: str
    prompt_id: str
    prompt: str
    completion: str
    full_text: str
    prompt_index: int
    sample_index: int
    params: dict[str, Any]
    model: dict[str, Any]
    seed: int | None = None
    n_prompt_tokens: int | None = None
    n_new_tokens: int | None = None
    finish_reason: str = "unknown"
    schema_version: str = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_generated(
        cls,
        generated: GeneratedText,
        metadata: ModelMetadata,
        params: GenerationParams,
        run_id: str,
        timestamp: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "GenerationRecord":
        return cls(
            record_id=make_record_id(
                metadata.model_id, generated.prompt, generated.sample_index, params
            ),
            run_id=run_id,
            timestamp_utc=timestamp or utc_now_iso(),
            prompt_id=make_prompt_id(generated.prompt),
            prompt=generated.prompt,
            completion=generated.completion,
            full_text=generated.full_text,
            prompt_index=generated.prompt_index,
            sample_index=generated.sample_index,
            seed=generated.seed,
            n_prompt_tokens=generated.n_prompt_tokens,
            n_new_tokens=generated.n_new_tokens,
            finish_reason=generated.finish_reason,
            params=params.to_dict(),
            model=metadata.to_dict(),
            extra=extra or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationRecord":
        known = {f for f in cls.__dataclass_fields__}
        unknown = {k: v for k, v in data.items() if k not in known}
        record = cls(**{k: v for k, v in data.items() if k in known})
        if unknown:  # forward compatibility: never drop data silently
            record.extra = {**record.extra, "_unknown_fields": unknown}
        return record


def build_records(
    generated: Sequence[GeneratedText],
    metadata: ModelMetadata,
    params: GenerationParams,
    run_id: str,
    extra: dict[str, Any] | None = None,
) -> list[GenerationRecord]:
    """Turn generator output into serializable records with one timestamp."""
    timestamp = utc_now_iso()
    return [
        GenerationRecord.from_generated(
            g, metadata, params, run_id, timestamp=timestamp, extra=extra
        )
        for g in generated
    ]


def write_jsonl(
    records: Iterable[GenerationRecord | dict[str, Any]],
    path: str | os.PathLike[str],
    append: bool = False,
) -> Path:
    """Write records as JSON Lines. Creates parent directories as needed.

    `ensure_ascii=False` keeps Amharic Ge'ez text readable on disk for Phase 3.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with out.open(mode, encoding="utf-8") as fh:
        for record in records:
            data = record.to_dict() if isinstance(record, GenerationRecord) else record
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return out


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[GenerationRecord]:
    """Stream records back, skipping blank lines. Raises on malformed JSON."""
    src = Path(path)
    with src.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield GenerationRecord.from_dict(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{src}:{line_no} is not valid JSON: {exc}") from exc
            except TypeError as exc:
                # Valid JSON, wrong shape - e.g. a run killed mid-write, or a
                # file from a different schema. Name the line so it is findable.
                raise ValueError(
                    f"{src}:{line_no} is valid JSON but not a GenerationRecord "
                    f"(schema {SCHEMA_VERSION}): {exc}"
                ) from exc


def write_json(payload: Any, path: str | os.PathLike[str]) -> Path:
    """Write a pretty-printed JSON side-car (run manifests, summaries)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    return out


def existing_record_ids(path: str | os.PathLike[str]) -> set[str]:
    """Record ids already present in a file; empty if it does not exist.

    Lets a later Gemini-judge or long generation run resume instead of redoing
    work that was already paid for.
    """
    if not Path(path).exists():
        return set()
    return {record.record_id for record in read_jsonl(path)}
