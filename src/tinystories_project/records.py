"""The on-disk record schema shared by every task in the project.

ONE JSONL LINE = ONE COMPLETION. Not one line per prompt: the proposal calls
for 10 completions per prompt for the Gemini judge, and ROUGE scores each
completion separately. A small atom means Tasks 2-4 and the Phase 2
architecture table all read the same file without reshaping it.

Schema version 2.0
------------------
    schema_version : str   this document's version
    record_id      : str   deterministic, derived from the full generation
                           request (model, revision, tokenizer, prompt, sample
                           index, seed, params) but NOT from the generated
                           text, so an interrupted run resumes by skipping the
                           ids a file already has. Different seeds give
                           different ids; see make_record_id.
    run_id         : str   groups records written by one invocation
    timestamp_utc  : str   ISO-8601, UTC
    prompt_id      : str   stable id for the prompt, for cross-model joins
    prompt         : str   the input text
    completion     : str   ONLY the newly generated text
    prompt_index   : int   position in the prompt list
    sample_index   : int   which of the N completions for this prompt
    seed           : int?  the seed used for this batch
    n_prompt_tokens: int?
    n_new_tokens   : int?
    finish_reason  : str   "eos" | "length" | "unknown"
    params         : obj   the GenerationParams used, verbatim
    model          : obj   the ModelMetadata, verbatim
    extra          : obj   free-form; later tasks add scores here

`full_text` is NOT stored - it is exactly `prompt + completion` and was 27% of
every file. `GenerationRecord.full_text` still returns it as a property, so
reading code is unchanged.

Stability contract: fields may be ADDED in a minor version. Existing names and
meanings do not change without a major bump, because Tasks 3 and 4 read files
written by Task 1. Later tasks attach their outputs under `extra` rather than
restructuring the record.

Version 2.0 made `record_id` seed-aware: under 1.x two runs differing only by
seed shared an id, which would make a resumed judge pass skip work it never
did. Do not mix 1.x and 2.x files in one directory.
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

SCHEMA_VERSION = "2.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_prompt_id(prompt: str) -> str:
    """Stable id for a prompt, so the same prompt joins across models."""
    return "p_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def make_record_id(
    model_id: str,
    prompt: str,
    sample_index: int,
    params: GenerationParams,
    seed: int | None = None,
    revision: str | None = None,
    tokenizer_id: str | None = None,
    run_id: str | None = None,
) -> str:
    """Deterministic id for one generation request.

    The id covers everything that decides the output - model, revision,
    tokenizer, prompt, which sample, the seed, and the sampling params - but
    NOT the generated text, so it can be computed BEFORE generating. That is
    what makes resume work: skip the ids a file already has.

    Two runs that differ only by seed get different ids (they are different
    work). Two runs with an identical request get the same id (they are the
    same work, and would produce the same text).

    Unseeded runs are not reproducible, so there is nothing to resume; `run_id`
    is mixed in instead to keep their ids unique.
    """
    request = {
        "model_id": model_id,
        "revision": revision,
        "tokenizer_id": tokenizer_id,
        "prompt": prompt,
        "sample_index": sample_index,
        "seed": seed,
        "params": params.to_dict(),
    }
    if seed is None:
        request["run_id"] = run_id
    payload = json.dumps(request, sort_keys=True)
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

    @property
    def full_text(self) -> str:
        """Prompt and continuation joined, as a reader sees the story."""
        return self.prompt + self.completion

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
                model_id=metadata.model_id,
                prompt=generated.prompt,
                sample_index=generated.sample_index,
                params=params,
                seed=generated.seed,
                revision=metadata.revision,
                tokenizer_id=metadata.tokenizer_id,
                run_id=run_id,
            ),
            run_id=run_id,
            timestamp_utc=timestamp or utc_now_iso(),
            prompt_id=make_prompt_id(generated.prompt),
            prompt=generated.prompt,
            completion=generated.completion,
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
        data = {k: v for k, v in data.items() if k != "full_text"}  # derivable
        known = set(cls.__dataclass_fields__)
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
    """Turn generator output into records sharing one timestamp."""
    timestamp = utc_now_iso()
    return [
        GenerationRecord.from_generated(g, metadata, params, run_id, timestamp, extra)
        for g in generated
    ]


def write_jsonl(
    records: Iterable[GenerationRecord | dict[str, Any]],
    path: str | os.PathLike[str],
    append: bool = False,
) -> Path:
    """Write records as JSON Lines, creating parent directories as needed.

    `ensure_ascii=False` keeps Amharic Ge'ez text readable for Phase 3.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if append else "w", encoding="utf-8") as fh:
        for record in records:
            data = record.to_dict() if isinstance(record, GenerationRecord) else record
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
    return out


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[GenerationRecord]:
    """Stream records back, skipping blank lines. Raises on malformed input."""
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
                # Valid JSON, wrong shape - e.g. a run killed mid-write.
                raise ValueError(
                    f"{src}:{line_no} is valid JSON but not a GenerationRecord "
                    f"(schema {SCHEMA_VERSION}): {exc}"
                ) from exc


def write_json(payload: Any, path: str | os.PathLike[str]) -> Path:
    """Write a pretty-printed JSON sidecar (run manifests, summaries)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return out


def existing_record_ids(path: str | os.PathLike[str]) -> set[str]:
    """Record ids already in a file; empty if it does not exist.

    Lets a long generation or judge run resume instead of re-spending budget.
    """
    if not Path(path).exists():
        return set()
    return {record.record_id for record in read_jsonl(path)}
