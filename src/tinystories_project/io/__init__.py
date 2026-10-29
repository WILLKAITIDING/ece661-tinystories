"""Shared on-disk formats.

`records` defines the JSONL schema that Task 1 writes and that Tasks 2-4, the
Phase 2 architecture comparison, and the Phase 3 Amharic evaluation all read.
Change it carefully: other people's code depends on these field names.
"""

from tinystories_project.io.records import (
    SCHEMA_VERSION,
    GenerationRecord,
    build_records,
    existing_record_ids,
    make_prompt_id,
    make_record_id,
    read_jsonl,
    write_json,
    write_jsonl,
)

__all__ = [
    "SCHEMA_VERSION",
    "GenerationRecord",
    "build_records",
    "existing_record_ids",
    "make_prompt_id",
    "make_record_id",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]
