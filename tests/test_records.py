"""Record schema and JSON/JSONL serialization.

Tasks 2-4 read files written by Task 1, so these guarantees are contractual.
"""

from __future__ import annotations

import json

import pytest

from tinystories_project.inference.interface import (
    GeneratedText,
    GenerationParams,
    ModelMetadata,
)
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


@pytest.fixture
def generated() -> list[GeneratedText]:
    return [
        GeneratedText(
            prompt="Once upon a time, there was a little girl named Lily.",
            completion=" She found a red ball.",
            prompt_index=0,
            sample_index=i,
            seed=42,
            n_prompt_tokens=13,
            n_new_tokens=6,
            finish_reason="eos",
        )
        for i in range(2)
    ]


@pytest.fixture
def records(generated, tiny_metadata):
    return build_records(generated, tiny_metadata, GenerationParams(), run_id="run-1")


def test_build_records_produces_one_per_generation(records, generated):
    assert len(records) == len(generated)
    assert all(r.schema_version == SCHEMA_VERSION for r in records)


def test_record_captures_params_and_model_metadata(records):
    r = records[0]
    assert r.params["temperature"] == 1.0
    assert r.model["model_id"] == "test/tiny-gpt-neo"
    assert r.model["n_parameters_total"] == 1234
    assert r.model["n_parameters_non_embedding"] == 234


def test_full_text_is_prompt_plus_completion(records):
    for r in records:
        assert r.full_text == r.prompt + r.completion


def test_all_records_in_a_run_share_one_timestamp(records):
    assert len({r.timestamp_utc for r in records}) == 1
    assert len({r.run_id for r in records}) == 1


# --- id stability ----------------------------------------------------------

def test_prompt_id_is_stable_and_content_addressed():
    assert make_prompt_id("hello") == make_prompt_id("hello")
    assert make_prompt_id("hello") != make_prompt_id("hello ")


def test_record_id_is_stable_across_calls():
    p = GenerationParams()
    a = make_record_id("m", "prompt", 0, p)
    b = make_record_id("m", "prompt", 0, p)
    assert a == b


def test_record_id_does_not_depend_on_generated_text(tiny_metadata):
    """Resume logic relies on ids being known BEFORE generating."""
    params = GenerationParams()
    one = build_records(
        [GeneratedText(prompt="p", completion="AAA", sample_index=0)],
        tiny_metadata, params, run_id="r1",
    )[0]
    two = build_records(
        [GeneratedText(prompt="p", completion="ZZZ different", sample_index=0)],
        tiny_metadata, params, run_id="r2",
    )[0]
    assert one.record_id == two.record_id


def test_record_id_varies_with_sample_model_and_params():
    p = GenerationParams()
    base = make_record_id("m", "prompt", 0, p)
    assert base != make_record_id("m", "prompt", 1, p)
    assert base != make_record_id("other-model", "prompt", 0, p)
    assert base != make_record_id("m", "prompt", 0, GenerationParams(temperature=0.5))


# --- serialization ---------------------------------------------------------

def test_jsonl_roundtrip(records, tmp_path):
    path = write_jsonl(records, tmp_path / "out.jsonl")
    restored = list(read_jsonl(path))
    assert len(restored) == len(records)
    assert [r.to_dict() for r in restored] == [r.to_dict() for r in records]


def test_jsonl_is_one_json_object_per_line(records, tmp_path):
    path = write_jsonl(records, tmp_path / "out.jsonl")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(records)
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_write_creates_missing_directories(records, tmp_path):
    path = write_jsonl(records, tmp_path / "deep" / "nested" / "out.jsonl")
    assert path.is_file()


def test_append_mode_adds_without_truncating(records, tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl(records, path)
    write_jsonl(records, path, append=True)
    assert len(list(read_jsonl(path))) == 2 * len(records)


def test_non_ascii_survives_roundtrip(tiny_metadata, tmp_path):
    """Phase 3 writes Amharic Ge'ez text; it must stay readable and intact."""
    amharic = " አንድ ልጅ ነበረ።"
    recs = build_records(
        [GeneratedText(prompt="ጊዜ", completion=amharic)],
        tiny_metadata, GenerationParams(), run_id="am",
    )
    path = write_jsonl(recs, tmp_path / "am.jsonl")
    assert amharic in path.read_text(encoding="utf-8"), "must not be \\u-escaped"
    assert list(read_jsonl(path))[0].completion == amharic


def test_blank_lines_are_skipped(records, tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl(records, path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n")
    assert len(list(read_jsonl(path))) == len(records)


def test_invalid_json_raises_with_line_number(records, tmp_path):
    path = tmp_path / "bad.jsonl"
    write_jsonl(records, path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
    with pytest.raises(ValueError, match=f"bad.jsonl:{len(records) + 1}"):
        list(read_jsonl(path))


def test_wrong_shape_line_raises_with_line_number(tmp_path):
    """A run killed mid-write leaves valid JSON of the wrong shape."""
    path = tmp_path / "partial.jsonl"
    path.write_text('{"ok": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="partial.jsonl:1"):
        list(read_jsonl(path))


def test_unknown_fields_are_preserved_not_dropped(tmp_path):
    """Forward compatibility: a newer writer must not lose data on an old reader."""
    path = tmp_path / "future.jsonl"
    payload = {
        "record_id": "r_x", "run_id": "r", "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "prompt_id": "p_x", "prompt": "a", "completion": "b", "full_text": "ab",
        "prompt_index": 0, "sample_index": 0, "params": {}, "model": {},
        "judge_score_from_task4": 9.5,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rec = list(read_jsonl(path))[0]
    assert rec.extra["_unknown_fields"]["judge_score_from_task4"] == 9.5


# --- resume support --------------------------------------------------------

def test_existing_record_ids_supports_resume(records, tmp_path):
    path = write_jsonl(records, tmp_path / "out.jsonl")
    assert existing_record_ids(path) == {r.record_id for r in records}


def test_existing_record_ids_is_empty_for_missing_file(tmp_path):
    assert existing_record_ids(tmp_path / "nothing.jsonl") == set()


def test_write_json_manifest(tmp_path):
    path = write_json({"run_id": "x", "n": 3}, tmp_path / "manifest.json")
    assert json.loads(path.read_text(encoding="utf-8")) == {"run_id": "x", "n": 3}
