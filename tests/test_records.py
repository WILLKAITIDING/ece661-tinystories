"""Record schema and JSONL serialization.

Tasks 2-4 read files written by Task 1, so these guarantees are contractual.
"""

from __future__ import annotations

import json

import pytest

from tinystories_project.inference.interface import GeneratedText, GenerationParams
from tinystories_project.records import (
    SCHEMA_VERSION,
    build_records,
    existing_record_ids,
    make_prompt_id,
    make_record_id,
    read_jsonl,
    write_json,
    write_jsonl,
)


@pytest.fixture
def records(tiny_metadata):
    generated = [
        GeneratedText(
            prompt="Once upon a time, there was a little girl named Lily.",
            completion=" She found a red ball.",
            prompt_index=0, sample_index=i, seed=42,
            n_prompt_tokens=13, n_new_tokens=6, finish_reason="eos",
        )
        for i in range(2)
    ]
    return build_records(generated, tiny_metadata, GenerationParams(), run_id="run-1")


def test_records_capture_params_and_model_metadata(records):
    r = records[0]
    assert r.schema_version == SCHEMA_VERSION
    assert r.params["temperature"] == 1.0
    assert r.model["n_parameters_non_embedding"] == 234


def test_full_text_is_a_property_not_a_stored_field(records):
    """Dropped from storage (27% of file size); still available to readers."""
    r = records[0]
    assert r.full_text == r.prompt + r.completion
    assert "full_text" not in r.to_dict()


def test_all_records_in_a_run_share_one_timestamp_and_run_id(records):
    assert len({r.timestamp_utc for r in records}) == 1
    assert len({r.run_id for r in records}) == 1


# --- id stability ----------------------------------------------------------

def test_prompt_id_is_stable_and_content_addressed():
    assert make_prompt_id("hello") == make_prompt_id("hello")
    assert make_prompt_id("hello") != make_prompt_id("hello ")


def test_record_id_does_not_depend_on_generated_text(tiny_metadata):
    """Resume logic relies on ids being known BEFORE generating."""
    params = GenerationParams()
    def build(completion):
        return build_records(
            [GeneratedText(prompt="p", completion=completion, sample_index=0)],
            tiny_metadata, params, run_id="r",
        )[0].record_id
    assert build("AAA") == build("ZZZ totally different")


def test_record_id_varies_with_every_part_of_the_request():
    """Anything that changes the output must change the id."""
    base = dict(model_id="m", prompt="prompt", sample_index=0,
                params=GenerationParams(), seed=42)
    ref = make_record_id(**base)
    assert ref != make_record_id(**{**base, "sample_index": 1})
    assert ref != make_record_id(**{**base, "model_id": "other"})
    assert ref != make_record_id(**{**base, "seed": 999})
    assert ref != make_record_id(**{**base, "revision": "abc123"})
    assert ref != make_record_id(**{**base, "tokenizer_id": "amharic-sp-8k"})
    assert ref != make_record_id(**{**base, "params": GenerationParams(temperature=0.5)})


def test_identical_requests_share_an_id_so_resume_works():
    base = dict(model_id="m", prompt="p", sample_index=0,
                params=GenerationParams(), seed=42)
    assert make_record_id(**base) == make_record_id(**base)


def test_unseeded_runs_get_unique_ids_per_run():
    """Unseeded output is not reproducible, so ids must not collide."""
    base = dict(model_id="m", prompt="p", sample_index=0,
                params=GenerationParams(), seed=None)
    assert make_record_id(**base, run_id="run-a") != make_record_id(**base, run_id="run-b")


def test_two_runs_differing_only_by_seed_do_not_collide(tiny_metadata):
    """The 1.x bug: a resumed judge pass would skip work it never did."""
    params = GenerationParams()
    def build(seed):
        gen = [GeneratedText(prompt="p", completion="c", sample_index=0, seed=seed)]
        return build_records(gen, tiny_metadata, params, run_id=f"run-{seed}")[0].record_id
    assert build(42) != build(999)


# --- serialization ---------------------------------------------------------

def test_jsonl_roundtrip_one_object_per_line(records, tmp_path):
    path = write_jsonl(records, tmp_path / "deep" / "out.jsonl")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(records)
    assert all(isinstance(json.loads(l), dict) for l in lines)
    assert [r.to_dict() for r in read_jsonl(path)] == [r.to_dict() for r in records]


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


@pytest.mark.parametrize(
    "line, name", [("not json at all", "invalid"), ('{"ok": 1}', "wrong-shape")]
)
def test_malformed_lines_raise_with_file_and_line_number(tmp_path, line, name):
    """A run killed mid-write must fail loudly, pointing at the bad line."""
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        list(read_jsonl(path))


def test_unknown_fields_are_preserved_not_dropped(records, tmp_path):
    """Forward compatibility: a newer writer must not lose data on an old reader."""
    path = tmp_path / "future.jsonl"
    payload = {**records[0].to_dict(), "judge_score_from_task4": 9.5}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert list(read_jsonl(path))[0].extra["_unknown_fields"]["judge_score_from_task4"] == 9.5


def test_legacy_full_text_field_is_accepted_and_ignored(records, tmp_path):
    """Schema 1.0 files carried full_text; reading them must not break."""
    path = tmp_path / "legacy.jsonl"
    payload = {**records[0].to_dict(), "full_text": "prompt + completion"}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rec = list(read_jsonl(path))[0]
    assert "_unknown_fields" not in rec.extra
    assert rec.full_text == rec.prompt + rec.completion


# --- resume support --------------------------------------------------------

def test_existing_record_ids_supports_resume(records, tmp_path):
    path = write_jsonl(records, tmp_path / "out.jsonl")
    assert existing_record_ids(path) == {r.record_id for r in records}
    assert existing_record_ids(tmp_path / "nothing.jsonl") == set()


def test_write_json_manifest(tmp_path):
    path = write_json({"run_id": "x", "n": 3}, tmp_path / "manifest.json")
    assert json.loads(path.read_text(encoding="utf-8")) == {"run_id": "x", "n": 3}
