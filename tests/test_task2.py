"""Task 2: prompt sets, the qualitative runner, and the paper extraction.

The runner is exercised with a FakeGenerator that implements the shared
`TextGenerator` protocol and imports nothing from Hugging Face - the same path
Rachel's minGRU and Luel's Amharic models will take.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinystories_project.config import find_project_root, load_task2_config
from tinystories_project.evaluation import (
    CATEGORIES,
    EvalPrompt,
    PromptSet,
    answer_span,
    load_prompt_set,
    run_prompt_set,
    summarise,
)
from tinystories_project.evaluation.qualitative import PROTOCOLS, run_fingerprint
from tinystories_project.inference.interface import (
    GeneratedText,
    GenerationParams,
    ModelMetadata,
    TextGenerator,
)
from tinystories_project.records import read_jsonl, write_jsonl

ROOT = find_project_root()
PAPER_SET = ROOT / "data" / "prompts" / "paper_figures_9_11.json"
CUSTOM_SET = ROOT / "data" / "prompts" / "custom_20.json"
REFERENCE = ROOT / "data" / "reference" / "paper_figure_completions.json"


# --------------------------------------------------------------------------
# A model that is not a Hugging Face model, to prove the interface is honoured
# --------------------------------------------------------------------------
class FakeGenerator:
    """Deterministic stand-in satisfying `TextGenerator`. No torch, no HF."""

    def __init__(self, model_id: str = "fake/tiny-1M", n_params: int = 1234):
        self._meta = ModelMetadata(
            model_id=model_id, architecture="FakeLM", source="custom",
            tokenizer_id="fake-tokenizer", n_parameters_total=n_params,
            n_parameters_non_embedding=n_params // 2, n_layers=2, hidden_size=8,
        )
        self.calls: list[tuple[int, GenerationParams, int | None]] = []

    @property
    def metadata(self) -> ModelMetadata:
        return self._meta

    def generate(self, prompts, params, seed=None):
        self.calls.append((len(prompts), params, seed))
        out = []
        for i, prompt in enumerate(prompts):
            for s in range(params.num_return_sequences):
                out.append(GeneratedText(
                    prompt=prompt,
                    completion=f" continuation {s} for prompt {i}. And more text.",
                    prompt_index=i, sample_index=s, seed=seed,
                    n_prompt_tokens=len(prompt.split()), n_new_tokens=9,
                    finish_reason="eos",
                ))
        return out


@pytest.fixture
def fake():
    return FakeGenerator()


@pytest.fixture
def small_set():
    return PromptSet(name="unit", prompts=[
        EvalPrompt(id="a1", text="Alice went", category="factual", origin="team32",
                   expected="to bed"),
        EvalPrompt(id="b1", text="Because it rained, Ben", category="reasoning",
                   origin="team32", expected="got wet"),
        EvalPrompt(id="c1", text="Max had two dogs and then", category="context",
                   origin="team32", expected="two dogs"),
    ])


# --- prompt sets ----------------------------------------------------------

def test_fake_generator_satisfies_the_shared_protocol(fake):
    assert isinstance(fake, TextGenerator)


@pytest.mark.parametrize("path", [PAPER_SET, CUSTOM_SET])
def test_shipped_prompt_sets_load(path):
    ps = load_prompt_set(path)
    assert len(ps) > 0
    assert all(p.text.strip() for p in ps)
    assert all(p.category in CATEGORIES for p in ps)


def test_paper_set_has_the_expected_figure_breakdown():
    ps = load_prompt_set(PAPER_SET)
    assert len(ps) == 24
    assert ps.counts() == {"factual": 9, "reasoning": 9, "context": 6}
    assert {p.figure for p in ps} == {9, 10, 11}
    assert all(p.origin == "paper" for p in ps)


def test_custom_set_matches_the_proposal_commitment():
    """The proposal commits to 20 new prompts: 7 factual, 7 reasoning, 6 context."""
    ps = load_prompt_set(CUSTOM_SET)
    assert len(ps) == 20
    assert ps.counts() == {"factual": 7, "reasoning": 7, "context": 6}
    assert all(p.origin == "team32" for p in ps)
    assert all(p.expected and p.capability for p in ps)


def test_prompt_ids_are_unique_within_and_across_shipped_sets():
    ids = [p.id for path in (PAPER_SET, CUSTOM_SET) for p in load_prompt_set(path)]
    assert len(ids) == len(set(ids)) == 44


def test_duplicate_prompt_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate prompt id"):
        PromptSet(name="x", prompts=[
            EvalPrompt(id="dup", text="one", category="factual", origin="team32"),
            EvalPrompt(id="dup", text="two", category="factual", origin="team32"),
        ])


def test_duplicate_prompt_text_is_rejected():
    """Identical text would collide on record_id and break resume."""
    with pytest.raises(ValueError, match="identical text"):
        PromptSet(name="x", prompts=[
            EvalPrompt(id="a", text="same", category="factual", origin="team32"),
            EvalPrompt(id="b", text="same", category="reasoning", origin="team32"),
        ])


@pytest.mark.parametrize("kwargs, match", [
    ({"category": "nonsense"}, "category"),
    ({"origin": "somewhere"}, "origin"),
    ({"text": "   "}, "empty"),
])
def test_invalid_prompts_are_rejected(kwargs, match):
    base = dict(id="x", text="hello", category="factual", origin="team32")
    with pytest.raises(ValueError, match=match):
        EvalPrompt(**{**base, **kwargs})


def test_missing_prompt_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompt_set(tmp_path / "nope.json")


def test_by_category_and_lookup(small_set):
    assert [p.id for p in small_set.by_category("factual")] == ["a1"]
    assert small_set["b1"].category == "reasoning"
    with pytest.raises(KeyError):
        small_set["missing"]


# --- answer span ----------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    (" to bed.", "to bed."),
    ("but no one answered. He knocked again.", "but no one answered."),
    ('said, "Yes!" Then she left.', 'said, "Yes!"'),
    ("no terminal punctuation", "no terminal punctuation"),
    ("", ""),
])
def test_answer_span_takes_the_first_sentence(text, expected):
    assert answer_span(text) == expected


def test_answer_span_skips_punctuation_that_closes_the_prompt():
    """Several Figure 9-11 prompts end without punctuation, so the model's
    first token is the full stop closing the PROMPT. That is not the answer."""
    assert answer_span('. His mother said, "It\'s called French."') == (
        '. His mother said, "It\'s called French."')
    assert answer_span("...") == "..."


# --- the runner -----------------------------------------------------------

def test_every_generation_maps_back_to_its_prompt(fake, small_set):
    records = run_prompt_set(fake, small_set, protocol="A_greedy", run_id="r")
    assert len(records) == 3
    for rec, prompt in zip(records, small_set.prompts):
        assert rec.extra["prompt_id"] == prompt.id
        assert rec.extra["category"] == prompt.category
        assert rec.prompt == prompt.text


def test_prompt_and_continuation_stay_separate(fake, small_set):
    for rec in run_prompt_set(fake, small_set, protocol="A_greedy", run_id="r"):
        assert not rec.completion.startswith(rec.prompt)
        assert rec.full_text == rec.prompt + rec.completion


def test_multiple_completions_per_prompt(fake, small_set):
    records = run_prompt_set(fake, small_set, protocol="B_sampling", run_id="r")
    assert len(records) == 3 * PROTOCOLS["B_sampling"].num_return_sequences
    for prompt in small_set.prompts:
        mine = [r for r in records if r.extra["prompt_id"] == prompt.id]
        assert sorted(r.sample_index for r in mine) == [0, 1, 2]


def test_record_ids_are_unique_across_samples_and_protocols(fake, small_set):
    a = run_prompt_set(fake, small_set, protocol="A_greedy", run_id="r")
    b = run_prompt_set(fake, small_set, protocol="B_sampling", run_id="r")
    ids = [r.record_id for r in a + b]
    assert len(ids) == len(set(ids))


def test_evaluation_metadata_is_attached(fake, small_set):
    rec = run_prompt_set(fake, small_set, protocol="A_greedy", run_id="r",
                         extra={"figure_column": "1M/8L"})[0]
    for key in ("prompt_id", "category", "figure", "origin", "protocol",
                "prompt_set", "answer_span", "figure_column"):
        assert key in rec.extra, key
    assert rec.extra["protocol"] == "A_greedy"
    assert rec.extra["answer_span"] == "continuation 0 for prompt 0."


def test_model_metadata_is_recorded(fake, small_set):
    rec = run_prompt_set(fake, small_set, protocol="A_greedy", run_id="r")[0]
    assert rec.model["model_id"] == "fake/tiny-1M"
    assert rec.model["n_parameters_total"] == 1234


def test_seed_is_passed_through_and_recorded(fake, small_set):
    run_prompt_set(fake, small_set, protocol="A_greedy", run_id="r", seed=7)
    assert fake.calls[-1][2] == 7


def test_same_seed_gives_identical_records(fake, small_set):
    a = run_prompt_set(fake, small_set, protocol="B_sampling", run_id="r1", seed=99)
    b = run_prompt_set(fake, small_set, protocol="B_sampling", run_id="r2", seed=99)
    assert [r.completion for r in a] == [r.completion for r in b]
    assert [r.record_id for r in a] == [r.record_id for r in b]


def test_unknown_protocol_is_rejected(fake, small_set):
    with pytest.raises(ValueError, match="Unknown protocol"):
        run_prompt_set(fake, small_set, protocol="C_made_up", run_id="r")


def test_mismatched_generator_output_is_caught(small_set):
    """A broken custom generator must fail loudly, not mislabel results."""

    class Broken(FakeGenerator):
        def generate(self, prompts, params, seed=None):
            out = super().generate(prompts, params, seed)
            out[0].prompt = "a completely different prompt"
            return out

    with pytest.raises(ValueError, match="does not match prompt"):
        run_prompt_set(Broken(), small_set, protocol="A_greedy", run_id="r")


def test_records_round_trip_through_jsonl(fake, small_set, tmp_path):
    records = run_prompt_set(fake, small_set, protocol="A_greedy", run_id="r")
    path = write_jsonl(records, tmp_path / "out.jsonl")
    restored = list(read_jsonl(path))
    assert len(restored) == len(records)
    assert restored[0].extra["prompt_id"] == records[0].extra["prompt_id"]
    assert restored[0].extra["category"] in CATEGORIES


def test_summarise_counts_without_judging(fake, small_set):
    s = summarise(run_prompt_set(fake, small_set, protocol="A_greedy", run_id="r"))
    assert s["n_records"] == 3
    assert s["by_category"] == {"factual": 1, "reasoning": 1, "context": 1}
    assert s["n_empty_completions"] == 0


# --- resume fingerprint ---------------------------------------------------

def test_fingerprint_distinguishes_every_run_dimension(fake, small_set):
    base = dict(protocol="A_greedy", seed=42, batch_size=1, prompt_set=small_set)
    params = PROTOCOLS["A_greedy"]
    ref = run_fingerprint(fake.metadata, params, **base)

    assert ref == run_fingerprint(fake.metadata, params, **base)   # stable
    assert ref != run_fingerprint(fake.metadata, params, **{**base, "seed": 7})
    assert ref != run_fingerprint(fake.metadata, params, **{**base, "batch_size": 8})
    assert ref != run_fingerprint(fake.metadata, params, **{**base, "protocol": "B_sampling"})
    assert ref != run_fingerprint(fake.metadata, PROTOCOLS["B_sampling"], **base)
    assert ref != run_fingerprint(FakeGenerator("other/model").metadata, params, **base)
    other = PromptSet(name="unit", prompts=small_set.prompts[:2])
    assert ref != run_fingerprint(fake.metadata, params, **{**base, "prompt_set": other})


def test_fingerprint_expects_one_record_per_prompt_per_sample(fake, small_set):
    fp = run_fingerprint(fake.metadata, PROTOCOLS["B_sampling"], protocol="B_sampling",
                         seed=42, batch_size=1, prompt_set=small_set)
    assert fp["n_expected"] == 3 * 3


# --- config ---------------------------------------------------------------

def test_shipped_task2_config_is_valid():
    cfg = load_task2_config(ROOT / "configs" / "task2.yaml")
    assert cfg.models and all("model_id" in m for m in cfg.models)
    assert set(cfg.protocols) == {"A_greedy", "B_sampling"}
    assert cfg.protocols["A_greedy"]["do_sample"] is False, "paper used zero temperature"
    assert cfg.batch_size == 1
    assert len(cfg.prompt_sets) == 2


def test_task2_config_rejects_duplicate_models(tmp_path):
    import yaml
    from tinystories_project.config import load_task2_config
    bad = {"models": [{"model_id": "a"}, {"model_id": "a"}],
           "prompt_sets": ["x"], "protocols": {"A_greedy": {}}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate model_id"):
        load_task2_config(p)


# --- the paper extraction -------------------------------------------------

def test_reference_completions_cover_every_paper_prompt():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    prompts = load_prompt_set(PAPER_SET)
    assert {r["id"] for r in ref["rows"]} == {p.id for p in prompts}
    assert all(len(r["completions"]) == 8 for r in ref["rows"])
    assert all(len(r["paper_verdicts"]) == 8 for r in ref["rows"])


def test_reference_is_labelled_as_the_papers_own_output():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert "NOT generated by Team 32" in ref["description"]
    assert ref["source"]["arxiv_id"] == "2305.07759"


def test_extracted_verdicts_match_two_explicit_claims_in_the_paper():
    """Section 4.2 states these outright; they validate the colour extraction.

    p.9  "the model with 1M parameters and 8 layers fails to answer any
          factual prompt correctly"
    p.10 "the model that has only 1 layer does not get any consistency
          prompt right"
    """
    rows = json.loads(REFERENCE.read_text(encoding="utf-8"))["rows"]
    factual = [r for r in rows if r["figure"] == 9]
    context = [r for r in rows if r["figure"] == 11]
    assert sum(r["paper_verdicts"]["1M/8L"] == "correct" for r in factual) == 0
    assert sum(r["paper_verdicts"]["21M/1L"] == "correct" for r in context) == 0


def test_paper_prompt_text_is_not_silently_rewritten():
    """We normalise quotes; everything else must survive verbatim."""
    for p in load_prompt_set(PAPER_SET):
        raw = p.source_ref
        assert raw and "arXiv:2305.07759v2" in raw
    data = json.loads(PAPER_SET.read_text(encoding="utf-8"))
    entry = next(e for e in data["prompts"] if e["id"] == "fig10_02")
    # Capital "Banana" is in the paper; it must not be "corrected".
    assert "her Banana so" in entry["text"]


# --- final annotations ----------------------------------------------------

ANNOTATIONS = ROOT / "data" / "annotations" / "task2_final.csv"


def test_final_annotations_load_with_the_expected_counts():
    from tinystories_project.evaluation.annotations import load_annotations
    ann = load_annotations(ANNOTATIONS)
    assert len(ann) == 176
    assert ann.counts() == {"correct": 82, "partial": 25, "incorrect": 69}


def test_final_annotation_record_ids_are_unique():
    from tinystories_project.evaluation.annotations import load_annotations
    ann = load_annotations(ANNOTATIONS)
    assert len({a.record_id for a in ann}) == 176


def test_annotation_provenance_is_recoverable():
    """The report must be able to state how many labels a human changed."""
    from tinystories_project.evaluation.annotations import load_annotations
    prov = load_annotations(ANNOTATIONS).provenance()
    assert prov["n_total"] == 176
    assert prov["n_human_overrides"] == 9
    assert prov["n_accepted_ai_drafts"] == 167
    assert prov["annotators"] == ["Will"]


def test_annotations_cover_exactly_the_greedy_records():
    """Guards against labels drifting out of sync with the generations."""
    from tinystories_project.evaluation.annotations import load_annotations
    from tinystories_project.records import read_jsonl
    greedy_dir = ROOT / "results" / "task2" / "A_greedy"
    if not greedy_dir.exists():
        pytest.skip("Task 2 generations not present; run scripts/run_task2.py")
    ids = [r.record_id for f in greedy_dir.glob("*.jsonl") for r in read_jsonl(f)]
    load_annotations(ANNOTATIONS).validate_against(ids)   # raises on mismatch


def test_validate_against_rejects_a_mismatch():
    from tinystories_project.evaluation.annotations import load_annotations
    ann = load_annotations(ANNOTATIONS)
    with pytest.raises(ValueError, match="do not match"):
        ann.validate_against(["r_not_a_real_record"])


def test_empty_label_is_rejected(tmp_path):
    """An unfinished annotation file must fail loudly, not score as 0."""
    from tinystories_project.evaluation.annotations import load_annotations
    p = tmp_path / "part.csv"
    p.write_text("record_id,label,annotator,notes\nr_1,correct,W,\nr_2,,W,\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="empty label"):
        load_annotations(p)


def test_invalid_label_value_is_rejected(tmp_path):
    from tinystories_project.evaluation.annotations import load_annotations
    p = tmp_path / "bad.csv"
    p.write_text("record_id,label\nr_1,greatish\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not one of"):
        load_annotations(p)


def test_accuracy_maths():
    from tinystories_project.evaluation.annotations import accuracy
    a = accuracy(["correct", "correct", "partial", "incorrect"])
    assert a["n"] == 4 and a["correct"] == 2
    assert a["strict"] == 0.5
    assert a["lenient"] == 0.625           # partial counts half
    assert a["partial_rate"] == 0.25
    assert accuracy([])["strict"] is None


def test_ambiguous_is_excluded_from_the_denominator():
    from tinystories_project.evaluation.annotations import accuracy
    a = accuracy(["correct", "ambiguous", "incorrect"])
    assert a["n"] == 2 and a["strict"] == 0.5
