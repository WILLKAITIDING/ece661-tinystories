"""Integration tests against the REAL TinyStories checkpoint.

Skipped by default (they download ~20 MB and need network). Run them with:

    pytest -m integration

These are the tests that justify the claim "Task 1 works". The unit tests use a
random model and prove plumbing only; these prove the actual released
checkpoint loads and produces English.
"""

from __future__ import annotations

import pytest

from tinystories_project.inference import (
    GenerationParams,
    HFTextGenerator,
    generate_stories,
    load_causal_lm,
)
from tinystories_project.inference.loader import (
    GPT_NEO_TOKENIZER_ID,
    verify_tokenizer_matches,
)
from tinystories_project.io import build_records, read_jsonl, write_jsonl

MODEL_ID = "roneneldan/TinyStories-1M"
PROMPT = "Once upon a time, there was a little girl named Lily."

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bundle():
    try:
        return load_causal_lm(MODEL_ID, device="auto")
    except Exception as exc:  # offline / hub down
        pytest.skip(f"could not download {MODEL_ID}: {exc}")


def test_checkpoint_loads_with_expected_architecture(bundle):
    assert bundle.metadata.architecture == "GPTNeoForCausalLM"
    assert bundle.metadata.extra["model_type"] == "gpt_neo"


def test_measured_architecture_matches_the_published_config(bundle):
    """Values verified against the Hub API on 2026-10-28."""
    m = bundle.metadata
    assert m.n_layers == 8
    assert m.hidden_size == 64
    assert m.vocab_size == 50257
    assert m.max_position_embeddings == 2048


def test_measured_parameter_counts(bundle):
    """TinyStories-1M is ~3.75M params total; only ~0.40M are non-embedding.

    Pinned exactly so that a silent checkpoint change is caught rather than
    quietly reported in our results table.
    """
    m = bundle.metadata
    assert m.n_parameters_total == 3_745_984
    assert m.n_parameters_embedding == 3_347_520
    assert m.n_parameters_non_embedding == 398_464


def test_model_is_in_eval_mode(bundle):
    assert not bundle.model.training


def test_tokenizer_is_padded_and_left_aligned(bundle):
    assert bundle.tokenizer.pad_token is not None
    assert bundle.tokenizer.padding_side == "left"


def test_bundled_tokenizer_matches_the_readme_recommendation():
    """Upstream README says use EleutherAI/gpt-neo-125M; the repo ships its own.

    We verify rather than assume the two agree. If this ever fails, set
    `tokenizer_id: EleutherAI/gpt-neo-125M` in configs/task1.yaml.
    """
    assert verify_tokenizer_matches(MODEL_ID, GPT_NEO_TOKENIZER_ID)


def test_generates_nonempty_english(bundle):
    out = generate_stories(bundle, [PROMPT], max_new_tokens=60, seed=42)
    assert len(out) == 1
    text = out[0].completion
    assert len(text.strip()) > 20, f"suspiciously short completion: {text!r}"
    assert not text.startswith(PROMPT)
    # A coherent TinyStories continuation should be ordinary ASCII English.
    assert sum(c.isalpha() for c in text) > 0.5 * len(text.replace(" ", ""))


def test_generation_is_reproducible_with_a_seed(bundle):
    a = generate_stories(bundle, [PROMPT], max_new_tokens=40, seed=1234)
    b = generate_stories(bundle, [PROMPT], max_new_tokens=40, seed=1234)
    assert a[0].completion == b[0].completion


def test_greedy_decoding_is_deterministic(bundle):
    params = GenerationParams(max_new_tokens=30, do_sample=False)
    a = generate_stories(bundle, [PROMPT], params=params)
    b = generate_stories(bundle, [PROMPT], params=params)
    assert a[0].completion == b[0].completion


def test_batched_prompts_of_different_lengths_all_produce_text(bundle):
    """Catches right-padding corruption, which only shows up when batching."""
    prompts = [PROMPT, "The cat sat.", "Tom found a key in the park one day."]
    gen = HFTextGenerator.from_bundle(bundle, batch_size=3)
    out = gen.generate(prompts, GenerationParams(max_new_tokens=40), seed=7)
    assert len(out) == 3
    for item in out:
        assert len(item.completion.strip()) > 10, f"empty for {item.prompt!r}"
        assert item.n_prompt_tokens and item.n_prompt_tokens > 0


def test_multiple_completions_per_prompt_differ(bundle):
    """The judge pass needs 10 distinct samples per prompt at temperature 1."""
    params = GenerationParams(max_new_tokens=40, temperature=1.0, num_return_sequences=5)
    out = generate_stories(bundle, [PROMPT], params=params, seed=99)
    assert len(out) == 5
    assert len({o.completion for o in out}) > 1, "sampling produced identical text"


def test_end_to_end_write_and_read_back(bundle, tmp_path):
    params = GenerationParams(max_new_tokens=40)
    out = generate_stories(bundle, [PROMPT], params=params, seed=42)
    records = build_records(out, bundle.metadata, params, run_id="integration")
    path = write_jsonl(records, tmp_path / "out.jsonl")
    restored = list(read_jsonl(path))
    assert len(restored) == 1
    assert restored[0].model["model_id"] == MODEL_ID
    assert restored[0].completion == out[0].completion
    assert restored[0].model["n_parameters_total"] == 3_745_984
