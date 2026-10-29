"""Generation behaviour, using a tiny randomly initialised model."""

from __future__ import annotations

import pytest
import torch

from tinystories_project.inference.generate import HFTextGenerator, generate_stories
from tinystories_project.inference.interface import (
    GeneratedText,
    GenerationParams,
    ModelMetadata,
    TextGenerator,
)
from tinystories_project.inference.loader import ModelBundle


@pytest.fixture
def generator(tiny_model, tiny_tokenizer, tiny_metadata):
    return HFTextGenerator(tiny_model, tiny_tokenizer, metadata=tiny_metadata, batch_size=2)


def test_generator_satisfies_the_published_protocol(generator):
    assert isinstance(generator, TextGenerator)


def test_returns_one_item_per_prompt(generator, prompts, short_params):
    out = generator.generate(prompts, short_params)
    assert len(out) == len(prompts)
    assert all(isinstance(o, GeneratedText) for o in out)


def test_returns_n_items_per_prompt(generator, prompts):
    out = generator.generate(prompts, GenerationParams(max_new_tokens=5, num_return_sequences=3))
    assert len(out) == len(prompts) * 3
    for i in range(len(prompts)):
        samples = [o for o in out if o.prompt_index == i]
        assert sorted(s.sample_index for s in samples) == [0, 1, 2]


def test_prompt_is_not_echoed_in_completion(generator, prompts, short_params):
    """The single most important output property for ROUGE and the judge."""
    for item in generator.generate(prompts, short_params):
        assert not item.completion.startswith(item.prompt)
        assert item.full_text == item.prompt + item.completion


def test_prompt_indices_are_correct_across_batches(generator, prompts, short_params):
    """batch_size=2 with 3 prompts means the offset maths must be right."""
    out = generator.generate(prompts, short_params)
    assert [o.prompt_index for o in out] == [0, 1, 2]
    assert all(item.prompt == prompts[item.prompt_index] for item in out)


def test_true_prompt_token_counts_ignore_padding(generator, prompts, short_params):
    """Left-padding must not inflate the recorded prompt length."""
    counts = [o.n_prompt_tokens for o in generator.generate(prompts, short_params)]
    assert counts[1] < counts[0] < counts[2], f"got {counts}"


def test_respects_max_new_tokens_and_reports_finish_reason(generator, prompts):
    for item in generator.generate(prompts, GenerationParams(max_new_tokens=6)):
        assert item.n_new_tokens <= 6
        assert item.finish_reason in {"eos", "length"}


def test_empty_prompt_list_returns_empty(generator, short_params):
    assert generator.generate([], short_params) == []


def test_bare_string_is_rejected(generator, short_params):
    """Passing a str would silently generate one completion per character."""
    with pytest.raises(TypeError, match="sequence of strings"):
        generator.generate("a single prompt", short_params)


# --- reproducibility -------------------------------------------------------

def test_same_seed_gives_identical_text(generator, prompts, short_params):
    a = generator.generate(prompts, short_params, seed=123)
    b = generator.generate(prompts, short_params, seed=123)
    assert [x.completion for x in a] == [x.completion for x in b]
    assert all(isinstance(x.seed, int) for x in a)


def test_different_seeds_give_different_text(generator, prompts):
    params = GenerationParams(max_new_tokens=20)
    a = generator.generate(prompts, params, seed=1)
    b = generator.generate(prompts, params, seed=999)
    assert [x.completion for x in a] != [x.completion for x in b]


def test_greedy_is_deterministic_without_a_seed(generator, prompts):
    params = GenerationParams(max_new_tokens=8, do_sample=False)
    a = generator.generate(prompts, params)
    b = generator.generate(prompts, params)
    assert [x.completion for x in a] == [x.completion for x in b]


# --- the generate_stories wrapper -----------------------------------------

def test_keyword_overrides_win_over_params(generator, prompts):
    base = GenerationParams(max_new_tokens=50, temperature=0.5)
    out = generate_stories(generator, prompts, params=base, max_new_tokens=3)
    assert all(o.n_new_tokens <= 3 for o in out)


def test_accepts_a_model_bundle(tiny_model, tiny_tokenizer, tiny_metadata, prompts):
    bundle = ModelBundle(tiny_model, tiny_tokenizer, tiny_metadata, torch.device("cpu"))
    assert len(generate_stories(bundle, prompts, max_new_tokens=4, seed=1)) == len(prompts)


def test_works_on_a_custom_non_hf_generator(prompts):
    """A model with no .generate() must still work - the Phase 2/3 contract."""

    class FakeGenerator:
        @property
        def metadata(self):
            return ModelMetadata(
                model_id="rachel/min-gru-5M", architecture="MinGRU", source="custom"
            )

        def generate(self, prompts, params, seed=None):
            return [
                GeneratedText(
                    prompt=p, completion=" and then everyone went home.",
                    prompt_index=i, sample_index=0, seed=seed,
                    n_prompt_tokens=len(p.split()), n_new_tokens=6,
                    finish_reason="eos",
                )
                for i, p in enumerate(prompts)
            ]

    fake = FakeGenerator()
    assert isinstance(fake, TextGenerator)
    out = generate_stories(fake, prompts, max_new_tokens=10, seed=42)
    assert len(out) == len(prompts)
    assert out[0].completion == " and then everyone went home."
