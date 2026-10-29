"""Task 3: the Section 4.4.1 ROUGE definitions.

These test the maths against the paper's formulas, with no model and no
dataset download, so they stay fast and offline.
"""

from __future__ import annotations

import pytest

from tinystories_project.evaluation.rouge import (
    compare_with_rouge_score,
    describe,
    figure14_scores,
    figure15_scores,
    ngram_set,
    rouge_fmeasure,
    rouge_precision,
    split_prompt,
    tokenize,
)


# --- tokenisation ---------------------------------------------------------

def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize('The CAT sat, "Wow!"') == ["the", "cat", "sat", "wow"]


def test_tokenize_handles_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("!!! ... ???") == []


def test_ngram_set_is_a_set_not_a_multiset():
    """The paper sums an indicator over t in G_k(T1), so repeats count once."""
    toks = tokenize("the cat the cat the cat")
    assert ngram_set(toks, 2) == {("the", "cat"), ("cat", "the")}


def test_ngram_set_empty_when_text_shorter_than_k():
    assert ngram_set(["only"], 2) == set()
    assert ngram_set([], 2) == set()


def test_ngram_set_rejects_bad_k():
    with pytest.raises(ValueError, match="k must be"):
        ngram_set(["a", "b"], 0)


# --- the ROUGE formulas ---------------------------------------------------

def test_precision_is_the_fraction_of_a_ngrams_found_in_b():
    a, b = "the cat sat down", "the cat ran away"
    # G2(a) = {the cat, cat sat, sat down}; only "the cat" is in G2(b)
    assert rouge_precision(a, b, 2) == pytest.approx(1 / 3)


def test_precision_is_directional():
    a, b = "the cat sat on the mat", "the cat"
    assert rouge_precision(b, a, 2) == 1.0      # all of b's 2-grams are in a
    assert rouge_precision(a, b, 2) < 1.0


def test_identical_text_scores_one():
    t = "once upon a time there was a little girl"
    assert rouge_precision(t, t, 2) == 1.0
    assert rouge_fmeasure(t, t, 2) == 1.0


def test_disjoint_text_scores_zero():
    assert rouge_precision("alpha beta gamma", "delta epsilon zeta", 2) == 0.0
    assert rouge_fmeasure("alpha beta gamma", "delta epsilon zeta", 2) == 0.0


def test_empty_candidate_scores_zero_rather_than_crashing():
    assert rouge_precision("", "the cat sat", 2) == 0.0
    assert rouge_fmeasure("", "the cat sat", 2) == 0.0


def test_fmeasure_is_the_harmonic_mean_of_both_precisions():
    a, b = "the cat sat on the mat", "the cat ran on the rug"
    p_ab, p_ba = rouge_precision(a, b, 2), rouge_precision(b, a, 2)
    assert rouge_fmeasure(a, b, 2) == pytest.approx(
        2 * p_ab * p_ba / (p_ab + p_ba))


def test_fmeasure_is_symmetric():
    a, b = "the cat sat on the mat", "a dog sat on the mat today"
    assert rouge_fmeasure(a, b, 2) == pytest.approx(rouge_fmeasure(b, a, 2))


def test_our_definition_differs_from_the_rouge_score_package_as_documented():
    """The paper uses SET k-grams; the package intersects multisets.

    We measure the gap rather than assume it is zero, so the deviation is a
    documented fact rather than a surprise.
    """
    a = "the cat the cat the cat sat"
    b = "the cat sat"
    got = compare_with_rouge_score(a, b, 2)
    assert got["ours_precision"] != got["package_precision"]
    assert got["ours_precision"] == pytest.approx(2 / 3)   # {the cat, cat the, cat sat}


# --- prompt splitting -----------------------------------------------------

def test_split_prompt_keeps_roughly_the_first_40_percent():
    story = " ".join(f"w{i}" for i in range(100))
    prompt, rest = split_prompt(story, 0.4)
    assert len(prompt.split()) == 40
    assert len(rest.split()) == 60
    assert prompt.split()[0] == "w0" and rest.split()[0] == "w40"


def test_split_prompt_rejoins_to_the_original_words():
    story = "Once upon a time there was a little girl named Lily who loved cake"
    prompt, rest = split_prompt(story, 0.4)
    assert (prompt + " " + rest).split() == story.split()


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_split_prompt_rejects_bad_fractions(fraction):
    with pytest.raises(ValueError, match="fraction"):
        split_prompt("a b c d e", fraction)


def test_split_prompt_rejects_a_story_that_is_too_short():
    with pytest.raises(ValueError, match="too short"):
        split_prompt("word", 0.4)


def test_split_prompt_never_empties_either_side():
    prompt, rest = split_prompt("alpha beta", 0.4)
    assert prompt and rest


# --- the per-figure measures ---------------------------------------------

def test_figure14_pairs_each_generation_with_its_own_original():
    gen = ["the cat sat", "a dog ran"]
    orig = ["the cat sat quietly", "a bird flew"]
    s = figure14_scores(gen, orig, 2)
    assert s[0] == pytest.approx(1.0)   # both 2-grams appear in its original
    assert s[1] == pytest.approx(0.0)


def test_figure14_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one original per generation"):
        figure14_scores(["a b"], ["a b", "c d"], 2)


def test_figure15_takes_the_max_over_other_generations_only():
    gen = ["the cat sat on the mat",
           "the cat sat on the rug",     # near-duplicate of the first
           "zebras orbit purple moons"]
    r = figure15_scores(gen, 2)
    assert r[0] > 0.5 and r[1] > 0.5     # they find each other
    assert r[2] == pytest.approx(0.0)    # nothing resembles the third
    assert all(v <= 1.0 for v in r)


def test_figure15_needs_at_least_two_generations():
    with pytest.raises(ValueError, match="at least two"):
        figure15_scores(["only one"], 2)


def test_figure15_of_identical_texts_is_one():
    assert figure15_scores(["the cat sat", "the cat sat"], 2) == [1.0, 1.0]


# --- summary statistics ---------------------------------------------------

def test_describe_reports_quantiles():
    d = describe([0.0, 0.25, 0.5, 0.75, 1.0])
    assert d["n"] == 5
    assert d["mean"] == pytest.approx(0.5)
    assert d["median"] == pytest.approx(0.5)
    assert d["min"] == 0.0 and d["max"] == 1.0
    assert d["p25"] == pytest.approx(0.25)


def test_describe_handles_empty_and_single():
    assert describe([])["n"] == 0
    d = describe([0.7])
    assert d["n"] == 1 and d["median"] == 0.7


# --- the one-pass corpus scan (Figures 16 and 17) -------------------------

def test_scan_corpus_matches_a_brute_force_implementation():
    """The fast one-pass scan must agree exactly with the naive definition."""
    import random
    from tinystories_project.evaluation.rouge import ngram_set, scan_corpus

    rng = random.Random(0)
    vocab = "cat dog ran sat jumped happy sad red blue the a on under".split()
    corpus = [" ".join(rng.choice(vocab) for _ in range(rng.randint(6, 20)))
              for _ in range(200)]
    gens = [" ".join(rng.choice(vocab) for _ in range(12)) for _ in range(8)]

    got = scan_corpus(gens, corpus, k=2)

    # Figure 17 by brute force: max over every story of R_2,p(T_i, S_j)
    for i, g in enumerate(gens):
        gi = ngram_set(tokenize(g), 2)
        expect = max(len(gi & ngram_set(tokenize(s), 2)) / len(gi) for s in corpus)
        assert got["figure17"][i] == pytest.approx(expect), f"generation {i}"

    # Figure 16 by brute force: total occurrences, and stories containing it
    for ng, count in got["ngram_counts"].items():
        toks_per_story = [tokenize(s) for s in corpus]
        occurrences = sum(
            sum(1 for i in range(len(t) - 1) if (t[i], t[i + 1]) == ng)
            for t in toks_per_story)
        docs = sum(1 for t in toks_per_story if ng in ngram_set(t, 2))
        assert count == occurrences, f"term freq {ng}"
        assert got["doc_counts"][ng] == docs, f"doc freq {ng}"


def test_scan_corpus_separates_term_and_document_frequency():
    """Figure 16 counts occurrences; Figure 17's overlap counts distinct k-grams."""
    from tinystories_project.evaluation.rouge import scan_corpus
    got = scan_corpus(["the cat"], ["the cat the cat the cat"], k=2)
    assert got["ngram_counts"][("the", "cat")] == 3    # three occurrences
    assert got["doc_counts"][("the", "cat")] == 1      # in one story
    assert got["figure17"] == [1.0]


def test_scan_corpus_reports_ngrams_absent_from_the_corpus():
    from tinystories_project.evaluation.rouge import scan_corpus
    got = scan_corpus(["zebra orbits"], ["the cat sat on the mat"], k=2)
    assert got["ngram_counts"][("zebra", "orbits")] == 0
    assert got["figure17"] == [0.0]


def test_scan_corpus_on_an_empty_corpus():
    from tinystories_project.evaluation.rouge import scan_corpus
    got = scan_corpus(["the cat sat"], [], k=2)
    assert got["n_stories"] == 0
    assert got["figure17"] == [0.0]
    assert set(got["ngram_counts"].values()) == {0}
