"""ROUGE measures of Section 4.4.1 of the TinyStories paper.

This is NOT generic ROUGE. Section 4.4.1 defines its own quantities and the
four figures each measure something different, so the definitions are
implemented literally from the paper (arXiv:2305.07759v2, pp. 17-18):

    Rouge-k precision   R_k,p(T1, T2) = |G_k(T1) & G_k(T2)| / |G_k(T1)|
    Rouge-k f-measure   R_k(T1, T2)   = 2 P(T1,T2) P(T2,T1) / (P(T1,T2)+P(T2,T1))

`G_k(T)` is the SET of k-grams of T: the paper sums an indicator over
`t in G_k(T1)`, so a k-gram occurring twice counts once. The widely used
`rouge_score` package instead intersects multisets, which gives slightly
different numbers; `compare_with_rouge_score()` quantifies the gap so the
deviation is measured rather than assumed.

Note a typo in the paper: the f-measure line prints `R_k` on the right-hand
side where `R_k,p` is meant. Read as written it is circular; the intended
quantity is the harmonic mean of the two precisions, which is the standard
ROUGE F1 (precision in one direction is recall in the other).

The four experiments (100 stories, prompted with their first ~40%):

    Figure 14  s_i = R_2,p(T_i, T'_i)          generated vs the true continuation
    Figure 15  r_i = max_{j!=i} R_2(T_i, T_j)  generated vs the other generations
    Figure 16  g_r = corpus frequency of each generated k-gram
    Figure 17  h_i = max_j R_2,p(T_i, S_j)     over every story S_j in the corpus

Figures 16 and 17 need the whole training corpus; see
`scripts/make_task3_tables.py` for the single-pass streaming implementation.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# Matches the normalisation used by the `rouge_score` package: lowercase, then
# everything that is not a letter or digit becomes a separator.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric word tokens."""
    return [t for t in _NON_ALNUM.sub(" ", text.lower()).split() if t]


def ngram_set(tokens: Sequence[str], k: int = 2) -> set[tuple[str, ...]]:
    """G_k(T): the SET of k-grams, as the paper's indicator sum implies."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(tokens) < k:
        return set()
    return {tuple(tokens[i: i + k]) for i in range(len(tokens) - k + 1)}


def rouge_precision(text_a: str, text_b: str, k: int = 2) -> float:
    """R_k,p(A, B): the fraction of A's k-grams that also occur in B.

    Returns 0.0 when A has no k-grams, so a degenerate generation scores as
    "nothing copied" rather than crashing or scoring 1.
    """
    ga = ngram_set(tokenize(text_a), k)
    if not ga:
        return 0.0
    gb = ngram_set(tokenize(text_b), k)
    return len(ga & gb) / len(ga)


def rouge_fmeasure(text_a: str, text_b: str, k: int = 2) -> float:
    """R_k(A, B): harmonic mean of R_k,p(A,B) and R_k,p(B,A)."""
    p_ab = rouge_precision(text_a, text_b, k)
    p_ba = rouge_precision(text_b, text_a, k)
    if p_ab + p_ba == 0:
        return 0.0
    return 2 * p_ab * p_ba / (p_ab + p_ba)


def split_prompt(story: str, fraction: float = 0.4) -> tuple[str, str]:
    """Cut a story at ~`fraction` of its words: (prompt, true continuation).

    Section 4.4.1: "we cut each story in the middle, keeping roughly the first
    40%, and use it as a prompt". The cut is on whitespace tokens of the
    ORIGINAL text (not the lowercased ROUGE tokens) so the prompt stays
    readable and correctly punctuated for the model.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    words = story.split()
    if len(words) < 2:
        raise ValueError("story is too short to split")
    cut = max(1, min(len(words) - 1, round(len(words) * fraction)))
    return " ".join(words[:cut]), " ".join(words[cut:])


# --- the per-figure measures ---------------------------------------------

def figure14_scores(generated: Sequence[str], originals: Sequence[str],
                    k: int = 2) -> list[float]:
    """s_i = R_k,p(T_i, T'_i): how much of the generation is in the real ending."""
    if len(generated) != len(originals):
        raise ValueError(
            f"need one original per generation, got {len(generated)} vs {len(originals)}")
    return [rouge_precision(g, o, k) for g, o in zip(generated, originals)]


def figure15_scores(generated: Sequence[str], k: int = 2) -> list[float]:
    """r_i = max_{j != i} R_k(T_i, T_j): similarity to the most similar sibling.

    O(n^2) in the number of stories, which is fine for the paper's n = 100.
    """
    n = len(generated)
    if n < 2:
        raise ValueError("need at least two generations to compare")
    tokens = [ngram_set(tokenize(t), k) for t in generated]

    def pairwise(i: int, j: int) -> float:
        gi, gj = tokens[i], tokens[j]
        if not gi or not gj:
            return 0.0
        inter = len(gi & gj)
        p_ij, p_ji = inter / len(gi), inter / len(gj)
        return 0.0 if p_ij + p_ji == 0 else 2 * p_ij * p_ji / (p_ij + p_ji)

    return [max(pairwise(i, j) for j in range(n) if j != i) for i in range(n)]


def describe(values: Iterable[float]) -> dict[str, float]:
    """Summary statistics for a figure's distribution."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return {"n": 0}

    def q(p: float) -> float:
        if n == 1:
            return vals[0]
        pos = p * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return vals[lo] + (pos - lo) * (vals[hi] - vals[lo])

    return {
        "n": n, "mean": sum(vals) / n, "min": vals[0], "max": vals[-1],
        "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
    }


def compare_with_rouge_score(text_a: str, text_b: str, k: int = 2) -> dict[str, float]:
    """Our set-based R_k,p beside the `rouge_score` package's multiset version.

    Used by the tests to measure, rather than assume, how far the paper's
    definition is from the common implementation.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer([f"rouge{k}"], use_stemmer=False)
    pkg = scorer.score(target=text_b, prediction=text_a)[f"rouge{k}"]
    return {
        "ours_precision": rouge_precision(text_a, text_b, k),
        "package_precision": pkg.precision,
        "ours_fmeasure": rouge_fmeasure(text_a, text_b, k),
        "package_fmeasure": pkg.fmeasure,
    }


# --- Figures 16 and 17: one streaming pass over a whole corpus ------------

def scan_corpus(
    generations: Sequence[str],
    corpus: Iterable[str],
    k: int = 2,
    progress_every: int = 0,
) -> dict[str, object]:
    """Compute Figures 16 and 17 in a single pass over the corpus.

    The naive reading of Figure 17 is 100 x |corpus| ROUGE computations. It does
    not need to be: both figures only ever ask about k-grams that the models
    GENERATED, so we index those once and then ask of each corpus story "which
    generated k-grams do you contain?".

    Returns:
        ngram_counts: generated k-gram -> how many TIMES it occurs in the corpus
                      (Figure 16 term frequency; 0 means it never occurs). The
                      paper normalises this by the corpus k-gram total, which
                      `n_ngrams` provides.
        doc_counts:   generated k-gram -> how many corpus STORIES contain it
        best_overlap: per generation, the largest number of its k-grams found in
                      any single corpus story
        figure17:     best_overlap divided by |G_k(T_i)| - that is
                      h_i = max_j R_k,p(T_i, S_j)
        n_stories / n_ngrams: what was scanned

    Two different counts are kept on purpose. Figure 16 is defined over the
    corpus k-gram multiset, so `ngram_counts` counts every occurrence. Figure 17
    needs |G_k(T_i) & G_k(S_j)|, a SET intersection, so the per-story overlap
    counts each distinct k-gram once.
    """
    gen_sets = [ngram_set(tokenize(g), k) for g in generations]
    owners: dict[tuple[str, ...], list[int]] = {}
    for slot, grams in enumerate(gen_sets):
        for ng in grams:
            owners.setdefault(ng, []).append(slot)

    ngram_counts = dict.fromkeys(owners, 0)   # occurrences (term frequency)
    doc_counts = dict.fromkeys(owners, 0)     # stories containing it
    best_overlap = [0] * len(gen_sets)
    hits = [0] * len(gen_sets)
    n_stories = n_ngrams = 0

    for story in corpus:
        n_stories += 1
        tokens = tokenize(story)
        n_ngrams += max(0, len(tokens) - k + 1)
        touched: list[int] = []
        seen: set[tuple[str, ...]] = set()
        for i in range(len(tokens) - k + 1):
            ng = tuple(tokens[i: i + k])
            owner = owners.get(ng)
            if owner is None:
                continue
            ngram_counts[ng] += 1          # every occurrence
            if ng in seen:
                continue
            seen.add(ng)
            doc_counts[ng] += 1            # first time in THIS story
            for slot in owner:             # set semantics for the overlap
                if hits[slot] == 0:
                    touched.append(slot)
                hits[slot] += 1
        for slot in touched:
            if hits[slot] > best_overlap[slot]:
                best_overlap[slot] = hits[slot]
            hits[slot] = 0
        if progress_every and n_stories % progress_every == 0:
            print(f"   scanned {n_stories:,} stories")

    figure17 = [
        best_overlap[s] / len(gen_sets[s]) if gen_sets[s] else 0.0
        for s in range(len(gen_sets))
    ]
    return {
        "ngram_counts": ngram_counts,
        "doc_counts": doc_counts,
        "best_overlap": best_overlap,
        "figure17": figure17,
        "n_stories": n_stories,
        "n_ngrams": n_ngrams,
        "gen_sizes": [len(g) for g in gen_sets],
    }
