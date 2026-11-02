#!/usr/bin/env python
"""Compute Figures 14-17 of Section 4.4.1 and write Task 3 tables and figures.

    python scripts/make_task3_tables.py --config configs/task3.yaml
    python scripts/make_task3_tables.py --skip-corpus     # figures 14 & 15 only

Figures 14 and 15 only need the 100 generated completions. Figures 16 and 17
need the ENTIRE training corpus, which the paper scans naively (100 x 2.1M
ROUGE computations). That is not tractable on a laptop, but it does not need to
be: both quantities only ever ask about k-grams the MODELS generated, of which
there are a few tens of thousands. So we

  1. collect every generated k-gram once, over all models and splits,
  2. make ONE streaming pass over the 2.12M training stories,
  3. for each story, count how many of its k-grams belong to each generation.

That yields Figure 16 (corpus frequency of each generated k-gram) and Figure 17
(max Rouge-2 precision against any training story) exactly as defined, in one
pass and in memory proportional to the generated text, not the corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from itertools import islice
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from tinystories_project.config import resolve_path  # noqa: E402
from tinystories_project.evaluation.corpus import iter_stories, local_paths  # noqa: E402
from tinystories_project.evaluation.rouge import (  # noqa: E402
    describe, figure14_scores, figure15_scores, ngram_set, scan_corpus, tokenize,
)
from tinystories_project.records import read_jsonl, write_json  # noqa: E402

K = 2  # Figures 14, 15 and 17 are all defined on Rouge-2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/task3.yaml")
    p.add_argument("--skip-corpus", action="store_true",
                   help="skip Figures 16/17 (they scan the whole training corpus)")
    p.add_argument("--corpus-limit", type=int, default=None,
                   help="scan only the first N training stories (for a quick check)")
    return p.parse_args(argv)


def load_runs(root: Path) -> dict[tuple[str, str], list]:
    """(split, model) -> records, from the Task 3 JSONL output."""
    runs: dict[tuple[str, str], list] = {}
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if split_dir.name in {"prompts", "analysis"}:
            continue
        for path in sorted(split_dir.glob("*.jsonl")):
            recs = list(read_jsonl(path))
            if recs:
                runs[(split_dir.name, path.stem)] = recs
    return runs


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    root = resolve_path(cfg["output_dir"])
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    if not runs:
        print("No Task 3 results. Run scripts/run_task3.py first.")
        return 1
    models = sorted({m for _, m in runs}, key=lambda m: [
        i for i, s in enumerate(cfg["models"]) if s["model_id"].endswith(m)] or [99])
    splits = sorted({s for s, _ in runs})
    print(f"loaded {len(runs)} runs: {len(models)} models x {len(splits)} splits")

    # ---------------------------------------------------- Figures 14 and 15
    per_story_rows, summary_rows = [], []
    fig14: dict[tuple[str, str], list[float]] = {}
    fig15: dict[tuple[str, str], list[float]] = {}

    for (split, model), recs in runs.items():
        gen = [r.completion for r in recs]
        orig = [r.extra["original_continuation"] for r in recs]
        s = figure14_scores(gen, orig, K)
        r_ = figure15_scores(gen, K)
        fig14[(split, model)] = s
        fig15[(split, model)] = r_
        for rec, si, ri in zip(recs, s, r_):
            per_story_rows.append({
                "split": split, "model": model, "story_id": rec.extra["story_id"],
                "corpus_index": rec.extra["corpus_index"],
                "fig14_rouge2_precision_vs_original": round(si, 6),
                "fig15_max_rouge2_f_vs_other_generations": round(ri, 6),
                "n_new_tokens": rec.n_new_tokens,
            })

    # The dataset's own stories are the control for Figure 15: if real stories
    # are as similar to each other as the generations are, "diverse" means
    # nothing. The paper plots this as the "original model" row.
    for split in splits:
        recs = next(r for (s, _), r in runs.items() if s == split)
        originals = [r.extra["original_continuation"] for r in recs]
        fig15[(split, "DATASET (real continuations)")] = figure15_scores(originals, K)

    for (split, model), vals in sorted(fig14.items()):
        summary_rows.append({"figure": 14, "split": split, "model": model,
                             "metric": "rouge2_precision_vs_original",
                             **{k: round(v, 6) if isinstance(v, float) else v
                                for k, v in describe(vals).items()}})
    for (split, model), vals in sorted(fig15.items()):
        summary_rows.append({"figure": 15, "split": split, "model": model,
                             "metric": "max_rouge2_f_vs_other_generations",
                             **{k: round(v, 6) if isinstance(v, float) else v
                                for k, v in describe(vals).items()}})

    # ---------------------------------------------- Figures 16 and 17 (corpus)
    fig16: dict[tuple[str, str], list[int]] = {}
    fig17: dict[tuple[str, str], list[float]] = {}
    corpus_stats = None
    if not args.skip_corpus:
        # Flatten every generation into one list so a SINGLE corpus pass serves
        # all models and splits at once.
        slot_of: list[tuple[str, str, int]] = []
        flat: list[str] = []
        for (split, model), recs in sorted(runs.items()):
            for i, rec in enumerate(recs):
                slot_of.append((split, model, i))
                flat.append(rec.completion)

        ds = cfg["dataset"]
        paths = local_paths(ds["repo_id"], ds["train_files"])

        # The corpus pass takes ~10 minutes, and nothing about it changes when
        # only the tables or figures are edited. Cache it against a digest of
        # exactly the things that would change the answer.
        digest = hashlib.sha256(json.dumps(
            {"k": K, "limit": args.corpus_limit,
             "files": [p_.name for p_ in paths],
             "generations": flat}, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = out / "corpus_scan_cache.json"
        scan = None
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("digest") == digest:
                scan = cached["scan"]
                scan["ngram_counts"] = {tuple(k_.split("\t")): v
                                        for k_, v in scan["ngram_counts"].items()}
                elapsed = cached["seconds"]
                print(f"reusing cached corpus scan ({cache_path.name}, "
                      f"digest {digest})")

        if scan is None:
            corpus = iter_stories(paths)
            if args.corpus_limit:
                corpus = islice(corpus, args.corpus_limit)
            print(f"scanning the training corpus for {len(flat)} generations "
                  f"(k={K}) - this is one pass, not {len(flat)} x |corpus| ...")
            t0 = time.time()
            scan = scan_corpus(flat, corpus, k=K, progress_every=500_000)
            elapsed = time.time() - t0
            serialisable = dict(scan)
            serialisable["ngram_counts"] = {"\t".join(k_): v
                                            for k_, v in scan["ngram_counts"].items()}
            serialisable.pop("doc_counts", None)
            cache_path.write_text(json.dumps(
                {"digest": digest, "seconds": round(elapsed, 1),
                 "scan": serialisable}), encoding="utf-8")
        print(f"corpus pass: {scan['n_stories']:,} stories, "
              f"{scan['n_ngrams']:,} {K}-grams, {elapsed:.0f}s")
        corpus_stats = {"n_stories": scan["n_stories"], "n_ngrams": scan["n_ngrams"],
                        "k": K, "seconds": round(elapsed, 1),
                        "limited_to": args.corpus_limit}

        # Figure 17: one value per generation.
        for slot, (split, model, _i) in enumerate(slot_of):
            fig17.setdefault((split, model), []).append(scan["figure17"][slot])

        # Figure 16: the corpus count of every k-gram each run generated.
        gen_sets = [ngram_set(tokenize(t), K) for t in flat]
        for slot, (split, model, _i) in enumerate(slot_of):
            bucket = fig16.setdefault((split, model), [])
            bucket.extend(scan["ngram_counts"][ng] for ng in gen_sets[slot])

        pos: dict[tuple[str, str], int] = defaultdict(int)
        for row, slot in zip(per_story_rows, range(len(slot_of))):
            key = (row["split"], row["model"])
            row["fig17_max_rouge2_precision_vs_corpus"] = round(
                fig17[key][pos[key]], 6)
            pos[key] += 1
        # A generation with very few 2-grams can hit h_i = 1.0 trivially, so
        # record the denominator alongside the score.
        size_of = {(sp, mo, i): n for (sp, mo, i), n
                   in zip(slot_of, scan["gen_sizes"])}
        idx: dict[tuple[str, str], int] = defaultdict(int)
        for row in per_story_rows:
            key = (row["split"], row["model"])
            row["n_generated_2grams"] = size_of[(key[0], key[1], idx[key])]
            idx[key] += 1

        total_ngrams = scan["n_ngrams"]
        for (split, model), counts in sorted(fig16.items()):
            zero = sum(1 for c in counts if c == 0)
            med = sorted(counts)[len(counts) // 2]
            summary_rows.append({
                "figure": 16, "split": split, "model": model,
                "metric": "occurrences_per_generated_2gram_in_corpus",
                "n": len(counts), "mean": round(sum(counts) / len(counts), 3),
                "median": med,
                # g_r as the paper defines it: occurrences / corpus k-gram total
                "median_frequency_g_r": med / total_ngrams,
                "frac_never_in_corpus": round(zero / len(counts), 6),
                "min": min(counts), "max": max(counts),
            })
        for (split, model), vals in sorted(fig17.items()):
            summary_rows.append({"figure": 17, "split": split, "model": model,
                                 "metric": "max_rouge2_precision_vs_corpus",
                                 **{k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in describe(vals).items()}})

    # ------------------------------------------------------------- outputs
    with (out / "per_story_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = list(per_story_rows[0])
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(per_story_rows)
    with (out / "figure_summaries.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = sorted({k for r in summary_rows for k in r})
        order = ["figure", "split", "model", "metric", "n", "mean", "median",
                 "p25", "p75", "min", "max", "frac_never_in_corpus"]
        cols = [c for c in order if c in cols] + [c for c in cols if c not in order]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)
    write_json({"k": K, "models": models, "splits": splits,
                "corpus": corpus_stats,
                "generation": yaml.safe_load(
                    resolve_path(args.config).read_text(encoding="utf-8"))["generation"]},
               out / "run_info.json")

    _figures(out, models, splits, fig14, fig15, fig16, fig17)
    _summary(out, cfg, models, splits, fig14, fig15, fig16, fig17, corpus_stats)

    print(f"\nWrote to {out}:")
    for f in sorted(out.iterdir()):
        print(f"   {f.name}")
    return 0


def _hist(ax, data, title, xlabel, bins=20, rng=(0, 1)):
    ax.hist(data, bins=bins, range=rng, color="#4c78a8", edgecolor="white")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)


def _figures(out, models, splits, fig14, fig15, fig16, fig17):
    for split in splits:
        panels = [("Figure 14", fig14, "Rouge-2 precision vs the true ending"),
                  ("Figure 15", fig15, "max Rouge-2 F vs other generations")]
        if fig17:
            panels.append(("Figure 17", fig17,
                           "max Rouge-2 precision vs any training story"))
        rows = len(panels)
        keys = [m for m in models] + (
            ["DATASET (real continuations)"] if any(
                k[1].startswith("DATASET") for k in fig15) else [])
        fig, axes = plt.subplots(rows, len(keys), figsize=(3.1 * len(keys), 2.5 * rows),
                                 squeeze=False)
        for r, (name, data, xlabel) in enumerate(panels):
            for c, model in enumerate(keys):
                ax = axes[r][c]
                vals = data.get((split, model))
                if vals is None:
                    ax.axis("off")
                    continue
                _hist(ax, vals, model, xlabel)
                if c == 0:
                    # Name the row, so each band of panels is identifiable.
                    ax.set_ylabel(f"{name}\nstories", fontsize=8)
        fig.suptitle(f"Section 4.4.1 replication - {split} prompts "
                     f"(n=100, greedy decoding)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"fig_14_15_17_{split}.png", dpi=140)
        plt.close(fig)

    if fig16:
        for split in splits:
            keys = [m for m in models if (split, m) in fig16]
            fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 2.8),
                                     squeeze=False)
            for c, model in enumerate(keys):
                counts = fig16[(split, model)]
                capped = [min(c_, 10) for c_ in counts]
                axes[0][c].hist(capped, bins=range(12), color="#72b7b2",
                                edgecolor="white", align="left")
                axes[0][c].set_title(model, fontsize=9)
                axes[0][c].set_xlabel("times the 2-gram occurs in the corpus\n"
                                      "(capped at 10)", fontsize=7)
                axes[0][c].tick_params(labelsize=7)
            fig.suptitle(f"Figure 16 - corpus frequency of each generated 2-gram "
                         f"({split} prompts)", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            fig.savefig(out / f"fig_16_{split}.png", dpi=140)
            plt.close(fig)


def _summary(out, cfg, models, splits, fig14, fig15, fig16, fig17, corpus_stats):
    L = ["# Task 3 - Section 4.4.1 quantitative replication", "",
         "Four measures, as defined on pp. 17-18 of arXiv:2305.07759v2. "
         f"{cfg['sampling']['n_stories']} stories per split, cut at "
         f"{cfg['sampling']['prompt_fraction']:.0%} of their words, completed with "
         "greedy decoding. The paper does not state a temperature for 4.4.1; we "
         "chose greedy because it is reproducible. We have not measured how "
         "sampling would change these overlaps.", ""]
    if corpus_stats:
        L += [f"Figures 16 and 17 scan the training corpus: "
              f"**{corpus_stats['n_stories']:,} stories, "
              f"{corpus_stats['n_ngrams']:,} 2-grams**, in one pass "
              f"({corpus_stats['seconds']:.0f}s).", ""]

    L += ["## Figure 14 - is the generation contained in the real ending?", "",
          "`s_i = R_2,p(T_i, T'_i)`. Low means the model wrote something new "
          "rather than reproducing the story it was trained on.", "",
          "| split | model | mean | median | p75 | max |", "|---|---|---:|---:|---:|---:|"]
    for split in splits:
        for m in models:
            d = describe(fig14.get((split, m), []))
            if d["n"]:
                L.append(f"| {split} | {m} | {d['mean']:.3f} | {d['median']:.3f} "
                         f"| {d['p75']:.3f} | {d['max']:.3f} |")

    L += ["", "## Figure 15 - are the 100 generations similar to each other?", "",
          "`r_i = max_{j != i} R_2(T_i, T_j)` (f-measure). The DATASET row is the "
          "same measure over the 100 real continuations, which is the control: it "
          "shows how similar genuine TinyStories are to one another.", "",
          "| split | model | mean | median | p75 | max |", "|---|---|---:|---:|---:|---:|"]
    for split in splits:
        for m in models + ["DATASET (real continuations)"]:
            d = describe(fig15.get((split, m), []))
            if d["n"]:
                L.append(f"| {split} | {m} | {d['mean']:.3f} | {d['median']:.3f} "
                         f"| {d['p75']:.3f} | {d['max']:.3f} |")

    if fig16:
        L += ["", "## Figure 16 - how often does each generated 2-gram occur in the corpus?", "",
              "`g_r` is the paper's quantity: a generated 2-gram's occurrences divided by "
              "the corpus 2-gram total. `frac never` is the share that appear "
              "**nowhere** in the corpus.", "",
              "| split | model | 2-grams | median occurrences | median g_r | frac never in corpus |",
              "|---|---|---:|---:|---:|---:|"]
        total = corpus_stats["n_ngrams"]
        for split in splits:
            for m in models:
                counts = fig16.get((split, m))
                if not counts:
                    continue
                zero = sum(1 for c in counts if c == 0)
                med = sorted(counts)[len(counts) // 2]
                L.append(f"| {split} | {m} | {len(counts):,} | {med:,} | "
                         f"{med/total:.2e} | {zero/len(counts):.1%} |")

    if fig17:
        L += ["", "## Figure 17 - closest story in the entire training corpus", "",
              "`h_i = max_j R_2,p(T_i, S_j)` over all 2.12M training stories. "
              "High values would mean the generation closely matches something the "
              "model was trained on.", "",
              "| split | model | mean | median | p75 | max |",
              "|---|---|---:|---:|---:|---:|"]
        for split in splits:
            for m in models:
                d = describe(fig17.get((split, m), []))
                if d["n"]:
                    L.append(f"| {split} | {m} | {d['mean']:.3f} | {d['median']:.3f} "
                             f"| {d['p75']:.3f} | {d['max']:.3f} |")

    # --- the memorisation question, which is what 4.4.1 is actually for ---
    if fig14:
        L += ["", "## Train vs validation - is the model memorising?", "",
              "The paper runs this on TRAINING stories: if a model had memorised "
              "them, its completion would match the real ending. We add the "
              "validation split, whose prefixes the models never trained on. "
              "If training-set scores are not higher than validation scores, we "
              "have no evidence of copying for the prefixes we tested. That is "
              "weaker than a claim that these models do not memorise at all.", "",
              "| model | Fig 14 train | Fig 14 validation | difference |",
              "|---|---:|---:|---:|"]
        for m in models:
            tr = describe(fig14.get(("train", m), []))
            va = describe(fig14.get(("validation", m), []))
            if tr["n"] and va["n"]:
                L.append(f"| {m} | {tr['mean']:.3f} | {va['mean']:.3f} | "
                         f"{tr['mean'] - va['mean']:+.3f} |")
        if fig17:
            L += ["", "| model | Fig 17 train | Fig 17 validation | difference |",
                  "|---|---:|---:|---:|"]
            for m in models:
                tr = describe(fig17.get(("train", m), []))
                va = describe(fig17.get(("validation", m), []))
                if tr["n"] and va["n"]:
                    L.append(f"| {m} | {tr['mean']:.3f} | {va['mean']:.3f} | "
                             f"{tr['mean'] - va['mean']:+.3f} |")

    L += ["", "## Limitations", "",
          "- The paper does not state the decoding temperature for 4.4.1. We used "
          "greedy because it is reproducible. We did not test sampling, so we "
          "cannot say how it would change these numbers.",
          "- We cannot verify which corpus snapshot each released checkpoint was "
          "trained on. We use the current `roneneldan/TinyStories` parquet config; "
          "if a checkpoint saw different data, Figures 16/17 are measured against "
          "a near-but-not-identical corpus.",
          "- `R_k,p` is implemented on k-gram SETS, exactly as the paper defines "
          "it. The common `rouge_score` package intersects multisets and gives "
          "slightly different numbers; `compare_with_rouge_score()` quantifies it.",
          "- A very short completion can reach h_i = 1.0 trivially, because all of "
          "its handful of 2-grams fit inside one training story. "
          "`per_story_scores.csv` records `n_generated_2grams` so these are "
          "identifiable; they are a small minority.",
          "- n = 100 stories per split, as in the paper. These are distributions "
          "to be read as such, not precise point estimates.",
          "- The validation split and the Figure 15 comparison against real "
          "dataset continuations are OUR additions, not part of the paper's "
          "Section 4.4.1.",
          ""]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
