#!/usr/bin/env python
"""Build Task 4 judge tables, figures and the methodology note.

    python scripts/make_task4_tables.py
    python scripts/make_task4_tables.py --scores path/to/scores.jsonl   # e.g. a self-test

Reads the per-record judge scores written by run_task4.py and produces, in
results/task4/analysis/:

    scores_by_model.csv        mean grammar / creativity / consistency per model
    scores_by_category.csv     the same, split by train vs validation
    age_distribution.csv       estimated reading-age buckets per model
    examples.md                the best- and worst-scored story per model
    usage_summary.csv          tokens and cost actually spent
    summary.md                 results plus methodology and limitations
    fig_judge_scores.png       the three rubric dimensions by model
    fig_age_distribution.png   reading age by model

Reports only what was actually scored. Dry-run rows are refused outright, so a
placeholder can never be mistaken for a judgement.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tinystories_project.config import find_project_root, resolve_path  # noqa: E402
from tinystories_project.evaluation.judge import AGE_BUCKETS, SCORE_FIELDS  # noqa: E402
from tinystories_project.records import read_jsonl  # noqa: E402

ROOT = find_project_root()
MODEL_ORDER = ["TinyStories-1M", "TinyStories-8M", "TinyStories-33M",
               "TinyStories-1Layer-21M"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", default="results/task4/scores.jsonl")
    p.add_argument("--usage", default="results/task4/usage_log.jsonl")
    p.add_argument("--out", default="results/task4/analysis")
    p.add_argument("--source-dir", default="results/task3")
    return p.parse_args(argv)


def load_scores(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    dry = [r for r in rows if r.get("dry_run")]
    if dry:
        raise SystemExit(
            f"{path} contains {len(dry)} dry-run rows. Those are placeholders, not "
            "judgements - point --scores at a real run instead.")
    ok = [r for r in rows if r.get("ok") and r.get("grammar") is not None]
    return ok


def agg(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows)}
    for f in SCORE_FIELDS:
        vals = [r[f] for r in rows if isinstance(r.get(f), (int, float))]
        out[f] = {"mean": mean(vals) if vals else None,
                  "sd": stdev(vals) if len(vals) > 1 else 0.0,
                  "n": len(vals)}
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    scores_path = resolve_path(args.scores)
    if not scores_path.exists():
        print(f"No judge scores at {scores_path}.\n"
              f"Run: python scripts/run_task4.py --config configs/task4.yaml")
        return 1
    rows = load_scores(scores_path)
    if not rows:
        print(f"{scores_path} has no successfully judged records yet.")
        return 1

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    models += sorted({r["model"] for r in rows} - set(models))
    splits = sorted({r["split"] for r in rows})
    judge_models = sorted({r["judge_model"] for r in rows})
    rubrics = sorted({r.get("rubric_version", "?") for r in rows})

    by_model = {m: [r for r in rows if r["model"] == m] for m in models}
    by_model_split = {(m, s): [r for r in rows if r["model"] == m and r["split"] == s]
                      for m in models for s in splits}

    # ------------------------------------------------------------ tables
    with (out / "scores_by_model.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n"] + [f"{f}_{k}" for f in SCORE_FIELDS
                                     for k in ("mean", "sd")])
        for m in models:
            a = agg(by_model[m])
            row = [m, a["n"]]
            for f in SCORE_FIELDS:
                row += [f"{a[f]['mean']:.3f}" if a[f]["mean"] is not None else "",
                        f"{a[f]['sd']:.3f}"]
            w.writerow(row)

    with (out / "scores_by_category.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "n"] + [f"{f}_mean" for f in SCORE_FIELDS])
        for m in models:
            for s in splits:
                a = agg(by_model_split[(m, s)])
                if not a["n"]:
                    continue
                w.writerow([m, s, a["n"]] + [
                    f"{a[f]['mean']:.3f}" if a[f]["mean"] is not None else ""
                    for f in SCORE_FIELDS])

    age_counts = {m: Counter(r.get("age_group") or "unparsed" for r in by_model[m])
                  for m in models}
    with (out / "age_distribution.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        buckets = list(AGE_BUCKETS) + ["unparsed"]
        w.writerow(["model", "n"] + buckets)
        for m in models:
            w.writerow([m, len(by_model[m])] + [age_counts[m].get(b, 0) for b in buckets])

    # ------------------------------------------------------- usage / cost
    usage_path = resolve_path(args.usage)
    totals = {"calls": 0, "ok": 0, "prompt_tokens": 0, "completion_tokens": 0,
              "cost_usd": 0.0, "key_spend_usd": None}
    if usage_path.exists():
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            u = json.loads(line)
            totals["calls"] += 1
            totals["ok"] += bool(u.get("ok"))
            totals["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            totals["completion_tokens"] += int(u.get("completion_tokens") or 0)
            totals["cost_usd"] += float(u.get("response_cost_usd") or 0.0)
            if u.get("key_spend_usd") is not None:
                totals["key_spend_usd"] = u["key_spend_usd"]
    with (out / "usage_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(totals))
        w.writerow([totals[k] for k in totals])

    # --------------------------------------------------------- examples
    source = {}
    src_dir = resolve_path(args.source_dir)
    if src_dir.exists():
        for f in src_dir.rglob("*.jsonl"):
            if f.parent.name in {"prompts", "analysis"}:
                continue
            for rec in read_jsonl(f):
                source[rec.record_id] = rec
    lines = ["# Task 4 - representative judged stories", "",
             f"Judge: `{', '.join(judge_models)}` at temperature 0, rubric "
             f"`{', '.join(rubrics)}`.", ""]
    for m in models:
        ranked = sorted(by_model[m],
                        key=lambda r: sum(r[f] for f in SCORE_FIELDS))
        for label, r in (("lowest-scored", ranked[0]), ("highest-scored", ranked[-1])):
            rec = source.get(r["record_id"])
            lines += [f"## {m} - {label}", "",
                      f"- grammar **{r['grammar']}**, creativity **{r['creativity']}**, "
                      f"consistency **{r['consistency']}**, age **{r.get('age_group') or '?'}**",
                      f"- judge note: {r.get('note', '')!r}", ""]
            if rec is not None:
                lines += ["**Prompt (first 40% of a real story):**", "",
                          "> " + " ".join(rec.prompt.split())[:400], "",
                          "**Model completion:**", "",
                          "> " + " ".join(rec.completion.split())[:600], ""]
    (out / "examples.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / len(SCORE_FIELDS)
    xs = range(len(models))
    for i, f in enumerate(SCORE_FIELDS):
        vals = [agg(by_model[m])[f]["mean"] or 0 for m in models]
        errs = [agg(by_model[m])[f]["sd"] for m in models]
        ax.bar([x + i * width for x in xs], vals, width, yerr=errs, capsize=3,
               label=f.capitalize())
    ax.set_xticks([x + width for x in xs])
    ax.set_xticklabels(models, fontsize=8, rotation=12)
    ax.set_ylabel("judge score (1-10)")
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8)
    ax.set_title(f"Task 4 - {', '.join(judge_models)} as judge, temperature 0\n"
                 f"mean score over {len(rows)} Task 3 story completions "
                 f"(error bars = 1 SD)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_judge_scores.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    buckets = [b for b in AGE_BUCKETS
               if any(age_counts[m].get(b) for m in models)]
    bottom = [0.0] * len(models)
    for b in buckets:
        vals = [100 * age_counts[m].get(b, 0) / max(1, len(by_model[m])) for m in models]
        ax.bar(models, vals, bottom=bottom, label=b)
        bottom = [x + v for x, v in zip(bottom, vals)]
    ax.set_ylabel("% of stories")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7, title="estimated reading age")
    ax.set_title("Task 4 - estimated reading age group by model", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_age_distribution.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- summary
    L = ["# Task 4 - Gemini-as-judge results", "",
         f"- **{len(rows)}** story completions judged by "
         f"`{', '.join(judge_models)}` at **temperature 0**, rubric "
         f"`{', '.join(rubrics)}`.",
         f"- Source: the Task 3 generations, so all four checkpoints answered "
         f"the SAME story prefixes - the scores are matched across models.",
         f"- API cost: **${totals['cost_usd']:.4f}** over {totals['calls']} calls "
         f"({totals['prompt_tokens']:,} input + {totals['completion_tokens']:,} "
         f"output tokens).", "",
         "## Mean judge scores by model", "",
         "| checkpoint | n | grammar | creativity | consistency |",
         "|---|---:|---:|---:|---:|"]
    for m in models:
        a = agg(by_model[m])
        L.append(f"| {m} | {a['n']} | " + " | ".join(
            f"{a[f]['mean']:.2f}" if a[f]["mean"] is not None else "-"
            for f in SCORE_FIELDS) + " |")

    L += ["", "## Train vs validation", "",
          "| checkpoint | split | n | grammar | creativity | consistency |",
          "|---|---|---:|---:|---:|---:|"]
    for m in models:
        for s in splits:
            a = agg(by_model_split[(m, s)])
            if not a["n"]:
                continue
            L.append(f"| {m} | {s} | {a['n']} | " + " | ".join(
                f"{a[f]['mean']:.2f}" if a[f]["mean"] is not None else "-"
                for f in SCORE_FIELDS) + " |")

    L += ["", "## Methodology", "",
          "- Prompt follows the GPT-Eval idea of Section 3 of the TinyStories "
          "paper: the judge sees the story beginning and the model's completion, "
          "and grades grammar, creativity and consistency out of 10 plus a "
          "reading-age bucket.",
          "- The paper used GPT-4; we use a Gemini Flash-tier model through the "
          "ECE 661 course gateway, as the proposal specifies.",
          "- Temperature 0 for judging, so the grading is deterministic and a "
          "rerun reproduces it.",
          "- The rubric text is byte-identical for every story and every model; "
          "only the prompt and completion differ.",
          "- Replies must be JSON and are validated: scores must be integers in "
          "1-10 and the age group must be one of the listed buckets. A reply "
          "that cannot be parsed is recorded as a failure, never guessed at.",
          "- Resume is by `judge_id` (record + judge model + temperature + rubric "
          "version), so a rerun never re-pays for a completed score.",
          "",
          "## Limitations", "",
          "- An LLM judge is not a human judge. Scores are comparable BETWEEN our "
          "models (identical rubric, identical prompts, same judge) but are not "
          "calibrated against the paper's GPT-4 numbers and should not be "
          "compared to them directly.",
          "- A single judge model has its own biases; the proposal pairs these "
          "scores with the Task 3 reference-free metrics and, for Phase 3, with "
          "native-speaker ratings.",
          "- One judgement per story (the paper averages 10 completions per "
          "prompt for its own evaluation set). These are single-sample scores on "
          "a shared prompt set, so model means are the unit to read, not "
          "individual stories.",
          "- Reading-age buckets are coarse and self-reported by the judge.",
          "", "## Files", ""]
    L += [f"- `{f.name}`" for f in sorted(out.iterdir())]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    try:
        shown = out.relative_to(ROOT)
    except ValueError:      # --out may point outside the repo (self-tests)
        shown = out
    print(f"Wrote {len(list(out.iterdir()))} files to {shown}:")
    for f in sorted(out.iterdir()):
        print(f"   {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
