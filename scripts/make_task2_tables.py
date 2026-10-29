#!/usr/bin/env python
"""Build Task 2 comparison tables, annotation templates and figures.

    python scripts/make_task2_tables.py

Reads the JSONL written by run_task2.py and writes into results/task2/analysis/:

    comparison_<set>.md / .csv    our models side by side, with the paper's
                                  own published answer for the same checkpoint
    annotation_template.csv       one blank row per completion, for HUMAN rating
    auto_screen.csv               conservative lexical screen (NOT human judgement)
    consistency_protocolB.csv     agreement across the 3 sampled completions
    summary.md                    descriptive statistics and caveats
    fig_paper_verdicts.png        the PAPER's own verdicts (its data, not ours)
    fig_auto_screen.png           our lexical screen rates

Nothing here produces a quality judgement. The screen reports whether an
expected word appears; it cannot tell "no, cows can't fly" from "yes, cows can
fly", which is why prompts needing negation or role tracking are excluded from
it and left to human annotation.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tinystories_project.config import find_project_root  # noqa: E402
from tinystories_project.evaluation import load_prompt_set  # noqa: E402
from tinystories_project.records import read_jsonl  # noqa: E402

ROOT = find_project_root()
RESULTS = ROOT / "results" / "task2"
OUT = RESULTS / "analysis"
CATEGORY_ORDER = ["factual", "reasoning", "context"]
LABELS = ["correct", "partial", "incorrect"]
COLORS = {"correct": "#5fbf5f", "partial": "#f3e699", "incorrect": "#f08080"}


def short(text: str, n: int = 58) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def load_records(protocol: str) -> list:
    out = []
    for path in sorted((RESULTS / protocol).glob("*.jsonl")):
        out.extend(read_jsonl(path))
    return out


def screen(completion: str, terms) -> bool:
    low = completion.lower()
    return any(t.lower() in low for t in terms)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = {}
    for name in ("paper_figures_9_11", "custom_20"):
        for p in load_prompt_set(ROOT / "data" / "prompts" / f"{name}.json"):
            prompts[p.id] = p
    reference = {
        r["id"]: r
        for r in json.loads(
            (ROOT / "data" / "reference" / "paper_figure_completions.json")
            .read_text(encoding="utf-8")
        )["rows"]
    }

    greedy = load_records("A_greedy")
    sampled = load_records("B_sampling")
    if not greedy:
        print("No A_greedy results found - run scripts/run_task2.py first.")
        return 1

    # Model order: as configured (smallest first, shallow variant last).
    manifest = sorted(RESULTS.glob("manifest_*.json"))[-1]
    model_meta = {m["model_id"]: m for m in
                  json.loads(manifest.read_text(encoding="utf-8"))["models"]}
    models = list(model_meta)
    col_of = {m: model_meta[m]["figure_column"] for m in models}
    name_of = {m: m.split("/")[-1] for m in models}

    # ---------------------------------------------------------------- tables
    by_prompt: dict[str, dict[str, object]] = defaultdict(dict)
    for r in greedy:
        by_prompt[r.extra["prompt_id"]][r.model["model_id"]] = r

    for set_name in ("paper_figures_9_11", "custom_20"):
        ids = [pid for pid in prompts
               if any(rec.extra["prompt_set"] == set_name
                      for rec in by_prompt[pid].values())]
        ids.sort(key=lambda i: (CATEGORY_ORDER.index(prompts[i].category), i))
        if not ids:
            continue

        csv_path = OUT / f"comparison_{set_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            head = ["prompt_id", "category", "prompt", "expected"]
            head += [f"ours:{name_of[m]}" for m in models]
            if set_name == "paper_figures_9_11":
                head += [f"paper:{col_of[m]}" for m in models]
            w.writerow(head)
            for pid in ids:
                p = prompts[pid]
                row = [pid, p.category, p.text, p.expected]
                row += [by_prompt[pid][m].extra["answer_span"] if m in by_prompt[pid]
                        else "" for m in models]
                if set_name == "paper_figures_9_11":
                    row += [reference[pid]["completions"].get(col_of[m], "")
                            for m in models]
                w.writerow(row)

        md = [f"# Task 2 comparison - `{set_name}`", "",
              "Protocol A (greedy / zero temperature), first sentence of each "
              "continuation. Generated locally on Apple MPS.", ""]
        if set_name == "paper_figures_9_11":
            md += ["**`ours:` columns are our own generations. `paper:` columns are "
                   "the completions printed in the TinyStories paper for the same "
                   "checkpoint - they were NOT generated by us.**", ""]
        for cat in CATEGORY_ORDER:
            cat_ids = [i for i in ids if prompts[i].category == cat]
            if not cat_ids:
                continue
            md += [f"## {cat.capitalize()}", ""]
            header = ["Prompt", "Expected"] + [f"ours: {name_of[m]}" for m in models]
            if set_name == "paper_figures_9_11":
                header += [f"paper: {col_of[m]}" for m in models]
            md += ["| " + " | ".join(header) + " |",
                   "|" + "---|" * len(header)]
            for pid in cat_ids:
                p = prompts[pid]
                cells = [short(p.text, 70), short(p.expected, 40)]
                cells += [short(by_prompt[pid][m].extra["answer_span"], 46)
                          if m in by_prompt[pid] else "" for m in models]
                if set_name == "paper_figures_9_11":
                    cells += [short(reference[pid]["completions"].get(col_of[m], ""), 46)
                              for m in models]
                md += ["| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"]
            md += [""]
        (OUT / f"comparison_{set_name}.md").write_text("\n".join(md), encoding="utf-8")

    # ------------------------------------------------- annotation + screening
    ann_path = OUT / "annotation_template.csv"
    scr_path = OUT / "auto_screen.csv"
    screen_rows = []
    with ann_path.open("w", newline="", encoding="utf-8") as af, \
         scr_path.open("w", newline="", encoding="utf-8") as sf:
        aw = csv.writer(af)
        aw.writerow(["record_id", "model", "figure_column", "protocol", "prompt_id",
                     "category", "origin", "prompt", "expected", "answer_span",
                     "full_completion", "label", "annotator", "notes"])
        sw = csv.writer(sf)
        sw.writerow(["record_id", "model", "prompt_id", "category", "origin",
                     "auto_screenable", "hit_in_answer_span", "hit_in_full_completion",
                     "answer_span"])
        for r in greedy:
            p = prompts[r.extra["prompt_id"]]
            mid = r.model["model_id"]
            aw.writerow([r.record_id, name_of.get(mid, mid), r.extra.get("figure_column"),
                         r.extra["protocol"], p.id, p.category, p.origin, p.text,
                         p.expected, r.extra["answer_span"], r.completion.strip(),
                         "", "", ""])   # label / annotator / notes left blank
            has = bool(p.auto_accept)
            span_hit = screen(r.extra["answer_span"], p.auto_accept) if has else ""
            full_hit = screen(r.completion, p.auto_accept) if has else ""
            sw.writerow([r.record_id, name_of.get(mid, mid), p.id, p.category,
                         p.origin, has, span_hit, full_hit, r.extra["answer_span"]])
            if has:
                screen_rows.append((mid, p.category, p.origin, bool(span_hit)))

    # ------------------------------------------------------- consistency (B)
    cons_path = OUT / "consistency_protocolB.csv"
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for r in sampled:
        grouped[(r.model["model_id"], r.extra["prompt_id"])].append(r)
    with cons_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "prompt_id", "category", "origin", "n_samples",
                    "n_distinct_spans", "auto_screenable", "n_screen_hits",
                    "all_hit", "none_hit"])
        for (mid, pid), rows in sorted(grouped.items()):
            p = prompts[pid]
            spans = [r.extra["answer_span"] for r in rows]
            has = bool(p.auto_accept)
            hits = sum(screen(s, p.auto_accept) for s in spans) if has else ""
            w.writerow([name_of.get(mid, mid), pid, p.category, p.origin, len(rows),
                        len(set(spans)), has, hits,
                        (hits == len(rows)) if has else "",
                        (hits == 0) if has else ""])

    # ------------------------------------------------------------- figures
    paper_tally = {c: Counter() for c in
                   ["1M/8L", "2.5M/8L", "8.3M/8L", "28M/8L", "33M/4L",
                    "21M/1L", "33M/2L", "GPT2-XL(1.5B)"]}
    for row in reference.values():
        for col, verdict in row["paper_verdicts"].items():
            paper_tally[col][verdict] += 1

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = list(paper_tally)
    bottom = [0] * len(cols)
    for label in LABELS:
        vals = [paper_tally[c][label] for c in cols]
        ax.bar(cols, vals, bottom=bottom, label=label, color=COLORS[label],
               edgecolor="#444", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("prompts (of 24)")
    ax.set_title("TinyStories paper's OWN verdicts on Figures 9-11\n"
                 "(extracted from the published figure colours - not our results)")
    ax.legend(title="paper's colour", fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_paper_verdicts.png", dpi=150)
    plt.close(fig)

    rate: dict[tuple[str, str], list] = defaultdict(list)
    for mid, cat, _origin, hit in screen_rows:
        rate[(mid, cat)].append(hit)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(models)
    xs = range(len(CATEGORY_ORDER))
    for i, m in enumerate(models):
        vals = [100 * sum(rate[(m, c)]) / len(rate[(m, c)]) if rate[(m, c)] else 0
                for c in CATEGORY_ORDER]
        ax.bar([x + i * width for x in xs], vals, width,
               label=f"{name_of[m]} ({col_of[m]})")
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in xs])
    ax.set_xticklabels([c.capitalize() for c in CATEGORY_ORDER])
    ax.set_ylabel("% of screenable prompts containing an expected word")
    ax.set_ylim(0, 100)
    ax.set_title("Conservative LEXICAL SCREEN, protocol A (greedy)\n"
                 "a keyword heuristic, NOT a human or LLM quality judgement")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_auto_screen.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------- summary
    n_screenable = sum(1 for p in prompts.values() if p.auto_accept)
    lines = [
        "# Task 2 summary", "",
        f"- models: {len(models)}  |  prompts: {len(prompts)} "
        f"(24 from the paper's Figures 9-11, 20 original)",
        f"- completions: {len(greedy)} greedy (protocol A) + {len(sampled)} sampled "
        f"(protocol B, 3 per prompt)",
        f"- auto-screenable prompts: {n_screenable}/{len(prompts)}; the remaining "
        f"{len(prompts) - n_screenable} need human judgement "
        "(negation, role tracking, or open-ended).", "",
        "## Measured model sizes", "",
        "| checkpoint | figure column | layers | hidden | total params | non-embedding |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m in models:
        d = model_meta[m]
        lines.append(
            f"| {name_of[m]} | {d['figure_column']} | {d['n_layers']} | "
            f"{d['hidden_size']} | {d['n_parameters_total']:,} | "
            f"{d['n_parameters_non_embedding']:,} |")
    lines += ["", "The figure-column names do not match either measured count; we "
              "report measured values throughout.", "",
              "## Lexical screen, protocol A (NOT a quality judgement)", "",
              "| checkpoint | " + " | ".join(c.capitalize() for c in CATEGORY_ORDER)
              + " | overall |", "|---|" + "---|" * (len(CATEGORY_ORDER) + 1)]
    for m in models:
        cells = []
        allhits = []
        for c in CATEGORY_ORDER:
            vals = rate[(m, c)]
            allhits += vals
            cells.append(f"{sum(vals)}/{len(vals)}" if vals else "-")
        lines.append(f"| {name_of[m]} | " + " | ".join(cells)
                     + f" | {sum(allhits)}/{len(allhits)} |")
    # Section 4.2 claims width drives factual recall and depth drives context
    # tracking. The factual-minus-context gap is the cleanest way to see it:
    # a wide but shallow model should show a LARGER gap than a deep one.
    lines += ["", "## Width vs depth (Section 4.2 claim)", "",
              "The paper argues embedding dimension drives factual recall while "
              "the number of layers drives context tracking. If so, a wide but "
              "shallow model should show a larger factual-minus-context gap.", "",
              "| checkpoint | layers | hidden | factual % | context % | gap |",
              "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        d = model_meta[m]
        pct = {}
        for c in ("factual", "context"):
            vals = rate[(m, c)]
            pct[c] = 100 * sum(vals) / len(vals) if vals else float("nan")
        lines.append(
            f"| {name_of[m]} | {d['n_layers']} | {d['hidden_size']} | "
            f"{pct['factual']:.0f} | {pct['context']:.0f} | "
            f"{pct['factual'] - pct['context']:+.0f} |")

    lines += ["", "## Files", ""]
    for f in sorted(OUT.iterdir()):
        lines.append(f"- `{f.name}`")
    lines += ["", "## Caveats", "",
              "- The screen checks whether an expected word appears. It cannot "
              "distinguish \"no, cows can't fly\" from \"yes, cows can fly\"; such "
              "prompts are excluded from it and left blank for human annotation.",
              "- `annotation_template.csv` has an empty `label` column. No row has "
              "been rated by a human yet, and none of these numbers is a human rating.",
              "- Protocol B samples are NOT reproductions of the paper's "
              "deterministic figures and must not be presented as such."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(list(OUT.iterdir()))} files to {OUT.relative_to(ROOT)}")
    for f in sorted(OUT.iterdir()):
        print(f"   {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
