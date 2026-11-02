#!/usr/bin/env python
"""Build Task 2 result tables and figures from the FINAL annotations.

    python scripts/make_task2_tables.py

Reads the greedy (protocol A) generations plus the final labels in
data/annotations/task2_final.csv, and writes into results/task2/analysis/:

    results_by_model.csv / _by_category.csv   strict accuracy and partial rates
    comparison_<set>.md / .csv                outputs side by side, with labels
    ours_vs_paper.csv                         our labels vs the paper's verdicts
                                              on the same 24 prompts
    consistency_protocolB.csv                 variation across 3 sampled runs
    summary.md                                the report-ready write-up
    fig_results_by_category.png               our labelled results
    fig_ours_vs_paper.png                     ours vs the paper, same checkpoints

The labels are HUMAN-REVIEWED AI DRAFTS (9 overridden, 167 accepted) - the
summary states this. Protocol B has no labels, so its 528 sampled completions
are described with a keyword proxy that is clearly marked as such.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tinystories_project.config import find_project_root  # noqa: E402
from tinystories_project.evaluation import load_prompt_set  # noqa: E402
from tinystories_project.evaluation.annotations import (  # noqa: E402
    SCORED_LABELS,
    accuracy,
    load_annotations,
)
from tinystories_project.records import read_jsonl  # noqa: E402

ROOT = find_project_root()
RESULTS = ROOT / "results" / "task2"
OUT = RESULTS / "analysis"
ANNOTATIONS = ROOT / "data" / "annotations" / "task2_final.csv"
CATEGORIES = ["factual", "reasoning", "context"]
COLORS = {"correct": "#5fbf5f", "partial": "#f3e699", "incorrect": "#f08080"}


def short(text: str, n: int = 58) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", default=str(ANNOTATIONS),
                   help="label CSV to score against (default: the Task 2 labels)")
    p.add_argument("--out", default=str(OUT), help="where to write the tables")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    annotations_path = Path(args.annotations)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = {}
    for name in ("paper_figures_9_11", "custom_20"):
        for p in load_prompt_set(ROOT / "data" / "prompts" / f"{name}.json"):
            prompts[p.id] = p
    reference = {
        r["id"]: r for r in json.loads(
            (ROOT / "data" / "reference" / "paper_figure_completions.json")
            .read_text(encoding="utf-8"))["rows"]
    }

    greedy = []
    for path in sorted((RESULTS / "A_greedy").glob("*.jsonl")):
        greedy.extend(read_jsonl(path))
    if not greedy:
        print("No A_greedy results. Run scripts/run_task2.py first.")
        return 1

    ann = load_annotations(annotations_path)
    label_of = {a.record_id: a.label for a in ann}
    prov = ann.provenance()

    # Model metadata comes from the records themselves - every GenerationRecord
    # carries the ModelMetadata its generator reported - so a teammate who drops
    # in their own JSONL gets layer counts and parameter counts for free. Run
    # manifests only add `figure_column`, which is specific to our baselines.
    model_meta: dict[str, dict] = {}
    for r in greedy:
        model_meta.setdefault(r.model["model_id"], dict(r.model))
    baseline_order: list[str] = []      # the order our own runs were configured in
    for mf in sorted(RESULTS.glob("manifest_*.json")):
        for m in json.loads(mf.read_text(encoding="utf-8")).get("models", []):
            model_meta.setdefault(m["model_id"], {}).update(
                {k: v for k, v in m.items() if v is not None})
            if m["model_id"] not in baseline_order:
                baseline_order.append(m["model_id"])

    # Records without a label are not an error: a new model simply has not been
    # annotated yet. Score what is labelled and say plainly what is not, rather
    # than refusing to produce the baseline tables at all.
    greedy_all = list(greedy)      # every generation, labelled or not
    unlabelled = [r for r in greedy if r.record_id not in label_of]
    if unlabelled:
        missing_by_model = defaultdict(int)
        for r in unlabelled:
            missing_by_model[r.model["model_id"]] += 1
        print("NOTE: these models have generations but no labels yet, so they "
              "are omitted from the accuracy tables:")
        for mid, n in sorted(missing_by_model.items()):
            print(f"   {mid}: {n} unlabelled completions")
        print("   Label them in annotation_template.csv and add them to "
              f"{annotations_path.name} to include them.")
        greedy = [r for r in greedy if r.record_id in label_of]
        if not greedy:
            print("No labelled completions at all - nothing to aggregate.")
            return 1

    models = []
    for r in greedy:
        mid = r.model["model_id"]
        if mid not in models:
            models.append(mid)
    # Keep the configured baselines first, then any extra models, for stable
    # column order across runs.
    # Baselines keep their configured order; any extra model follows, by name.
    models.sort(key=lambda m: (baseline_order.index(m)
                               if m in baseline_order else len(baseline_order), m))
    name_of = {m: m.split("/")[-1] for m in models}
    for r in greedy_all:                       # include unlabelled models too
        name_of.setdefault(r.model["model_id"], r.model["model_id"].split("/")[-1])
    col_of = {m: (model_meta.get(m, {}) or {}).get("figure_column") or "-"
              for m in models}

    recs = defaultdict(dict)        # prompt_id -> model_id -> record
    labels = defaultdict(list)      # (model, category|None, origin|None) -> labels
    for r in greedy:
        mid = r.model["model_id"]
        pid = r.extra["prompt_id"]
        recs[pid][mid] = r
        lab = label_of[r.record_id]
        labels[(mid, None, None)].append(lab)
        labels[(mid, r.extra["category"], None)].append(lab)
        labels[(mid, None, r.extra["origin"])].append(lab)
        labels[(mid, r.extra["category"], r.extra["origin"])].append(lab)

    # ------------------------------------------------------- result tables
    with (out / "results_by_model.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "figure_column", "layers", "hidden",
                    "n_params_total", "n_params_non_embedding", "n",
                    "correct", "partial", "incorrect",
                    "strict_accuracy", "lenient_accuracy", "partial_rate"])
        for m in models:
            a = accuracy(labels[(m, None, None)])
            d = model_meta.get(m, {})
            w.writerow([name_of[m], col_of[m], d.get("n_layers", ""),
                        d.get("hidden_size", ""),
                        d.get("n_parameters_total", ""),
                        d.get("n_parameters_non_embedding", ""),
                        a["n"], a["correct"], a["partial"], a["incorrect"],
                        f"{a['strict']:.4f}", f"{a['lenient']:.4f}",
                        f"{a['partial_rate']:.4f}"])

    with (out / "results_by_category.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "category", "origin", "n", "correct", "partial",
                    "incorrect", "strict_accuracy", "lenient_accuracy"])
        for m in models:
            for cat in CATEGORIES:
                for origin in (None, "paper", "team32"):
                    a = accuracy(labels[(m, cat, origin)])
                    if a["n"] == 0:
                        continue
                    w.writerow([name_of[m], cat, origin or "all", a["n"],
                                a["correct"], a["partial"], a["incorrect"],
                                f"{a['strict']:.4f}", f"{a['lenient']:.4f}"])

    # ---- annotation template: every generation, with any existing label ----
    # This is how a newly added model gets annotated: its rows appear here with
    # an empty `label` for a human to fill in.
    with (out / "annotation_template.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["record_id", "model", "figure_column", "protocol", "prompt_id",
                    "category", "origin", "prompt", "expected", "answer_span",
                    "full_completion", "label", "annotator", "notes"])
        for r in greedy_all:
            pr = prompts[r.extra["prompt_id"]]
            mid = r.model["model_id"]
            w.writerow([r.record_id, name_of.get(mid, mid.split("/")[-1]),
                        r.extra.get("figure_column"), r.extra["protocol"], pr.id,
                        pr.category, pr.origin, pr.text, pr.expected,
                        r.extra["answer_span"], r.completion.strip(),
                        label_of.get(r.record_id, ""), "", ""])

    # --------------------------------------------- ours vs the paper's verdicts
    paper_rows = []
    agree = disagree = 0
    for pid, row in reference.items():
        for m in models:
            rec = recs.get(pid, {}).get(m)
            if rec is None:
                continue
            ours = label_of[rec.record_id]
            theirs = row["paper_verdicts"].get(col_of[m], "")
            paper_rows.append({
                "prompt_id": pid, "category": row["category"],
                "model": name_of[m], "figure_column": col_of[m],
                "our_label": ours, "paper_verdict": theirs,
                "match": ours == theirs,
                "our_answer_span": rec.extra["answer_span"],
                "paper_completion": row["completions"].get(col_of[m], ""),
            })
            agree += ours == theirs
            disagree += ours != theirs
    with (out / "ours_vs_paper.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(paper_rows[0]))
        w.writeheader()
        w.writerows(paper_rows)

    # ------------------------------------------------------ comparison tables
    for set_name in ("paper_figures_9_11", "custom_20"):
        ids = sorted(
            (pid for pid in recs
             if any(r.extra["prompt_set"] == set_name for r in recs[pid].values())),
            key=lambda i: (CATEGORIES.index(prompts[i].category), i))
        if not ids:
            continue
        is_paper = set_name == "paper_figures_9_11"

        with (out / f"comparison_{set_name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            head = ["prompt_id", "category", "prompt", "expected"]
            for m in models:
                head += [f"ours:{name_of[m]}", f"label:{name_of[m]}"]
            if is_paper:
                head += [f"paper:{col_of[m]}" for m in models]
                head += [f"paper_verdict:{col_of[m]}" for m in models]
            w.writerow(head)
            for pid in ids:
                p = prompts[pid]
                row = [pid, p.category, p.text, p.expected]
                for m in models:
                    r = recs[pid].get(m)
                    row += [r.extra["answer_span"] if r else "",
                            label_of[r.record_id] if r else ""]
                if is_paper:
                    row += [reference[pid]["completions"].get(col_of[m], "") for m in models]
                    row += [reference[pid]["paper_verdicts"].get(col_of[m], "") for m in models]
                w.writerow(row)

        md = [f"# Task 2 comparison - `{set_name}`", "",
              "Protocol A (greedy / zero temperature), first sentence of each "
              "continuation, with its final label in brackets.", ""]
        if is_paper:
            md += ["**`ours:` columns are our generations and our labels. "
                   "`paper:` columns are the completions and colour verdicts "
                   "printed in the TinyStories paper - NOT generated or judged "
                   "by us.**", ""]
        for cat in CATEGORIES:
            cat_ids = [i for i in ids if prompts[i].category == cat]
            if not cat_ids:
                continue
            header = ["Prompt", "Expected"] + [f"ours: {name_of[m]}" for m in models]
            if is_paper:
                header += [f"paper: {col_of[m]}" for m in models]
            md += [f"## {cat.capitalize()}", "",
                   "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            for pid in cat_ids:
                p = prompts[pid]
                cells = [short(p.text, 66), short(p.expected, 36)]
                for m in models:
                    r = recs[pid].get(m)
                    cells.append(
                        f"{short(r.extra['answer_span'], 40)} **[{label_of[r.record_id]}]**"
                        if r else "")
                if is_paper:
                    for m in models:
                        cells.append(
                            f"{short(reference[pid]['completions'].get(col_of[m], ''), 40)} "
                            f"*[{reference[pid]['paper_verdicts'].get(col_of[m], '')}]*")
                md.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
            md.append("")
        (out / f"comparison_{set_name}.md").write_text("\n".join(md), encoding="utf-8")

    # --------------------------------------------------- protocol B (unlabelled)
    sampled = []
    for path in sorted((RESULTS / "B_sampling").glob("*.jsonl")):
        sampled.extend(read_jsonl(path))
    grouped = defaultdict(list)
    for r in sampled:
        grouped[(r.model["model_id"], r.extra["prompt_id"])].append(r)
    with (out / "consistency_protocolB.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "prompt_id", "category", "n_samples",
                    "n_distinct_spans", "keyword_proxy_hits",
                    "greedy_label_for_same_prompt"])
        for (mid, pid), rows in sorted(grouped.items()):
            p = prompts[pid]
            spans = [r.extra["answer_span"] for r in rows]
            hits = (sum(any(t.lower() in s.lower() for t in p.auto_accept)
                        for s in spans) if p.auto_accept else "")
            g = recs.get(pid, {}).get(mid)
            w.writerow([name_of.get(mid, mid), pid, p.category, len(rows),
                        len(set(spans)), hits,
                        label_of[g.record_id] if g else ""])

    # ------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, cat in zip(axes, CATEGORIES):
        bottom = [0.0] * len(models)
        for lab in SCORED_LABELS:
            vals = []
            for m in models:
                group = labels[(m, cat, None)]
                vals.append(100 * group.count(lab) / len(group) if group else 0)
            ax.bar([name_of[m] for m in models], vals, bottom=bottom, label=lab,
                   color=COLORS[lab], edgecolor="#444", linewidth=0.5)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(cat.capitalize())
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of prompts")
    # Bars stack to 100%, so there is no free space inside the axes - put the
    # legend above the panels rather than over a bar.
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("Task 2 final labels by category (protocol A, greedy; n=44 prompts "
                 "per model)\nhuman-reviewed AI-drafted labels", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "fig_results_by_category.png", dpi=150)
    plt.close(fig)

    # Only models with a counterpart column in the paper can be compared to it;
    # a teammate's own model has no paper verdict and is simply left out here.
    comparable = [m for m in models
                  if accuracy([r["paper_verdict"] for r in paper_rows
                               if r["model"] == name_of[m]])["strict"] is not None]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    xs = range(len(comparable))
    width = 0.38
    ours, theirs = [], []
    for m in comparable:
        mine = [r["our_label"] for r in paper_rows if r["model"] == name_of[m]]
        pap = [r["paper_verdict"] for r in paper_rows if r["model"] == name_of[m]]
        ours.append(100 * accuracy(mine)["strict"])
        theirs.append(100 * accuracy(pap)["strict"])
    ax.bar([x - width / 2 for x in xs], ours, width, label="ours (our labels)",
           color="#4c78a8")
    ax.bar([x + width / 2 for x in xs], theirs, width,
           label="the paper (its own verdicts)", color="#bbb")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{name_of[m]}\n{col_of[m]}" for m in comparable], fontsize=8)
    ax.set_ylabel("% judged correct (strict)")
    ax.set_ylim(0, 100)
    ax.set_title("Same 24 Figure 9-11 prompts, same checkpoints:\n"
                 "our greedy reproduction vs the completions published in the paper")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_ours_vs_paper.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------- summary
    overall = accuracy([a.label for a in ann])
    L = ["# Task 2 results (final annotations)", "",
         f"- **{len(ann)}** greedy (protocol A) completions: "
         f"{len(models)} checkpoints x {len(prompts)} prompts.",
         f"- Labels: **{overall['correct']} correct, {overall['partial']} partial, "
         f"{overall['incorrect']} incorrect**.",
         f"- Provenance: labels were **drafted by an AI assistant and reviewed by "
         f"{', '.join(prov['annotators'])}**, who overrode "
         f"**{prov['n_human_overrides']}** and accepted "
         f"**{prov['n_accepted_ai_drafts']}**. They are human-reviewed AI drafts, "
         f"not independent human annotation.", "",
         "## Results by model", "",
         "| checkpoint | figure col | layers | hidden | non-emb params | correct | partial | incorrect | strict | lenient |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        a = accuracy(labels[(m, None, None)])
        d = model_meta.get(m, {})
        nonemb = d.get("n_parameters_non_embedding")
        L.append(f"| {name_of[m]} | {col_of[m]} | {d.get('n_layers', '?')} | "
                 f"{d.get('hidden_size', '?')} "
                 f"| {f'{nonemb:,}' if nonemb else '?'} | {a['correct']} | "
                 f"{a['partial']} | {a['incorrect']} | **{pct(a['strict'])}** | "
                 f"{pct(a['lenient'])} |")

    L += ["", "## Strict accuracy by category", "",
          "| checkpoint | " + " | ".join(c.capitalize() for c in CATEGORIES) + " | all |",
          "|---|" + "---:|" * (len(CATEGORIES) + 1)]
    for m in models:
        cells = [pct(accuracy(labels[(m, c, None)])["strict"]) for c in CATEGORIES]
        L.append(f"| {name_of[m]} | " + " | ".join(cells)
                 + f" | {pct(accuracy(labels[(m, None, None)])['strict'])} |")

    L += ["", "## Paper prompts vs our own prompts", "",
          "| checkpoint | paper Figures 9-11 (24) | our new prompts (20) |",
          "|---|---:|---:|"]
    for m in models:
        L.append(f"| {name_of[m]} | {pct(accuracy(labels[(m, None, 'paper')])['strict'])} "
                 f"| {pct(accuracy(labels[(m, None, 'team32')])['strict'])} |")

    L += ["", "## Our reproduction vs the paper's published verdicts", "",
          f"On the same 24 prompts and the same four checkpoints, our label "
          f"agrees with the paper's colour on **{agree}/{agree + disagree}** cells "
          f"({100 * agree / (agree + disagree):.0f}%).", "",
          "**Read this comparison carefully.** The two columns below are judged "
          "by DIFFERENT judges: ours by our reviewer, the paper's by its own "
          "authors. A gap therefore mixes a real model difference with a "
          "difference in how strictly each judge marked, and our greedy outputs "
          "are not the same strings the paper printed. Treat it as a sanity "
          "check on the trend, not as a measured delta.", "",
          "| checkpoint | ours (strict) | the paper (strict) |", "|---|---:|---:|"]
    for m in comparable:
        mine = [r["our_label"] for r in paper_rows if r["model"] == name_of[m]]
        pap = [r["paper_verdict"] for r in paper_rows if r["model"] == name_of[m]]
        L.append(f"| {name_of[m]} ({col_of[m]}) | {pct(accuracy(mine)['strict'])} "
                 f"| {pct(accuracy(pap)['strict'])} |")

    L += ["", "## Width vs depth (Section 4.2)", "",
          "The paper argues embedding dimension drives factual recall while depth "
          "drives context tracking, so a wide but shallow model should show a "
          "larger factual-minus-context gap.", "",
          "| checkpoint | layers | hidden | factual | context | gap |",
          "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        d = model_meta.get(m, {})
        f = accuracy(labels[(m, "factual", None)])["strict"]
        c = accuracy(labels[(m, "context", None)])["strict"]
        gap = f"{100 * (f - c):+.0f} pts" if (f is not None and c is not None) else "-"
        L.append(f"| {name_of[m]} | {d.get('n_layers', '?')} | "
                 f"{d.get('hidden_size', '?')} | "
                 f"{pct(f)} | {pct(c)} | {gap} |")

    L += ["", "## Protocol B (sampling) - NOT labelled", "",
          f"{len(sampled)} completions (3 per prompt, T=1.0, top-p 0.95, seed 42) "
          "have no human labels. `consistency_protocolB.csv` reports how many of "
          "the three sampled answers differ, with a keyword proxy that is a crude "
          "heuristic, not a judgement.", "",
          ]
    for stale in ("auto_screen.csv", "fig_auto_screen.png"):
        (out / stale).unlink(missing_ok=True)
    L += ["", "## Caveats", "",
          "- Labels are human-reviewed AI drafts; disclose this in the report.",
          "- Protocol B is unlabelled and must not be reported as accuracy.",
          "- We reproduce the paper's qualitative trend, not its exact strings "
          "(0/24 exact matches on TinyStories-1M).",
          "- Our labels and the paper's colours come from different judges and "
          "different generated text, so the two are not directly comparable.",
          f"- n is small: {len(prompts)} prompts per model. Differences of a few "
          "points are not meaningful; no significance testing was done."]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    try:
        shown = out.relative_to(ROOT)
    except ValueError:      # --out may point outside the repo
        shown = out
    print(f"Wrote {len(list(out.iterdir()))} files to {shown}")
    for f in sorted(out.iterdir()):
        print(f"   {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
