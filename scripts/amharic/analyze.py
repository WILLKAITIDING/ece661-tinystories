"""
Statistics and figures for the Amharic experiments.

    python analyze.py --runs runs

Reads (produced by train_lm.py, eval_quality.py, judge_gemini.py and make_rating_sheet.py):
    runs/results.csv                       perplexity for every run
    runs/quality/completions.jsonl         continuations with name-drift flags
    runs/quality/judge.jsonl               Gemini judge scores
    runs/quality/rating_key.csv            which model wrote each rating item
    runs/quality/ratings/*.csv             filled-in rating sheets, one file per person (optional)

Writes into runs/quality/analysis/:
    stats.txt        confidence intervals, scratch-vs-transfer tests, judge/human agreement
    fig_size.png     perplexity by model size
    fig_data.png     perplexity by amount of training data (equal training steps)
    fig_drift.png    name drift by model size
    fig_judge.png    judge scores by model size

Uncertainty is computed over the 50 story openings (the continuations of one opening are not
independent): 95% intervals by bootstrap, p-values by a paired sign-flip permutation test.
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = ["grammar", "creativity", "consistency", "naturalness"]
SIZES = ["1M", "3M", "8M"]
C = {"scratch": "#2a78d6", "transfer": "#e8710a"}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
rng = np.random.default_rng(0)


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired_p(d, n=20000):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    flips = rng.choice([-1, 1], (n, len(d)))
    return float(((flips * d).mean(1).__abs__() >= obs - 1e-12).mean())


def spearman(a, b):
    def rank(v):
        v = np.asarray(v, float); o = v.argsort(); r = np.empty(len(v)); r[o] = np.arange(len(v))
        for u in np.unique(v):
            r[v == u] = r[v == u].mean()
        return r
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def per_prompt(recs, value):
    """model -> {prompt_id: mean value over that opening's continuations}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in recs:
        acc[r["model"]][r["prompt_id"]].append(value(r))
    return {m: {p: float(np.mean(v)) for p, v in d.items()} for m, d in acc.items()}


def style(ax, ylabel, title):
    ax.set_facecolor(SURF); ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def two_lines(ax, xs, ys, lo=None, hi=None, fmt="{:.1f}", label_ends=True):
    """ys: {'scratch': [...], 'transfer': [...]}; optional lo/hi with the same shape."""
    pos = np.arange(len(xs))
    for k in ("scratch", "transfer"):
        y = np.array(ys[k], float)
        ax.plot(pos, y, color=C[k], linewidth=2, marker="o", markersize=7, markeredgecolor=SURF,
                markeredgewidth=2, label="from scratch" if k == "scratch" else "from English checkpoint")
        if lo is not None:
            ax.vlines(pos, lo[k], hi[k], color=C[k], linewidth=1.2, alpha=0.6)
    if label_ends:
        for i in range(len(xs)):
            a_, b_ = ys["scratch"][i], ys["transfer"][i]
            up, dn = ("scratch", "transfer") if a_ >= b_ else ("transfer", "scratch")
            ax.annotate(fmt.format(ys[up][i]), (pos[i], ys[up][i]), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
            ax.annotate(fmt.format(ys[dn][i]), (pos[i], ys[dn][i]), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(pos); ax.set_xticklabels(xs); ax.set_xlim(-0.35, len(xs) - 0.65); ax.margins(y=0.18)


def save(fig, path):
    fig.patch.set_facecolor(SURF); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); a = ap.parse_args()
    R = Path(a.runs); Q = R / "quality"; out = Q / "analysis"; out.mkdir(parents=True, exist_ok=True)
    L = []
    say = lambda s="": (L.append(s), print(s))

    # ---------------- perplexity ----------------
    res = list(csv.DictReader(open(R / "results.csv", encoding="utf-8")))
    full = defaultdict(list)          # (size, init) -> valid ppl over seeds, full corpus
    data = defaultdict(list)          # (n stories, init) -> valid ppl, equal-step runs
    n_full = max(int(r["train_stories"]) for r in res)
    for r in res:
        n, init, size, ppl = int(r["train_stories"]), r["init"], r["size"], float(r["valid_ppl"])
        if n == n_full:
            full[(size, init)].append(ppl)
            if size == "3M":
                data[(n, init)].append(ppl)
        elif r["run"].endswith("_long"):
            data[(n, init)].append(ppl)
    say("PERPLEXITY on held-out synthetic stories (mean over seeds; n = number of seeds)")
    for s in SIZES:
        sc, tr = full[(s, "scratch")], full[(s, "transfer")]
        if sc and tr:
            say(f"  {s}: scratch {np.mean(sc):6.2f} (n={len(sc)})  transfer {np.mean(tr):6.2f} (n={len(tr)})  "
                f"change {np.mean(tr) / np.mean(sc) - 1:+.1%}")
    sizes = [s for s in SIZES if full[(s, "scratch")] and full[(s, "transfer")]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    two_lines(ax, sizes, {k: [np.mean(full[(s, k)]) for s in sizes] for k in C})
    style(ax, "validation perplexity (lower is better)", "Perplexity by model size"); ax.set_xlabel("model size", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED); save(fig, out / "fig_size.png")

    ns = sorted({n for n, _ in data})
    ns = [n for n in ns if data[(n, "scratch")] and data[(n, "transfer")]]
    if ns:
        say("\nDATA SCALING at 3M, equal training steps (validation perplexity)")
        for n in ns:
            say(f"  {n:6d} stories: scratch {np.mean(data[(n, 'scratch')]):6.2f}  transfer {np.mean(data[(n, 'transfer')]):6.2f}")
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        two_lines(ax, [f"{n:,}" for n in ns], {k: [np.mean(data[(n, k)]) for n in ns] for k in C})
        style(ax, "validation perplexity (lower is better)", "Perplexity by training-set size (3M model)")
        ax.set_xlabel("training stories", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED); save(fig, out / "fig_data.png")

    # ---------------- name drift ----------------
    comp = [json.loads(l) for l in open(Q / "completions.jsonl", encoding="utf-8")]
    drift = per_prompt([c for c in comp if c["model"] != "reference"], lambda c: float(bool(c.get("new_names"))))
    name = lambda s, k: f"am_{s}_{k}"
    say("\nNAME DRIFT (share of continuations with a new character name), 95% interval over openings")
    dm, dlo, dhi = ({k: [] for k in C} for _ in range(3))
    dsizes = [s for s in SIZES if name(s, "scratch") in drift and name(s, "transfer") in drift]
    for s in dsizes:
        pids = sorted(drift[name(s, "scratch")])
        v = {k: np.array([drift[name(s, k)][p] for p in pids]) for k in C}
        for k in C:
            m, lo, hi = boot_ci(v[k]); dm[k].append(100 * m); dlo[k].append(100 * lo); dhi[k].append(100 * hi)
        d = v["transfer"] - v["scratch"]; m, lo, hi = boot_ci(d)
        say(f"  {s}: scratch {v['scratch'].mean():.1%}  transfer {v['transfer'].mean():.1%}  "
            f"difference {m:+.1%} [{lo:+.1%}, {hi:+.1%}]  p={paired_p(d):.4f}")
    if dsizes:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        two_lines(ax, dsizes, dm, dlo, dhi, fmt="{:.0f}%")
        style(ax, "continuations with a new name, % (lower is better)", "Character-name drift by model size")
        ax.set_xlabel("model size", color=MUTED, fontsize=9); ax.set_ylim(0, 108)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower left"); save(fig, out / "fig_drift.png")

    # ---------------- judge ----------------
    jp = Q / "judge.jsonl"
    judge = [json.loads(l) for l in open(jp, encoding="utf-8")] if jp.exists() else []
    if judge:
        say("\nGEMINI JUDGE, mean score 1-10 with 95% interval over openings")
        pp = {k: per_prompt(judge, lambda r, k=k: r[k]) for k in KEYS}
        for m in sorted(pp[KEYS[0]]):
            say(f"  {m:18s} " + "  ".join("{} {:.2f} [{:.2f}, {:.2f}]".format(k[:5], *boot_ci(list(pp[k][m].values()))) for k in KEYS))
        say("\nJUDGE: transfer minus scratch (paired by opening)")
        jsizes = [s for s in SIZES if name(s, "scratch") in pp[KEYS[0]] and name(s, "transfer") in pp[KEYS[0]]]
        for s in jsizes:
            for k in KEYS:
                A, B = pp[k][name(s, "scratch")], pp[k][name(s, "transfer")]
                d = np.array([B[p] - A[p] for p in sorted(set(A) & set(B))]); m, lo, hi = boot_ci(d)
                say(f"  {s} {k:12s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  p={paired_p(d):.4f}")
        # does the judge's consistency score agree with the automatic drift flag?
        flag = {(c["model"], c["prompt_id"], c["k"]): bool(c.get("new_names")) for c in comp}
        yes = [r["consistency"] for r in judge if r["model"] != "reference" and flag.get((r["model"], r["prompt_id"], r["k"]))]
        no = [r["consistency"] for r in judge if r["model"] != "reference" and flag.get((r["model"], r["prompt_id"], r["k"])) is False]
        if yes and no:
            say(f"\nJUDGE vs NAME DRIFT: mean consistency {np.mean(no):.2f} when no new name appears (n={len(no)}), "
                f"{np.mean(yes):.2f} when one does (n={len(yes)})")
        if jsizes:
            fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
            for ax, k in zip(axes, KEYS):
                ys, lo, hi = ({i: [] for i in C} for _ in range(3))
                for s in jsizes:
                    for i in C:
                        m, l_, h_ = boot_ci(list(pp[k][name(s, i)].values())); ys[i].append(m); lo[i].append(l_); hi[i].append(h_)
                two_lines(ax, jsizes, ys, lo, hi, label_ends=False)
                style(ax, "judge score (1-10)" if k == KEYS[0] else "", k.capitalize()); ax.set_ylim(1, 10)
                if "reference" in pp[k]:
                    ref = np.mean(list(pp[k]["reference"].values()))
                    ax.axhline(ref, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
                    ax.annotate(f"Gemini {ref:.1f}", (len(jsizes) - 1, ref), xytext=(0, -11), textcoords="offset points",
                                ha="right", fontsize=8, color=MUTED)
            axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="center left")
            fig.supxlabel("model size", color=MUTED, fontsize=9); save(fig, out / "fig_judge.png")

    # ---------------- human ratings ----------------
    HK = ["grammar", "consistency", "naturalness"]
    sheets = sorted((Q / "ratings").glob("*.csv")) if (Q / "ratings").exists() else []
    if sheets and (Q / "rating_key.csv").exists():
        key = {r["item"]: r for r in csv.DictReader(open(Q / "rating_key.csv", encoding="utf-8"))}
        raters = {}
        for f in sheets:
            d = {}
            for r in csv.DictReader(open(f, encoding="utf-8-sig")):
                try:
                    d[r["item"].strip()] = {k: float(r[k]) for k in HK}
                except (ValueError, KeyError, AttributeError):
                    continue                                   # row not filled in
            raters[f.stem] = d
        say(f"\nHUMAN RATINGS (1-5): {len(raters)} rater(s): " + ", ".join(f"{n} ({len(d)} items)" for n, d in raters.items()))
        bym = defaultdict(lambda: defaultdict(list)); item_mean = defaultdict(dict)
        for it in key:
            got = [d[it] for d in raters.values() if it in d]
            if got:
                for k in HK:
                    item_mean[it][k] = float(np.mean([g[k] for g in got])); bym[key[it]["model"]][k].append(item_mean[it][k])
        for m in sorted(bym):
            say(f"  {m:18s} n={len(bym[m][HK[0]]):3d}  " + "  ".join(f"{k} {np.mean(bym[m][k]):.2f}" for k in HK))
        names = list(raters)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sh = sorted(set(raters[names[i]]) & set(raters[names[j]]))
                if len(sh) >= 5:
                    say(f"  agreement {names[i]} vs {names[j]} on {len(sh)} shared items (Spearman): " + "  ".join(
                        f"{k} {spearman([raters[names[i]][t][k] for t in sh], [raters[names[j]][t][k] for t in sh]):.2f}" for k in HK))
        jmap = {(r["model"], str(r["prompt_id"])): r for r in judge if r["k"] == 0}
        pairs = [(item_mean[it], jmap[(key[it]["model"], str(key[it]["prompt_id"]))]) for it in item_mean
                 if (key[it]["model"], str(key[it]["prompt_id"])) in jmap]
        if len(pairs) >= 5:
            say(f"  human vs Gemini judge on {len(pairs)} items (Spearman): " + "  ".join(
                f"{k} {spearman([h[k] for h, _ in pairs], [j[k] for _, j in pairs]):.2f}" for k in HK))
    else:
        say("\nHUMAN RATINGS: none found yet. Put each person's filled-in sheet, saved as CSV, in "
            f"{Q / 'ratings'}/ (one file per person) and run this again.")

    (out / "stats.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {out}/stats.txt and figures")


if __name__ == "__main__":
    main()
