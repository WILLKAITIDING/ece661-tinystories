"""
Do the models copy the training stories? (TinyStories paper, Section 4.4)

    python copy_check.py --runs runs --train data/clean/train.jsonl

Uses the continuations already saved in runs/quality/completions.jsonl. For every continuation:

  seen_5gram     share of its 5-word sequences that occur anywhere in the training stories
  seen_8gram     the same for 8-word sequences
  longest_copy   longest run of consecutive words that appears word-for-word in training (average, and maximum)
  nearest_story  the largest share of its 5-word sequences found in any ONE training story (average),
                 and how many continuations take more than half of theirs from one story

The row "reference" is Gemini's own held-out continuations, measured the same way. Those were never
in the training set, so that row shows how much overlap comes just from common phrases.
Writes runs/quality/copy_check.csv. Needs a few GB of memory.
"""
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter, defaultdict
from pathlib import Path
from common import normalize, WORD
words = lambda s: WORD.findall(normalize(s))
grams = lambda w, n: [hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--train", required=True)
    a = ap.parse_args()
    Q = Path(a.runs) / "quality"
    idx5, set8 = defaultdict(set), set()
    n_train = 0
    for l in open(a.train, encoding="utf-8"):
        if l.strip():
            w = words(json.loads(l)["story"])
            for h in grams(w, 5):
                idx5[h].add(n_train)
            set8.update(grams(w, 8)); n_train += 1
    print(f"training stories indexed: {n_train:,}")

    acc = defaultdict(lambda: defaultdict(list))
    for l in open(Q / "completions.jsonl", encoding="utf-8"):
        c = json.loads(l); w = words(c["continuation"])
        g5, g8 = grams(w, 5), grams(w, 8)
        if not g5:
            continue
        hit = [h in idx5 for h in g5]
        run = best = 0                      # longest copied run, in words (k consecutive 5-grams = k+4 words)
        for x in hit:
            run = run + 1 if x else 0; best = max(best, run)
        per_story = Counter(s for h in g5 for s in idx5.get(h, ()))
        m = acc[c["model"]]
        m["s5"].append(sum(hit) / len(g5)); m["s8"].append(sum(h in set8 for h in g8) / max(1, len(g8)))
        m["run"].append(best + 4 if best else 0)
        m["near"].append((per_story.most_common(1)[0][1] / len(g5)) if per_story else 0.0)

    mean = lambda v: sum(v) / len(v)
    rows = [{"model": k, "n": len(v["s5"]), "seen_5gram": round(mean(v["s5"]), 4), "seen_8gram": round(mean(v["s8"]), 4),
             "longest_copy_mean": round(mean(v["run"]), 1), "longest_copy_max": max(v["run"]),
             "nearest_story_mean": round(mean(v["near"]), 4),
             "over_half_from_one_story": sum(x > 0.5 for x in v["near"])} for k, v in sorted(acc.items())]
    with open(Q / "copy_check.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\n{'model':28s} {'n':>5s} {'5-gram seen':>12s} {'8-gram seen':>12s} {'longest copy':>18s} {'nearest story':>14s} {'>50% one story':>15s}")
    for r in rows:
        print(f"{r['model']:28s} {r['n']:5d} {r['seen_5gram']:12.1%} {r['seen_8gram']:12.1%} "
              f"{r['longest_copy_mean']:7.1f} (max {r['longest_copy_max']:3d}) {r['nearest_story_mean']:14.1%} {r['over_half_from_one_story']:15d}")
    print(f"\nwrote {Q}/copy_check.csv")


if __name__ == "__main__":
    main()
