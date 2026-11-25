"""
Build a blind rating sheet for the native-speaker evaluation.

    python make_rating_sheet.py runs/quality/completions.jsonl --out runs/quality \
        --models am_1M_scratch am_3M_scratch am_8M_scratch am_3M_transfer am_8M_transfer

Takes the first 30 openings and, for each chosen model plus the reference, one continuation.
Rows are shuffled and the model name is hidden.

  rating_sheet.csv   item, opening, continuation, grammar, consistency, naturalness  (fill in 1-5)
  rating_key.csv     item -> model  (do not open until the ratings are done)
"""
import argparse, csv, json, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("completions"); ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--n-prompts", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

recs = [json.loads(l) for l in open(a.completions, encoding="utf-8")]
have = {r["model"] for r in recs}
missing = [m for m in a.models if m not in have]
if missing:
    raise SystemExit(f"not in completions file: {missing}\navailable: {sorted(have)}")
systems = ["reference"] + a.models
pids = list(dict.fromkeys(r["prompt_id"] for r in recs))[:a.n_prompts]
items = [r for r in recs if r["k"] == 0 and r["model"] in systems and r["prompt_id"] in pids]
random.Random(a.seed).shuffle(items)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "rating_sheet.csv", "w", newline="", encoding="utf-8-sig") as f, \
     open(out / "rating_key.csv", "w", newline="", encoding="utf-8") as g:
    w = csv.writer(f); k = csv.writer(g)
    w.writerow(["item", "opening", "continuation", "grammar", "consistency", "naturalness"])
    k.writerow(["item", "model", "prompt_id"])
    for i, r in enumerate(items, 1):
        w.writerow([i, r["opening"], r["continuation"], "", "", ""])
        k.writerow([i, r["model"], r["prompt_id"]])
print(f"{len(items)} items ({len(pids)} openings x {len(systems)} systems) -> {out}/rating_sheet.csv")
