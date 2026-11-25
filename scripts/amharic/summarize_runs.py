"""
Collect every runs/*/metrics.json into one table (printed and saved as results.csv).

    python summarize_runs.py runs
"""
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
rows = []
for f in sorted(root.glob("*/metrics.json")):
    m = json.loads(f.read_text())
    a, fin, h = m["args"], m["final"], m["history"]
    best = min(h, key=lambda r: r["valid_loss"])
    rows.append({
        "run": f.parent.name,
        "init": "transfer" if a.get("init") else "scratch",
        "size": (a["init"].split("-")[-1] if a.get("init") else a["size"]),
        "train_stories": m["train_stories"], "train_tokens": m["train_tokens"],
        "params_total": m["params"]["total"], "params_non_emb": m["params"]["non_embedding"],
        "valid_loss": round(fin["synthetic_valid"]["loss"], 4), "valid_ppl": round(fin["synthetic_valid"]["ppl"], 2),
        "valid_bpb": round(fin["synthetic_valid"]["bits_per_byte"], 4),
        "natural_loss": round(fin["natural"]["loss"], 4) if "natural" in fin else "",
        "natural_ppl": round(fin["natural"]["ppl"], 1) if "natural" in fin else "",
        "natural_bpb": round(fin["natural"]["bits_per_byte"], 4) if "natural" in fin else "",
        "best_step": best["step"], "last_step": h[-1]["step"], "minutes": h[-1]["minutes"],
    })
if not rows:
    sys.exit(f"no metrics.json found under {root}")
cols = list(rows[0])
with open(root / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
show = ["run", "init", "size", "train_stories", "params_total", "valid_loss", "valid_ppl", "valid_bpb", "natural_ppl", "natural_bpb", "best_step", "minutes"]
width = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in show}
print("  ".join(c.ljust(width[c]) for c in show))
for r in rows:
    print("  ".join(str(r[c]).ljust(width[c]) for c in show))
print(f"\nwrote {root / 'results.csv'}")
