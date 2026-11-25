"""
Automatic quality measures for the trained Amharic models (no API needed).

    python eval_quality.py --runs runs --valid data/clean/valid.jsonl --train data/clean/train.jsonl \
        --names names.csv --out runs/quality

For each model in --runs (every folder that has best/), it takes the same held-out story openings,
lets the model continue each one several times, and measures:

  name_drift   share of continuations that bring in a character name (from the 100-name list)
               that was not in the opening. Lower is better.
  real_words   share of generated words that occur somewhere in the training corpus.
               Catches broken word forms. Higher is better.
  finished     share of continuations that reach the end-of-story token before the length limit.
  words        average continuation length in words.

The row "reference" is Gemini's own continuation of the same openings, measured the same way.
It shows what a perfect model would score (the measures are not exactly 0% / 100% even then).

Writes into --out:
  quality.csv         one row per model
  completions.jsonl   every opening and continuation (also the input for the judge and rating sheet)
"""
from __future__ import annotations
import argparse, csv, json, random, re
from collections import Counter
from pathlib import Path
import torch
from transformers import GPTNeoForCausalLM
from common import SPTok, normalize, PAD, EOS, WORD, GEEZ

SENT_END = re.compile(r"(?<=[።!?»])\s+")
SENT_END_EN = re.compile(r"(?<=[.!?\"”])\s+")
LATIN = re.compile(r"[A-Za-z]+")


def present(names, text: str) -> list[str]:
    """Names that occur in the text. Amharic names take prefixes and suffixes (ለአበበ, አበበን), so they are
    matched anywhere inside a word. English names must be a whole word (Tom, not Tomorrow)."""
    latin = set(LATIN.findall(text))
    return [n for n in names if (n in latin if n.isascii() else n in text)]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def make_opening(story: str, names: list[str], frac=0.4):
    """Cut at a sentence end near `frac` of the story. Keep only cuts where every character
    of the story has already been named, so any new name in a continuation is a real drift."""
    flat = story
    cuts = [m.end() for m in (SENT_END if GEEZ.search(flat) else SENT_END_EN).finditer(flat)]
    cuts = [c for c in cuts if 0.25 * len(flat) <= c <= 0.6 * len(flat)]
    cuts.sort(key=lambda c: abs(c - frac * len(flat)))
    for c in cuts:
        head = flat[:c].rstrip()
        if len(present(names, head)) == len(names):
            return head, flat[c:].strip()
    return None


def new_names(text: str, opening: str, name_list: list[str]) -> list[str]:
    seen = set(present(name_list, opening))
    return [n for n in present(name_list, text) if n not in seen]


def measure(conts: list[dict], vocab: set, name_list: list[str]) -> dict:
    drift = words = known = fin = 0
    for c in conts:
        nn = new_names(c["continuation"], c["opening"], name_list)
        c["new_names"] = nn
        drift += bool(nn)
        ws = WORD.findall(c["continuation"])
        words += len(ws); known += sum(w in vocab for w in ws)
        fin += bool(c.get("finished", True))
    n = max(1, len(conts))
    return {"n": len(conts), "name_drift": round(drift / n, 4), "real_words": round(known / max(1, words), 4),
            "finished": round(fin / n, 4), "words": round(words / n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--valid", required=True, help="clean/valid.jsonl")
    ap.add_argument("--train", required=True, help="clean/train.jsonl")
    ap.add_argument("--names", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", help="run names to evaluate (default: all)")
    ap.add_argument("--n-prompts", type=int, default=50); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    all_names = [r["name"] for r in csv.DictReader(open(a.names, encoding="utf-8"))]
    train = read_jsonl(a.train)
    vocab = set()
    appears, sampled = Counter(), Counter()
    for r in train:
        s = normalize(r["story"]); vocab.update(WORD.findall(s))
        for n in present(all_names, s):
            appears[n] += 1; sampled[n] += n in (r.get("names") or [])
    # Some names are also ordinary words or parts of words. Leave out any name that often
    # shows up in stories where it was not a character, so it cannot cause false alarms.
    ambiguous = [n for n in all_names if (appears[n] - sampled[n]) > 0.005 * len(train)]
    name_list = [n for n in all_names if n not in ambiguous]
    print(f"names used for drift: {len(name_list)} of {len(all_names)}"
          + (f" (left out as ordinary words: {' '.join(ambiguous)})" if ambiguous else ""))

    rng = random.Random(a.seed)
    valid = read_jsonl(a.valid); rng.shuffle(valid)
    prompts = []
    for r in valid:
        names = r.get("names") or []
        if not names or any(n in ambiguous for n in names):
            continue
        cut = make_opening(normalize(r["story"]), names)
        if cut:
            prompts.append({"prompt_id": r["id"], "opening": cut[0], "reference": cut[1]})
        if len(prompts) == a.n_prompts:
            break
    print(f"openings: {len(prompts)}  x {a.k} continuations per model")

    rows, allc = [], []
    ref = [{"model": "reference", "prompt_id": p["prompt_id"], "k": 0, "opening": p["opening"],
            "continuation": p["reference"], "finished": True} for p in prompts]
    rows.append({"model": "reference", **measure(ref, vocab, name_list)}); allc += ref

    runs = sorted(d for d in Path(a.runs).iterdir() if (d / "best" / "config.json").exists())
    if a.models:
        runs = [d for d in runs if d.name in a.models]
    for d in runs:
        tok = SPTok(d / "best" / "tokenizer.model")
        model = GPTNeoForCausalLM.from_pretrained(d / "best").to(device).eval()
        torch.manual_seed(a.seed)
        conts = []
        for p in prompts:
            ids = tok.encode(p["opening"], eos=False)
            x = torch.tensor([ids], device=device)
            with torch.no_grad():
                y = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=a.max_new_tokens,
                                   do_sample=True, temperature=a.temperature, top_p=a.top_p,
                                   num_return_sequences=a.k, pad_token_id=PAD, eos_token_id=EOS)
            for j, seq in enumerate(y.tolist()):
                new = seq[len(ids):]
                conts.append({"model": d.name, "prompt_id": p["prompt_id"], "k": j, "opening": p["opening"],
                              "continuation": tok.decode(new), "finished": EOS in new})
        row = {"model": d.name, **measure(conts, vocab, name_list)}
        rows.append(row); allc += conts
        print(f"{d.name:28s} drift {row['name_drift']:.1%}  real words {row['real_words']:.1%}  "
              f"finished {row['finished']:.1%}  words {row['words']}")
        del model

    with open(out / "quality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "completions.jsonl", "w", encoding="utf-8") as f:
        for c in allc:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nreference (the original story's own continuation): drift {rows[0]['name_drift']:.1%}  real words {rows[0]['real_words']:.1%}")
    print(f"wrote {out}/quality.csv and {out}/completions.jsonl")


if __name__ == "__main__":
    main()
