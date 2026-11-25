"""
Build the English comparison set: a random sample of TinyStories the same size as the Amharic corpus.

    python prepare_english.py --out data/english

Downloads the original TinyStories training file from Hugging Face (about 2 GB, once), keeps
clean stories of 100-220 words, and takes a random 48,452 for training and 983 for validation
(the same counts as the Amharic split). Also builds a list of the 100 most common character
names, which the name-drift measure needs.

Writes into --out:
    clean/train.jsonl, valid.jsonl, train.txt, valid.txt    same format as the Amharic files
    en_names.csv                                             name, stories (how many stories use it)
    report.txt

Why 100-220 words: Amharic packs more into each word, so the Amharic stories (about 100 words)
correspond to roughly 150 English words. This keeps the stories about the same length in content.
"""
from __future__ import annotations
import argparse, csv, json, random, re
from collections import Counter
from pathlib import Path

OK = re.compile(r"^[A-Za-z0-9\s.,!?;:'\"()\-]+$")
MID_CAP = re.compile(r"[a-z,;] ([A-Z][a-z]+)\b")          # capitalised word in the middle of a sentence
LATIN = re.compile(r"[A-Za-z]+")
NOT_NAMES = set("""Mom Dad Mommy Daddy Mama Papa Mum Mummy Momma Grandma Grandpa Granny Grandmother Grandfather
Nana Aunt Auntie Uncle Mr Mrs Ms Miss Sir Doctor Dr Teacher Farmer King Queen Prince Princess Captain Officer
God Santa Christmas Halloween Easter Birthday English Monday Tuesday Wednesday Thursday Friday Saturday Sunday
January February March April May June July August September October November December
Yes No Oh Wow Hello Hi Thank Thanks Please Okay Ok Look Let Can What Why Do You We It The A I He She They But And
So Then Now Maybe Sorry Come Go Stop Wait Help Good Bye Goodbye Hooray Yay Ouch Hey Well Sure Of That This There
Earth Moon Sun Sky America""".split())


def stories_from_hub():
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download("roneneldan/TinyStories", "TinyStories-train.txt", repo_type="dataset")
        return stories_from_file(p)
    except Exception as e:
        print(f"could not fetch TinyStories-train.txt ({str(e)[:80]}); trying the datasets library")
        from datasets import load_dataset
        return (r["text"] for r in load_dataset("roneneldan/TinyStories", split="train"))


def stories_from_file(path):
    buf = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                yield "".join(buf); buf = []
            else:
                buf.append(line)
    if buf:
        yield "".join(buf)


def clean(s: str):
    for a_, b_ in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("—", "-"), ("–", "-"), ("…", "...")):
        s = s.replace(a_, b_)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", s) if p.strip()]
    return "\n\n".join(paras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=None, help="local TinyStories-train.txt (default: download)")
    ap.add_argument("--n-train", type=int, default=48452); ap.add_argument("--n-valid", type=int, default=983)
    ap.add_argument("--min-words", type=int, default=100); ap.add_argument("--max-words", type=int, default=220)
    ap.add_argument("--n-names", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); (out / "clean").mkdir(parents=True, exist_ok=True)

    seen, pool, total, drop = set(), [], 0, Counter()
    for raw in (stories_from_file(a.source) if a.source else stories_from_hub()):
        total += 1
        s = clean(raw)
        if not s or not OK.match(s):
            drop["odd_characters_or_empty"] += 1; continue
        n = len(LATIN.findall(s))
        if not (a.min_words <= n <= a.max_words):
            drop["length"] += 1; continue
        if s in seen:
            drop["duplicate"] += 1; continue
        seen.add(s); pool.append(s)
    need = a.n_train + a.n_valid
    if len(pool) < need:
        raise SystemExit(f"only {len(pool)} usable stories, need {need}")
    random.Random(a.seed).shuffle(pool)
    chosen = pool[:need]

    # character names: capitalised mid-sentence words, by number of stories they appear in
    cnt = Counter()
    for s in chosen:
        cnt.update(set(MID_CAP.findall(s)) - NOT_NAMES)
    names = [n for n, _ in cnt.most_common(a.n_names)]
    with open(out / "en_names.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name", "stories"]); w.writerows((n, cnt[n]) for n in names)

    recs = []
    for i, s in enumerate(chosen):
        latin = set(LATIN.findall(s))
        recs.append({"id": i, "story": s, "names": [n for n in names if n in latin]})
    parts = {"train": recs[:a.n_train], "valid": recs[a.n_train:]}
    for name, part in parts.items():
        with open(out / "clean" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / "clean" / f"{name}.txt").write_text("\n\n\n".join(r["story"] for r in part) + "\n", encoding="utf-8")

    words = [len(LATIN.findall(s)) for s in chosen]
    rep = [f"TinyStories stories read: {total:,}   usable: {len(pool):,}   dropped: " + ", ".join(f"{k} {v:,}" for k, v in drop.most_common()),
           f"sampled: train {len(parts['train']):,}   valid {len(parts['valid']):,}   (seed {a.seed})",
           f"words: {sum(words):,}   mean words/story: {sum(words) / len(words):.0f}   mean chars/story: {sum(map(len, chosen)) / len(chosen):.0f}",
           f"stories with at least one listed name: {sum(bool(r['names']) for r in recs) / len(recs):.1%}",
           "top names: " + ", ".join(f"{n} {cnt[n]}" for n in names[:15]),
           "last names on the list (check these are really names): " + ", ".join(names[-15:])]
    (out / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep)); print(f"wrote {out}/clean/ and {out}/en_names.csv")


if __name__ == "__main__":
    main()
