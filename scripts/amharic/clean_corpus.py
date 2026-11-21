"""
Clean the generated Amharic stories and write train/validation splits.

    python clean_corpus.py corpus/amharic_tinystories.jsonl --out corpus/clean

Drops: truncated stories, stories with non-Ge'ez characters, stories missing the words line,
stories that are too short/long or not 3-5 paragraphs, exact duplicates, and near-duplicates
(same opening sentence, or very high word-5-gram overlap with an earlier story).

Writes into --out:
    train.jsonl / valid.jsonl   full records (id, story, sampled inputs)
    train.txt   / valid.txt     one story per block, blank line between stories (for the tokenizer / LM)
    report.txt                  what was dropped and why, plus repetition statistics
The split is by a hash of the story id, so it is stable when more stories are added later.
"""

import argparse, hashlib, json, re, sys
from collections import Counter
from pathlib import Path

GEEZ_OK = re.compile(r"^[ሀ-፿\s«»\"'!?.,:;()\-–—…0-9]+$")
TOK = re.compile(r"[ሀ-ፚ]+")


def norm(text: str) -> str:
    return " ".join(TOK.findall(text))


def first_sentence(text: str) -> str:
    return norm(re.split(r"[።!?]|፡፡", text, maxsplit=1)[0])


def shingles(words, n=5):
    return {hash(tuple(words[i:i + n])) for i in range(max(0, len(words) - n + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--out", default="clean")
    ap.add_argument("--valid-frac", type=float, default=0.02)
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument("--near-dup", type=float, default=0.5, help="5-gram Jaccard above this = near-duplicate")
    ap.add_argument("--max-same-opening", type=int, default=0,
                    help="keep at most this many stories per identical first sentence (0 = no limit). "
                         "Stock openings are normal in TinyStories-style data, so the default keeps them.")
    a = ap.parse_args()

    recs, bad_lines, seen_ids = [], 0, set()
    for line in open(a.inp, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            bad_lines += 1
            continue
        if r["id"] in seen_ids:
            continue
        seen_ids.add(r["id"])
        recs.append(r)
    recs.sort(key=lambda r: r["id"])

    drop = Counter()
    kept, seen_text, seen_open, index = [], set(), Counter(), {}
    for r in recs:
        s = (r.get("story") or "").strip()
        s = re.sub(r"[ \t]+\n", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        r["story"] = s
        paras = [p for p in s.split("\n\n") if p.strip()]
        if r.get("truncated") or (r.get("usage") or {}).get("finish_reason") not in (None, "stop"):
            drop["truncated"] += 1; continue
        if not s or not GEEZ_OK.match(s):
            drop["non_geez_or_empty"] += 1; continue
        if len(r.get("words_used") or []) != 3:
            drop["no_words_line"] += 1; continue
        if not (a.min_chars <= len(s) <= a.max_chars):
            drop["length"] += 1; continue
        if not (3 <= len(paras) <= 5):
            drop["paragraph_count"] += 1; continue
        if any(n not in s for n in r.get("names") or []):
            drop["sampled_name_missing"] += 1; continue
        key = norm(s)
        if key in seen_text:
            drop["exact_duplicate"] += 1; continue
        op = first_sentence(s)
        if a.max_same_opening and seen_open[op] >= a.max_same_opening:
            drop["opening_over_limit"] += 1; continue
        words = key.split()
        sh = shingles(words)
        # candidate earlier stories sharing any 5-gram (inverted index keeps this fast)
        cand = Counter()
        for h in sh:
            for j in index.get(h, ()):
                cand[j] += 1
        dup = False
        for j, inter in cand.most_common(3):
            other = kept[j]["_sh"]
            if inter / max(1, len(sh | other)) > a.near_dup:
                dup = True; break
        if dup:
            drop["near_duplicate"] += 1; continue
        r["_sh"] = sh
        for h in sh:
            index.setdefault(h, []).append(len(kept))
        seen_text.add(key); seen_open[op] += 1
        kept.append(r)

    for r in kept:
        r.pop("_sh", None)

    def is_valid(i):
        h = int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 10_000
        return h < a.valid_frac * 10_000

    train = [r for r in kept if not is_valid(r["id"])]
    valid = [r for r in kept if is_valid(r["id"])]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("valid", valid)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n\n\n".join(r["story"] for r in part) + "\n")

    chars = sum(len(r["story"]) for r in kept)
    words = sum(len(TOK.findall(r["story"])) for r in kept)
    vocab = Counter(w for r in kept for w in TOK.findall(r["story"]))
    first = Counter(TOK.findall(r["story"])[0] for r in kept)
    names = Counter(n for r in kept for n in r.get("names") or [])
    sett = Counter(r.get("setting", "") for r in kept)
    feats = Counter(f for r in kept for f in r.get("features") or [])
    think = sum(((r.get("usage") or {}).get("thought_tokens") or 0) for r in recs)
    tin = sum(((r.get("usage") or {}).get("input_tokens") or 0) for r in recs)
    tout = sum(((r.get("usage") or {}).get("output_tokens") or 0) for r in recs)

    L = []
    L.append(f"input records: {len(recs)}   unreadable lines: {bad_lines}")
    L.append(f"kept: {len(kept)} ({len(kept) / max(1, len(recs)):.1%})   train {len(train)}   valid {len(valid)}")
    L.append("dropped: " + (", ".join(f"{k} {v}" for k, v in drop.most_common()) or "none"))
    L.append(f"characters: {chars:,}   words: {words:,}   distinct word forms: {len(vocab):,}")
    L.append(f"mean chars/story: {chars / max(1, len(kept)):.0f}   mean words/story: {words / max(1, len(kept)):.0f}")
    L.append(f"generation tokens: in {tin:,}  out {tout:,}  thinking {think:,}  "
             f"=> ~${tin * 0.75e-6 + (tout + think) * 3.75e-6:.2f} at 3.8-flash standard rates")
    L.append("top opening words: " + ", ".join(f"{w} {c / len(kept):.1%}" for w, c in first.most_common(8)))
    rep = sum(c for c in seen_open.values() if c > 1)
    L.append(f"opening sentences: {len(seen_open):,} distinct; {rep / max(1, len(kept)):.1%} of stories share theirs with another; "
             + "most common: " + " | ".join(f"{o} ({c})" for o, c in seen_open.most_common(5)))
    L.append("top names: " + ", ".join(f"{w} {c}" for w, c in names.most_common(6))
             + f"   (distinct: {len(names)})")
    L.append("settings used: " + f"{len(sett)} distinct; most common {sett.most_common(1)[0][1] / len(kept):.1%}" if sett else "")
    L.append("features: " + ", ".join(f"{k} {v / len(kept):.0%}" for k, v in feats.most_common()))
    L.append("most frequent words: " + " ".join(w for w, _ in vocab.most_common(40)))
    report = "\n".join(L)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
