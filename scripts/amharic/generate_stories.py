"""
Generate Amharic TinyStories with Gemini.

Samples one verb, one noun, one adjective (uniformly over DISTINCT Amharic words) plus 1-3
story features, fills the prompt template, calls the model, and appends one JSON line per
story with the sampled inputs, the story, the words the model says it used, and token usage.

Usage
    export GEMINI_API_KEY=...                      # public Gemini API (default)
    python generate_stories.py --words amharic_wordlist.csv --template generation_prompt.txt \
        --n 100 --out stories.jsonl

    # any other OpenAI-compatible endpoint (same script):
    export OPENAI_BASE_URL=https://<host>/v1 OPENAI_API_KEY=...

Inputs
    --words      CSV with columns amharic,pos (pos is verb, noun or adj)
    --template   prompt text with the placeholders {VERB} {NOUN} {ADJECTIVE} {FEATURES}
    --names      CSV with columns name,gender (default: names.csv next to this script)

Resumable: re-running with the same --out skips story ids already present.
"""

import argparse, csv, json, os, random, re, sys, time
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- features
# Explicit instructions, not labels (labels were ignored in testing).
FEATURES = {
    "dialogue":      "The story contains at least one conversation between two characters, written in «».",
    "bad_ending":    "The story ends badly or sadly for the main character. Do NOT add a happy resolution or a comforting last line.",
    "moral_value":   "The story clearly teaches a simple moral lesson (for example honesty, sharing, patience, kindness), and the last sentence states the lesson in plain words a child understands.",
    "plot_twist":    "Something unexpected happens near the end that changes how the story turns out.",
    "foreshadowing": "Early in the story there is a small hint about what will happen at the end.",
    "conflict":      "Two characters want different things and disagree or argue before the problem is resolved.",
}
FEATURE_WEIGHTS = {"dialogue": 3, "bad_ending": 1, "moral_value": 2, "plot_twist": 2, "foreshadowing": 1, "conflict": 2}

WORDS_FOOTER = (
    "\n\nAfter the story, on a new line, write exactly: ቃላት: followed by the three required words "
    "in the form you used them in the story, separated by « | ». Nothing else after that line."
)

def load_words(path):
    by_pos = {"verb": [], "noun": [], "adj": []}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by_pos[r["pos"]].append(r["amharic"])
    for k, v in by_pos.items():
        assert v, f"no {k}s in {path}"
    return by_pos

def sample_features(rng):
    k = rng.choices([1, 2, 3], weights=[0.35, 0.45, 0.20])[0]
    names = list(FEATURES)
    chosen = []
    pool = names[:]
    w = [FEATURE_WEIGHTS[n] for n in pool]
    while len(chosen) < k:
        pick = rng.choices(pool, weights=w)[0]
        i = pool.index(pick); pool.pop(i); w.pop(i)
        chosen.append(pick)
    return chosen

# Character names, their relationship and the setting are sampled as well. Left to the model, a few
# names, openings and scenes dominate the corpus.
SETTINGS = [
    "at home in the living room", "in the kitchen while food is being prepared", "in the family compound (ግቢ)",
    "at the neighborhood shop (ሱቅ)", "at the market (ገበያ)", "at school or kindergarten", "on the way to school",
    "at grandmother's house in the countryside", "on a farm with animals", "by a river", "in a field (ሜዳ) near the village",
    "at church or a holiday celebration", "during a coffee ceremony at home", "in a minibus taxi or bus", "at a wedding or family feast (ድግስ)",
    "at the clinic (ሐኪም ቤት)", "at the grain mill (ወፍጮ ቤት)", "on a rainy day in the rainy season (ክረምት)", "on a hot sunny day",
    "at night before bedtime", "early in the morning", "in the forest (ጫካ) near home", "on a hill or mountain path",
    "at a neighbor's house", "in the garden behind the house (ጓሮ)", "at a playground", "in the street in front of the house",
    "at a relative's house in the city", "while herding sheep or goats", "at a holiday such as Enkutatash, Meskel, Timket or Eid",
]
RELATIONS_TWO = [   # both named characters are children
    "two friends", "siblings", "two neighbors' children", "two classmates", "two cousins",
    "an older child looking after a younger one",
]
RELATIONS_ONE = [   # one named child; the other character is not given a personal name
    "a child and their mother", "a child and their father", "a child and a grandmother or grandfather",
    "a child and an animal", "a child and an aunt or uncle", "a child and a teacher",
    "a child and a shopkeeper or neighbor", "a child on their own",
]
AVOID_OPENING = ("Do not begin the story with the words «ትንሹ» or «ትንሿ», and do not begin it with a character's name; "
                 "start in a different way (with the place, the time, an action, or something a character says).")

def load_names(path):
    if not path or not os.path.exists(path): return None
    return [(r["name"], r.get("gender", "")) for r in csv.DictReader(open(path, encoding="utf-8")) if r["name"].strip()]

def sample_cast(rng, names):
    """Returns (cast, relation, setting, instruction text). Drawn after the words and features, so the
    word triple of a story id does not depend on the name list."""
    k = rng.choices([1, 2], weights=[0.45, 0.55])[0]
    cast = rng.sample(names, k) if names else []
    relation = rng.choice(RELATIONS_TWO if k == 2 else RELATIONS_ONE)
    setting = rng.choice(SETTINGS)
    g = {"f": "girl", "m": "boy"}
    text = ""
    if cast:
        who = " and ".join(f"«{n}» ({g.get(x, 'child')})" for n, x in cast)
        text += (f"\nCharacters: {relation}. Use the name(s) {who} for the child(ren); "
                 "do not give personal names to anyone else (say እናቱ, አያቷ, መምህሩ and so on).")
    text += f"\nSetting: {setting}.\n{AVOID_OPENING}"
    return cast, relation, setting, text

def build_prompt(template, verb, noun, adj, feats, cast_text=""):
    text = " ".join(FEATURES[f] for f in feats)
    return (template.replace("{VERB}", verb).replace("{NOUN}", noun)
                    .replace("{ADJECTIVE}", adj).replace("{FEATURES}", text) + cast_text + WORDS_FOOTER)

def parse_output(text):
    text = text.strip()
    m = re.search(r"\n?\s*ቃላት\s*[:፡]\s*(.+?)\s*$", text, re.S)
    words_used = []
    if m:
        words_used = [w.strip(" «»\"'") for w in re.split(r"\||،|,|፣", m.group(1)) if w.strip()]
        text = text[: m.start()].rstrip()
    return text, words_used

def client_from_env():
    if os.environ.get("OPENAI_BASE_URL") and os.environ.get("OPENAI_API_KEY"):
        return OpenAI()
    return OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                  base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

def call(client, model, prompt, max_tokens=4000, effort="minimal"):
    """Thinking is billed as output and counts against max_tokens, so keep it minimal.
    reasoning_effort goes through extra_body so old openai packages can't silently drop it."""
    efforts = [effort] + [e for e in ("low",) if e != effort]      # Pro models reject "minimal"
    kw = dict(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    ei = 0
    for attempt in range(5):
        try:
            r = client.chat.completions.create(extra_body={"reasoning_effort": efforts[ei]}, **kw)
            choice = r.choices[0] if r.choices else None
            if choice is None or choice.message is None or not choice.message.content:
                reason = getattr(choice, "finish_reason", "no choice")   # blocked or empty reply
                print(f"    empty reply (finish_reason={reason}); retrying", file=sys.stderr)
                time.sleep(2)
                continue
            u = r.usage
            det = getattr(u, "completion_tokens_details", None)
            thought = getattr(det, "reasoning_tokens", None) if det else None
            if thought is None:                                     # endpoint reports it only via the total
                thought = max(0, u.total_tokens - u.prompt_tokens - u.completion_tokens)
            return r.choices[0].message.content or "", {
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "thought_tokens": thought,
                "total_tokens": u.total_tokens,
                "finish_reason": r.choices[0].finish_reason,
                "effort": efforts[ei],
            }
        except Exception as e:
            msg = str(e)
            if ("reasoning" in msg.lower() or "thinking" in msg.lower()) and ei + 1 < len(efforts):
                ei += 1; continue
            wait = 10 * (attempt + 1)
            print(f"    error ({msg[:90]}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--out", default="stories.jsonl", help="output JSONL; existing story ids are skipped")
    ap.add_argument("--words", required=True, help="CSV with columns amharic,pos")
    ap.add_argument("--names", default=str(HERE / "names.csv"), help="CSV with columns name,gender (f/m); skipped if missing")
    ap.add_argument("--template", required=True, help="prompt template file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait after each call, per worker")
    ap.add_argument("--start", type=int, default=0, help="first story id")
    ap.add_argument("--effort", default="minimal", help="thinking level: minimal | low | medium | high")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests")
    a = ap.parse_args()

    import threading
    from concurrent.futures import ThreadPoolExecutor

    words = load_words(a.words)
    names = load_names(a.names)
    print(f"words: { {k: len(v) for k, v in words.items()} } | names: {len(names) if names else 'none (model picks names)'}")
    template = Path(a.template).read_text(encoding="utf-8")
    client = client_from_env()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    todo = [i for i in range(a.start, a.start + a.n) if i not in done]
    print(f"{len(todo)} stories to generate ({len(done)} already in {out})")

    GEEZ_ONLY = re.compile(r"^[\u1200-\u137F\s«»\"'!?.,:;()\-–—…0-9]+$")
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "in": 0, "out": 0, "think": 0, "trunc": 0}
    f = out.open("a", encoding="utf-8")
    t0 = time.time()

    def work(i):
        rng = random.Random(a.seed * 1_000_003 + i)             # per-story seed -> reproducible
        verb, noun, adj = (rng.choice(words["verb"]), rng.choice(words["noun"]), rng.choice(words["adj"]))
        feats = sample_features(rng)
        cast, relation, setting, cast_text = sample_cast(rng, names)
        prompt = build_prompt(template, verb, noun, adj, feats, cast_text)
        raw, usage = call(client, a.model, prompt, effort=a.effort)
        if raw is None:
            with lock: stats["fail"] += 1
            print(f"[{i}] FAILED", file=sys.stderr); return
        story, used = parse_output(raw)
        rec = {
            "id": i, "model": a.model, "seed": a.seed,
            "verb": verb, "noun": noun, "adj": adj, "features": feats,
            "names": [n for n, _ in cast], "relation": relation, "setting": setting,
            "story": story, "words_used": used,
            "n_chars": len(story), "n_paragraphs": story.count("\n\n") + 1 if story else 0,
            "geez_only": bool(GEEZ_ONLY.match(story)) if story else False,
            "truncated": usage.get("finish_reason") != "stop" or not used,
            "usage": usage, "ts": time.time(),
        }
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            stats["ok"] += 1; stats["in"] += usage["input_tokens"]; stats["out"] += usage["output_tokens"]
            stats["think"] += usage["thought_tokens"] or 0; stats["trunc"] += rec["truncated"]
            n = stats["ok"]
            if a.workers == 1 or n % 50 == 0 or n == len(todo):
                cost = stats["in"] * 0.75e-6 + (stats["out"] + stats["think"]) * 3.75e-6
                rate = n / max(time.time() - t0, 1e-9)
                print(f"[{n}/{len(todo)}] last id {i}: {verb} / {noun} / {adj} | {','.join(feats)} | "
                      f"{rec['n_paragraphs']}p {rec['n_chars']}ch | think {usage['thought_tokens']} | "
                      f"~${cost:.2f} so far, {rate*60:.0f}/min, trunc {stats['trunc']}, fail {stats['fail']}")
        if a.sleep: time.sleep(a.sleep)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    f.close()
    cost = stats["in"] * 0.75e-6 + (stats["out"] + stats["think"]) * 3.75e-6
    print(f"\n{stats['ok']} stories written to {out} | failed {stats['fail']} | truncated {stats['trunc']} | "
          f"tokens in {stats['in']:,} out {stats['out']:,} think {stats['think']:,} | est. cost ${cost:.2f} (3.8-flash standard rates)")

if __name__ == "__main__":
    main()
