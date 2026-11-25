"""
Gemini as a judge of the model continuations (the TinyStories paper's GPT-Eval, Section 3).

    export GEMINI_API_KEY=...        # or OPENAI_BASE_URL + OPENAI_API_KEY; the endpoint in use is printed
    python judge_gemini.py runs/quality/completions.jsonl --out runs/quality

Reads the continuations written by eval_quality.py. For each one, the judge sees the story with
the opening and the continuation marked, and scores the continuation 1-10 on grammar, creativity
and consistency (same three scores as the paper), plus naturalness of the Amharic.

By default it judges the six full-corpus models and the reference, 5 continuations per opening
(about 1,550 calls). Finished items are skipped, so it is safe to stop and re-run.

  judge.jsonl         one line per judged continuation
  judge_summary.csv   average scores per model
"""
from __future__ import annotations
import argparse, csv, json, os, re, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_MODELS = ["reference", "am_1M_scratch", "am_1M_transfer", "am_3M_scratch", "am_3M_transfer",
                  "am_8M_scratch", "am_8M_transfer"]
DEFAULT_MODELS_EN = ["reference", "en_1M_scratch", "en_3M_scratch", "en_8M_scratch"]
KEYS = ["grammar", "creativity", "consistency", "naturalness"]

PROMPT = """You are grading an exercise by a student who is learning to write children's stories in {language}.
The student was given the beginning of a story and had to write the rest.
The beginning (given to the student) is between <beginning> tags. The student's part is between <student> tags.

<beginning>
{opening}
</beginning>

<student>
{continuation}
</student>

Grade ONLY the student's part, as a fluent native {language} reader would. Give each score as an integer from 1 (very poor) to 10 (excellent):
- grammar: are the {language} words, verb forms, agreement and sentences correct? Penalise broken or non-existent words.
- creativity: is the continuation imaginative, or is it bland and generic?
- consistency: does it follow from the beginning? Same characters (no new names appearing from nowhere, no swapped names or genders), same situation, and events that make sense together.
- naturalness: does it read like {language} {who} would write for a small child, not like a translation?

Reply with JSON only, in exactly this form:
{{"grammar": 0, "creativity": 0, "consistency": 0, "naturalness": 0, "comment": "one short sentence in English"}}"""


def parse(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        s = {k: int(d[k]) for k in KEYS}
    except Exception:
        return None
    if not all(1 <= v <= 10 for v in s.values()):
        return None
    s["comment"] = str(d.get("comment", ""))[:300]
    return s


def make_caller(model: str, effort: str):
    from openai import OpenAI
    # Same rule as generate_stories.py: GEMINI_API_KEY first, then OPENAI_BASE_URL + OPENAI_API_KEY.
    other = os.environ.get("OPENAI_BASE_URL")
    if os.environ.get("GEMINI_API_KEY"):
        print("endpoint: Gemini API (GEMINI_API_KEY)" + ("; OPENAI_BASE_URL is set but not used" if other else ""))
        client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif other and os.environ.get("OPENAI_API_KEY"):
        u = urlsplit(other)
        print(f"endpoint: {u.scheme}://{u.hostname}{u.path} (OPENAI_BASE_URL)")
        client = OpenAI()
    else:
        raise SystemExit("set GEMINI_API_KEY (or OPENAI_BASE_URL and OPENAI_API_KEY for another OpenAI-compatible endpoint)")

    def call(prompt: str):
        last = ""
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                   temperature=0, max_tokens=4000,
                                                   extra_body={"reasoning_effort": effort})
                msg = r.choices[0].message if r.choices else None
                s = parse(msg.content if msg else "")
                if s:
                    u = r.usage
                    s["tokens_in"] = getattr(u, "prompt_tokens", 0) or 0
                    s["tokens_out"] = (getattr(u, "total_tokens", 0) or 0) - s["tokens_in"]
                    return s
                last = "unreadable reply"
            except Exception as e:
                last = str(e)[:200]
                if "quota" in last.lower():
                    time.sleep(30)
            time.sleep(2 * (attempt + 1))
        return {"error": last}
    return call


def summarize(path: Path, out: Path):
    agg = defaultdict(lambda: defaultdict(list))
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        for k in KEYS:
            agg[r["model"]][k].append(r[k])
    rows = [{"model": m, "n": len(v[KEYS[0]]), **{k: round(sum(v[k]) / len(v[k]), 2) for k in KEYS}}
            for m, v in sorted(agg.items())]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n"] + KEYS); w.writeheader(); w.writerows(rows)
    print(f"\n{'model':28s} {'n':>5s} " + " ".join(f"{k:>12s}" for k in KEYS))
    for r in rows:
        print(f"{r['model']:28s} {r['n']:5d} " + " ".join(f"{r[k]:12.2f}" for k in KEYS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("completions"); ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="gemini-3.1-pro-preview")
    ap.add_argument("--effort", default="low", help="reasoning effort for the judge")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--language", default="Amharic", choices=["Amharic", "English"])
    ap.add_argument("--k", type=int, default=5, help="continuations per opening to judge (max 10)")
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="judge only this many items (for a first test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jpath = out / "judge.jsonl"
    if a.summary_only:
        return summarize(jpath, out / "judge_summary.csv")

    if a.models is None:
        a.models = DEFAULT_MODELS if a.language == "Amharic" else DEFAULT_MODELS_EN
    who = "an Ethiopian adult" if a.language == "Amharic" else "a native English-speaking adult"
    recs = [json.loads(l) for l in open(a.completions, encoding="utf-8")]
    have = {r["model"] for r in recs}
    missing = [m for m in a.models if m not in have]
    if missing:
        raise SystemExit(f"not in completions file: {missing}")
    pids = list(dict.fromkeys(r["prompt_id"] for r in recs))[:a.n_prompts]
    todo = [r for r in recs if r["model"] in a.models and r["prompt_id"] in pids and r["k"] < a.k]
    done = set()
    if jpath.exists():
        done = {(r["model"], r["prompt_id"], r["k"]) for r in map(json.loads, open(jpath, encoding="utf-8"))}
    todo = [r for r in todo if (r["model"], r["prompt_id"], r["k"]) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"judge: {a.judge} | to judge: {len(todo)} | already done: {len(done)}")
    if not todo:
        return summarize(jpath, out / "judge_summary.csv")

    call = make_caller(a.judge, a.effort)
    lock = threading.Lock(); n = {"ok": 0, "fail": 0, "tin": 0, "tout": 0}
    f = open(jpath, "a", encoding="utf-8")

    def work(r):
        s = call(PROMPT.format(opening=r["opening"], continuation=r["continuation"], language=a.language, who=who))
        with lock:
            if "error" in s:
                n["fail"] += 1
                if n["fail"] <= 5:
                    print("  failed:", s["error"])
            else:
                n["ok"] += 1; n["tin"] += s["tokens_in"]; n["tout"] += s["tokens_out"]
                f.write(json.dumps({"model": r["model"], "prompt_id": r["prompt_id"], "k": r["k"],
                                    "judge": a.judge, **s}, ensure_ascii=False) + "\n"); f.flush()
            if (n["ok"] + n["fail"]) % 50 == 0:
                print(f"  {n['ok'] + n['fail']}/{len(todo)}  failed {n['fail']}  "
                      f"tokens in {n['tin']:,} out {n['tout']:,}")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    f.close()
    print(f"done: {n['ok']} judged, {n['fail']} failed (re-run to retry) | tokens in {n['tin']:,} out {n['tout']:,}")
    summarize(jpath, out / "judge_summary.csv")


if __name__ == "__main__":
    main()
