"""
Train SentencePiece tokenizers for the Amharic corpus and compare vocabulary sizes.

    python train_tokenizer.py --train corpus/clean/train.txt --valid corpus/clean/valid.txt \
        --natural natural_stories/extracted --out tokenizers --sizes 4000 8000

For each size it writes <out>/am_<type>_<V>.model and prints, on the held-out synthetic stories
and on the human-written stories: tokens per word (fertility), characters per token, and the
share of unknown tokens. tokenizer_report.json holds the same numbers.
"""
import argparse, json, re, tempfile
from pathlib import Path
import sentencepiece as spm
from common import load_stories, to_line, SPTok, NL, UNK

WORD = re.compile(r"[ሀ-ፚ]+")


def stats(tok: SPTok, stories):
    n_tok = n_unk = n_word = n_char = 0
    for s in stories:
        ids = tok.encode(s, bos=False, eos=False)
        n_tok += len(ids); n_unk += sum(i == UNK for i in ids)
        n_word += len(WORD.findall(s)); n_char += len(s)
    return {"stories": len(stories), "tokens": n_tok, "tokens_per_word": round(n_tok / max(1, n_word), 3),
            "chars_per_token": round(n_char / max(1, n_tok), 3), "tokens_per_story": round(n_tok / max(1, len(stories)), 1),
            "unk_rate": round(n_unk / max(1, n_tok), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--natural", default=None, help="folder of human-written stories (optional)")
    ap.add_argument("--out", default="tokenizers")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4000, 8000])
    ap.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    a = ap.parse_args()

    train, valid = load_stories(a.train), load_stories(a.valid)
    natural = load_stories(a.natural) if a.natural else []
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"train {len(train)} stories | valid {len(valid)} | natural {len(natural)}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for s in train:
            f.write(to_line(s) + "\n")
        corpus = f.name

    report = {}
    for V in a.sizes:
        prefix = out / f"am_{a.model_type}_{V}"
        spm.SentencePieceTrainer.train(
            input=corpus, model_prefix=str(prefix), vocab_size=V, model_type=a.model_type,
            character_coverage=1.0, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            user_defined_symbols=[NL], input_sentence_size=2_000_000, shuffle_input_sentence=True,
            max_sentence_length=20000, num_threads=8, minloglevel=2,
        )
        tok = SPTok(f"{prefix}.model")
        r = {"synthetic_valid": stats(tok, valid)}
        if natural:
            r["natural"] = stats(tok, natural)
        r["train_tokens"] = sum(len(tok.encode(s)) for s in train)
        report[f"{a.model_type}_{V}"] = r
        print(f"\nV={V}  ({prefix}.model)   training tokens incl. BOS/EOS: {r['train_tokens']:,}")
        for k in ("synthetic_valid", "natural"):
            if k in r:
                x = r[k]
                print(f"  {k:16s} tokens/word {x['tokens_per_word']:.2f}  chars/token {x['chars_per_token']:.2f}  "
                      f"tokens/story {x['tokens_per_story']:.0f}  unk {x['unk_rate']:.3%}")
    (out / "tokenizer_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nwrote {out}/tokenizer_report.json")


if __name__ == "__main__":
    main()
