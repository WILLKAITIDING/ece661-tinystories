"""Shared helpers for the Amharic tokenizer / training / generation scripts."""
from __future__ import annotations
import json, re
from pathlib import Path

NL = "<nl>"            # paragraph break, kept as one token (SentencePiece drops raw newlines)
PAD, UNK, BOS, EOS = 0, 1, 2, 3
GEEZ = re.compile(r"[ሀ-ፚ]")
WORD = re.compile(r"[ሀ-ፚ]+|[A-Za-z]+")      # a word: Ge'ez letters, or Latin letters (English baseline)


def load_stories(path: str | Path) -> list[str]:
    """Stories from: a .jsonl with a 'story' field, a .txt with stories separated by a blank
    line pair (as written by clean_corpus.py), or a directory with one story per .txt file."""
    p = Path(path)
    if p.is_dir():
        out = [f.read_text(encoding="utf-8").strip() for f in sorted(p.rglob("*.txt"))]
    elif p.suffix == ".jsonl":
        out = [json.loads(l)["story"].strip() for l in p.open(encoding="utf-8") if l.strip()]
    else:
        out = [s.strip() for s in p.read_text(encoding="utf-8").split("\n\n\n")]
    return [s for s in out if s]


def normalize(text: str) -> str:
    """Light, meaning-preserving normalization so human-written text matches the synthetic corpus:
    the old two-colon full stop (፡፡) becomes ።, every style of double quote becomes « », ዉ becomes ው."""
    if not GEEZ.search(text):                   # not Amharic (English baseline): leave untouched
        return text
    text = text.replace("፡፡", "።").replace("ዉ", "ው")
    for a_, b_ in (("‹‹", "«"), ("››", "»"), ("<<", "«"), (">>", "»"), ("“", "«"), ("”", "»"), ("„", "«")):
        text = text.replace(a_, b_)
    for q in ("''", '"'):                       # unpaired straight quotes: alternate open / close
        parts = text.split(q)
        if len(parts) > 1:
            text = "".join(p + ("" if i == len(parts) - 1 else ("«" if i % 2 == 0 else "»")) for i, p in enumerate(parts))
    return text


def to_line(story: str) -> str:
    """One story on one line, paragraph breaks as the <nl> symbol."""
    story = normalize(story)
    paras = [" ".join(p.split()) for p in story.split("\n\n") if p.strip()]
    return f" {NL} ".join(paras)


def from_line(text: str) -> str:
    return "\n\n".join(p.strip() for p in text.split(NL)).strip()


class SPTok:
    """Thin wrapper around a SentencePiece model with BOS/EOS and paragraph handling."""

    def __init__(self, model_path: str | Path):
        import sentencepiece as spm
        self.path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.path)
        self.vocab_size = self.sp.get_piece_size()
        assert (self.sp.pad_id(), self.sp.unk_id(), self.sp.bos_id(), self.sp.eos_id()) == (PAD, UNK, BOS, EOS)

    def encode(self, story: str, bos: bool = True, eos: bool = True) -> list[int]:
        ids = self.sp.encode(to_line(story))
        return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids if int(i) not in (PAD, BOS, EOS)]
        return from_line(self.sp.decode(ids))
