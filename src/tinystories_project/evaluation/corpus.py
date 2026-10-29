"""Loading the TinyStories corpus for the Section 4.4.1 experiments.

Kept separate from `rouge.py` so the ROUGE maths stays testable without any
dataset download, and so Luel can point the same code at an Amharic corpus.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def local_paths(repo_id: str, files: Sequence[str]) -> list[Path]:
    """Resolve dataset files, downloading them to the HF cache if needed."""
    return [Path(hf_hub_download(repo_id, f, repo_type="dataset")) for f in files]


def read_stories(paths: Sequence[Path], column: str = "text") -> list[str]:
    """Read every story from the given parquet files into memory."""
    out: list[str] = []
    for p in paths:
        out.extend(pq.read_table(p, columns=[column]).column(column).to_pylist())
    return out


def iter_stories(paths: Sequence[Path], column: str = "text",
                 batch_size: int = 20_000) -> Iterator[str]:
    """Stream stories a batch at a time.

    Figures 16 and 17 scan the whole ~2.1M-story training corpus; streaming
    keeps peak memory at one batch instead of the entire corpus.
    """
    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=batch_size,
                                                    columns=[column]):
            yield from batch.column(column).to_pylist()


def sample_stories(stories: Sequence[str], n: int, seed: int,
                   min_words: int = 60) -> list[tuple[int, str]]:
    """Pick `n` stories at random, reproducibly, skipping ones that are too short.

    Returns (index_in_corpus, story) so a sample can be traced back to its row.
    A story shorter than `min_words` would leave a prompt of only a few words,
    which is not a meaningful test of completion.
    """
    eligible = [i for i, s in enumerate(stories) if len(s.split()) >= min_words]
    if len(eligible) < n:
        raise ValueError(
            f"only {len(eligible)} stories have >= {min_words} words, need {n}")
    rng = random.Random(seed)
    return [(i, stories[i]) for i in sorted(rng.sample(eligible, n))]
