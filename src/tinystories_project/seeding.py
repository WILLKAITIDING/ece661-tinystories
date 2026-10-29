"""Reproducibility helpers.

Hugging Face `model.generate()` does not accept a `torch.Generator`, so the only
way to make sampling reproducible is to seed the global RNGs immediately before
each call. `derive_seed` gives every batch its own deterministic seed derived
from one base seed, so a run can be stopped and resumed without repeating or
re-rolling samples.
"""

from __future__ import annotations

import hashlib
import random

import torch

MAX_SEED = 2**31 - 1


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy (if installed), and Torch RNGs. Returns the seed."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS reads from the global torch seed, so manual_seed above covers it.

    try:  # NumPy is not a Task 1 dependency; seed it only if present.
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    return seed


def derive_seed(base_seed: int, *parts: object) -> int:
    """Deterministically derive a sub-seed from a base seed and some labels.

    Stable across processes and Python versions (unlike `hash()`), so
    `derive_seed(42, "batch", 3)` is the same value on every machine.
    """
    payload = "|".join([str(base_seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (MAX_SEED + 1)
