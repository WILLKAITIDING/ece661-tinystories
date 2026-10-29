"""Device selection and reproducible seeding.

Both concern *how a run executes* rather than what it computes, so they live
together. Never hardcode `cuda`: this project is developed on Apple Silicon
(MPS) and will later run on course GPU machines.
"""

from __future__ import annotations

import logging
import random

import torch

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cpu", "cuda", "mps")
MAX_SEED = 2**31 - 1


# --- devices ---------------------------------------------------------------

def available_devices() -> list[str]:
    """Accelerators actually usable on this machine, best first."""
    found = []
    if torch.cuda.is_available():
        found.append("cuda")
    if torch.backends.mps.is_available():
        found.append("mps")
    found.append("cpu")
    return found


def resolve_device(requested: str = "auto") -> torch.device:
    """Turn a device string into a concrete `torch.device`.

    `auto` picks CUDA, then MPS, then CPU. An explicitly requested accelerator
    that is unavailable falls back to CPU with a warning rather than crashing,
    so a config written on a GPU box still runs on a laptop.
    """
    requested = (requested or "auto").lower()
    if requested not in VALID_DEVICES:
        raise ValueError(f"Unknown device {requested!r}. Expected one of {VALID_DEVICES}.")

    usable = available_devices()
    if requested == "auto":
        return torch.device(usable[0])
    if requested not in usable:
        logger.warning(
            "Device %r requested but unavailable (have: %s). Falling back to CPU.",
            requested, ", ".join(usable),
        )
        return torch.device("cpu")
    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    """Human-readable description, recorded in output metadata."""
    if device.type == "cuda":
        idx = device.index or 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


# --- seeding ---------------------------------------------------------------

def seed_everything(seed: int) -> int:
    """Seed Python, NumPy (if present), and Torch RNGs. Returns the seed."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")

    random.seed(seed)
    torch.manual_seed(seed)  # MPS reads the global torch seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    return seed


def batch_seed(base_seed: int, batch_index: int) -> int:
    """Seed for one batch, so a run can be stopped and resumed deterministically.

    `model.generate()` accepts no `torch.Generator`, so the only way to make
    sampling reproducible is to seed globals before each batch.
    """
    return (base_seed + batch_index) % (MAX_SEED + 1)
