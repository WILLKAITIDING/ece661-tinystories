"""Device selection.

Never hardcode `cuda`. This project is developed on Apple Silicon (MPS) and will
later run on course GPU/TPU machines, so device choice is resolved at runtime.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def available_devices() -> list[str]:
    """Return the accelerators actually usable on this machine, best first."""
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
        raise ValueError(
            f"Unknown device {requested!r}. Expected one of {VALID_DEVICES}."
        )

    usable = available_devices()
    if requested == "auto":
        return torch.device(usable[0])

    if requested not in usable:
        logger.warning(
            "Device %r was requested but is not available on this machine "
            "(available: %s). Falling back to CPU.",
            requested,
            ", ".join(usable),
        )
        return torch.device("cpu")

    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    """Human-readable device description, recorded in output metadata."""
    if device.type == "cuda":
        idx = device.index or 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
