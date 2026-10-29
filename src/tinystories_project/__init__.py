"""ECE 661 Team 32 - Tiny Language Models.

Shared library for all three project phases. The package is deliberately split
so that one person's work does not block another's:

    inference/   generation interfaces + checkpoint loading   (Will, Phase 1)
    evaluation/  qualitative, ROUGE, and LLM-judge scoring    (Will, Phase 1)
    io/          on-disk record schemas and (de)serialization (shared)
    models/      custom architectures                         (Rachel, Phase 2)
    training/    training loops and configs                   (Rachel, Phase 2)
    data/        corpus generation, filtering, tokenizers     (Luel, Phase 3)

Only `inference` and `io` are implemented at this point (Task 1).
"""

__version__ = "0.1.0"

from tinystories_project.devices import resolve_device
from tinystories_project.seeding import seed_everything

__all__ = ["__version__", "resolve_device", "seed_everything"]
