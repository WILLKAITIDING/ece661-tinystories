"""ECE 661 Team 32 - Tiny Language Models.

Shared library for all three project phases:

    inference/   generation interfaces + checkpoint loading   (Will, Phase 1)
    records.py   the JSONL schema every task reads and writes (shared)
    runtime.py   device selection and seeding                 (shared)
    config.py    YAML experiment configs                      (shared)

Rachel's architectures and Luel's Amharic data get their own modules when that
work starts; see the README for the layout and the TextGenerator contract.
"""

__version__ = "0.1.0"
