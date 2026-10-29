"""Custom architectures (Rachel, Phase 2). NOT IMPLEMENTED YET.

Planned: GPT-Neo baseline, deep-narrow vs shallow-wide, a modernised
Transformer (RoPE + RMSNorm + GeGLU + GQA), and a parameter-matched
non-attention baseline (minGRU or Mamba), at ~5M and ~10M parameters.

To evaluate a model from here with the Phase 1 pipeline, expose a class
satisfying `inference.interface.TextGenerator`. Models that do not implement
the Hugging Face `.generate()` API are explicitly supported - that is why the
protocol exists. See the docstring in `inference/interface.py`.
"""
