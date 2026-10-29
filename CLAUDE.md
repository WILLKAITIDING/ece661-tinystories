# CLAUDE.md — instructions for AI coding agents

Read this before changing anything. It should be enough to work here without
any chat history.

## Project

**Tiny Language Models: Architecture Exploration and Cross-Lingual Transfer to
Amharic** — Duke ECE 661, Fall 2026, **Team 32**.

Three phases: (1) replicate TinyStories evaluation, (2) compare tiny
architectures, (3) English→Amharic transfer. Authoritative scope lives in
`docs/proposal.md` (transcribed from the proposal PDF in the parent directory;
**the PDF wins** on any disagreement).

Paper: TinyStories, Eldan & Li, https://arxiv.org/abs/2305.07759
Checkpoints: https://huggingface.co/roneneldan

## Team and ownership

| Member | Area | Paths |
|---|---|---|
| **Will Ding** | Phase 1 replication + all evaluation | `inference/`, `evaluation/`, `io/` |
| **Rachel Odonkor** | Phase 2 architectures + training | `models/`, `training/` |
| **Luel Hagos** | Phase 3 Amharic data, tokenizer, training | `data/` |

**Do not modify a teammate's module unless the task is explicitly about it.**
If a shared interface needs to change, say so in your summary rather than
quietly refactoring. `io/records.py` is a **contract**: other people's code
reads those field names. Additive changes only.

## Current status

**Task 1 is done.** Tasks 2–5 are not started. Do not implement them
speculatively.

| Task | Owner | Status |
|---|---|---|
| 1 — Load checkpoint, generate, save JSONL | Will | **Done** |
| 2 — Figures 9–11 + 20 new prompts | Will | Not started |
| 3 — ROUGE (Section 4.4.1, Figures 14–17) | Will | Not started |
| 4 — Gemini judge | Will | Not started |
| Phase 2 — architectures | Rachel | Not started |
| Phase 3 — Amharic | Luel | Not started |

## Layout

```
configs/        YAML experiment configs (task1.yaml)
docs/           proposal.md, collaboration.md, experiment_protocol.md
notebooks/      thin demos only — logic belongs in src/
results/        git-ignored except results/examples/
scripts/        run_task1.py (CLI runner), smoke_test.py (health check)
src/tinystories_project/
  config.py       YAML -> dataclasses; paths resolve against project root
  devices.py      cpu/cuda/mps resolution — never hardcode cuda
  seeding.py      seed_everything, derive_seed
  inference/
    interface.py  GenerationParams, ModelMetadata, GeneratedText, TextGenerator
    loader.py     load_causal_lm -> ModelBundle; parameter counting
    generate.py   HFTextGenerator, generate_stories
  io/records.py   GenerationRecord schema v1.0 + JSONL read/write
  evaluation/ models/ training/ data/   placeholders for later phases
tests/          unit tests (fast, offline) + integration tests (opt-in)
```

## The shared interface — the most important design rule

Evaluation code must work on **any** model: released TinyStories checkpoints,
Rachel's custom Transformers, her minGRU/Mamba baselines, and Luel's Amharic
checkpoints.

**Never assume a model has Hugging Face `.generate()`.** The contract is the
`TextGenerator` protocol in `inference/interface.py`:

```python
@property
def metadata(self) -> ModelMetadata: ...
def generate(self, prompts, params, seed=None) -> list[GeneratedText]: ...
```

`HFTextGenerator` is the reference implementation. If a future architecture
needs an adapter, **document the expected interface rather than writing
speculative adapters.**

Loading (`loader.py`) stays separate from generation (`generate.py`).

## Verified checkpoint facts (checked 2026-10-28; do not re-guess)

- All 14 `roneneldan/TinyStories-*` repos are `GPTNeoForCausalLM`, public, ungated.
- `gpt_neo` is native to Transformers → **`trust_remote_code` is NOT needed.**
  Do not add it. If some future checkpoint seems to need it, ask the user first
  and explain why.
- Weights are `pytorch_model.bin` only (no safetensors). Loads fine under
  transformers 5.21.
- Tokenizer: GPT-2 BPE, vocab 50257, **`pad_token` is null** → must be set
  before batching. Verified identical to `EleutherAI/gpt-neo-125M`
  (`verify_tokenizer_matches`).
- Loading logs `attn.attention.bias` / `masked_bias` as **UNEXPECTED**. These
  are obsolete causal-mask buffers from transformers 4.28, not learned weights.
  **Harmless — do not try to "fix" it.**
- `TinyStories-1M` measures **3,745,984 params total / 398,464 non-embedding**.
  The name matches neither. **Always report measured counts, never the name.**

## Two correctness traps

1. **Decoder-only models must be LEFT-padded.** Right padding silently produces
   garbage for every prompt shorter than the longest in the batch — no error,
   just bad data. `prepare_tokenizer_for_generation` forces this. Never undo it.
2. **`completion` must contain only NEW text.** ROUGE and the judge both break
   if the prompt is echoed. There are tests for this; keep them.

## Standards

- Python 3.11+. Type hints on public functions. Small modules.
- **No hardcoded absolute paths.** Use `config.resolve_path` / `find_project_root`.
- **No hardcoded `cuda`.** Use `devices.resolve_device`.
- Experiment settings come from `configs/*.yaml`, never from literals in code.
- Record seeds, model ids, and measured parameter counts with every result.
- Don't silently change an experiment setting. Add a new config instead.

## Testing

```bash
pytest                      # unit tests: fast, offline, tiny random model
pytest -m integration       # real checkpoint; downloads ~20 MB
python scripts/smoke_test.py
```

- Unit tests must not hit the network. Use the fixtures in `tests/conftest.py`.
- **Run the tests after implementing anything.** Report real counts and real
  output — never claim something works without having executed it. If a
  download or a test fails, say so plainly and leave the code ready to retry.

## Security and budget — hard rules

- **Do not call the course Gemini API** until the judge task is explicitly
  started. It spends a **shared $250 team budget** split with TPU compute.
- **Do not use course TPU resources** without being asked to.
- **Never commit secrets.** `.env` is git-ignored; `.env.example` holds
  placeholder names only. Never print a key value, and never put one in a URL,
  header, log line, or commit.
- Keep checkpoints and datasets out of git (`.gitignore` covers them).
- Do not push, create repos, or change collaborator access without explicit
  approval from the user.
- Do not modify files outside this repository without asking.

## Reporting

State what you actually ran and what it actually printed. Distinguish
**completed** / **unverified** / **blocked**. Document limitations and
assumptions rather than papering over them.
