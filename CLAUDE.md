# CLAUDE.md — instructions for AI coding agents

Enough to work here without any chat history. Read before changing anything.

## Project

**Tiny Language Models: Architecture Exploration and Cross-Lingual Transfer to
Amharic** — Duke ECE 661 Fall 2026, **Team 32**.

Three phases: (1) replicate TinyStories evaluation, (2) compare tiny
architectures, (3) English→Amharic transfer. Authoritative scope is
`docs/proposal.md`, transcribed from the proposal PDF in the parent directory —
**the PDF wins** on any disagreement.

Paper: https://arxiv.org/abs/2305.07759 · Checkpoints: https://huggingface.co/roneneldan

## Ownership

| Member | Area | Paths |
|---|---|---|
| **Will Ding** | Phase 1 replication + all evaluation | `inference/`, `records.py`, future `evaluation/` |
| **Rachel Odonkor** | Phase 2 architectures + training | future `models/`, `training/` |
| **Luel Hagos** | Phase 3 Amharic data, tokenizer, training | future `data/` |

Those three future modules **do not exist yet** — create them when the work
starts rather than leaving empty packages around.

**Do not modify a teammate's module unless the task is about it.** If a shared
interface must change, say so in your summary instead of quietly refactoring.
`records.py` is a **contract**: other people's code reads those field names.
Additive changes only. `record_id` covers the whole generation request (model,
revision, tokenizer, prompt, sample index, seed, params) but never the output
text, so it can be computed before generating - that is what lets an
interrupted judge run resume by skipping ids a file already has. Different
seeds give different ids. Prompts must be unique within a config (enforced in
`config.py`), since two identical prompts would collide.

## Status

**Task 1 is done. Tasks 2–4 and Phases 2–3 are not started — do not implement
them speculatively.**

| Task | Owner | Status |
|---|---|---|
| 1 — Load checkpoint, generate, save JSONL | Will | **Done** |
| 2 — Figures 9–11 + 20 new prompts | Will | Not started |
| 3 — ROUGE (Section 4.4.1, Figures 14–17) | Will | Not started |
| 4 — Gemini judge | Will | Not started |
| Phase 2 / Phase 3 | Rachel / Luel | Not started |

## Layout

```
configs/task1.yaml       experiment settings (never hardcode these in code)
docs/proposal.md         authoritative scope
notebooks/               thin demos only — logic belongs in src/
results/                 git-ignored except results/examples/
scripts/run_task1.py     CLI runner
src/tinystories_project/
  config.py              YAML -> dataclasses; paths resolve to project root
  runtime.py             device resolution + seeding
  records.py             GenerationRecord schema v2.0 + JSONL read/write
  inference/
    interface.py         GenerationParams, ModelMetadata, GeneratedText,
                         TextGenerator protocol
    loader.py            load_causal_lm -> ModelBundle; parameter counting
    generate.py          HFTextGenerator, generate_stories
tests/                   test_core, test_generation, test_records, test_integration
```

## The shared interface — the most important design rule

Evaluation code must work on **any** model: released checkpoints, Rachel's
Transformers and minGRU/Mamba baselines, Luel's Amharic checkpoints.

**Never assume a model has Hugging Face `.generate()`.** The contract is the
`TextGenerator` protocol in `inference/interface.py`:

```python
@property
def metadata(self) -> ModelMetadata: ...
def generate(self, prompts, params, seed=None) -> list[GeneratedText]: ...
```

`HFTextGenerator` is the reference implementation. Requirements: return
`len(prompts) * num_return_sequences` items; `completion` holds **only** new
text; set `prompt_index`, `sample_index`, `finish_reason`. If a future
architecture needs an adapter, **document the expected interface rather than
writing speculative adapters.**

Loading (`loader.py`) stays separate from generation (`generate.py`).

## Verified checkpoint facts (checked 2026-10-28 — do not re-guess)

- All 14 `roneneldan/TinyStories-*` repos are `GPTNeoForCausalLM`, public, ungated.
- `gpt_neo` is native to Transformers → **`trust_remote_code` is NOT needed.**
  Do not add it; if some checkpoint seems to need it, ask the user and explain why.
- Weights are `pytorch_model.bin` only (no safetensors). Fine under transformers 5.21.
- Tokenizer: GPT-2 BPE, vocab 50257, **`pad_token` is null** → must be set before
  batching. Verified identical to `EleutherAI/gpt-neo-125M` (see `test_integration.py`).
- Loading logs `attn.attention.bias` / `masked_bias` as **UNEXPECTED**: obsolete
  causal-mask buffers from transformers 4.28, not learned weights. **Harmless —
  do not try to "fix" it.**
- `TinyStories-1M` measures **3,745,984 params total / 398,464 non-embedding**.
  The name matches neither. **Always report measured counts, never the name.**

## Two correctness traps

1. **Decoder-only models must be LEFT-padded.** Right padding silently produces
   garbage for every prompt shorter than the longest in the batch — no error,
   just bad data. `prepare_tokenizer_for_generation` forces it. Never undo it.
2. **`completion` must contain only NEW text.** ROUGE and the judge both break
   if the prompt is echoed. Tests guard both; keep them.

## Reproducibility, stated honestly

`model.generate()` takes no `torch.Generator`, so globals are seeded before each
batch via `runtime.batch_seed(base_seed, batch_index)`.

Identical given the same model+revision, params, prompt list **and order**,
`batch_size`, device, and library versions. **Changing `batch_size` regroups
prompts and changes sampled text** — record it with results. Greedy decoding
(`do_sample=False`) is unaffected.

**Which parameter count do we report? Both, always.** For the Phase 2 "~5M and
~10M parameter-matched" comparison the team must agree which number is being
matched **before** training. Non-embedding is the more meaningful basis, since
vocabulary size is a tokenizer choice — this matters most in Phase 3, where the
Amharic SentencePiece vocab (4K/8K) is far smaller than GPT-Neo's 50,257.

## Standards

- Python 3.11+. Type hints on public functions. Small modules.
- **No hardcoded absolute paths** (`config.resolve_path`) and **no hardcoded
  `cuda`** (`runtime.resolve_device`).
- Experiment settings live in `configs/*.yaml`, never as literals in code.
- Record seeds, model ids, and measured parameter counts with every result.
- Don't silently change an experiment setting — add a new config instead.

## Testing

```bash
pytest                   # 72 unit tests, offline, tiny random model, ~2 s
pytest -m integration    # 9 tests against the real checkpoint (~20 MB download)
```

Unit tests must not hit the network; use the fixtures in `tests/conftest.py`.
**Run them after implementing anything, and report real counts and real
output** — never claim something works without having executed it. If a
download or test fails, say so plainly and leave the code ready to retry.

## Security and budget — hard rules

- **Do not call the course Gemini API** until the judge task is explicitly
  started: it spends a **shared $250 team budget** split with TPU compute.
- **Do not use course TPU resources** unless asked.
- **Never commit secrets.** `.env` is git-ignored; `.env.example` holds
  placeholder names only. Never print a key, or put one in a URL, header, log,
  or commit.
- Keep checkpoints and datasets out of git (`.gitignore` covers them).
- Do not push, create repos, or change collaborator access without explicit
  approval. Do not modify files outside this repository without asking.

## Reporting

State what you actually ran and what it printed. Distinguish **completed** /
**unverified** / **blocked**. Document limitations rather than papering over them.
