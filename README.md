# ece661-tinystories

**Tiny Language Models: Architecture Exploration and Cross-Lingual Transfer to Amharic**
Duke ECE 661 — Fall 2026 — **Team 32**

Will Ding · Rachel Odonkor · Luel Hagos

Reproduces the evaluation from [TinyStories (Eldan & Li, 2023)](https://arxiv.org/abs/2305.07759),
compares tiny architectures, and tests English→Amharic transfer.
Full scope: [`docs/proposal.md`](docs/proposal.md).

---

## Quick start

```bash
python3 -m venv .venv                 # Python 3.11+
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebook]"

python scripts/smoke_test.py          # downloads ~20 MB on first run
```

Expected final line: `All 9 checks passed. Task 1 pipeline is working.`

## Run Task 1

```bash
python scripts/run_task1.py --config configs/task1.yaml
```

Override anything from the CLI (overrides are recorded in the run manifest):

```bash
python scripts/run_task1.py --config configs/task1.yaml \
    --model-id roneneldan/TinyStories-33M \
    --prompt "Once upon a time, there was a little girl named Lily." \
    --max-new-tokens 200 --temperature 1.0 --num-return-sequences 10 --seed 42
```

Or from Python / a notebook:

```python
from tinystories_project.inference import load_causal_lm, generate_stories

bundle = load_causal_lm("roneneldan/TinyStories-1M", device="auto")
stories = generate_stories(
    bundle,
    ["Once upon a time, there was a little girl named Lily."],
    max_new_tokens=200, temperature=1.0, top_p=0.95, seed=42,
)
print(stories[0].completion)
```

See [`notebooks/task1_demo.ipynb`](notebooks/task1_demo.ipynb) for a walkthrough.

## Example output

`roneneldan/TinyStories-1M` on Apple M1 Pro (MPS), seed 42, temperature 1.0, top-p 0.95:

> **Prompt:** Once upon a time, there was a little girl named Lily.
>
> **Continuation:** She loved to play outside in the dirt and trees. One day, she saw a shiny rock at her house. She asked her mommy, "What is that?" / Mommy replied, "I'm making a cool rock in the sand. It's very special!"

## Status

| Task | Owner | Status |
|---|---|---|
| 1 — Load checkpoint, generate, save | Will | ✅ **Done** |
| 2 — Figures 9–11 + 20 new prompts | Will | ⬜ Not started |
| 3 — ROUGE (Section 4.4.1, Figures 14–17) | Will | ⬜ Not started |
| 4 — Gemini judge | Will | ⬜ Not started |
| Phase 2 — architecture exploration | Rachel | ⬜ Not started |
| Phase 3 — Amharic transfer | Luel | ⬜ Not started |

## Layout

```
configs/      YAML experiment configs
docs/         proposal, collaboration guide, experiment protocol
notebooks/    thin demos — logic lives in src/
results/      generated outputs (git-ignored except results/examples/)
scripts/      run_task1.py, smoke_test.py
src/tinystories_project/
              config, devices, seeding
              inference/  interface, loader, generate
              io/         record schema + JSONL
              evaluation/ models/ training/ data/   (later phases)
tests/        unit tests (offline) + integration tests (opt-in)
```

## Using your own model with this pipeline

Rachel and Luel: your models do **not** need Hugging Face `.generate()`.
Implement the `TextGenerator` protocol
([`inference/interface.py`](src/tinystories_project/inference/interface.py)) and
every evaluation in this repo works on your model unchanged:

```python
class MyMinGRUGenerator:
    @property
    def metadata(self) -> ModelMetadata: ...
    def generate(self, prompts, params, seed=None) -> list[GeneratedText]: ...
```

Details in [`docs/collaboration.md`](docs/collaboration.md).

## Output format

One JSONL line per completion, schema v1.0, documented in
[`io/records.py`](src/tinystories_project/io/records.py). Each record carries the
prompt, the completion (new text only), generation params, seed, timestamp, and
full model metadata including **measured** parameter counts. Every run also
writes a `*_manifest.json` with the config, CLI overrides, git commit, and
library versions.

## Verified checkpoint facts

Checked against the Hugging Face API on 2026-10-28 — not assumed:

- All 14 `roneneldan/TinyStories-*` repos are `GPTNeoForCausalLM`, public, ungated.
- `gpt_neo` is native to Transformers, so **`trust_remote_code` is not required** and is not used.
- Weights ship as `pytorch_model.bin`; tokenizer is GPT-2 BPE, vocab 50257, with **`pad_token` null** (we set it to EOS).
- The bundled tokenizer is verified identical to `EleutherAI/gpt-neo-125M`.
- `TinyStories-1M` measures **3,745,984 parameters total**, of which 3,347,520 are the embedding table — leaving **398,464 non-embedding**. The "1M" in the name matches neither; we report measured counts everywhere.

## Testing

```bash
pytest                   # 90 unit tests, offline, ~2 s
pytest -m integration    # 12 tests against the real checkpoint
```

## Requirements

Python 3.11+, PyTorch, Transformers. Runs on CUDA, Apple Silicon (MPS), or CPU —
the device is resolved at runtime and never hardcoded. Developed on an M1 Pro
using MPS.

## Security

Never commit API keys, checkpoints, or datasets. `.env` is git-ignored;
`.env.example` lists placeholder names only. The course Gemini API spends a
shared **$250** team budget and **is not called by any code in this repo yet**.
