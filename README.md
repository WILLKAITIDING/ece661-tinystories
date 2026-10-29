# ece661-tinystories

**Tiny Language Models: Architecture Exploration and Cross-Lingual Transfer to Amharic**
Duke ECE 661 — Fall 2026 — **Team 32** · Will Ding · Rachel Odonkor · Luel Hagos

Reproduces the evaluation from [TinyStories (Eldan & Li, 2023)](https://arxiv.org/abs/2305.07759),
compares tiny architectures, and tests English→Amharic transfer.
Full scope: [`docs/proposal.md`](docs/proposal.md). Engineering reference: [`CLAUDE.md`](CLAUDE.md).

## Quick start

```bash
python3 -m venv .venv                 # Python 3.11+
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebook]"

pytest -m integration                 # downloads ~20 MB; proves the pipeline works
python scripts/run_task1.py --config configs/task1.yaml
```

Task 2 (qualitative replication, ~4 min on an M1 Pro, ~720 MB of checkpoints):

```bash
pip install -e ".[analysis]"
python scripts/run_task2.py --config configs/task2.yaml
python scripts/make_task2_tables.py      # tables, annotation CSV, figures
```

Task 3 (Section 4.4.1 ROUGE replication, ~16 min to generate + ~10 min to scan
the corpus; downloads ~1 GB of TinyStories data):

```bash
pip install -e ".[task3,analysis]"
python scripts/run_task3.py --config configs/task3.yaml
python scripts/make_task3_tables.py --config configs/task3.yaml
```

Runs on CUDA, Apple Silicon (MPS), or CPU — resolved at runtime, never hardcoded.
Developed on an M1 Pro using MPS.

From Python or a notebook:

```python
from tinystories_project.inference import load_causal_lm, generate_stories

bundle = load_causal_lm("roneneldan/TinyStories-1M", device="auto")
stories = generate_stories(
    bundle, ["Once upon a time, there was a little girl named Lily."],
    max_new_tokens=200, temperature=1.0, top_p=0.95, seed=42,
)
print(stories[0].completion)
```

See [`notebooks/task1_demo.ipynb`](notebooks/task1_demo.ipynb).

> **Example** (TinyStories-1M, seed 42, T=1.0, top-p 0.95)
> *Prompt:* Once upon a time, there was a little girl named Lily.
> *Continuation:* She loved to play outside in the dirt and trees. One day, she saw a shiny rock at her house. She asked her mommy, "What is that?"

## Status

| Task | Owner | Status |
|---|---|---|
| 1 — Load checkpoint, generate, save | Will | ✅ **Done** |
| 2 — Figures 9–11 + 20 new prompts | Will | ✅ **Done** |
| 3 — ROUGE (Section 4.4.1, Figures 14–17) | Will | ✅ **Done** |
| 4 — Gemini judge | Will | ⬜ Not started |
| Phase 2 — architecture exploration | Rachel | ⬜ Not started |
| Phase 3 — Amharic transfer | Luel | ⬜ Not started |

## Layout

```
configs/              task1.yaml, task2.yaml      docs/proposal.md   authoritative scope
data/prompts/         paper_figures_9_11.json (24), custom_20.json (20)
data/reference/       the paper's OWN completions and verdicts (not ours)
scripts/              run_task1, run_task2, make_task2_tables, extract_paper_figures
results/              git-ignored except results/examples/
src/tinystories_project/
  config.py  runtime.py  records.py   inference/{interface,loader,generate}.py
                                      evaluation/{prompts,qualitative}.py
tests/                test_core, test_generation, test_records, test_task2,
                      test_integration
```

## Task 2 at a glance

44 prompts (24 recovered from the paper's Figures 9-11, 20 written by us) run
across four checkpoints under two protocols: **A** greedy, reproducing the
paper's zero-temperature setting, and **B** three sampled completions per prompt
at T=1.0 / top-p 0.95, answering whether output is *consistently* good.

All 176 greedy completions were labelled correct / partial / incorrect, the
same three-way scheme the paper colour-codes with. **Strict accuracy
(correct / 44):**

| checkpoint | layers | hidden | factual | reasoning | context | **overall** |
|---|---:|---:|---:|---:|---:|---:|
| TinyStories-1M | 8 | 64 | 38% | 6% | 17% | **20%** |
| TinyStories-8M | 8 | 256 | 62% | 56% | 42% | **55%** |
| TinyStories-33M | 4 | 768 | 88% | 62% | 67% | **73%** |
| TinyStories-1Layer-21M | 1 | 1024 | 56% | 31% | 25% | **39%** |

Accuracy rises monotonically with size, and the shallow-but-wide model shows
the largest factual-minus-context gap (+31 pts vs +21 for every deeper model) —
the width/depth dissociation of Section 4.2.

Those labels were **drafted by an AI assistant and reviewed by Will**, who
overrode 9 and accepted 167 — human-reviewed AI drafts, not independent human
annotation. Details in [`data/annotations/`](data/annotations/); result tables
in [`results/examples/task2/`](results/examples/task2/).

## Using your own model with this pipeline

Rachel and Luel: your models do **not** need Hugging Face `.generate()`.
Implement the `TextGenerator` protocol
([`inference/interface.py`](src/tinystories_project/inference/interface.py)) and every
evaluation here works on your model unchanged:

```python
class MyMinGRUGenerator:
    @property
    def metadata(self) -> ModelMetadata: ...
    def generate(self, prompts, params, seed=None) -> list[GeneratedText]: ...
```

`completion` must hold **only** new text — never echo the prompt. Full contract
in the module docstring.

## Working together

| Branch | Purpose |
|---|---|
| `main` | Stable. Never commit directly. |
| `phase1-eval-will` / `phase2-arch-rachel` / `phase3-amharic-luel` | One per phase |

```bash
git checkout main && git pull
git checkout -b phase2-arch-rachel/mingru
pytest                                      # must pass before pushing
git commit -am "feat(models): add minGRU baseline"
git push -u origin phase2-arch-rachel/mingru   # then open a PR into main
```

In the PR say what changed, what you ran, and the actual output. Get one review
from whoever owns the code you touched; squash and merge. **Never commit**
checkpoints, datasets, `.env`, or API keys — `.gitignore` covers these, and
`.env.example` holds placeholder names only. Don't refactor a teammate's module
to suit your own; raise it at the weekly sync instead.

## Task 3 at a glance

Section 4.4.1's four measures, implemented from the paper's formulas rather
than as generic ROUGE. 100 story prefixes per split (train and validation),
cut at 40%, completed greedily by the same four checkpoints.

| measure | what it asks | result |
|---|---|---|
| Fig 14 | is the generation inside the real ending? | mean 0.13–0.17 |
| Fig 15 | are the 100 generations alike? | 1M 0.27 → 33M 0.14; real stories 0.09 |
| Fig 16 | how common are the generated 2-grams? | ~0.2% occur nowhere in the corpus |
| Fig 17 | closest story in the whole corpus | mean 0.37–0.52 |

**Train minus validation is ≈0 or negative on every model and both measures**,
so the models do not reproduce training prefixes more faithfully than unseen
ones — the memorisation control, which the paper does not run.

Figures 16 and 17 are defined over the entire 2.12 M-story corpus. Rather than
the naive 800 × 2.12 M comparisons, we index only the ~20 k k-grams the models
generated and make **one streaming pass**, verified against a brute-force
implementation in the tests.

## Output format

One JSONL line per completion (schema v2.0), documented in
[`records.py`](src/tinystories_project/records.py). Each record carries the prompt,
the completion, generation params, seed, timestamp, and model metadata including
**measured** parameter counts. Every run also writes a `*_manifest.json` with the
config, CLI overrides, git commit, and library versions.

Note: `roneneldan/TinyStories-1M` measures **3,745,984 parameters** (398,464
non-embedding) — the name matches neither, so we always report measured counts.
Other verified checkpoint facts are in [`CLAUDE.md`](CLAUDE.md).

## Testing

```bash
pytest                   # 152 unit tests, offline, ~3 s
pytest -m integration    # 12 tests against real checkpoints
```

The course Gemini API spends a shared **$250** team budget and **is not called by
any code in this repo yet**.
