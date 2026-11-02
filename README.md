# ece661-tinystories

**Tiny Language Models: Architecture Exploration and Cross-Lingual Transfer to Amharic**
Duke ECE 661 — Machine Learning — Fall 2026 — **Team 32**

Will Ding · Rachel Odonkor · Luel Hagos

---

## Project overview

[TinyStories (Eldan & Li, 2023)](https://arxiv.org/abs/2305.07759) showed that
language models with **fewer than 10 million parameters** can write coherent,
grammatical short stories — provided the training data is restricted to the
vocabulary and situations a three-year-old would understand. That result makes
small models a practical object of study: you can train and evaluate them on
modest hardware and still ask real questions about architecture.

This project asks two of those questions:

1. **Which architectural choices matter at this scale?** The paper reports that
   embedding width drives factual recall while depth drives context tracking.
   We reproduce that evaluation and then test it on architectures the paper did
   not try.
2. **Does English pre-training transfer to a very different language?** We
   build a TinyStories-style corpus for **Amharic** (Ge'ez script, SOV word
   order, root-and-pattern morphology) and compare training from scratch
   against initialising from English weights.

Full scope and timeline: [`docs/proposal.md`](docs/proposal.md).

## Team responsibilities

| Member | Phase | Scope |
|---|---|---|
| **Will Ding** | **Phase 1** | Pretrained baselines and the shared evaluation pipeline (Tasks 1–4) |
| **Rachel Odonkor** | **Phase 2** | Architecture exploration — GPT-Neo baseline, deep-narrow vs shallow-wide, a modernised Transformer (RoPE/RMSNorm/GeGLU/GQA), and a non-attention baseline (minGRU or Mamba) |
| **Luel Hagos** | **Phase 3** | Amharic corpus generation, tokenizer, transfer experiments, and native-speaker evaluation |

Phase 1 exists so that Phases 2 and 3 do not each have to build an evaluation
harness. **Everything here is designed to run on someone else's model.** See
[Team integration guide](#team-integration-guide).

## Current progress

| Task | Status | What exists |
|---|---|---|
| **1 — Model loading & generation** | ✅ Complete | Loads any released TinyStories checkpoint, generates, writes structured records |
| **2 — Qualitative evaluation** | ✅ Complete | Figures 9–11 reproduced + 20 original prompts; **176 completions labelled** |
| **3 — Quantitative ROUGE** | ✅ Complete | Section 4.4.1 / Figures 14–17; **800 completions**, full 2.12 M-story corpus scan |
| **4 — Gemini judge** | 🟡 **Code complete, execution blocked** | Client, rubric, validation, budget guard, 37 offline tests — but **0 of 800 stories have been judged** |

> ### ⚠️ Task 4 is blocked, not finished
>
> Every chat model on the ECE 661 course gateway returns an upstream Vertex AI
> `403 PERMISSION_DENIED` ("Request rejected in accordance with our terms of use
> and policies"), while the embeddings route works normally. We tested 9 models,
> both base URLs, `curl` and the OpenAI SDK, and minimal payloads — all 403.
>
> **No stories have been scored. API spend to date: $0.00.** The code is
> finished and tested; it needs a working gateway, which is a course-side fix.
> Diagnosis and a one-line reproduction: [`docs/task4_status.md`](docs/task4_status.md).

## Project structure

```
ece661-tinystories/
├── configs/                      one YAML per task — all experiment settings live here
│   ├── task1.yaml                checkpoint, device, generation parameters
│   ├── task2.yaml                models, protocols (greedy / sampling), prompt sets
│   ├── task3.yaml                dataset shards, story sampling, ROUGE run
│   └── task4.yaml                judge model, temperature, budget cap
│
├── data/                         inputs, version-controlled
│   ├── prompts/                  paper_figures_9_11.json (24) · custom_20.json (20)
│   ├── reference/                the PAPER's own completions and verdicts — not ours
│   └── annotations/              task2_final.csv — the 176 Task 2 labels
│
├── src/tinystories_project/      the importable package
│   ├── config.py                 YAML → dataclasses; paths resolve to the repo root
│   ├── runtime.py                device selection (cuda → mps → cpu) and seeding
│   ├── records.py                GenerationRecord schema v2.0 + JSONL read/write
│   ├── inference/
│   │   ├── interface.py          ★ TextGenerator protocol — the integration point
│   │   ├── loader.py             load_causal_lm() → ModelBundle, parameter counting
│   │   └── generate.py           HFTextGenerator (reference impl), generate_stories()
│   └── evaluation/
│       ├── prompts.py            EvalPrompt / PromptSet loading and validation
│       ├── qualitative.py        Task 2 protocols and the prompt-set runner
│       ├── annotations.py        Task 2 label loading, accuracy, provenance
│       ├── rouge.py              Task 3 Section 4.4.1 measures + scan_corpus()
│       ├── corpus.py             TinyStories parquet loading / streaming
│       └── judge.py              Task 4 Gemini client, rubric, budget guard
│
├── scripts/                      one runner + one analysis script per task
│   ├── run_task1.py              …  run_task2.py  ·  run_task3.py  ·  run_task4.py
│   ├── make_task2_tables.py      …  make_task3_tables.py  ·  make_task4_tables.py
│   └── extract_paper_figures.py  rebuilds the prompt files from the paper PDF
│
├── tests/                        189 offline unit tests + 12 integration tests
├── notebooks/task1_demo.ipynb    thin demo — all logic lives in src/
├── docs/                         proposal.md · task4_status.md
└── results/                      git-ignored, EXCEPT results/examples/
    └── examples/                 small committed samples of every finished task
```

**Why `results/` is mostly ignored:** a full run writes ~1 MB of JSONL per task
plus a 2 GB model cache. Committed under `results/examples/` are the small,
readable summaries (tables and the `summary.md` for Tasks 2 and 3) so you can
read the findings without running anything. Everything else regenerates from
the commands below.

## Installation

Requires **Python 3.11+**. Works on CUDA, Apple Silicon (MPS), or CPU — the
device is chosen at runtime and never hardcoded. Developed on an M1 Pro.

```bash
git clone https://github.com/WILLKAITIDING/ece661-tinystories.git
cd ece661-tinystories

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev,notebook]"   # core + tests + notebook
```

Optional extras, installed per task:

| Extra | Needed for |
|---|---|
| `.[analysis]` | Task 2/3/4 tables and figures (`matplotlib`, `pymupdf`) |
| `.[task3]` | Task 3 corpus reading and the ROUGE cross-check (`pyarrow`, `rouge-score`) |
| `.[judge]` | Task 4 Gemini judge (`openai`) |

`requirements-lock.txt` pins the exact versions this was validated against.

**Models and data download themselves.** `load_causal_lm()` pulls checkpoints
from the Hugging Face Hub into `~/.cache/huggingface` on first use (~720 MB for
all four baselines); Task 3 pulls ~1 GB of TinyStories parquet the same way.
Nothing large is stored in the repository.

Verify the install:

```bash
pytest                    # 189 unit tests, offline, ~4 s
pytest -m integration     # 12 tests against the real checkpoints (downloads ~20 MB)
```

## How to run the evaluation

### Free / offline — no API key needed

```bash
# Task 1 — load a checkpoint and generate stories
python scripts/run_task1.py --config configs/task1.yaml

# Task 2 — qualitative evaluation (~4 min, downloads ~720 MB of checkpoints)
pip install -e ".[analysis]"
python scripts/run_task2.py --config configs/task2.yaml
python scripts/make_task2_tables.py

# Task 3 — ROUGE replication (~16 min generate + ~10 min corpus scan, ~1 GB download)
pip install -e ".[task3,analysis]"
python scripts/run_task3.py --config configs/task3.yaml
python scripts/make_task3_tables.py --config configs/task3.yaml

# Task 4 — dry run: exercises the whole judge pipeline, makes NO API calls
pip install -e ".[judge]"
python scripts/run_task4.py --config configs/task4.yaml --dry-run --limit 5
```

The Task 3 corpus scan is cached, so re-running `make_task3_tables.py` to adjust
a table takes seconds instead of ten minutes.

### 💸 Requires the paid course API

Put the key in **`.env`** (git-ignored) — never in `.env.example`, which is
tracked:

```bash
cp .env.example .env
# then edit .env and set COURSE_API_KEY=...
```

```bash
python scripts/run_task4.py --config configs/task4.yaml --limit 5   # pilot, ~$0.005
python scripts/run_task4.py --config configs/task4.yaml             # all 800, ~$0.75
python scripts/make_task4_tables.py
```

Both commands currently fail with the gateway 403 described above. The runner
stops after three failures rather than attempting all 800, and resume means a
retry never re-pays for a story already scored.

**Cost controls** (agreed Task 4 cap: **$10 total, including earlier calls**):
pricing is pinned in `judge.py` (`gemini-3.6-flash`, $0.75/$3.75 per 1M
input/output); the cap is checked before every call against the gateway's
cumulative key spend, so it covers previous runs and never resets; per-call cost
is `max(gateway header, our own calculation)` so an under-reporting gateway
cannot hide spend; and a call-count cap plus a circuit breaker bound the damage
from a bug. A personal "Unlimited" allowance on the course portal is **not**
treated as permission to exceed the cap.

## Baseline models and findings

Four released checkpoints, chosen to probe both of the paper's architectural
claims. Parameter counts are **measured by our loader**, not taken from the
checkpoint names — which match neither total.

| Checkpoint | Figure column | Layers | Hidden | Total params | Non-embedding |
|---|---|---:|---:|---:|---:|
| `roneneldan/TinyStories-1M` | 1M/8L | 8 | 64 | 3,745,984 | 398,464 |
| `roneneldan/TinyStories-8M` | 8.3M/8L | 8 | 256 | 19,702,528 | 6,312,448 |
| `roneneldan/TinyStories-33M` | 33M/4L | 4 | 768 | 68,514,048 | 28,343,808 |
| `roneneldan/TinyStories-1Layer-21M` | 21M/1L | 1 | 1024 | 66,155,520 | 12,595,200 |

### Task 2 — qualitative evaluation

44 prompts per model (24 recovered from the paper's Figures 9–11, 20 written by
us), greedy decoding to match the paper's stated zero temperature. All 176
completions were labelled `correct` / `partial` / `incorrect`, mirroring the
paper's green/yellow/red coding.

**Strict accuracy (correct ÷ 44):**

| Checkpoint | Factual | Reasoning | Context | **Overall** |
|---|---:|---:|---:|---:|
| TinyStories-1M | 38% | 6% | 17% | **20%** |
| TinyStories-8M | 62% | 56% | 42% | **55%** |
| TinyStories-33M | 88% | 62% | 67% | **73%** |
| TinyStories-1Layer-21M | 56% | 31% | 25% | **39%** |

Accuracy rises monotonically with size. The width-vs-depth claim of Section 4.2
is supported: the 1-layer/1024-wide model shows a **+31 pt** factual-minus-context
gap against **+21 pt** for all three deeper models — it recalls facts about as
well as the 8M model but tracks context markedly worse.

> **Annotation provenance — read before citing these numbers.** The labels were
> **drafted by an AI assistant and then reviewed by Will Ding**, who **overrode
> 9** and **explicitly accepted the remaining 167**. They are *human-reviewed AI
> drafts*, not independent human annotation. The `notes` column in
> [`data/annotations/task2_final.csv`](data/annotations/task2_final.csv) records
> which rows were overridden and why. Describe them this way in the report, and
> disclose the AI assistance as the course requires.

We also extracted the paper's own colour-coded verdicts from the published PDF.
On the same 24 prompts and checkpoints our labels agree with the paper's on
**59/96 cells (61%)** — but that comparison mixes a model difference with a
*judge* difference (different graders, different generated text), so it is a
sanity check on the trend, not a measured delta.

### Task 3 — quantitative ROUGE (Section 4.4.1)

100 story prefixes per split (train **and** validation), cut at 40% of their
words, completed greedily by all four checkpoints = **800 completions**.
Figures 16 and 17 scanned the **entire 2,119,719-story training corpus
(374,105,331 2-grams)** in a single pass.

| Measure | What it asks | Result |
|---|---|---|
| **Fig 14** | Is the generation contained in the real ending? | mean **0.13–0.17** |
| **Fig 15** | Are the 100 generations similar to each other? | 1M 0.27 → 33M 0.14; **real stories 0.09** |
| **Fig 16** | How common are the generated 2-grams in the corpus? | only **~0.2%** occur nowhere |
| **Fig 17** | Closest story in the whole corpus | mean **0.37–0.52** |

**Memorisation:** train-minus-validation is **≈0 or slightly negative** for every
model on both Figure 14 and Figure 17. Models do not reproduce training prefixes
more faithfully than unseen ones.

**Methodological deviations from the paper — please read before citing:**

- **The Figure 15 control is ours.** The paper compares models against GPT-3.5
  generations. We additionally compute the same measure over the 100 **real
  dataset continuations** (0.09), which is what makes "diverse" interpretable:
  every model is *more* self-similar than genuine TinyStories, and larger models
  move toward the dataset's diversity rather than reaching it.
- **The validation split is ours.** The paper runs 4.4.1 on training stories
  only. Adding validation is what turns "low overlap" into an actual
  memorisation control.
- **Memorisation conclusions are bounded.** A near-zero train/validation gap is
  evidence *against* verbatim memorisation of these 100 prefixes under greedy
  decoding. It is **not** proof that no memorisation exists: n = 100 per split,
  one decoding strategy, and we cannot verify which corpus snapshot each
  released checkpoint actually saw.
- **Decoding temperature is unspecified by the paper** for 4.4.1. We use greedy,
  which is reproducible and the strictest memorisation test; sampling would only
  lower the overlaps, so our conclusion is the conservative one.
- **ROUGE is implemented from the paper's formulas**, over k-gram *sets*, which
  differs slightly from the common `rouge_score` package (multiset
  intersection). `compare_with_rouge_score()` quantifies the gap rather than
  assuming it away. The paper's f-measure line also contains a typo (`R_k` where
  `R_k,p` is meant).
- **Figure 16 is a near-universal-hit measure** at this vocabulary size. ~99.8%
  of generated 2-grams appear in the corpus simply because TinyStories uses a
  deliberately tiny vocabulary — common word pairs are everywhere. Figure 17 and
  the train/validation gap are the measures that actually bear on memorisation.

### Task 4 — Gemini judge

**0 of 800 stories judged.** See the blocked-status box above and
[`docs/task4_status.md`](docs/task4_status.md). No results exist yet, and none
have been fabricated: `--dry-run` writes to a separate directory and
`make_task4_tables.py` refuses dry-run rows outright.

---

## Team integration guide

**Rachel and Luel: this is the part that matters for you.** The goal is that you
never rewrite an evaluation script. You implement one small class and every
Phase 1 evaluation runs on your model.

### 1. Clone and install

```bash
git clone https://github.com/WILLKAITIDING/ece661-tinystories.git
cd ece661-tinystories
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis]"
pytest                    # should report 189 passed
```

Work on your own branch, never on `main`:

```bash
git checkout -b phase2-arch-rachel      # or phase3-amharic-luel
```

### 2. Find the two things you need to know

| What | Where | Why |
|---|---|---|
| **`TextGenerator` protocol** | [`src/tinystories_project/inference/interface.py`](src/tinystories_project/inference/interface.py) | The *only* contract your model must satisfy |
| **`GenerationRecord` schema** | [`src/tinystories_project/records.py`](src/tinystories_project/records.py) | The JSONL format every evaluation reads |

Read the module docstring at the top of each. The reference implementation for
a Hugging Face model is `HFTextGenerator` in
[`inference/generate.py`](src/tinystories_project/inference/generate.py) — copy
its shape, not its internals.

### 3. Connect your own architecture

**Your model does not need Hugging Face `.generate()`.** That is the whole point
of the protocol — a minGRU or Mamba with a hand-written sampling loop works
exactly as well. You need two members: a `metadata` property and a `generate`
method.

```python
from tinystories_project.inference.interface import (
    GeneratedText, GenerationParams, ModelMetadata,
)


class MinGRUGenerator:
    """Wraps Rachel's minGRU so Phase 1 evaluations accept it."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    @property
    def metadata(self) -> ModelMetadata:
        # Report MEASURED parameter counts, not the name you gave the model.
        n_total = sum(p.numel() for p in self.model.parameters())
        return ModelMetadata(
            model_id="rachel/min-gru-5M",
            architecture="MinGRU",
            tokenizer_id="EleutherAI/gpt-neo-125M",
            source="custom",
            n_parameters_total=n_total,
            n_parameters_non_embedding=n_total,   # minus your embedding table
            n_layers=4,
            hidden_size=512,
        )

    def generate(self, prompts, params: GenerationParams, seed=None):
        out = []
        for i, prompt in enumerate(prompts):
            for s in range(params.num_return_sequences):
                text = my_sampling_loop(            # ← your code
                    self.model, self.tokenizer, prompt,
                    max_new_tokens=params.max_new_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    greedy=not params.do_sample,
                    seed=seed,
                )
                out.append(GeneratedText(
                    prompt=prompt,
                    completion=text,        # ONLY the new text — never echo the prompt
                    prompt_index=i,         # index into `prompts`
                    sample_index=s,         # which of the N samples
                    seed=seed,
                    n_prompt_tokens=len(self.tokenizer.encode(prompt)),
                    n_new_tokens=len(self.tokenizer.encode(text)),
                    finish_reason="eos",    # or "length"
                ))
        return out
```

Four rules, all enforced by the evaluation code:

1. Return exactly `len(prompts) * params.num_return_sequences` items.
2. `completion` holds **only newly generated text**. Echoing the prompt breaks
   ROUGE and the judge, and `run_prompt_set` raises if the prompt comes back
   altered.
3. Set `prompt_index` and `sample_index` correctly — that is how results map
   back to prompts.
4. Honour `params`. If you ignore `max_new_tokens` or `temperature`, your model
   is not being compared under the same settings as the baselines.

Check your class satisfies the protocol:

```python
from tinystories_project.inference.interface import TextGenerator
assert isinstance(MinGRUGenerator(model, tok), TextGenerator)
```

### 4–6. Generate, save, and evaluate — without touching the scripts

```python
from tinystories_project.evaluation import load_prompt_set, run_prompt_set
from tinystories_project.records import write_jsonl

gen = MinGRUGenerator(model, tokenizer)
prompts = load_prompt_set("data/prompts/paper_figures_9_11.json")

# Identical settings to the Phase 1 baselines: protocol "A_greedy" is greedy
# decoding, "B_sampling" is 3 samples at T=1.0 / top-p 0.95.
records = run_prompt_set(gen, prompts, protocol="A_greedy",
                         run_id="rachel-mingru-001", seed=42)

write_jsonl(records, "results/task2/A_greedy/min-gru-5M__paper_figures_9_11.jsonl")
```

That file is now in exactly the format the existing analysis reads:

```bash
python scripts/make_task2_tables.py     # your model appears beside the baselines
```

For Task 3, use `evaluation.rouge` the same way — `figure14_scores`,
`figure15_scores`, and `scan_corpus` all take plain strings and know nothing
about where the text came from.

### 7. Add a model configuration

Append to `configs/task2.yaml` (or copy it to `configs/task2_phase2.yaml`) — the
runner loads one model at a time and releases it, so adding models does not
increase peak memory:

```yaml
models:
  - model_id: roneneldan/TinyStories-1M
    figure_column: "1M/8L"
  - model_id: /path/to/your/checkpoint       # a local directory works
    figure_column: "minGRU-5M"
```

A local path works wherever a Hub id does, as long as
`AutoModelForCausalLM.from_pretrained` can read it. If your architecture is not
loadable that way, skip the config and construct your generator in Python as in
step 4.

### 8. Compare against the Phase 1 baselines

The baselines are already in the committed tables under
[`results/examples/task2/`](results/examples/task2/) and
[`results/examples/task3/`](results/examples/task3/). Once your JSONL sits
alongside theirs, `make_task2_tables.py` and `make_task3_tables.py` include your
model automatically — same prompts, same decoding, same metrics.

For Task 2 your completions need labels to appear in the accuracy tables.
`make_task2_tables.py` writes `annotation_template.csv`; fill the `label` column
with `correct` / `partial` / `incorrect` and record in `annotator` and `notes`
who judged it and how. Follow the provenance convention in
[`data/annotations/README.md`](data/annotations/README.md) — if labels are
AI-drafted, say so.

### Notes for Luel — what is and is not English-specific

**Language-independent — reuse as is:**

| Component | Why it transfers |
|---|---|
| `inference/interface.py`, `loader.py`, `generate.py` | No language assumptions at all |
| `records.py` | Writes with `ensure_ascii=False`, so **Ge'ez text stays readable** on disk. There is a test asserting Amharic survives a round-trip |
| `evaluation/prompts.py` | A prompt set is just JSON; only `category` is a controlled vocabulary |
| `evaluation/annotations.py` | Label scheme and provenance tracking are language-neutral |
| `evaluation/rouge.py` | The **formulas** are language-neutral |
| `evaluation/corpus.py` | Point it at your Amharic parquet instead |
| `runtime.py`, `config.py` | Devices and configs |

**English-specific — needs adaptation:**

| Component | What to change |
|---|---|
| **Tokenizer** | GPT-Neo's BPE has no Ge'ez coverage. `load_causal_lm(model_id, tokenizer_id=...)` already takes a separate tokenizer, so your SentencePiece model drops in — that split exists for exactly this |
| **Prompts** | `data/prompts/*.json` is English. Write `amharic_figures_9_11.json` and `amharic_custom_20.json` in the same shape; the loader validates them unchanged |
| **`rouge.tokenize()`** | ⚠️ Uses `[^a-z0-9]+` as a separator, which **deletes Ge'ez entirely**. This needs a Unicode-aware replacement before any Amharic ROUGE number is meaningful. Keep the `ngram_set` / precision / f-measure functions; replace only the tokenizer |
| **`auto_accept` keyword screen** | English substrings. Either translate them or leave the field empty so prompts go to human judgement |
| **Judge rubric** | `judge.py`'s prompt and reading-age buckets are English. The rubric needs translating, and the proposal already notes LLM judges may be lenient on non-English text — pair the scores with the blinded native-speaker rating |
| **Reference corpus** | Figures 16/17 compare against the *English* training corpus. Swap in the Amharic corpus via `configs/task3.yaml` |

The cleanest first step is a `tokenize_unicode()` alongside the existing one,
selected by config. Please raise it before changing `rouge.py` in place, since
Task 3's committed numbers depend on the current behaviour.

### Workflow

| Branch | Purpose |
|---|---|
| `main` | Stable. Never commit directly |
| `phase1-eval-will` · `phase2-arch-rachel` · `phase3-amharic-luel` | One per phase |

```bash
git checkout main && git pull
git checkout -b phase2-arch-rachel/mingru
pytest                                           # must pass before pushing
git commit -am "feat(models): add minGRU baseline at 5M params"
git push -u origin phase2-arch-rachel/mingru     # then open a PR into main
```

In the PR, say what changed, what you ran, and the actual output. Get a review
from whoever owns the code you touched. **Never commit** checkpoints, datasets,
`.env`, or API keys — `.gitignore` covers these. Don't refactor a teammate's
module to suit your own; raise it at the weekly sync.

Shared code to coordinate on before changing: `inference/`, `records.py`
(the schema is a contract — additive changes only), and `evaluation/`.

---

## Reproducibility

| Setting | Value |
|---|---|
| **Seed** | 42 everywhere (`configs/*.yaml`); recorded in every record and manifest |
| **Task 2 decoding** | Protocol A: greedy (`do_sample=false`). Protocol B: T=1.0, top-p 0.95, 3 samples |
| **Task 3 decoding** | Greedy, `max_new_tokens=200`, prompts cut at 40% of words |
| **Task 4 judging** | `gemini-3.6-flash` at temperature 0, rubric `v1` |
| **Batch size** | 1 for Tasks 2–3, deliberately — sampling seeds are derived per batch, so batch composition would otherwise change sampled text |
| **Source models** | `roneneldan/TinyStories-{1M,8M,33M,1Layer-21M}` from the Hugging Face Hub |
| **Source data** | `roneneldan/TinyStories` default parquet config (2,119,719 train / 21,990 validation) |
| **Paper source** | arXiv:2305.07759**v2**, SHA-256 pinned in `data/prompts/paper_figures_9_11.json` |
| **Provenance** | Every run writes a `manifest_*.json` with the config, git commit, and library versions |

Greedy decoding was verified numerically stable across batch sizes (identical at
batch 1 vs 8). Re-running Task 3 reproduced identical token counts.

## Known limitations

- **We do not reproduce the paper's exact strings** — 0/24 exact matches on
  TinyStories-1M (7/24 share a prefix). Ruled out quote normalisation and
  batching as causes; most likely hardware/library drift since 2023 plus
  hand-truncation in the published figures. We claim reproduction of the
  **trend**, never of the text.
- **Task 2 labels are human-reviewed AI drafts** (9 overridden, 167 accepted),
  not independent human annotation.
- **Small n.** 44 prompts per model in Task 2, 100 stories per split in Task 3.
  Differences of a few points are not meaningful; no significance testing was
  done.
- **Three of seven checkpoints were not evaluated** (`TinyStories-3M`, `-28M`,
  `-2Layers-33M`) to bound local compute. They are verified and listed in the
  configs — adding them is a one-line edit.
- **GPT-2 XL was not run.** The paper's column is carried as cited reference
  data only, clearly marked as not independently generated.
- **Task 4 produced no results.** Gateway blocked; $0.00 spent.
- **Protocol B (528 sampled completions) is unlabelled** and must never be
  reported as accuracy.

## API troubleshooting

If Task 4 fails, read [`docs/task4_status.md`](docs/task4_status.md) first — it
has the full diagnosis, a one-line `curl` reproduction, and the evidence that
the fault is upstream (embeddings work, all 9 chat models 403). **Do not retry
the endpoint in a loop**; the circuit breaker exists for that reason. If the
gateway has been fixed, the pilot command in the paid section above is the way
to confirm it.

## Citations

- Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models.* [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Checkpoints and dataset: [huggingface.co/roneneldan](https://huggingface.co/roneneldan)

Further references (RoPE, RMSNorm, GeGLU, GQA, minGRU, Mamba, SentencePiece,
XLM, XLM-R) are listed in [`docs/proposal.md`](docs/proposal.md).

## AI assistance disclosure

Parts of this repository — including code, the Task 2 draft labels, and
documentation — were produced with AI assistance (Claude Code), reviewed by the
team. Per-task provenance is recorded in `data/annotations/README.md` and in the
generated summaries. This is disclosed per course policy.
