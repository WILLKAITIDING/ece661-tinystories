# ece661-tinystories

Duke ECE 661, Fall 2026, Team 32.

**Tiny Language Models: Architecture Exploration and Cross-Lingual Transfer to Amharic**

The TinyStories paper ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759))
showed that language models with only a few million parameters can still write
coherent short stories if you restrict the training data to simple children's
vocabulary. We reproduce their evaluation, then use the same pipeline to compare
other small architectures and to test whether English pre-training helps when
transferring to Amharic.

Full project plan: `docs/proposal.md`.

## Team

- **Will Ding** — Phase 1: pretrained baselines and the shared evaluation pipeline
- **Rachel Odonkor** — Phase 2: architecture exploration (GPT-Neo baseline, deep vs wide, a modernised Transformer, a non-attention baseline)
- **Luel Hagos** — Phase 3: Amharic corpus, tokenizer, transfer experiments, native-speaker evaluation

Phase 1 exists so Phases 2 and 3 don't each have to build their own evaluation
code. See "Using your own model" below.

## Status

- **Task 1 — model loading and generation.** Done.
- **Task 2 — qualitative evaluation.** Done. Figures 9–11 from the paper plus 20 prompts of our own, 176 completions labelled.
- **Task 3 — quantitative ROUGE (Section 4.4.1).** Done. 800 completions, scanned against all 2.1M training stories.
- **Task 4 — Gemini judge.** Code written and tested, but **not run**. Every chat
  model on the course gateway returns `403 PERMISSION_DENIED` from Vertex AI
  while the embeddings endpoint works fine; we tried 9 models, both base URLs,
  curl and the OpenAI SDK. **0 of 800 stories judged, $0.00 spent.** Needs a fix
  on the course side. `python scripts/run_task4.py --config configs/task4.yaml
  --limit 1` is the quickest way to check if it's back.

## Structure

```
configs/    one YAML per task - all experiment settings live here
data/       prompts/ (24 paper + 20 ours), reference/ (the paper's own outputs),
            annotations/ (the 176 Task 2 labels)
src/tinystories_project/
  inference/  interface.py (the TextGenerator protocol), loader.py, generate.py
  evaluation/ prompts, qualitative (T2), rouge (T3), judge (T4), annotations
  records.py  the JSONL format every task reads and writes
scripts/    run_task{1,2,3,4}.py and make_task{2,3,4}_tables.py
tests/      190 offline tests + 12 that use the real checkpoints
results/    git-ignored except results/examples/ (small committed summaries)
```

## Install

Python 3.11+. Runs on CUDA, Apple Silicon (MPS), or CPU — picked at runtime.

```bash
git clone https://github.com/WILLKAITIDING/ece661-tinystories.git
cd ece661-tinystories
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis]"
pytest
```

Extras: `.[analysis]` for tables and figures, `.[task3]` for the corpus reading,
`.[judge]` for Task 4. `requirements-lock.txt` has the exact versions we used.

Models and datasets download themselves on first use (about 720 MB of
checkpoints, 1 GB of TinyStories data). Nothing large is in the repo.

## Running the experiments

Free and offline:

```bash
python scripts/run_task1.py --config configs/task1.yaml
python scripts/run_task2.py --config configs/task2.yaml            # ~4 min
python scripts/make_task2_tables.py
python scripts/run_task3.py --config configs/task3.yaml            # ~16 min
python scripts/make_task3_tables.py --config configs/task3.yaml    # ~10 min, then cached
python scripts/run_task4.py --config configs/task4.yaml --dry-run --limit 5
```

Only the last one touches Task 4, and `--dry-run` makes no API calls. For real
judging, put the key in `.env` (not `.env.example`, which is tracked), then
`--limit 5` as a pilot before the full run. We agreed a **$10 cap**; the runner
checks it before every call against the gateway's own reported spend, so it
covers earlier runs too. All 800 records should cost about $0.75.

## Results so far

Four released checkpoints, with parameter counts measured by our loader rather
than taken from the names (which match neither number). **Task 2**: 44 prompts
per model, greedy decoding, share judged correct.

| checkpoint | layers | hidden | non-emb params | factual | reasoning | context | overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| TinyStories-1M | 8 | 64 | 398K | 38% | 6% | 17% | 20% |
| TinyStories-8M | 8 | 256 | 6.3M | 62% | 56% | 42% | 55% |
| TinyStories-33M | 4 | 768 | 28.3M | 88% | 62% | 67% | 73% |
| TinyStories-1Layer-21M | 1 | 1024 | 12.6M | 56% | 31% | 25% | 39% |

Accuracy goes up with model size. The 1-layer model does about as well as the 8M
model on factual prompts but noticeably worse on context-tracking, which is the
direction the paper's width/depth argument predicts. With 44 prompts and one
decoding setting, our results are consistent with that argument rather than a
confirmation of it.

The labels were **drafted by an AI assistant and then reviewed by Will**, who
changed 9 and accepted the other 167. They are AI-drafted human-reviewed labels,
not independent human annotation — `data/annotations/README.md` has the detail.

**Task 3.** 100 story prefixes per split, cut at 40%, completed greedily.
Generated text overlaps the real ending by about 0.13–0.17 (ROUGE-2 precision);
the models resemble each other more (0.14–0.27) than real dataset stories do
(0.09); the closest training story scores about 0.37–0.52.

Train-minus-validation is roughly zero for every model. That is limited evidence
against copying the specific training prefixes we tested, under greedy decoding
with n=100 per split — not a general claim that these models don't memorise.

The validation split and the Figure 15 comparison against real continuations
are ours, not the paper's.

Tables and figures: `results/examples/task2/` and `results/examples/task3/`.

## Using your own model

Rachel and Luel: you shouldn't need to rewrite any evaluation code. Implement
one class with a `metadata` property and a `generate` method. The protocol is in
`src/tinystories_project/inference/interface.py`; `inference/generate.py` is the
Hugging Face version to copy the shape of.

```python
from tinystories_project.inference.interface import GeneratedText, ModelMetadata
from tinystories_project.evaluation import load_prompt_set, run_prompt_set
from tinystories_project.records import write_jsonl

class MyGenerator:
    @property
    def metadata(self):
        return ModelMetadata(model_id="rachel/min-gru-5M", architecture="MinGRU",
                             source="custom", n_parameters_total=5_000_000,
                             n_parameters_non_embedding=2_000_000,
                             n_layers=4, hidden_size=256)

    def generate(self, prompts, params, seed=None):
        out = []
        for i, prompt in enumerate(prompts):
            for s in range(params.num_return_sequences):
                text = my_sampling_loop(prompt, params)   # your code
                out.append(GeneratedText(
                    prompt=prompt, completion=text,   # new text only, don't echo the prompt
                    prompt_index=i, sample_index=s, seed=seed,
                    n_new_tokens=..., finish_reason="eos"))
        return out

prompts = load_prompt_set("data/prompts/paper_figures_9_11.json")
records = run_prompt_set(MyGenerator(), prompts, protocol="A_greedy",
                         run_id="mingru-001", seed=42)
write_jsonl(records, "results/task2/A_greedy/min-gru-5M__paper_figures_9_11.jsonl")
```

`python scripts/make_task2_tables.py` then picks your model up automatically.
We checked this with a dummy model, so it does work. Things to know:

- You show up in the comparison tables immediately, but the **accuracy** tables
  need labels. The script writes `annotation_template.csv` with your rows blank;
  fill in `label` (`correct`/`partial`/`incorrect`) and pass your file with
  `--annotations your_labels.csv` to keep it separate from ours.
- If your model loads with `AutoModelForCausalLM.from_pretrained`, skip all of
  this and just put the **local path** as `model_id` in `configs/task2.yaml`.
  Tested too.
- `run_task3.py` only loads Hugging Face checkpoints, so a non-HF model has to
  generate its Task 3 completions in Python (same pattern) and write the JSONL
  into `results/task3/<split>/`. Adding a model invalidates the corpus-scan
  cache, so the next `make_task3_tables.py` takes ~10 minutes again.
- Task 4's dry run accepts anything in `results/task3/`, filtered with `--models`.

For Amharic, most of this is language-neutral: the interface, loader, records
(JSONL uses `ensure_ascii=False` and there's a test that Ge'ez survives a
round-trip), prompt loading, annotations, and the ROUGE formulas. Needing work:
the prompt files, the judge rubric and age buckets, the `auto_accept` keyword
lists, the reference corpus in `configs/task3.yaml`, and above all
`rouge.tokenize()` — it splits on `[^a-z0-9]+`, which deletes Ge'ez entirely, so
it needs a Unicode-aware version before any Amharic ROUGE number means anything.
Please raise that before changing it in place; our Task 3 numbers depend on the
current behaviour.

## Limitations

- We don't reproduce the paper's exact generated text (0/24 exact matches on
  TinyStories-1M, 7/24 share a prefix) — probably library and hardware drift
  since 2023 plus hand-trimmed figures. We reproduce the trend, not the strings.
- Our ROUGE follows the paper's formulas over k-gram sets, which differs a
  little from the `rouge_score` package. `compare_with_rouge_score()` measures
  the gap.
- The paper doesn't give a decoding temperature for Section 4.4.1. We used
  greedy because it's reproducible; we don't know how sampling would change the
  overlap numbers.
- Our scores and the paper's GPT-4 scores come from different judges and
  different text, so they aren't directly comparable. The same will apply to
  Task 4's Gemini scores if we get any.
- Small samples (44 prompts per model, 100 stories per split) and no
  significance testing.
- Three of the seven checkpoints weren't evaluated; GPT-2 XL wasn't run (its
  column is quoted from the paper and marked as such).
- Protocol B (528 sampled completions) has no labels, so it isn't accuracy.

Seed is 42 everywhere and recorded in every output file. Tasks 2 and 3 use batch
size 1 on purpose, since sampling seeds are derived per batch. Every run writes a
manifest with the config, git commit, and library versions.

Parts of this repo — code, the Task 2 draft labels, and documentation — were
written with AI assistance (Claude Code) and reviewed by the team.
