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

**Tasks 1–3 are done. Task 4's code is complete but cannot run (gateway 403 —
see `docs/task4_status.md`). Phases 2–3 are not started; do not implement them
speculatively.**

| Task | Owner | Status |
|---|---|---|
| 1 — Load checkpoint, generate, save JSONL | Will | **Done** |
| 2 — Figures 9–11 + 20 new prompts | Will | **Done** |
| 3 — ROUGE (Section 4.4.1, Figures 14–17) | Will | **Done** |
| 4 — Gemini judge | Will | **Code complete, blocked on the gateway** |
| Phase 2 / Phase 3 | Rachel / Luel | Not started |

## Layout

```
configs/task1.yaml       Task 1 settings (never hardcode these in code)
configs/task2.yaml       Task 2: models, protocols, prompt sets
configs/task3.yaml       Task 3: dataset, sampling, ROUGE run
configs/task4.yaml       Task 4: judge model, budget cap
data/prompts/            paper_figures_9_11.json (24), custom_20.json (20)
data/reference/          the PAPER's own completions + verdicts (not ours)
data/annotations/        task2_final.csv - the final Task 2 labels
docs/proposal.md         authoritative scope
notebooks/               thin demos only — logic belongs in src/
results/                 git-ignored except results/examples/
scripts/run_task1.py     Task 1 CLI runner
scripts/run_task2.py     Task 2 runner (one model at a time, resumable)
scripts/make_task2_tables.py   comparison tables, annotation CSV, figures
scripts/run_task3.py     Task 3 runner (samples prefixes, generates)
scripts/make_task3_tables.py   Figures 14-17 (caches the corpus scan)
scripts/run_task4.py     Task 4 judge runner (resumable, budget-capped)
scripts/make_task4_tables.py   judge tables, figures, methodology note
scripts/extract_paper_figures.py   rebuilds the prompt files from the PDF
src/tinystories_project/
  config.py              YAML -> dataclasses; paths resolve to project root
  runtime.py             device resolution + seeding
  records.py             GenerationRecord schema v2.0 + JSONL read/write
  inference/
    interface.py         GenerationParams, ModelMetadata, GeneratedText,
                         TextGenerator protocol
    loader.py            load_causal_lm -> ModelBundle; parameter counting
    generate.py          HFTextGenerator, generate_stories
  evaluation/
    prompts.py           EvalPrompt, PromptSet, load_prompt_set
    qualitative.py       PROTOCOLS, run_prompt_set, answer_span
    annotations.py       load_annotations, accuracy, provenance
    rouge.py             Section 4.4.1 measures + scan_corpus
    corpus.py            TinyStories parquet loading / streaming
    judge.py             Gemini judge client, rubric, budget guard
tests/                   test_core, test_generation, test_records, test_task2,
                         test_integration
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
pytest                   # 186 unit tests, offline, tiny random model, ~3 s
pytest -m integration    # 12 tests against real checkpoints
```

Unit tests must not hit the network; use the fixtures in `tests/conftest.py`.
**Run them after implementing anything, and report real counts and real
output** — never claim something works without having executed it. If a
download or test fails, say so plainly and leave the code ready to retry.

## Task 2 facts (established 2026-10-29)

- Figures 9-11 hold **24** prompts: 9 factual, 9 reasoning, 6 context. They were
  recovered from the PDF by **layout-aware** extraction (`scripts/extract_paper_figures.py`),
  because reading the text in order cannot tell where a wrapped prompt cell ends.
  Do not re-extract by plain text dump.
- The paper's green/yellow/red verdicts are cell **background fills** and are
  extracted too, giving its own labels for all 192 cells.
- **Zero temperature = greedy.** Section 4 says examples from Figure 6 onward are
  generated "at zero temperature"; Figures 9-11 follow. Express this as
  `do_sample=False`, never `temperature=0` (which GenerationParams rejects).
- **We do NOT reproduce the paper's exact strings** (0/24 exact, 7/24 prefix on
  TinyStories-1M). Verified not to be caused by quote normalisation (ASCII beats
  the PDF's curly quotes, 7 vs 3) or by batching (greedy is identical at
  batch_size 1 vs 8). Most likely hardware/library drift since 2023, plus
  hand-truncation in the figures. Claim reproduction of the **trend**, never of
  the text.
- **Final labels live in `data/annotations/task2_final.csv`** (176 rows; 82
  correct / 25 partial / 69 incorrect). They were **drafted by an AI assistant
  and reviewed by Will**, who overrode 9 and accepted 167. Always describe them
  as *human-reviewed AI drafts*, never as independent human annotation — the
  course requires disclosing AI assistance. `AnnotationSet.provenance()` recomputes
  the split; do not hardcode it.
- Do **not** re-label or overwrite those annotations. `make_task2_tables.py`
  validates every record_id against the generations and fails loudly on drift.
- Protocol B (528 sampled completions) is **unlabelled**; never report it as
  accuracy. The keyword proxy in the consistency CSV is a heuristic only.
- Comparing our accuracy to the paper's colours mixes a model difference with a
  judge difference. Report it as a trend check, not a measured delta.

## Task 3 facts (established 2026-10-29)

- Section 4.4.1 defines its OWN quantities; this is not generic ROUGE. `R_k,p`
  is over k-gram **SETS**, and the f-measure is the harmonic mean of the two
  precisions. The paper has a typo there (`R_k` where `R_k,p` is meant).
- The four figures measure different things — see `rouge.py`. Figures 14, 15, 17
  are Rouge-2; Figure 16 is a normalised corpus frequency `g_r`, which is a term
  frequency (occurrences), not a document frequency.
- Figures 16/17 are defined over the whole corpus but do NOT need 800 x 2.12M
  comparisons: `scan_corpus()` indexes only the generated k-grams and makes one
  pass. It is verified against brute force in `tests/test_task3.py` — keep that
  test if you touch it.
- The corpus pass takes ~10 min and is cached in
  `results/task3/analysis/corpus_scan_cache.json`, keyed to the generations and
  corpus files. Delete it to force a rescan.
- The paper does not state a temperature for 4.4.1. We use **greedy**: strictest
  memorisation test and reproducible. Say so when reporting.
- Result: train-minus-validation is ~0 on every model, i.e. no memorisation.

## Task 4 facts (established 2026-11-02)

- **Blocked upstream.** Every chat model returns Vertex `403 PERMISSION_DENIED`
  while embeddings succeed. Tested 9 models, both base URLs, curl and the SDK.
  Do NOT keep retrying it — read `docs/task4_status.md` first and check whether
  the gateway has been fixed.
- **Spend to date: $0.00.** The agreed cap is **$10 total including earlier
  calls**; the portal's "Unlimited" personal allowance does not override it.
- Base URL is `https://34-127-166-57.sslip.io/api` (the gateway says so itself
  at `/api/v1/health`). Judge model `gemini-3.6-flash`, temperature 0.
- Pricing is pinned in `judge.py` ($0.75/$3.75 per 1M in/out). Cost per call is
  `max(gateway header, our calculation)` so an under-reporting gateway cannot
  hide spend. 800 records project to ~$0.75.
- The key lives in `.env` (git-ignored). It was once pasted into the TRACKED
  `.env.example` but never committed (verified across every blob). A test
  asserts `.env.example` stays key-free — keep it.
- Resume is by `judge_id` (record + judge model + temperature + rubric version).
  **Never re-run a successful paid call.**
- `--dry-run` writes to `results/task4/dry_run/` and the analysis script refuses
  dry-run rows outright, so placeholders can never become results.

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
