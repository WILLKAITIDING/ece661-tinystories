# Experiment Protocol

How we run experiments so results are comparable across people, models, and
weeks. Read this before adding a new experiment.

## Rules

1. **Settings live in `configs/*.yaml`, not in code or notebooks.** A result
   that cannot be traced to a config file cannot go in the report.
2. **Never silently change a setting.** Changing a config changes future runs;
   it does not retroactively explain an old result. Add a new config instead of
   editing one that produced results you are still citing.
3. **Every run records:** model id, measured parameter counts, tokenizer id,
   generation params, seed, device, library versions, git commit, timestamp.
   `scripts/run_task1.py` writes all of this to a `*_manifest.json` sidecar.
4. **Record `batch_size` with sampled results.** See the reproducibility note
   below — it affects the sampled text.
5. **Compare models only under identical prompts and generation settings.**
   The Phase 2 architecture table is meaningless otherwise.

## Reproducibility, stated honestly

`model.generate()` accepts no `torch.Generator`, so we seed global RNGs
immediately before each batch using `seeding.derive_seed(base_seed, "batch", i)`.

**Guaranteed identical** given the same: model + revision, generation params,
prompt list **and order**, `batch_size`, device, and library versions.

**Not guaranteed:** across different `batch_size` values (prompts regroup into
different batches, so sampling differs), across devices (MPS vs CUDA vs CPU
kernels differ), or across torch/transformers versions.

**Greedy decoding (`do_sample: false`) is deterministic regardless** of any of
the above except device/version. Use it when you need an exact match.

## Which parameter count do we report?

Report **both**, always. Tiny models are dominated by their embedding table:

| Checkpoint | Total | Embedding | Non-embedding |
|---|---:|---:|---:|
| `roneneldan/TinyStories-1M` | 3,745,984 | 3,347,520 | 398,464 |

*(measured 2026-10-28 with `count_parameters`, not taken from the model name)*

The released checkpoint names do **not** match either count exactly. For the
Phase 2 "~5M and ~10M parameter-matched" comparison, the team must agree on
which number is being matched **before** training starts, and state it in the
report. Non-embedding count is the more meaningful one for architecture
comparisons, because vocabulary size is a tokenizer choice, not an
architecture choice — this matters especially for Phase 3, where the Amharic
SentencePiece vocabulary (4K/8K) is far smaller than GPT-Neo's 50,257.

## Output layout

```
results/
  task1/
    <run_name>_<model>_<UTC timestamp>.jsonl           # one line per completion
    <run_name>_<model>_<UTC timestamp>_manifest.json   # full run provenance
  examples/      # small, deliberately committed samples (tracked in git)
```

Everything under `results/` is git-ignored **except** `results/examples/`.
Keep committed examples small; never commit a full generation sweep.

## Record schema

Defined and documented in `src/tinystories_project/io/records.py`
(schema version 1.0). One JSONL line = one completion.

Fields may be **added** in a minor version. Existing names and meanings do not
change without a major bump, because Tasks 3 and 4 read Task 1's files.
Later tasks attach their own outputs under `extra` (e.g. judge scores) rather
than restructuring the record.

`record_id` is derived from model + prompt + sample index + params, and
**not** from the generated text, so an interrupted run can resume by skipping
ids already present (`io.existing_record_ids`). This matters most for the
Gemini judge, where repeated work costs real money.

## Status of each task

| Task | Owner | Status |
|---|---|---|
| 1 — Load checkpoint, generate, save | Will | **Done** (this session) |
| 2 — Figures 9–11 + 20 new prompts | Will | Not started |
| 3 — ROUGE, Section 4.4.1 / Figures 14–17 | Will | Not started |
| 4 — Gemini judge | Will | Not started; **do not call the API yet** |
| Phase 2 — architectures | Rachel | Not started |
| Phase 3 — Amharic | Luel | Not started |

## Open decisions (not fixed by the proposal)

- The exact 20 additional prompts for Task 2.
- Which checkpoint sizes are used for each reproduced figure.
- Generation length for ROUGE prefix completions.
- Exact judge prompt wording and score scale.

Record each here once decided, with the date and who decided it.
