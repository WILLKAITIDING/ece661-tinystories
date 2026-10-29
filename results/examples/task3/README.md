# Task 3 example outputs

Committed snapshots of the Section 4.4.1 replication. Regenerate everything
(per-story CSVs and PNG histograms included, which are not committed) with:

    pip install -e ".[task3,analysis]"
    python scripts/run_task3.py          --config configs/task3.yaml   # ~16 min
    python scripts/make_task3_tables.py  --config configs/task3.yaml   # ~10 min

Full outputs land in `results/task3/`, which is git-ignored.

The second step scans all 2,119,719 training stories once and caches the result
in `results/task3/analysis/corpus_scan_cache.json`, so re-running it to tweak a
table or figure takes seconds instead of ten minutes. The cache is keyed to the
generations and corpus files, so it is rebuilt automatically if either changes.

## The headline

Four measures over 100 story prefixes per split, 4 checkpoints, greedy decoding:

- **Figure 14** (generation vs the real ending): mean 0.13-0.17. The models
  write something new rather than reproducing the training story.
- **Figure 15** (generations vs each other): 1M 0.27, 33M 0.14, and the real
  dataset continuations 0.09. Bigger models are more diverse and approach the
  diversity of genuine TinyStories.
- **Figure 16**: ~99.8% of generated 2-grams do occur in the corpus, but they
  are very common pairs (median g_r ~1e-4); only ~0.2% are novel.
- **Figure 17** (closest story in the whole corpus): mean 0.37-0.52.

**Train vs validation differences are ~0 or slightly negative on every model**,
so training prefixes are not reproduced more faithfully than unseen ones. That
is the memorisation control, and it supports the paper's conclusion.
