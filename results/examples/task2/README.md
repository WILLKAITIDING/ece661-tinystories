# Task 2 example outputs

Committed snapshots of the finalised Task 2 analysis so teammates can read the
results without running anything. Regenerate everything (plus the CSVs and PNG
figures, which are not committed) with:

    python scripts/run_task2.py --config configs/task2.yaml
    python scripts/make_task2_tables.py

Full outputs land in `results/task2/`, which is git-ignored.

Two figures are committed here: `fig_results_by_category.png` (our labelled
results per category) and `fig_ours_vs_paper.png` (our reproduction beside the
paper's published verdicts on the same 24 prompts). The remaining figures and
all per-record CSVs regenerate with the commands above.

## What the numbers are

Strict accuracy over 176 greedy (protocol A) completions - 4 checkpoints x 44
prompts - scored against the final labels in `data/annotations/task2_final.csv`.

**Those labels were drafted by an AI assistant and reviewed by Will Ding**, who
overrode 9 and accepted 167. They are human-reviewed AI drafts, not independent
human annotation; say so in the report.

`comparison_paper_figures_9_11.md` holds BOTH our generations with our labels
(`ours:`) and the completions and colour verdicts printed in the TinyStories
paper (`paper:`). The `paper:` columns were extracted from the published PDF and
were neither generated nor judged by us.
