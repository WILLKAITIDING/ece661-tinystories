# Task 2 summary

- models: 4  |  prompts: 44 (24 from the paper's Figures 9-11, 20 original)
- completions: 176 greedy (protocol A) + 528 sampled (protocol B, 3 per prompt)
- auto-screenable prompts: 39/44; the remaining 5 need human judgement (negation, role tracking, or open-ended).

## Measured model sizes

| checkpoint | figure column | layers | hidden | total params | non-embedding |
|---|---|---:|---:|---:|---:|
| TinyStories-1M | 1M/8L | 8 | 64 | 3,745,984 | 398,464 |
| TinyStories-8M | 8.3M/8L | 8 | 256 | 19,702,528 | 6,312,448 |
| TinyStories-33M | 33M/4L | 4 | 768 | 68,514,048 | 28,343,808 |
| TinyStories-1Layer-21M | 21M/1L | 1 | 1024 | 66,155,520 | 12,595,200 |

The figure-column names do not match either measured count; we report measured values throughout.

## Lexical screen, protocol A (NOT a quality judgement)

| checkpoint | Factual | Reasoning | Context | overall |
|---|---|---|---|---|
| TinyStories-1M | 5/15 | 2/13 | 3/11 | 10/39 |
| TinyStories-8M | 9/15 | 7/13 | 5/11 | 21/39 |
| TinyStories-33M | 12/15 | 9/13 | 7/11 | 28/39 |
| TinyStories-1Layer-21M | 9/15 | 5/13 | 4/11 | 18/39 |

## Width vs depth (Section 4.2 claim)

The paper argues embedding dimension drives factual recall while the number of layers drives context tracking. If so, a wide but shallow model should show a larger factual-minus-context gap.

| checkpoint | layers | hidden | factual % | context % | gap |
|---|---:|---:|---:|---:|---:|
| TinyStories-1M | 8 | 64 | 33 | 27 | +6 |
| TinyStories-8M | 8 | 256 | 60 | 45 | +15 |
| TinyStories-33M | 4 | 768 | 80 | 64 | +16 |
| TinyStories-1Layer-21M | 1 | 1024 | 60 | 36 | +24 |

## Files

- `annotation_template.csv`
- `auto_screen.csv`
- `comparison_custom_20.csv`
- `comparison_custom_20.md`
- `comparison_paper_figures_9_11.csv`
- `comparison_paper_figures_9_11.md`
- `consistency_protocolB.csv`
- `fig_auto_screen.png`
- `fig_paper_verdicts.png`
- `summary.md`

## Caveats

- The screen checks whether an expected word appears. It cannot distinguish "no, cows can't fly" from "yes, cows can fly"; such prompts are excluded from it and left blank for human annotation.
- `annotation_template.csv` has an empty `label` column. No row has been rated by a human yet, and none of these numbers is a human rating.
- Protocol B samples are NOT reproductions of the paper's deterministic figures and must not be presented as such.
