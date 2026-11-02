# Task 2 results (final annotations)

- **176** greedy (protocol A) completions: 4 checkpoints x 44 prompts.
- Labels: **82 correct, 25 partial, 69 incorrect**.
- Provenance: labels were **drafted by an AI assistant and reviewed by Will**, who overrode **9** and accepted **167**. They are human-reviewed AI drafts, not independent human annotation.

## Results by model

| checkpoint | figure col | layers | hidden | non-emb params | correct | partial | incorrect | strict | lenient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TinyStories-1M | 1M/8L | 8 | 64 | 398,464 | 9 | 6 | 29 | **20%** | 27% |
| TinyStories-8M | 8.3M/8L | 8 | 256 | 6,312,448 | 24 | 6 | 14 | **55%** | 61% |
| TinyStories-33M | 33M/4L | 4 | 768 | 28,343,808 | 32 | 6 | 6 | **73%** | 80% |
| TinyStories-1Layer-21M | 21M/1L | 1 | 1024 | 12,595,200 | 17 | 7 | 20 | **39%** | 47% |

## Strict accuracy by category

| checkpoint | Factual | Reasoning | Context | all |
|---|---:|---:|---:|---:|
| TinyStories-1M | 38% | 6% | 17% | 20% |
| TinyStories-8M | 62% | 56% | 42% | 55% |
| TinyStories-33M | 88% | 62% | 67% | 73% |
| TinyStories-1Layer-21M | 56% | 31% | 25% | 39% |

## Paper prompts vs our own prompts

| checkpoint | paper Figures 9-11 (24) | our new prompts (20) |
|---|---:|---:|
| TinyStories-1M | 12% | 30% |
| TinyStories-8M | 58% | 50% |
| TinyStories-33M | 75% | 70% |
| TinyStories-1Layer-21M | 21% | 60% |

## Our reproduction vs the paper's published verdicts

On the same 24 prompts and the same four checkpoints, our label agrees with the paper's colour on **59/96** cells (61%).

**Read this comparison carefully.** The two columns below are judged by DIFFERENT judges: ours by our reviewer, the paper's by its own authors. A gap therefore mixes a real model difference with a difference in how strictly each judge marked, and our greedy outputs are not the same strings the paper printed. Treat it as a sanity check on the trend, not as a measured delta.

| checkpoint | ours (strict) | the paper (strict) |
|---|---:|---:|
| TinyStories-1M (1M/8L) | 12% | 4% |
| TinyStories-8M (8.3M/8L) | 58% | 38% |
| TinyStories-33M (33M/4L) | 75% | 100% |
| TinyStories-1Layer-21M (21M/1L) | 21% | 8% |

## Width vs depth (Section 4.2)

The paper argues embedding dimension drives factual recall while depth drives context tracking, so a wide but shallow model should show a larger factual-minus-context gap.

| checkpoint | layers | hidden | factual | context | gap |
|---|---:|---:|---:|---:|---:|
| TinyStories-1M | 8 | 64 | 38% | 17% | +21 pts |
| TinyStories-8M | 8 | 256 | 62% | 42% | +21 pts |
| TinyStories-33M | 4 | 768 | 88% | 67% | +21 pts |
| TinyStories-1Layer-21M | 1 | 1024 | 56% | 25% | +31 pts |

## Protocol B (sampling) - NOT labelled

528 completions (3 per prompt, T=1.0, top-p 0.95, seed 42) have no human labels. `consistency_protocolB.csv` reports how many of the three sampled answers differ, with a keyword proxy that is a crude heuristic, not a judgement.


## Caveats

- Labels are human-reviewed AI drafts; disclose this in the report.
- Protocol B is unlabelled and must not be reported as accuracy.
- We reproduce the paper's qualitative trend, not its exact strings (0/24 exact matches on TinyStories-1M).
- Our labels and the paper's colours come from different judges and different generated text, so the two are not directly comparable.
- n is small: 44 prompts per model. Differences of a few points are not meaningful; no significance testing was done.
