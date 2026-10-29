# Task 3 - Section 4.4.1 quantitative replication

Four measures, as defined on pp. 17-18 of arXiv:2305.07759v2. 100 stories per split, cut at 40% of their words, completed with greedy decoding (the paper does not state a temperature; greedy is the strictest memorisation test and is reproducible).

Figures 16 and 17 scan the training corpus: **2,119,719 stories, 374,105,331 2-grams**, in one pass (603s).

## Figure 14 - is the generation contained in the real ending?

`s_i = R_2,p(T_i, T'_i)`. Low means the model wrote something new rather than reproducing the story it was trained on.

| split | model | mean | median | p75 | max |
|---|---|---:|---:|---:|---:|
| train | TinyStories-1M | 0.131 | 0.121 | 0.164 | 0.417 |
| train | TinyStories-8M | 0.161 | 0.128 | 0.202 | 0.455 |
| train | TinyStories-33M | 0.165 | 0.157 | 0.215 | 0.414 |
| train | TinyStories-1Layer-21M | 0.142 | 0.127 | 0.180 | 0.476 |
| validation | TinyStories-1M | 0.145 | 0.130 | 0.196 | 0.500 |
| validation | TinyStories-8M | 0.159 | 0.131 | 0.200 | 0.818 |
| validation | TinyStories-33M | 0.167 | 0.156 | 0.231 | 0.475 |
| validation | TinyStories-1Layer-21M | 0.146 | 0.118 | 0.176 | 1.000 |

## Figure 15 - are the 100 generations similar to each other?

`r_i = max_{j != i} R_2(T_i, T_j)` (f-measure). The DATASET row is the same measure over the 100 real continuations, which is the control: it shows how similar genuine TinyStories are to one another.

| split | model | mean | median | p75 | max |
|---|---|---:|---:|---:|---:|
| train | TinyStories-1M | 0.269 | 0.256 | 0.306 | 0.545 |
| train | TinyStories-8M | 0.158 | 0.148 | 0.184 | 0.354 |
| train | TinyStories-33M | 0.139 | 0.136 | 0.156 | 0.274 |
| train | TinyStories-1Layer-21M | 0.156 | 0.154 | 0.182 | 0.246 |
| train | DATASET (real continuations) | 0.091 | 0.089 | 0.108 | 0.133 |
| validation | TinyStories-1M | 0.278 | 0.283 | 0.319 | 0.624 |
| validation | TinyStories-8M | 0.162 | 0.151 | 0.182 | 0.351 |
| validation | TinyStories-33M | 0.142 | 0.140 | 0.171 | 0.255 |
| validation | TinyStories-1Layer-21M | 0.170 | 0.170 | 0.202 | 0.361 |
| validation | DATASET (real continuations) | 0.096 | 0.091 | 0.112 | 0.197 |

## Figure 16 - how often does each generated 2-gram occur in the corpus?

`g_r` is the paper's quantity: a generated 2-gram's occurrences divided by the corpus 2-gram total. `frac never` is the share that appear **nowhere** in the corpus.

| split | model | 2-grams | median occurrences | median g_r | frac never in corpus |
|---|---|---:|---:|---:|---:|
| train | TinyStories-1M | 5,451 | 61,981 | 1.66e-04 | 0.2% |
| train | TinyStories-8M | 7,361 | 33,684 | 9.00e-05 | 0.3% |
| train | TinyStories-33M | 7,952 | 26,646 | 7.12e-05 | 0.3% |
| train | TinyStories-1Layer-21M | 6,948 | 33,622 | 8.99e-05 | 0.3% |
| validation | TinyStories-1M | 4,724 | 66,292 | 1.77e-04 | 0.3% |
| validation | TinyStories-8M | 6,869 | 36,105 | 9.65e-05 | 0.2% |
| validation | TinyStories-33M | 7,700 | 27,269 | 7.29e-05 | 0.2% |
| validation | TinyStories-1Layer-21M | 6,393 | 35,629 | 9.52e-05 | 0.1% |

## Figure 17 - closest story in the entire training corpus

`h_i = max_j R_2,p(T_i, S_j)` over all 2.12M training stories. High values would mean the generation closely matches something the model was trained on.

| split | model | mean | median | p75 | max |
|---|---|---:|---:|---:|---:|
| train | TinyStories-1M | 0.486 | 0.465 | 0.551 | 1.000 |
| train | TinyStories-8M | 0.391 | 0.389 | 0.439 | 0.690 |
| train | TinyStories-33M | 0.367 | 0.354 | 0.417 | 0.532 |
| train | TinyStories-1Layer-21M | 0.414 | 0.389 | 0.466 | 0.800 |
| validation | TinyStories-1M | 0.522 | 0.485 | 0.556 | 1.000 |
| validation | TinyStories-8M | 0.427 | 0.394 | 0.473 | 1.000 |
| validation | TinyStories-33M | 0.393 | 0.365 | 0.446 | 0.867 |
| validation | TinyStories-1Layer-21M | 0.433 | 0.406 | 0.493 | 1.000 |

## Train vs validation - is the model memorising?

The paper runs this on TRAINING stories: if a model had memorised them, its completion would match the real ending. We add the validation split, whose prefixes the models never trained on. **If training-set scores are not higher than validation scores, there is no memorisation to find.**

| model | Fig 14 train | Fig 14 validation | difference |
|---|---:|---:|---:|
| TinyStories-1M | 0.131 | 0.145 | -0.014 |
| TinyStories-8M | 0.161 | 0.159 | +0.002 |
| TinyStories-33M | 0.165 | 0.167 | -0.002 |
| TinyStories-1Layer-21M | 0.142 | 0.146 | -0.004 |

| model | Fig 17 train | Fig 17 validation | difference |
|---|---:|---:|---:|
| TinyStories-1M | 0.486 | 0.522 | -0.036 |
| TinyStories-8M | 0.391 | 0.427 | -0.036 |
| TinyStories-33M | 0.367 | 0.393 | -0.027 |
| TinyStories-1Layer-21M | 0.414 | 0.433 | -0.019 |

## Limitations

- The paper does not state the decoding temperature for 4.4.1. We used greedy; sampling would lower the overlaps further, so our conclusion is the conservative one.
- We cannot verify which corpus snapshot each released checkpoint was trained on. We use the current `roneneldan/TinyStories` parquet config; if a checkpoint saw different data, Figures 16/17 are measured against a near-but-not-identical corpus.
- `R_k,p` is implemented on k-gram SETS, exactly as the paper defines it. The common `rouge_score` package intersects multisets and gives slightly different numbers; `compare_with_rouge_score()` quantifies it.
- A very short completion can reach h_i = 1.0 trivially, because all of its handful of 2-grams fit inside one training story. `per_story_scores.csv` records `n_generated_2grams` so these are identifiable; they are a small minority.
- n = 100 stories per split, as in the paper. These are distributions to be read as such, not precise point estimates.

## Files

- `corpus_scan_cache.json`
- `fig_14_15_17_train.png`
- `fig_14_15_17_validation.png`
- `fig_16_train.png`
- `fig_16_validation.png`
- `figure_summaries.csv`
- `per_story_scores.csv`
- `run_info.json`
- `summary.md`
