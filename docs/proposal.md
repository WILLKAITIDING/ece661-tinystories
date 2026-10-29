# Project Proposal (transcribed)

**Source of truth:** `ECE 661 Project Proposal .pdf`, located in the parent
directory of this repository (`ECE661/FinalProject/`). This file is a faithful
transcription of the commitments in that PDF so that code and agents can read
them. **If the two ever disagree, the PDF wins.** Nothing here is invented; where
the proposal is silent, this file says so explicitly.

- **Title:** Tiny Language Models: Architecture Exploration and Cross-Lingual Transfer to Amharic
- **Topic:** Project 9 (Tiny Language Models), with a custom optional extension
- **Course:** Duke ECE 661, Fall 2026 — **Group 32**

## Members and responsibilities

| Member | Responsibilities |
|---|---|
| **Will Ding** | Phase 1 replication, ROUGE and judge pipeline, Phase 1/2 evaluation |
| **Rachel Odonkor** | Phase 2 training infrastructure, architecture variants, English matched-subset baselines |
| **Luel Hagos** (native Amharic speaker) | Amharic vocabulary and prompts, pilot audit, tokenizer, Phase 3 training and native-speaker rating |
| All | Weekly sync, report, poster; each member presents their own phase at the poster session |

## Phase 1 — Replication and evaluation pipeline

- Use the released TinyStories checkpoints (**1M–33M, 1-layer, 2-layer**) and dataset.
- Reproduce the factual, reasoning, and context-tracking prompts of **Figures 9–11**, plus **20 prompts of our own**.
- Following **Section 4.4.1**: sample **100 training stories**, prompt with the **first 40%**, and compute the **four ROUGE-based measures** (Figures 14–17) to confirm the models are not memorising.
- Implement the **GPT-Eval judge (Section 3) with Gemini**: **10 completions per prompt at temperature 1**, graded at **temperature 0** on **grammar, creativity, consistency, and an estimated age group**.

## Phase 2 — Architecture exploration (English)

Parameter count (**~5M and ~10M**), token budget, and optimiser fixed; four variants trained from scratch:

- **A.** GPT-Neo baseline, matching the architecture used in the paper
- **B.** Deep-narrow vs shallow-wide (tests the Section 4.2 claim that width drives factual recall, depth drives context tracking)
- **C.** Modernised transformer with **RoPE, RMSNorm, GeGLU, GQA**
- **D.** Parameter-matched non-attention baseline (**minGRU or Mamba**)

All are evaluated with the Phase 1 pipeline.

## Phase 3 — TinyStories for Amharic (+ optional Kinyarwanda)

- Re-create the Section 2 recipe for Amharic: native-speaker-curated **~1,500-word** child vocabulary; one noun, verb, and adjective sampled per prompt, plus random story features.
- A **Gemini Flash-tier** model generates **on the order of 100K stories** via the Batch API; final count set from a **100-story pilot** audited by our native Amharic speaker first.
- Corpus filtered by **Ge'ez script validation** and **n-gram deduplication**.
- GPT-Neo vocabulary has no Ge'ez coverage → train an Amharic **SentencePiece** tokenizer (**V = 4K or 8K**, chosen by a fertility pilot).
- Compare **training from scratch** vs **initialising from English checkpoints with a re-initialised embedding table**.
- Evaluate with translated Figure 9–11 prompts, the Gemini judge, and a **blinded native-speaker rating on a 50-prompt set**.
- **Kinyarwanda** extension only if Amharic is on schedule and under budget; otherwise reported as future work.

## Risks named in the proposal

Corpus size · synthetic data quality · morphology and tokenisation · negative cross-script transfer · judge bias · budget (Ge'ez tokenisation may be 2–3× less efficient than English).

## Timeline

| Week | Dates | Focus |
|---|---|---|
| 1 | Oct 26 – Nov 1 | **Replication and setup** — read the paper; reproduce Figures 9–11; design 20 additional prompts; verify toolchain; begin Amharic vocabulary |
| 2 | Nov 2 – 8 | Quantitative replication (ROUGE), Gemini judge pipeline; Amharic 100-story pilot + full generation |
| 3 | Nov 9 – 15 | Architecture exploration: train four variants at ~5M and ~10M; re-run evaluations |
| 4 | Nov 16 – 22 | Amharic training; English matched-subset baselines; midterm TA check-in; Kinyarwanda go/no-go |
| 5 | Nov 23 – 29 | Amharic evaluation, judge–human agreement, report draft |
| 6 | Nov 30 – Dec 6 | Analysis, tables/figures, poster, report |
| — | Dec 9 – 11 | Poster session; final report |

## Success criteria

1. **Replication:** Figures 9–11 and 14–17 reproduced side by side with the paper's results.
2. **Architecture comparison:** one table, four variants × two sizes — validation loss/perplexity, judge scores, efficiency (params, throughput, latency).
3. **Cross-language transfer:** Amharic from-scratch vs English-initialised on bits-per-byte, judge scores, native-speaker ratings, plus judge–human agreement.
4. **Artifacts:** a public code repository, the Amharic corpus and tokenizer, and trained checkpoints.

A negative result counts as success if it is measured carefully and explained.

## Budget note

The course provides a personal API key per student with a **shared team budget of $250** split between Gemini API calls and TPU compute. See `CLAUDE.md` for the hard rules about not spending it.

## Not specified by the proposal

The proposal does **not** fix: the exact 20 additional prompts; which specific checkpoint sizes are used for each figure; generation length for ROUGE prefixes; or the judge prompt wording. These are open decisions, to be recorded here as they are made.

## References

See the PDF for the full list: TinyStories (2305.07759), Chinchilla (2203.15556), XLM (1901.07291), XLM-R (1911.02116), RoFormer/RoPE (2104.09864), RMSNorm (1910.07467), GLU Variants (2002.05202), GQA (2305.13245), minGRU (2410.01201), Mamba (2312.00752), SentencePiece (1808.06226).
