"""Evaluation (Will, Phase 1). NOT IMPLEMENTED YET - Tasks 2-4.

Planned modules, all consuming `io.records.GenerationRecord` so they work
unchanged on Rachel's Phase 2 models and Luel's Phase 3 Amharic models:

    qualitative.py  Figures 9-11 prompt sets + our 20 additional prompts
    rouge.py        Section 4.4.1 / Figures 14-17 train-vs-validation ROUGE
    judge.py        Gemini judge via the OpenAI-compatible course proxy
                    (grammar, creativity, consistency, estimated age group),
                    with retries, cost logging, and resume support
    aggregate.py    cross-model tables for the architecture comparison

Do not call the course Gemini API from this package until the judge task is
explicitly started; it spends a shared team budget.
"""
