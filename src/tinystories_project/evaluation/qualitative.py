"""Qualitative evaluation: run a prompt set through any TextGenerator.

Works on anything satisfying `inference.interface.TextGenerator` - the released
TinyStories checkpoints today, Rachel's minGRU/Mamba models and Luel's Amharic
models later - because it never touches a model object directly.

Two protocols, kept separate on purpose
---------------------------------------
PROTOCOL_A (greedy) reproduces the paper. Section 4 of arXiv:2305.07759v2 says
"examples from this point on will [be] generated at zero temperature", and
Figures 9-11 come after that sentence. Our GenerationParams rejects
temperature=0 (it is not a valid sampling temperature), so zero-temperature
decoding is expressed the correct way: do_sample=False.

PROTOCOL_B (sampling) answers the course question "do the models CONSISTENTLY
produce coherent stories?", which a single greedy completion cannot. It draws
several completions per prompt at a fixed, recorded seed.

Protocol A results are the ones comparable to the paper. Protocol B samples
must never be presented as reproductions of the paper's deterministic examples.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from tinystories_project.evaluation.prompts import EvalPrompt, PromptSet
from tinystories_project.inference.interface import (
    GeneratedText,
    GenerationParams,
    TextGenerator,
)
from tinystories_project.records import GenerationRecord, build_records

# Figures 9-11 show short phrase- or sentence-length answers, so a small token
# budget is both faithful and cheap. The budget is identical for every model -
# never tune a stopping rule per model, that would manufacture a result.
DEFAULT_MAX_NEW_TOKENS = 60

PROTOCOLS: dict[str, GenerationParams] = {
    "A_greedy": GenerationParams(
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        do_sample=False,
        num_return_sequences=1,
    ),
    "B_sampling": GenerationParams(
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        num_return_sequences=3,
    ),
}

# First terminal punctuation, plus any closing quote/bracket glued to it.
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’\)\]]*")


def answer_span(completion: str, max_sentences: int = 1) -> str:
    """The first sentence(s) of a completion - the part a figure would show.

    The paper does not document how it truncated the completions it displays
    (its cells range from a two-word phrase to two sentences), so we cannot
    reproduce its exact rule. We therefore keep the FULL continuation in every
    record and derive this span only for display and annotation. The same rule
    is applied to every model.
    """
    text = completion.strip()
    if not text:
        return ""
    end = 0
    taken = 0
    while taken < max_sentences:
        match = _SENTENCE_END.search(text, end)
        if match is None:
            return text
        end = match.end()
        # Several Figure 9-11 prompts end without punctuation, so the model's
        # first token is often the full stop that closes the PROMPT's sentence.
        # A span with no letters or digits is that stray punctuation, not an
        # answer - keep reading.
        if any(ch.isalnum() for ch in text[:end]):
            taken += 1
    return text[:end].strip()


def run_prompt_set(
    generator: TextGenerator,
    prompt_set: PromptSet,
    *,
    protocol: str,
    run_id: str,
    params: GenerationParams | None = None,
    seed: int | None = 42,
    extra: dict[str, Any] | None = None,
) -> list[GenerationRecord]:
    """Generate completions for every prompt and return JSONL-ready records.

    Each record's `extra` carries the evaluation metadata (prompt id, category,
    figure, origin, protocol, prompt set, answer span) so later tasks can join
    on it without re-reading the prompt files.
    """
    if protocol not in PROTOCOLS and params is None:
        raise ValueError(
            f"Unknown protocol {protocol!r}; expected one of {sorted(PROTOCOLS)} "
            "or an explicit params="
        )
    params = params or PROTOCOLS[protocol]

    generated = generator.generate(prompt_set.texts, params, seed=seed)
    return _records_for(generated, prompt_set.prompts, generator, params,
                        protocol, prompt_set, run_id, extra)


def _records_for(
    generated: Sequence[GeneratedText],
    prompts: Sequence[EvalPrompt],
    generator: TextGenerator,
    params: GenerationParams,
    protocol: str,
    prompt_set: PromptSet,
    run_id: str,
    extra: dict[str, Any] | None,
) -> list[GenerationRecord]:
    records = build_records(generated, generator.metadata, params, run_id=run_id)
    for record, item in zip(records, generated):
        if not 0 <= item.prompt_index < len(prompts):
            raise IndexError(
                f"generator returned prompt_index {item.prompt_index} for "
                f"{len(prompts)} prompts"
            )
        prompt = prompts[item.prompt_index]
        if item.prompt != prompt.text:
            raise ValueError(
                f"generator returned text for prompt_index {item.prompt_index} "
                f"that does not match prompt {prompt.id!r}"
            )
        record.extra = {
            **(extra or {}),
            "prompt_id": prompt.id,
            "category": prompt.category,
            "figure": prompt.figure,
            "origin": prompt.origin,
            "protocol": protocol,
            "prompt_set": prompt_set.name,
            "answer_span": answer_span(item.completion),
        }
    return records


def summarise(records: Sequence[GenerationRecord]) -> dict[str, Any]:
    """Descriptive counts for a run. No quality judgement is made here."""
    by_category: dict[str, int] = {}
    empty = 0
    lengths = []
    for r in records:
        cat = r.extra.get("category", "?")
        by_category[cat] = by_category.get(cat, 0) + 1
        if not r.completion.strip():
            empty += 1
        if r.n_new_tokens is not None:
            lengths.append(r.n_new_tokens)
    return {
        "n_records": len(records),
        "by_category": by_category,
        "n_empty_completions": empty,
        "mean_new_tokens": round(sum(lengths) / len(lengths), 1) if lengths else None,
        "n_hit_token_limit": sum(1 for r in records if r.finish_reason == "length"),
    }


def run_fingerprint(
    metadata: Any,
    params: GenerationParams,
    *,
    protocol: str,
    seed: int | None,
    batch_size: int,
    prompt_set: PromptSet,
) -> dict[str, Any]:
    """Everything that decides a run's output, for safe resume.

    A previous output file may be reused only if its fingerprint matches
    exactly. This covers every dimension that changes the generation request:
    model, revision, tokenizer, protocol, sampling params, seed, batch size
    (which shifts per-batch seeds), and the prompts themselves. Comparing
    filenames or timestamps would not.
    """
    return {
        "model_id": metadata.model_id,
        "revision": metadata.revision,
        "tokenizer_id": metadata.tokenizer_id,
        "protocol": protocol,
        "params": params.to_dict(),
        "seed": seed,
        "batch_size": batch_size,
        "prompt_set": prompt_set.name,
        "prompt_ids": [p.id for p in prompt_set.prompts],
        "n_expected": len(prompt_set) * params.num_return_sequences,
    }
