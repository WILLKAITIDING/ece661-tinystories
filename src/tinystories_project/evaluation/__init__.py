"""Evaluation (Will, Phase 1).

Task 2 (qualitative replication) is implemented. ROUGE (Task 3) and the Gemini
judge (Task 4) are not started; do not call the course API from here yet.

Everything consumes the shared `TextGenerator` protocol and writes the standard
`GenerationRecord` schema, so Rachel's and Luel's models work unchanged.
"""

from tinystories_project.evaluation.prompts import (
    CATEGORIES,
    EvalPrompt,
    PromptSet,
    load_prompt_set,
    load_prompt_sets,
)
from tinystories_project.evaluation.qualitative import (
    PROTOCOLS,
    answer_span,
    run_prompt_set,
    summarise,
)

__all__ = [
    "CATEGORIES", "EvalPrompt", "PROTOCOLS", "PromptSet", "answer_span",
    "load_prompt_set", "load_prompt_sets", "run_prompt_set", "summarise",
]
