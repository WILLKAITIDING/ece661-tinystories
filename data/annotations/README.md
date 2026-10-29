# Task 2 final annotations

`task2_final.csv` holds the final label for all **176** protocol-A (greedy)
completions: 4 checkpoints x 44 prompts. It is a complete export of the Apple
Numbers workbook Will annotated in (`task2_annotations_final.numbers`, kept
under the git-ignored `results/task2/analysis/`). The CSV is what the code
reads and the version under version control; keep a backup of the .numbers
file yourself if you want to re-open it in Numbers.

## Provenance — read before citing these numbers

Labels were **drafted by an AI assistant and then reviewed by Will Ding**, who
overrode **9** of them and explicitly accepted the remaining **167**. The
`notes` column records which: overridden rows read
`User override of AI draft label (<from> -> <to>)`, accepted rows keep the
`AI draft rationale: ...` text.

So these are **human-reviewed AI-drafted labels**, not independent human
annotation. Describe them that way in the report, and disclose the AI
assistance as the course rules require.

## Label scheme

Mirrors the colour coding of Figures 9-11 in the TinyStories paper:

| label | paper colour | meaning |
|---|---|---|
| `correct` | green | satisfies the expected answer or inference |
| `partial` | yellow | partly right, or right but flawed/incomplete |
| `incorrect` | red | contradicts the prompt or answers inappropriately |

(`ambiguous` is permitted by the loader but was not used.)

Counts: **82 correct, 25 partial, 69 incorrect**.

Join to the generation records on `record_id`; `scripts/make_task2_tables.py`
validates every id against `results/task2/A_greedy/*.jsonl` before aggregating.
