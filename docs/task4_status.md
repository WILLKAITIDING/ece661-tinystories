# Task 4 status — blocked on the course gateway

**Implementation: complete and tested. Execution: blocked upstream.**
Last checked 2026-11-02. **Total spend to date: $0.00.**

## The blocker

Every chat/completions request to the ECE 661 gateway is rejected by Google
Vertex AI before any token is generated:

```
403 PERMISSION_DENIED
litellm.BadRequestError: Vertex_aiException BadRequestError
  "Request rejected in accordance with our terms of use and policies."
  Received Model Group=gemini-3.6-flash
  Available Model Group Fallbacks=None
```

This is not our key, our code, or our payload. Evidence:

| test | result |
|---|---|
| `GET /models` with our key | **200** — key is valid |
| `GET /models` with a bogus key | 401 — so 403 is not an auth failure |
| `POST /embeddings` (`gemini-embedding-001`) | **200**, returns a real 3072-dim vector |
| `POST /chat/completions`, 9 different models | **403**, identical message every time |
| Both `…/api` and `…/api/v1` base URLs | 403 |
| `curl` and the OpenAI Python SDK | 403 |
| Minimal payload (no system message, no temperature) | 403 |

Models tried: `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`,
`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3-flash-preview`,
`gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gemini-4-argon`.

Embeddings succeeding while every chat model fails isolates the fault to the
gateway's generative-model configuration — most likely the upstream Vertex
project lacks generative-language entitlement, or is under a policy/billing
block. **A TA or the instructor has to fix this; there is no client-side
workaround.**

Note: `GET /api/v1/health` replies *"this path is not available. Use
https://34-127-166-57.sslip.io/api as the OpenAI-compatible base URL."* We
switched to `…/api` accordingly. It did not change the 403, but it is the
documented base and is what `.env.example` now records.

### Reproducing it in one command

```bash
curl -s -H "Authorization: Bearer $COURSE_API_KEY" -H "Content-Type: application/json" \
  -X POST https://34-127-166-57.sslip.io/api/chat/completions \
  -d '{"model":"gemini-3.6-flash","messages":[{"role":"user","content":"hi"}]}'
```

## What is ready to run

The moment the gateway works, this is a single command:

```bash
python scripts/run_task4.py --config configs/task4.yaml --limit 5   # pilot
python scripts/run_task4.py --config configs/task4.yaml             # all 800
python scripts/make_task4_tables.py                                 # tables + figures
```

- `evaluation/judge.py` — OpenAI-SDK client, the Section 3 rubric, strict JSON
  validation, retries, and cost accounting.
- `scripts/run_task4.py` — resume by `judge_id`, per-call usage log, budget guard.
- `scripts/make_task4_tables.py` — tables, figures, examples, methodology note.
- `tests/test_task4.py` — 34 offline tests; no network, no spend.

Verified without spending: the dry run exercises the whole loop, and the
analysis script was run end to end on synthetic scores (then deleted — no
synthetic numbers were kept anywhere under `results/`).

## Cost control

Agreed cap for Task 4 is **$10 total, including any earlier calls**. A personal
"Unlimited" allowance on the course portal is *not* treated as permission to
exceed it.

- Published pricing is pinned in code: `gemini-3.6-flash` at $0.75 / 1M input
  and $3.75 / 1M output.
- A typical judge call is ~650 input + ~120 output tokens ≈ **$0.00094**, so all
  800 records project to **≈ $0.75** — about 7.5% of the cap.
- Cost per call is taken as `max(gateway header, our own pricing calculation)`,
  so a gateway reporting `0` cannot hide real spend.
- The cap is checked **before every call** against the gateway's cumulative
  `x-litellm-key-spend`, so it covers earlier runs and never resets.
- A call-count cap and a circuit breaker (stop after 3 failures with no
  success) prevent a loop from spending anything. The breaker is what stopped
  the real pilot after 3 requests instead of attempting all 800.

## Measured spend

The gateway reported `x-litellm-key-spend: 1.5e-07` after all diagnostics —
about 0.000015 cents, from the one successful embeddings probe. Rejected chat
requests cost nothing. **Effective Task 4 spend: $0.00.**

## Key handling

The course key was found pasted into `COURSE_API_KEY` in **`.env.example`**,
which is tracked by git. It has been moved to `.env` (git-ignored, mode 600)
and `.env.example` restored to placeholders.

**It was never committed.** Verified by searching every blob in the object
store across all branches: 0 matches. The committed `.env.example` has always
had an empty placeholder, and the edit was unstaged.

Rotation is therefore not required for git exposure. It is still worth
rotating if the key was shared anywhere else, since it sat in a tracked file
and a routine `git add -A` would have committed it. `tests/test_task4.py` now
has a regression test asserting `.env.example` carries no key.
