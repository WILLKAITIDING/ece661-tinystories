"""Gemini-as-judge scoring through the ECE 661 course gateway.

Implements the GPT-Eval idea of Section 3 of the TinyStories paper: an LLM
grades a completion on grammar, creativity and consistency, and estimates the
age group the writer reads like. The paper used GPT-4; our proposal uses a
Gemini Flash-tier model through the course gateway, graded at temperature 0.

SPENDING IS REAL MONEY. Three guards, all on by default:

  1. A hard USD cap checked BEFORE every call. The gateway (LiteLLM) returns
     `x-litellm-key-spend`, the cumulative spend of the key, so the cap is
     enforced against the server's own figure and not just our local tally.
     The cap therefore covers earlier calls too - it never "resets".
  2. A maximum call count, so a bug cannot loop.
  3. Every call's tokens and cost are appended to a usage log before the next
     call is made, so an interrupted run never loses its accounting.

Resume is by `judge_id`, derived from the record, judge model, rubric version
and temperature - so re-running skips work already paid for.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

RUBRIC_VERSION = "v1"

# Published course pricing for the recommended judge, USD per 1M tokens. Used
# to compute cost locally: the gateway reports a per-response cost header, but
# we take the larger of the two so an under-reporting gateway cannot let the
# budget cap drift.
PRICING_USD_PER_MTOK = {
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
}
DEFAULT_PRICING = {"input": 0.75, "output": 3.75}


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int,
                      model: str) -> float:
    """Cost from published per-token pricing."""
    rates = PRICING_USD_PER_MTOK.get(model, DEFAULT_PRICING)
    return (prompt_tokens * rates["input"]
            + completion_tokens * rates["output"]) / 1_000_000


# Scored 1-10 like the paper's GPT-Eval prompt; the age group is a bucket.
SCORE_FIELDS = ("grammar", "creativity", "consistency")
AGE_BUCKETS = ("A: 3 or under", "B: 4-5", "C: 6-7", "D: 8-9", "E: 10-12", "F: 13-16")

SYSTEM_PROMPT = (
    "You are grading short stories written by small language models for young "
    "children. You are strict, consistent, and you answer only with JSON."
)

USER_TEMPLATE = """The following exercise: a student is given the beginning of a story. The student needs to complete it into a full story. The exercise tests the student's language abilities and creativity.

The beginning of the story is:
{prompt}

The student's completion is:
{completion}

Grade the student's completion. Give an integer from 1 to 10 for each of:
- grammar: is the writing grammatical and well-formed English?
- creativity: is the continuation imaginative rather than generic or repetitive?
- consistency: is it consistent with the beginning, keeping characters, objects and facts straight?

Then estimate the age group the student writes like, choosing exactly one of:
{age_buckets}

Answer with ONLY a JSON object, no prose and no code fence:
{{"grammar": <1-10>, "creativity": <1-10>, "consistency": <1-10>, "age_group": "<one bucket exactly as written>", "note": "<at most 15 words>"}}"""


class BudgetExceeded(RuntimeError):
    """Raised when a call would push spending past the configured cap."""


class JudgeError(RuntimeError):
    """A call failed in a way that is not worth retrying."""


def build_messages(prompt: str, completion: str) -> list[dict[str, str]]:
    """The judge prompt. Identical for every model, so scores are comparable."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            prompt=prompt.strip(),
            completion=completion.strip(),
            age_buckets="\n".join(AGE_BUCKETS),
        )},
    ]


def make_judge_id(record_id: str, judge_model: str, temperature: float,
                  rubric_version: str = RUBRIC_VERSION) -> str:
    """Stable id for one judging request, so a resumed run skips paid work."""
    payload = json.dumps({"record_id": record_id, "judge_model": judge_model,
                          "temperature": temperature, "rubric": rubric_version},
                         sort_keys=True)
    return "j_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def parse_scores(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a judge reply and validate it.

    Models sometimes wrap JSON in a code fence or add a sentence, so we locate
    the outermost braces rather than assuming the whole reply is JSON.
    """
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise JudgeError(f"no JSON object in judge reply: {text[:160]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge reply is not valid JSON: {exc}") from exc

    out: dict[str, Any] = {}
    for field_name in SCORE_FIELDS:
        if field_name not in data:
            raise JudgeError(f"judge reply is missing {field_name!r}")
        try:
            value = int(round(float(data[field_name])))
        except (TypeError, ValueError) as exc:
            raise JudgeError(f"{field_name} is not a number: {data[field_name]!r}") from exc
        if not 1 <= value <= 10:
            raise JudgeError(f"{field_name}={value} outside 1-10")
        out[field_name] = value

    age = str(data.get("age_group", "")).strip()
    if age not in AGE_BUCKETS:
        # Accept a bare letter ("E") or a loose match, else record it unparsed
        # rather than silently inventing a bucket.
        letter = age[:1].upper()
        matched = [b for b in AGE_BUCKETS if b.startswith(letter + ":")]
        age = matched[0] if matched else ""
    out["age_group"] = age
    out["note"] = str(data.get("note", ""))[:200]
    return out


@dataclass
class Usage:
    """Token and cost accounting for one call, as reported by the gateway."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_cost_usd: float = 0.0         # the figure the budget uses
    reported_cost_usd: float = 0.0         # what the gateway header said
    computed_cost_usd: float = 0.0         # from published per-token pricing
    key_spend_usd: float | None = None     # cumulative, from the gateway
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetGuard:
    """Enforces a hard USD cap across the whole run, earlier calls included."""

    max_spend_usd: float
    max_calls: int = 10_000
    spent_usd: float = 0.0            # our own tally for this run
    calls: int = 0
    server_spend_usd: float | None = None   # the gateway's cumulative figure
    _baseline: float | None = field(default=None, repr=False)

    def observed_total(self) -> float:
        """Best available total spend: prefer the gateway's own number."""
        if self.server_spend_usd is not None:
            return self.server_spend_usd
        return self.spent_usd

    def check(self, estimated_cost_usd: float = 0.0) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"call cap reached ({self.calls}/{self.max_calls})")
        projected = self.observed_total() + estimated_cost_usd
        if projected > self.max_spend_usd:
            raise BudgetExceeded(
                f"spend cap would be exceeded: observed ${self.observed_total():.4f} "
                f"+ estimated ${estimated_cost_usd:.4f} > cap ${self.max_spend_usd:.2f}")

    def record(self, usage: Usage) -> None:
        self.calls += 1
        self.spent_usd += usage.response_cost_usd
        if usage.key_spend_usd is not None:
            if self._baseline is None:
                self._baseline = usage.key_spend_usd - usage.response_cost_usd
            self.server_spend_usd = usage.key_spend_usd
        # A cap breach detected after the fact still stops the run.
        if self.observed_total() > self.max_spend_usd:
            raise BudgetExceeded(
                f"spend cap exceeded: ${self.observed_total():.4f} > "
                f"${self.max_spend_usd:.2f}")


class JudgeClient:
    """Chat client for the course gateway, via the OpenAI-compatible SDK."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        timeout: int = 60,
        max_retries: int = 3,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError(
                "No API key. Put COURSE_API_KEY in .env (git-ignored); never in "
                ".env.example, which is tracked.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._sleep = sleep
        if client is not None:
            self._client = client          # injected for offline tests
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=self.base_url,
                                  timeout=timeout, max_retries=0)

    def __repr__(self) -> str:   # never leak the key in a traceback or notebook
        return f"JudgeClient(base_url={self.base_url!r}, model={self.model!r})"

    def score(self, prompt: str, completion: str) -> tuple[dict[str, Any], Usage]:
        """Grade one completion. Returns (scores, usage)."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._client.chat.completions.with_raw_response.create(
                    model=self.model,
                    messages=build_messages(prompt, completion),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                # 400/401/403 will not fix themselves; fail fast rather than
                # burning retries on a broken configuration.
                if status in (400, 401, 403, 404):
                    raise JudgeError(f"HTTP {status}: {str(exc)[:400]}") from exc
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(2 ** attempt)
                continue

            latency = time.time() - t0
            parsed = resp.parse()
            usage = _usage_from_sdk(parsed, dict(resp.headers), latency, self.model)
            text = parsed.choices[0].message.content or ""
            return parse_scores(text), usage

        raise JudgeError(f"giving up after {self.max_retries} attempts: {last_error}")


def _usage_from_sdk(parsed: Any, headers: dict, latency: float,
                    model: str) -> Usage:
    """Token counts from the SDK response, cost from headers and pricing."""
    usage = Usage(latency_s=round(latency, 3))

    def header_float(name: str) -> float | None:
        raw = headers.get(name)
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    tokens = getattr(parsed, "usage", None)
    usage.prompt_tokens = int(getattr(tokens, "prompt_tokens", 0) or 0)
    usage.completion_tokens = int(getattr(tokens, "completion_tokens", 0) or 0)
    usage.total_tokens = int(getattr(tokens, "total_tokens", 0)
                             or usage.prompt_tokens + usage.completion_tokens)
    usage.key_spend_usd = header_float("x-litellm-key-spend")
    reported = header_float("x-litellm-response-cost") or 0.0
    computed = estimate_cost_usd(usage.prompt_tokens, usage.completion_tokens, model)
    # Take the larger: a gateway that reports 0 must not hide real spend.
    usage.response_cost_usd = max(reported, computed)
    usage.reported_cost_usd = reported
    usage.computed_cost_usd = computed
    return usage


def load_credentials(env_path: str | Path = ".env") -> dict[str, str]:
    """Read COURSE_* settings from the environment, falling back to .env.

    `.env` is git-ignored. We never read `.env.example`, which is tracked and
    must only ever hold placeholders.
    """
    values = {k: os.environ.get(k, "") for k in
              ("COURSE_API_KEY", "COURSE_API_BASE_URL", "COURSE_JUDGE_MODEL")}
    path = Path(env_path)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key in values and not values[key] and val:
                values[key] = val
    return values
