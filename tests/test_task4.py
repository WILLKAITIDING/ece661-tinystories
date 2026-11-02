"""Task 4: the Gemini judge client, score validation and budget guard.

Entirely offline. A FakeOpenAI stands in for the SDK, so the rubric, parsing,
retry, resume and spending logic are all exercised without a network call or a
cent of spend.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tinystories_project.evaluation.judge import (
    AGE_BUCKETS,
    PRICING_USD_PER_MTOK,
    RUBRIC_VERSION,
    BudgetExceeded,
    BudgetGuard,
    JudgeClient,
    JudgeError,
    Usage,
    build_messages,
    estimate_cost_usd,
    load_credentials,
    make_judge_id,
    parse_scores,
)

GOOD = '{"grammar": 7, "creativity": 4, "consistency": 8, "age_group": "C: 6-7", "note": "ok"}'


# --------------------------------------------------------------- fake SDK
class _Raw:
    def __init__(self, content, headers, usage):
        self._content = content
        self.headers = headers
        self._usage = usage

    def parse(self):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=SimpleNamespace(**self._usage),
        )


class FakeOpenAI:
    """Mimics `client.chat.completions.with_raw_response.create(...)`."""

    def __init__(self, content=GOOD, headers=None, usage=None, errors=None):
        self.content, self.calls, self.errors = content, [], list(errors or [])
        self.headers = headers if headers is not None else {
            "x-litellm-response-cost": "0.0009", "x-litellm-key-spend": "0.0009"}
        self.usage = usage or {"prompt_tokens": 650, "completion_tokens": 120,
                               "total_tokens": 770}
        inner = SimpleNamespace(with_raw_response=SimpleNamespace(create=self._create))
        self.chat = SimpleNamespace(completions=inner)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return _Raw(self.content, self.headers, self.usage)


def make_client(**kw):
    fake = FakeOpenAI(**kw)
    return JudgeClient(base_url="https://gw.example/api", api_key="sk-test",
                       model="gemini-3.6-flash", client=fake,
                       sleep=lambda _s: None), fake


# --------------------------------------------------------------- rubric
def test_rubric_asks_for_all_four_dimensions():
    msgs = build_messages("Once upon a time,", " the cat sat.")
    user = msgs[1]["content"]
    for term in ("grammar", "creativity", "consistency", "age group"):
        assert term in user.lower(), term
    assert all(b in user for b in AGE_BUCKETS)
    assert msgs[0]["role"] == "system"


def test_rubric_is_identical_for_every_story_except_the_text():
    a = build_messages("P1", "C1")[1]["content"]
    b = build_messages("P2", "C2")[1]["content"]
    assert a.replace("P1", "X").replace("C1", "Y") == b.replace("P2", "X").replace("C2", "Y")


# --------------------------------------------------------------- parsing
def test_parse_accepts_plain_json():
    got = parse_scores(GOOD)
    assert got["grammar"] == 7 and got["creativity"] == 4 and got["consistency"] == 8
    assert got["age_group"] == "C: 6-7"


def test_parse_accepts_a_code_fence_and_surrounding_prose():
    got = parse_scores('Sure!\n```json\n' + GOOD + '\n```\nHope that helps.')
    assert got["grammar"] == 7


def test_parse_recovers_a_bare_age_letter():
    got = parse_scores('{"grammar":5,"creativity":5,"consistency":5,"age_group":"E"}')
    assert got["age_group"] == "E: 10-12"


def test_parse_leaves_an_unrecognised_age_blank_rather_than_inventing_one():
    got = parse_scores('{"grammar":5,"creativity":5,"consistency":5,"age_group":"grown up"}')
    assert got["age_group"] == ""


def test_parse_rounds_a_float_score():
    assert parse_scores('{"grammar":6.6,"creativity":5,"consistency":5}')["grammar"] == 7


@pytest.mark.parametrize("bad, match", [
    ("no json here", "no JSON object"),
    ('{"grammar": 7, "creativity": 4}', "consistency"),
    ('{"grammar": 99, "creativity": 4, "consistency": 8}', "outside 1-10"),
    ('{"grammar": 0, "creativity": 4, "consistency": 8}', "outside 1-10"),
    ('{"grammar": "good", "creativity": 4, "consistency": 8}', "not a number"),
    ('{"grammar": 7, "creativity": 4, "consistency": }', "not valid JSON"),
    ('{"grammar": 7, "creativity": 4, "consistency": 8,', "no JSON object"),
])
def test_parse_rejects_malformed_replies(bad, match):
    with pytest.raises(JudgeError, match=match):
        parse_scores(bad)


# --------------------------------------------------------------- client
def test_score_returns_scores_and_usage():
    client, fake = make_client()
    scores, usage = client.score("Once upon a time,", " the cat sat.")
    assert scores["grammar"] == 7
    assert usage.prompt_tokens == 650 and usage.completion_tokens == 120
    assert fake.calls[0]["model"] == "gemini-3.6-flash"
    assert fake.calls[0]["temperature"] == 0.0, "judging must be deterministic"


def test_cost_takes_the_larger_of_reported_and_computed():
    """A gateway reporting 0 must not be able to hide real spend."""
    client, _ = make_client(headers={"x-litellm-response-cost": "0",
                                     "x-litellm-key-spend": "0"})
    _scores, usage = client.score("p", "c")
    assert usage.reported_cost_usd == 0.0
    assert usage.computed_cost_usd == pytest.approx(
        estimate_cost_usd(650, 120, "gemini-3.6-flash"))
    assert usage.response_cost_usd == usage.computed_cost_usd > 0


def test_published_pricing_is_pinned():
    assert PRICING_USD_PER_MTOK["gemini-3.6-flash"] == {"input": 0.75, "output": 3.75}
    assert estimate_cost_usd(1_000_000, 0, "gemini-3.6-flash") == pytest.approx(0.75)
    assert estimate_cost_usd(0, 1_000_000, "gemini-3.6-flash") == pytest.approx(3.75)


def test_client_fails_fast_on_403_without_retrying():
    """A misconfigured endpoint must not burn retries, or money, 800 times."""
    err = Exception("forbidden")
    err.status_code = 403
    client, fake = make_client(errors=[err])
    with pytest.raises(JudgeError, match="HTTP 403"):
        client.score("p", "c")
    assert len(fake.calls) == 1, "403 must not be retried"


def test_client_retries_a_transient_error_then_succeeds():
    transient = Exception("boom")
    transient.status_code = 503
    client, fake = make_client(errors=[transient])
    scores, _usage = client.score("p", "c")
    assert scores["grammar"] == 7
    assert len(fake.calls) == 2


def test_client_gives_up_after_max_retries():
    transient = Exception("boom")
    transient.status_code = 503
    client, _ = make_client(errors=[transient] * 5)
    with pytest.raises(JudgeError, match="giving up"):
        client.score("p", "c")


def test_client_requires_a_key():
    with pytest.raises(ValueError, match="No API key"):
        JudgeClient(base_url="https://x/api", api_key="", model="m", client=object())


def test_repr_never_leaks_the_key():
    client, _ = make_client()
    assert "sk-test" not in repr(client)


# --------------------------------------------------------------- budget
def test_guard_blocks_a_call_that_would_breach_the_cap():
    guard = BudgetGuard(max_spend_usd=0.01)
    guard.spent_usd = 0.0098
    with pytest.raises(BudgetExceeded, match="would be exceeded"):
        guard.check(0.0005)


def test_guard_allows_a_call_inside_the_cap():
    BudgetGuard(max_spend_usd=10.0).check(0.001)      # must not raise


def test_guard_enforces_a_call_cap():
    guard = BudgetGuard(max_spend_usd=10.0, max_calls=2)
    guard.calls = 2
    with pytest.raises(BudgetExceeded, match="call cap"):
        guard.check(0.0)


def test_guard_prefers_the_gateways_own_cumulative_spend():
    """The cap must cover earlier runs, so it never 'resets'."""
    guard = BudgetGuard(max_spend_usd=10.0)
    guard.record(Usage(response_cost_usd=0.001, key_spend_usd=4.0))
    assert guard.observed_total() == 4.0      # not 0.001
    with pytest.raises(BudgetExceeded):
        guard.check(7.0)                      # 4.0 + 7.0 > 10.0


def test_guard_stops_even_if_a_breach_is_only_seen_after_the_call():
    guard = BudgetGuard(max_spend_usd=1.0)
    with pytest.raises(BudgetExceeded, match="cap exceeded"):
        guard.record(Usage(response_cost_usd=0.5, key_spend_usd=1.5))


def test_full_run_cost_projection_stays_within_the_agreed_cap():
    """800 records at the pinned pricing must fit in the $10 Task 4 budget."""
    per_call = estimate_cost_usd(650, 120, "gemini-3.6-flash")
    assert per_call * 800 < 10.0
    assert per_call * 800 == pytest.approx(0.75, abs=0.01)


# --------------------------------------------------------------- resume
def test_judge_id_is_stable_and_separates_every_dimension():
    base = ("r_1", "gemini-3.6-flash", 0.0)
    ref = make_judge_id(*base)
    assert ref == make_judge_id(*base)
    assert ref != make_judge_id("r_2", "gemini-3.6-flash", 0.0)
    assert ref != make_judge_id("r_1", "gemini-2.5-flash", 0.0)
    assert ref != make_judge_id("r_1", "gemini-3.6-flash", 1.0)
    assert ref != make_judge_id("r_1", "gemini-3.6-flash", 0.0, rubric_version="v2")


def test_rubric_version_is_recorded_so_scores_are_traceable():
    assert RUBRIC_VERSION


# --------------------------------------------------------------- credentials
def test_credentials_are_read_from_a_dotenv(tmp_path, monkeypatch):
    for var in ("COURSE_API_KEY", "COURSE_API_BASE_URL", "COURSE_JUDGE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nCOURSE_API_KEY="sk-xyz"\n'
                   'COURSE_API_BASE_URL=https://gw/api\n\n', encoding="utf-8")
    got = load_credentials(env)
    assert got["COURSE_API_KEY"] == "sk-xyz"
    assert got["COURSE_API_BASE_URL"] == "https://gw/api"


def test_environment_wins_over_the_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_API_KEY", "sk-from-env")
    env = tmp_path / ".env"
    env.write_text("COURSE_API_KEY=sk-from-file\n", encoding="utf-8")
    assert load_credentials(env)["COURSE_API_KEY"] == "sk-from-env"


def test_missing_dotenv_is_not_an_error(tmp_path, monkeypatch):
    for var in ("COURSE_API_KEY", "COURSE_API_BASE_URL", "COURSE_JUDGE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert load_credentials(tmp_path / "nope.env")["COURSE_API_KEY"] == ""


def test_tracked_env_example_contains_no_real_key():
    """Regression guard: the key belongs in .env, which is git-ignored."""
    from tinystories_project.config import find_project_root
    text = (find_project_root() / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("COURSE_API_KEY="):
            assert line.strip() == "COURSE_API_KEY=", f"key leaked: {line!r}"


# --------------------------------------------------------------- resume
def _resume_sets(rows: list[dict]) -> tuple[set, int]:
    """Mirror the runner's resume rule: only SCORED rows count as done."""
    done, failed = set(), 0
    for row in rows:
        if row.get("ok") and row.get("grammar") is not None:
            done.add(row["judge_id"])
        else:
            failed += 1
    return done, failed


def test_resume_skips_scored_records_but_retries_failures():
    """Regression: a failed attempt must NOT be treated as already judged.

    Earlier this counted every row in scores.jsonl, so records that failed
    (e.g. during the gateway outage) would have been skipped for ever and
    silently missing from the final results.
    """
    rows = [
        {"judge_id": "j_ok", "ok": True, "grammar": 7},
        {"judge_id": "j_failed", "ok": False, "error": "HTTP 403"},
        {"judge_id": "j_dry", "ok": True, "grammar": None},   # dry-run placeholder
    ]
    done, failed = _resume_sets(rows)
    assert done == {"j_ok"}
    assert "j_failed" not in done, "a failed attempt must be retried"
    assert "j_dry" not in done, "a dry-run placeholder is not a score"
    assert failed == 2


def test_resume_never_repays_for_a_completed_score():
    rows = [{"judge_id": "j_a", "ok": True, "grammar": 8},
            {"judge_id": "j_b", "ok": True, "grammar": 5}]
    done, failed = _resume_sets(rows)
    assert done == {"j_a", "j_b"} and failed == 0


def test_analysis_refuses_dry_run_rows(tmp_path):
    """A placeholder must never be aggregated as if it were a judgement.

    Runs the real analysis script, which imports matplotlib from the optional
    `.[analysis]` extra, so it skips rather than fails on a bare install.
    """
    pytest.importorskip("matplotlib", reason="install .[analysis] for this check")
    import json as _json
    import subprocess
    import sys as _sys
    from tinystories_project.config import find_project_root

    path = tmp_path / "scores.jsonl"
    path.write_text(_json.dumps({
        "judge_id": "j_1", "record_id": "r_1", "ok": True, "dry_run": True,
        "grammar": None, "creativity": None, "consistency": None,
        "judge_model": "m", "split": "train", "model": "TinyStories-1M"}) + "\n",
        encoding="utf-8")
    proc = subprocess.run(
        [_sys.executable, "scripts/make_task4_tables.py", "--scores", str(path),
         "--out", str(tmp_path / "out")],
        cwd=find_project_root(), capture_output=True, text=True)
    assert proc.returncode != 0
    assert "dry-run" in (proc.stdout + proc.stderr)
