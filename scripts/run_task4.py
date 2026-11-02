#!/usr/bin/env python
"""Task 4 runner: Gemini-as-judge scoring of the Task 3 completions.

    python scripts/run_task4.py --config configs/task4.yaml --dry-run   # no API calls
    python scripts/run_task4.py --config configs/task4.yaml --limit 8   # small pilot
    python scripts/run_task4.py --config configs/task4.yaml             # full run

Spends real money. The run refuses to start without a key, checks a hard USD
cap before every call against the gateway's own cumulative spend figure, writes
the usage log after every call, and skips any record already judged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tinystories_project.config import resolve_path
from tinystories_project.evaluation.judge import (
    RUBRIC_VERSION, BudgetExceeded, BudgetGuard, JudgeClient, JudgeError, Usage,
    load_credentials, make_judge_id,
)
from tinystories_project.records import read_jsonl, write_json

logger = logging.getLogger("run_task4")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/task4.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="judge at most N records (per run, after resume)")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--splits", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="make NO API calls; emit placeholder scores to test the pipeline")
    p.add_argument("--max-spend", type=float, default=None,
                   help="override the USD cap (never raises it above the config)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def load_source_records(root: Path, models, splits):
    """Task 3 generations, as (split, model, record) triples."""
    out = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if split_dir.name in {"prompts", "analysis"}:
            continue
        if splits and split_dir.name not in splits:
            continue
        for path in sorted(split_dir.glob("*.jsonl")):
            if models and path.stem not in models:
                continue
            for rec in read_jsonl(path):
                out.append((split_dir.name, path.stem, rec))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    out_root = resolve_path(cfg["output_dir"])
    # Dry-run output goes somewhere separate so placeholder rows can never be
    # mistaken for, or merged with, real paid scores.
    if args.dry_run:
        out_root = out_root / "dry_run"
    out_root.mkdir(parents=True, exist_ok=True)
    scores_path = out_root / "scores.jsonl"
    usage_path = out_root / "usage_log.jsonl"

    creds = load_credentials(resolve_path(".env"))
    judge_cfg = dict(cfg["judge"])
    if creds.get("COURSE_JUDGE_MODEL"):
        judge_cfg["model"] = creds["COURSE_JUDGE_MODEL"]
    model_name = judge_cfg["model"]
    temperature = float(judge_cfg["temperature"])

    records = load_source_records(resolve_path(cfg["source_dir"]),
                                  args.models, args.splits)
    if not records:
        logger.error("no Task 3 records found under %s", cfg["source_dir"])
        return 1

    # ---- resume: skip anything already SCORED under this exact setup ----
    # Only ok=True rows count. A failed attempt is recorded for the audit trail
    # but must be retried, or a transient outage would silently drop records
    # from the final results for ever.
    done: set[str] = set()
    n_failed_before = 0
    if scores_path.exists():
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok") and row.get("grammar") is not None:
                done.add(row["judge_id"])
            else:
                n_failed_before += 1
    todo = [(s, m, r) for s, m, r in records
            if make_judge_id(r.record_id, model_name, temperature) not in done]
    print(f"source records : {len(records)}")
    print(f"already judged : {len(done)}  (resumed, not re-paid)")
    if n_failed_before:
        print(f"earlier failures: {n_failed_before}  (will be retried)")
    print(f"to judge       : {len(todo)}")
    if args.limit:
        todo = todo[: args.limit]
        print(f"limited to     : {len(todo)} this run")
    if not todo:
        print("Nothing to do - every record is already judged.")
        return 0

    # ---- budget ----
    bcfg = cfg["budget"]
    cap = float(bcfg["max_spend_usd"])
    if args.max_spend is not None:
        cap = min(cap, args.max_spend)      # a flag may lower the cap, never raise it
    est = float(bcfg["estimated_cost_per_call_usd"])
    guard = BudgetGuard(max_spend_usd=cap, max_calls=int(bcfg["max_calls"]))

    # Fold in spend already recorded locally, so the cap covers earlier runs
    # even before the gateway reports its own figure.
    prior = 0.0
    if usage_path.exists():
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prior += float(json.loads(line).get("response_cost_usd") or 0.0)
    guard.spent_usd = prior
    print(f"budget cap     : ${cap:.2f}   prior logged spend ${prior:.4f}   "
          f"worst-case this run ${est * len(todo):.4f}")
    if prior + est * len(todo) > cap:
        print(f"WARNING: the whole run may not fit under the cap; it will stop "
              f"cleanly when the cap is reached and can be resumed later.")

    client = None
    if not args.dry_run:
        if not creds["COURSE_API_KEY"]:
            logger.error("COURSE_API_KEY is empty. Put it in .env (git-ignored). "
                         "Re-run with --dry-run to exercise the pipeline for free.")
            return 2
        client = JudgeClient(
            base_url=creds["COURSE_API_BASE_URL"], api_key=creds["COURSE_API_KEY"],
            model=model_name, temperature=temperature,
            max_tokens=int(judge_cfg["max_tokens"]), timeout=int(judge_cfg["timeout"]),
            max_retries=int(judge_cfg["max_retries"]))
        print(f"judge          : {client}")
    else:
        print("judge          : DRY RUN - no API calls, placeholder scores")

    n_ok = n_failed = 0
    first_error: str | None = None
    stopped_for_budget = False
    t_start = time.time()

    with scores_path.open("a", encoding="utf-8") as sf, \
         usage_path.open("a", encoding="utf-8") as uf:
        for i, (split, model, rec) in enumerate(todo, start=1):
            judge_id = make_judge_id(rec.record_id, model_name, temperature)
            prompt = rec.prompt
            completion = rec.completion
            try:
                guard.check(est)
            except BudgetExceeded as exc:
                print(f"\nSTOPPING: {exc}")
                stopped_for_budget = True
                break

            stop_after_writing = False
            if args.dry_run:
                scores = {"grammar": None, "creativity": None, "consistency": None,
                          "age_group": "", "note": "DRY RUN - not scored"}
                usage = Usage()
                ok = True
            else:
                try:
                    scores, usage = client.score(prompt, completion)
                    ok = True
                except Exception as exc:   # noqa: BLE001
                    ok = False
                    n_failed += 1
                    if first_error is None:
                        first_error = f"{type(exc).__name__}: {exc}"
                    usage = Usage()
                    scores = {"error": str(exc)[:300]}
                    # A hard configuration error hits every record, so stop
                    # rather than hammering a broken endpoint 800 times. Both
                    # logs are still written below before we break, so the two
                    # files never disagree about how many attempts were made.
                    if n_failed >= 3 and n_ok == 0:
                        stop_after_writing = True

            row = {
                "judge_id": judge_id, "record_id": rec.record_id,
                "task": "task4", "rubric_version": RUBRIC_VERSION,
                "judge_model": model_name, "temperature": temperature,
                "split": split, "model": model,
                "story_id": rec.extra.get("story_id"),
                "figure_column": rec.extra.get("figure_column"),
                "ok": ok, "dry_run": bool(args.dry_run),
                "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **scores,
            }
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.flush()
            uf.write(json.dumps({"judge_id": judge_id, "ok": ok,
                                 "judge_model": model_name,
                                 **usage.to_dict()}) + "\n")
            uf.flush()

            if ok and not args.dry_run:
                try:
                    guard.record(usage)
                except BudgetExceeded as exc:
                    print(f"\nSTOPPING after recording: {exc}")
                    stopped_for_budget = True
                    break
                n_ok += 1
            elif ok:
                n_ok += 1

            if stop_after_writing:
                print(f"\nSTOPPING: {n_failed} failures with no success at all - "
                      f"the endpoint looks misconfigured, so the remaining "
                      f"{len(todo) - i} records were not attempted.\n"
                      f"  {first_error}")
                break

            if i % 25 == 0 or i == len(todo):
                print(f"  {i:>4}/{len(todo)}  ok={n_ok} failed={n_failed}  "
                      f"spend=${guard.observed_total():.4f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json({
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg, "judge_model": model_name, "temperature": temperature,
        "rubric_version": RUBRIC_VERSION, "dry_run": bool(args.dry_run),
        "n_source_records": len(records), "n_already_judged": len(done),
        "n_attempted": n_ok + n_failed, "n_ok": n_ok, "n_failed": n_failed,
        "n_failures_retried_from_earlier_runs": n_failed_before,
        "first_error": first_error,
        "stopped_for_budget": stopped_for_budget,
        "budget": {"cap_usd": cap, "prior_logged_usd": round(prior, 6),
                   "run_spend_usd": round(guard.spent_usd - prior, 6),
                   "observed_total_usd": round(guard.observed_total(), 6),
                   "gateway_key_spend_usd": guard.server_spend_usd,
                   "calls": guard.calls},
        "elapsed_seconds": round(time.time() - t_start, 1),
        "environment": {"python": sys.version.split()[0]},
    }, out_root / f"manifest_{stamp}.json")

    print(f"\njudged ok      : {n_ok}")
    print(f"failed         : {n_failed}")
    print(f"spend this run : ${guard.spent_usd - prior:.4f}")
    print(f"observed total : ${guard.observed_total():.4f} of ${cap:.2f} cap")
    if first_error:
        print(f"first error    : {first_error[:200]}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
