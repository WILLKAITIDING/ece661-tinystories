#!/usr/bin/env python
"""One-command check that the environment and Task 1 pipeline really work.

    python scripts/smoke_test.py

Downloads TinyStories-1M (~20 MB) on first run. Prints a PASS/FAIL table and
exits non-zero if anything is broken, so it is usable in CI later.

This is the first thing Rachel or Luel should run after cloning.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

MODEL_ID = "roneneldan/TinyStories-1M"
PROMPT = "Once upon a time, there was a little girl named Lily."

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(fn):
        try:
            detail = fn() or ""
            results.append((name, True, str(detail)))
        except Exception as exc:  # noqa: BLE001 - report, never crash the run
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn

    return decorator


def main() -> int:
    print(f"Smoke test: {MODEL_ID}\n" + "-" * 70)

    @check("python + core imports")
    def _():
        import torch, transformers, yaml  # noqa: F401
        return f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}"

    @check("project package importable")
    def _():
        import tinystories_project as tp
        return f"tinystories_project {tp.__version__}"

    @check("device resolution")
    def _():
        from tinystories_project.devices import available_devices, describe_device, resolve_device
        d = resolve_device("auto")
        return f"auto -> {describe_device(d)} (available: {', '.join(available_devices())})"

    @check("config loads")
    def _():
        from tinystories_project.config import find_project_root, load_task1_config
        cfg = load_task1_config(find_project_root() / "configs" / "task1.yaml")
        return f"{cfg.run_name}: {cfg.model.model_id}, {len(cfg.prompts)} prompts"

    from tinystories_project.inference import generate_stories, load_causal_lm
    from tinystories_project.inference.interface import GenerationParams

    bundle = None

    @check("checkpoint + tokenizer load")
    def _():
        nonlocal bundle
        bundle = load_causal_lm(MODEL_ID, device="auto")
        m = bundle.metadata
        return (f"{m.architecture}, {m.n_parameters_total:,} params "
                f"({m.n_parameters_non_embedding:,} non-embedding), on {m.device}")

    if bundle is not None:
        completion = ""

        @check("generation produces text")
        def _():
            nonlocal completion
            out = generate_stories(bundle, [PROMPT], max_new_tokens=60, seed=42)
            completion = out[0].completion
            if len(completion.strip()) < 20:
                raise AssertionError(f"completion too short: {completion!r}")
            return f"{out[0].n_new_tokens} tokens, finish={out[0].finish_reason}"

        @check("same seed reproduces text")
        def _():
            a = generate_stories(bundle, [PROMPT], max_new_tokens=30, seed=321)
            b = generate_stories(bundle, [PROMPT], max_new_tokens=30, seed=321)
            if a[0].completion != b[0].completion:
                raise AssertionError("same seed produced different text")
            return "identical across two runs"

        @check("batched mixed-length prompts")
        def _():
            out = generate_stories(
                bundle, [PROMPT, "The cat sat.", "Tom ran fast."],
                max_new_tokens=25, seed=5,
            )
            empty = [o.prompt for o in out if len(o.completion.strip()) < 5]
            if empty:
                raise AssertionError(f"empty completions for {empty}")
            return f"{len(out)} completions, all non-empty"

        @check("JSONL write + read back")
        def _():
            import tempfile
            from tinystories_project.io import build_records, read_jsonl, write_jsonl
            params = GenerationParams(max_new_tokens=20)
            out = generate_stories(bundle, [PROMPT], params=params, seed=42)
            recs = build_records(out, bundle.metadata, params, run_id="smoke")
            with tempfile.TemporaryDirectory() as td:
                path = write_jsonl(recs, Path(td) / "smoke.jsonl")
                back = list(read_jsonl(path))
                if len(back) != 1 or back[0].completion != out[0].completion:
                    raise AssertionError("round-trip mismatch")
            return "schema 1.0 round-trip OK"

        if completion:
            print(f"\nPROMPT: {PROMPT}")
            print(f"OUTPUT: {completion.strip()[:400]}\n")

    print("-" * 70)
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    failed = [n for n, ok, _ in results if not ok]
    print("-" * 70)
    if failed:
        print(f"{len(failed)} of {len(results)} checks FAILED: {', '.join(failed)}")
        print("Re-run with -v for tracebacks.")
        return 1
    print(f"All {len(results)} checks passed. Task 1 pipeline is working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
