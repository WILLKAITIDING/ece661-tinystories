#!/usr/bin/env python
"""Task 2 runner: qualitative replication of TinyStories Figures 9-11.

    python scripts/run_task2.py --config configs/task2.yaml
    python scripts/run_task2.py --config configs/task2.yaml --models roneneldan/TinyStories-1M
    python scripts/run_task2.py --config configs/task2.yaml --protocols A_greedy --force

Loads each model in turn, runs every prompt set under every protocol, writes
one JSONL per (model, protocol, prompt set) plus a fingerprint sidecar, then
releases the model before loading the next.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from tinystories_project.config import load_task2_config, resolve_path
from tinystories_project.evaluation import load_prompt_set, run_prompt_set, summarise
from tinystories_project.evaluation.qualitative import PROTOCOLS, run_fingerprint
from tinystories_project.inference import HFTextGenerator, load_causal_lm
from tinystories_project.inference.interface import GenerationParams
from tinystories_project.inference.loader import CheckpointLoadError
from tinystories_project.records import read_jsonl, write_json, write_jsonl

logger = logging.getLogger("run_task2")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/task2.yaml")
    p.add_argument("--models", nargs="*", default=None,
                   help="only run these model_ids (default: all in the config)")
    p.add_argument("--protocols", nargs="*", default=None,
                   help="only run these protocols (default: all in the config)")
    p.add_argument("--force", action="store_true",
                   help="regenerate even if a matching result already exists")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "-")


def release(bundle) -> None:
    """Drop a model from memory before loading the next one."""
    if bundle is None:
        return
    try:
        bundle.model.to("cpu")
    except Exception:  # noqa: BLE001 - best effort only
        pass
    del bundle
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg = load_task2_config(resolve_path(args.config))
    out_root = resolve_path(cfg.output_dir)

    prompt_sets = [load_prompt_set(resolve_path(p)) for p in cfg.prompt_sets]
    print("Prompt sets:")
    for ps in prompt_sets:
        print(f"  {ps.name:22} n={len(ps):3}  {ps.counts()}")

    protocols = {
        name: GenerationParams.from_dict({**PROTOCOLS[name].to_dict(), **(spec or {})})
        if name in PROTOCOLS else GenerationParams.from_dict(spec or {})
        for name, spec in cfg.protocols.items()
    }
    if args.protocols:
        unknown = set(args.protocols) - set(protocols)
        if unknown:
            logger.error("Unknown protocol(s): %s", ", ".join(sorted(unknown)))
            return 2
        protocols = {k: v for k, v in protocols.items() if k in args.protocols}

    models = cfg.models
    if args.models:
        models = [m for m in models if m["model_id"] in args.models]
        if not models:
            logger.error("No configured model matches %s", args.models)
            return 2

    run_id = f"{cfg.run_name}-{uuid.uuid4().hex[:8]}"
    written, reused, skipped_models, model_meta = [], [], [], []
    t_start = time.time()

    for spec in models:
        model_id = spec["model_id"]
        print(f"\n{'=' * 78}\nMODEL {model_id}")
        bundle = None
        try:
            bundle = load_causal_lm(
                model_id,
                tokenizer_id=spec.get("tokenizer_id"),
                revision=spec.get("revision"),
                device=cfg.device,
                dtype=cfg.dtype,
            )
        except CheckpointLoadError as exc:
            # Record and move on: one unavailable checkpoint must not sink the run.
            logger.error("SKIPPING %s: %s", model_id, exc)
            skipped_models.append({"model_id": model_id, "reason": str(exc)})
            release(bundle)
            continue

        meta = bundle.metadata
        print(f"  {meta.architecture}  {meta.n_layers} layers, hidden {meta.hidden_size}")
        print(f"  {meta.n_parameters_total:,} params total "
              f"| {meta.n_parameters_non_embedding:,} non-embedding")
        print(f"  figure column: {spec.get('figure_column', '-')}   device: {meta.device}")
        model_meta.append({
            "model_id": model_id,
            "figure_column": spec.get("figure_column"),
            **{k: meta.to_dict()[k] for k in
               ("architecture", "n_layers", "hidden_size", "vocab_size",
                "n_parameters_total", "n_parameters_non_embedding",
                "n_parameters_embedding", "device", "dtype")},
        })

        generator = HFTextGenerator.from_bundle(bundle, batch_size=cfg.batch_size)

        for protocol, params in protocols.items():
            for ps in prompt_sets:
                path = out_root / protocol / f"{slug(model_id)}__{ps.name}.jsonl"
                fp_path = path.with_suffix(".fingerprint.json")
                fingerprint = run_fingerprint(meta, params, protocol=protocol,
                                              seed=cfg.seed, batch_size=cfg.batch_size,
                                              prompt_set=ps)

                if not args.force and path.exists() and fp_path.exists():
                    old = json.loads(fp_path.read_text(encoding="utf-8"))
                    n_have = sum(1 for _ in read_jsonl(path))
                    if old == fingerprint and n_have == fingerprint["n_expected"]:
                        print(f"  [reuse] {protocol:11} {ps.name:22} "
                              f"{n_have} records already match")
                        reused.append(str(path))
                        continue

                t0 = time.time()
                records = run_prompt_set(
                    generator, ps, protocol=protocol, run_id=run_id,
                    params=params, seed=cfg.seed,
                    extra={"figure_column": spec.get("figure_column")},
                )
                write_jsonl(records, path)
                write_json(fingerprint, fp_path)
                s = summarise(records)
                print(f"  [run]   {protocol:11} {ps.name:22} "
                      f"{len(records):3} records in {time.time() - t0:5.1f}s  "
                      f"mean {s['mean_new_tokens']} new tokens, "
                      f"{s['n_empty_completions']} empty")
                written.append(str(path))

        release(bundle)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = out_root / f"manifest_{stamp}.json"
    write_json({
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "git_commit": git_commit(),
        "models": model_meta,
        "skipped_models": skipped_models,
        "protocols": {k: v.to_dict() for k, v in protocols.items()},
        "prompt_sets": [
            {"name": ps.name, "path": ps.path, "n": len(ps), "counts": ps.counts(),
             "source": ps.source}
            for ps in prompt_sets
        ],
        "files_written": written,
        "files_reused": reused,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }, manifest)

    print(f"\n{'=' * 78}")
    print(f"{len(written)} file(s) written, {len(reused)} reused, "
          f"{len(skipped_models)} model(s) skipped, "
          f"{time.time() - t_start:.1f}s total")
    print(f"manifest -> {manifest}")
    if skipped_models:
        print("SKIPPED:")
        for s in skipped_models:
            print(f"  {s['model_id']}: {s['reason'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
