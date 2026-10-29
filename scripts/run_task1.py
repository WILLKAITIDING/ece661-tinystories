#!/usr/bin/env python
"""Task 1 runner: load a TinyStories checkpoint, generate, save JSONL.

    python scripts/run_task1.py --config configs/task1.yaml
    python scripts/run_task1.py --config configs/task1.yaml --model-id roneneldan/TinyStories-33M

Settings come from the config. The few flags here are explicit overrides and
are recorded in the run manifest, so a result is never produced by settings
that were silently changed. For anything else, copy configs/task1.yaml.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tinystories_project.config import load_task1_config, resolve_path
from tinystories_project.inference import HFTextGenerator, load_causal_lm
from tinystories_project.inference.loader import CheckpointLoadError
from tinystories_project.records import build_records, write_json, write_jsonl

logger = logging.getLogger("run_task1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/task1.yaml")
    p.add_argument("--model-id", default=None, help="override model.model_id")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None, help="override output_dir")
    p.add_argument("--run-name", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def git_commit() -> str | None:
    """Current commit, so a result can be traced back to the code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5, check=False,
        )
        # With no commits yet, git exits non-zero and echoes "HEAD" on stdout.
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = load_task1_config(resolve_path(args.config))
    overrides: dict[str, object] = {}
    if args.model_id is not None:
        cfg.model.model_id = args.model_id
        overrides["model.model_id"] = args.model_id
    if args.seed is not None:
        cfg.seed = args.seed
        overrides["seed"] = args.seed
    if args.out_dir is not None:
        cfg.output_dir = args.out_dir
        overrides["output_dir"] = args.out_dir
    if args.run_name is not None:
        cfg.run_name = args.run_name
        overrides["run_name"] = args.run_name
    if overrides:
        logger.info("CLI overrides: %s", overrides)

    try:
        bundle = load_causal_lm(
            cfg.model.model_id,
            tokenizer_id=cfg.model.tokenizer_id,
            device=cfg.model.device,
            dtype=cfg.model.dtype,
            revision=cfg.model.revision,
        )
    except CheckpointLoadError as exc:
        logger.error("%s", exc)
        return 1

    meta = bundle.metadata
    print(f"\nLoaded {meta.model_id}")
    print(f"  architecture : {meta.architecture} ({meta.n_layers} layers, hidden {meta.hidden_size})")
    print(f"  parameters   : {meta.n_parameters_total:,} total "
          f"| {meta.n_parameters_non_embedding:,} non-embedding")
    print(f"  tokenizer    : {meta.tokenizer_id} (vocab {meta.vocab_size})")
    print(f"  device       : {meta.device}\n")

    generator = HFTextGenerator.from_bundle(bundle, batch_size=cfg.batch_size)
    generated = generator.generate(cfg.prompts, cfg.generation, seed=cfg.seed)

    run_id = f"{cfg.run_name}-{uuid.uuid4().hex[:8]}"
    records = build_records(generated, meta, cfg.generation, run_id=run_id)

    out_dir = resolve_path(cfg.output_dir)
    slug = cfg.model.model_id.split("/")[-1].replace(".", "-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = out_dir / f"{cfg.run_name}_{slug}_{stamp}.jsonl"
    write_jsonl(records, jsonl_path)
    write_json(
        {
            "run_id": run_id,
            "created_utc": records[0].timestamp_utc if records else None,
            "config": cfg.to_dict(),
            "cli_overrides": overrides,
            "git_commit": git_commit(),
            "n_records": len(records),
            "model": meta.to_dict(),
            "environment": {
                "python": sys.version.split()[0],
                "torch": __import__("torch").__version__,
                "transformers": __import__("transformers").__version__,
            },
            "records_file": jsonl_path.name,
        },
        jsonl_path.with_name(jsonl_path.stem + "_manifest.json"),
    )

    for item in generated:
        print("=" * 78)
        print(f"PROMPT       [{item.prompt_index}.{item.sample_index}]: {item.prompt}")
        print(f"CONTINUATION ({item.n_new_tokens} tokens, finish={item.finish_reason}):")
        print(item.completion.strip() or "<empty>")
    print("=" * 78)
    print(f"\nWrote {len(records)} records -> {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
