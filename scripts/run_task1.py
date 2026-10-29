#!/usr/bin/env python
"""Task 1 runner: load a TinyStories checkpoint, generate, save JSONL.

    python scripts/run_task1.py --config configs/task1.yaml
    python scripts/run_task1.py --config configs/task1.yaml \
        --model-id roneneldan/TinyStories-33M --num-return-sequences 3

Every setting comes from the config; CLI flags are explicit overrides and are
recorded in the run manifest, so a result is never produced by settings that
were silently changed.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tinystories_project.config import Task1Config, load_task1_config, resolve_path
from tinystories_project.inference import (
    CheckpointLoadError,
    GenerationParams,
    HFTextGenerator,
    load_causal_lm,
)
from tinystories_project.io import build_records, write_json, write_jsonl

logger = logging.getLogger("run_task1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/task1.yaml", help="path to the YAML config")
    p.add_argument("--model-id", default=None, help="override model.model_id")
    p.add_argument("--tokenizer-id", default=None, help="override model.tokenizer_id")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--num-return-sequences", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--prompt", action="append", default=None,
                   help="replace config prompts; repeat for several")
    p.add_argument("--out-dir", default=None, help="override output.dir")
    p.add_argument("--run-name", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def apply_overrides(cfg: Task1Config, args: argparse.Namespace) -> tuple[Task1Config, dict]:
    """Apply CLI overrides, returning the config and a record of what changed."""
    applied: dict[str, object] = {}

    def override(target: object, attr: str, value: object, label: str) -> None:
        if value is not None:
            applied[label] = value
            setattr(target, attr, value)

    override(cfg.model, "model_id", args.model_id, "model.model_id")
    override(cfg.model, "tokenizer_id", args.tokenizer_id, "model.tokenizer_id")
    override(cfg.model, "device", args.device, "model.device")
    override(cfg, "seed", args.seed, "seed")
    override(cfg, "batch_size", args.batch_size, "batch_size")
    override(cfg, "run_name", args.run_name, "run_name")
    override(cfg.output, "dir", args.out_dir, "output.dir")

    if args.prompt:
        applied["prompts"] = args.prompt
        cfg.prompts = args.prompt

    gen_overrides = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_return_sequences": args.num_return_sequences,
    }
    gen_overrides = {k: v for k, v in gen_overrides.items() if v is not None}
    if gen_overrides:
        applied.update({f"generation.{k}": v for k, v in gen_overrides.items()})
        cfg.generation = GenerationParams.from_dict(
            {**cfg.generation.to_dict(), **gen_overrides}
        )
    return cfg, applied


def git_commit() -> str | None:
    """Current commit, recorded so a result can be traced back to the code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5, check=False,
        )
        # In a repo with no commits yet, git exits non-zero and echoes "HEAD"
        # on stdout. Only a successful call yields a real commit hash.
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def output_paths(cfg: Task1Config) -> tuple[Path, Path]:
    out_dir = resolve_path(cfg.output.dir)
    if cfg.output.filename:
        stem = Path(cfg.output.filename).stem
        jsonl = out_dir / f"{stem}.jsonl"
    else:
        slug = cfg.model.model_id.split("/")[-1].replace(".", "-")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        jsonl = out_dir / f"{cfg.run_name}_{slug}_{stamp}.jsonl"
    return jsonl, jsonl.with_name(jsonl.stem + "_manifest.json")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = load_task1_config(resolve_path(args.config) if not Path(args.config).is_absolute() else args.config)
    cfg, applied = apply_overrides(cfg, args)
    if applied:
        logger.info("CLI overrides applied: %s", json.dumps(applied, default=str))

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
          f"| {meta.n_parameters_non_embedding:,} non-embedding "
          f"| {meta.n_parameters_embedding:,} embedding")
    print(f"  tokenizer    : {meta.tokenizer_id} (vocab {meta.vocab_size})")
    print(f"  device       : {meta.device}\n")

    generator = HFTextGenerator.from_bundle(bundle, batch_size=cfg.batch_size)
    generated = generator.generate(cfg.prompts, cfg.generation, seed=cfg.seed)

    run_id = f"{cfg.run_name}-{uuid.uuid4().hex[:8]}"
    records = build_records(generated, meta, cfg.generation, run_id=run_id)

    jsonl_path, manifest_path = output_paths(cfg)
    write_jsonl(records, jsonl_path)

    if cfg.output.write_manifest:
        write_json(
            {
                "run_id": run_id,
                "created_utc": records[0].timestamp_utc if records else None,
                "config": cfg.to_dict(),
                "cli_overrides": applied,
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
            manifest_path,
        )

    for item in generated:
        print("=" * 78)
        print(f"PROMPT     [{item.prompt_index}.{item.sample_index}]: {item.prompt}")
        print(f"CONTINUATION ({item.n_new_tokens} tokens, finish={item.finish_reason}):")
        print(item.completion.strip() or "<empty>")
    print("=" * 78)
    print(f"\nWrote {len(records)} records -> {jsonl_path}")
    if cfg.output.write_manifest:
        print(f"Wrote manifest        -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
