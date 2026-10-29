#!/usr/bin/env python
"""Task 3 runner: generate the completions for Section 4.4.1 / Figures 14-17.

    python scripts/run_task3.py --config configs/task3.yaml
    python scripts/run_task3.py --config configs/task3.yaml --models roneneldan/TinyStories-1M --limit 10

Samples stories, cuts each at ~40%, generates a completion for each prompt with
every configured model, and writes the standard JSONL records. The ROUGE
measures themselves are computed afterwards by make_task3_tables.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from tinystories_project.config import resolve_path
from tinystories_project.evaluation.corpus import (
    local_paths, read_stories, sample_stories,
)
from tinystories_project.evaluation.rouge import split_prompt
from tinystories_project.inference import HFTextGenerator, load_causal_lm
from tinystories_project.inference.interface import GenerationParams
from tinystories_project.inference.loader import CheckpointLoadError
from tinystories_project.records import build_records, write_json, write_jsonl

logger = logging.getLogger("run_task3")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/task3.yaml")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--splits", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N sampled stories (for a pilot)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def release(bundle) -> None:
    if bundle is None:
        return
    try:
        bundle.model.to("cpu")
    except Exception:  # noqa: BLE001
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
    cfg = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    out_root = resolve_path(cfg["output_dir"])
    samp, ds = cfg["sampling"], cfg["dataset"]
    params = GenerationParams.from_dict(cfg["generation"])

    splits = args.splits or samp["splits"]
    models = cfg["models"]
    if args.models:
        models = [m for m in models if m["model_id"] in args.models]
        if not models:
            logger.error("no configured model matches %s", args.models)
            return 2

    # ---- sample the prompts once, so every model sees identical inputs ----
    prompt_sets: dict[str, list[dict]] = {}
    for split in splits:
        files = ds["train_files"] if split == "train" else ds["validation_files"]
        logger.info("loading %s split (%d file(s))", split, len(files))
        stories = read_stories(local_paths(ds["repo_id"], files))
        picked = sample_stories(stories, samp["n_stories"], seed=cfg["seed"],
                                min_words=samp["min_words"])
        if args.limit:
            picked = picked[: args.limit]
        entries = []
        for corpus_index, story in picked:
            prompt, original = split_prompt(story, samp["prompt_fraction"])
            entries.append({
                "id": f"{split}_{corpus_index:08d}",
                "split": split,
                "corpus_index": corpus_index,
                "prompt": prompt,
                "original_continuation": original,
                "n_words_story": len(story.split()),
            })
        prompt_sets[split] = entries
        print(f"{split:11} {len(stories):>9,} stories -> sampled {len(entries)}"
              f"  (mean story {sum(e['n_words_story'] for e in entries) // len(entries)} words)")
        del stories
        gc.collect()

    # The prompts are part of the experiment: save them so the measures can be
    # recomputed later without re-sampling (and so a rerun is verifiable).
    for split, entries in prompt_sets.items():
        write_json({"split": split, "seed": cfg["seed"],
                    "prompt_fraction": samp["prompt_fraction"],
                    "min_words": samp["min_words"], "n": len(entries),
                    "prompts": entries},
                   out_root / "prompts" / f"{split}.json")

    run_id = f"{cfg['run_name']}-{uuid.uuid4().hex[:8]}"
    written, skipped, model_meta = [], [], []
    t_start = time.time()

    for spec in models:
        model_id = spec["model_id"]
        print(f"\n{'=' * 78}\nMODEL {model_id}")
        bundle = None
        try:
            bundle = load_causal_lm(model_id, device=cfg["device"], dtype=cfg["dtype"])
        except CheckpointLoadError as exc:
            logger.error("SKIPPING %s: %s", model_id, exc)
            skipped.append({"model_id": model_id, "reason": str(exc)})
            release(bundle)
            continue

        meta = bundle.metadata
        print(f"  {meta.n_parameters_total:,} params "
              f"({meta.n_parameters_non_embedding:,} non-embedding) on {meta.device}")
        model_meta.append({"model_id": model_id,
                           "figure_column": spec.get("figure_column"),
                           **{k: meta.to_dict()[k] for k in
                              ("architecture", "n_layers", "hidden_size",
                               "n_parameters_total", "n_parameters_non_embedding",
                               "device", "dtype")}})
        gen = HFTextGenerator.from_bundle(bundle, batch_size=cfg["batch_size"])

        for split, entries in prompt_sets.items():
            slug = model_id.split("/")[-1].replace(".", "-")
            path = out_root / split / f"{slug}.jsonl"
            if path.exists() and not args.force:
                print(f"  [reuse] {split:11} {sum(1 for _ in path.open()):>4} records")
                continue
            t0 = time.time()
            produced = gen.generate([e["prompt"] for e in entries], params,
                                    seed=cfg["seed"])
            records = build_records(produced, meta, params, run_id=run_id)
            for rec, item, entry in zip(records, produced, entries):
                if item.prompt != entry["prompt"]:
                    raise ValueError(f"prompt mismatch at {entry['id']}")
                rec.extra = {
                    "task": "task3",
                    "story_id": entry["id"],
                    "split": split,
                    "corpus_index": entry["corpus_index"],
                    "original_continuation": entry["original_continuation"],
                    "figure_column": spec.get("figure_column"),
                }
            write_jsonl(records, path)
            written.append(str(path))
            print(f"  [run]   {split:11} {len(records):>4} records in "
                  f"{time.time() - t0:5.1f}s  "
                  f"(mean {sum(r.n_new_tokens for r in records) // len(records)} new tokens)")

        release(bundle)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json({
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "models": model_meta,
        "skipped_models": skipped,
        "generation": params.to_dict(),
        "n_prompts": {s: len(e) for s, e in prompt_sets.items()},
        "files_written": written,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__},
    }, out_root / f"manifest_{stamp}.json")

    print(f"\n{len(written)} file(s) written, {len(skipped)} model(s) skipped, "
          f"{time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
