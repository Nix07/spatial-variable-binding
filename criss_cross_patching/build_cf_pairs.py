"""Generate the (clean, cf) 3-element tuple pool for ONE (model, setting)
of the criss-cross patching experiment, keeping only tuples the model
answers correctly on BOTH the clean and counterfactual prompts.

Usage:
    python -m criss_cross_patching.build_cf_pairs \\
        --model qwen2vl --setting shapes --target 50

Tuple semantics (3 elements, middle = index 1):
  - squares  : 3 color names
  - shapes   : 3 (shape, color) entities
  - objects  : 3 object names
  - whatsup  : (side0, middle_ref, side2); middle_ref ∈ {chair, table,
               armchair}. Horizontal (left/right) only; Qwen/Gemma only.

Output merged into ``cf_pairs/<setting>.json`` under the versioned model
slug (``{model: {direction: [tuples]}}``) — drop-in for `run_experiment`.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

from shared import models as _models
from shared.experiment_config import GEOM
from shared.paths import setup as _setup_paths

_setup_paths()

import torch    # noqa: E402

from criss_cross_patching.src.data import (    # noqa: E402
    CF_PAIRS_KEY, SETTINGS, build_vl_prompts, candidate_pool,
)
from last_token_exp.src.filter import passes_filter      # noqa: E402


HERE = Path(__file__).resolve().parent


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample_tuple(rng: random.Random, setting: str, pool):
    """Draw one 3-element candidate tuple for `setting`."""
    if setting == "whatsup":
        middle = rng.choice(list(pool.keys()))
        s0, s2 = rng.sample(pool[middle], 2)
        return [s0, middle, s2]
    return list(rng.sample(pool, 3))


def collect_survivors(
    model, processor, direction: str, geom, *,
    setting: str, model_key: str, pool,
    target: int, max_attempts: int, rng: random.Random,
) -> tuple[list[list], int]:
    survivors: list[list] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(survivors) < target and attempts < max_attempts:
        attempts += 1
        candidate = sample_tuple(rng, setting, pool)
        key = tuple(tuple(e) if isinstance(e, (list, tuple)) else e for e in candidate)
        if key in seen:
            continue
        seen.add(key)
        vlps = build_vl_prompts(processor, [candidate], direction, geom,
                                setting=setting, model_key=model_key)
        if passes_filter(model, processor, vlps[0]):
            survivors.append(candidate)
            if len(survivors) % 10 == 0 or len(survivors) == target:
                stamp(f"    {direction}: {len(survivors)}/{target} "
                      f"(attempts={attempts}, hit_rate={len(survivors)/attempts:.2f})")
        del vlps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return survivors, attempts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(GEOM))
    p.add_argument("--setting", required=True, choices=list(SETTINGS))
    p.add_argument("--target", type=int, default=50)
    p.add_argument("--max-attempts-per-direction", type=int, default=2000)
    p.add_argument("--output", type=Path, default=None,
                   help="output JSON path (default: cf_pairs/<setting>.json)")
    p.add_argument("--torch-dtype", default=None,
                   choices=("float32", "bfloat16", "float16"),
                   help="torch dtype (default: each model's `models.DEFAULT_DTYPE`)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.setting == "whatsup" and args.model == "pixtral":
        raise SystemExit("whatsup criss-cross has no Pixtral region maps; "
                         "supported for qwen2vl / gemma3 only.")
    rng = random.Random(args.seed)
    geom = GEOM[args.model]
    out_path = args.output or HERE / "cf_pairs" / f"{args.setting}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stamp(f"loading {args.model} (setting={args.setting}, "
          f"dtype={args.torch_dtype or 'default'}) ...")
    model, processor = _models.load(
        args.model,
        torch_dtype=getattr(torch, args.torch_dtype) if args.torch_dtype else None,
    )
    stamp(f"loaded; cfg.n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")

    pool = candidate_pool(args.setting, args.model, processor)
    if args.setting == "whatsup":
        stamp(f"whatsup pools: { {m: len(s) for m, s in pool.items()} }")
    else:
        stamp(f"candidate pool ({len(pool)}): {pool}")

    per_dir: dict[str, list[list]] = {}
    total_attempts = 0
    for direction in SETTINGS[args.setting].directions:
        stamp(f"\n== {direction} ==")
        survivors, attempts = collect_survivors(
            model, processor, direction, geom,
            setting=args.setting, model_key=args.model, pool=pool,
            target=args.target, max_attempts=args.max_attempts_per_direction, rng=rng,
        )
        per_dir[direction] = survivors
        total_attempts += attempts
        stamp(f"  {direction}: {len(survivors)}/{args.target} survivors "
              f"after {attempts} attempts")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged = json.loads(out_path.read_text()) if out_path.exists() else {}
    merged[CF_PAIRS_KEY[args.model]] = per_dir
    out_path.write_text(json.dumps(merged, indent=2))
    stamp(f"\nwrote {out_path}  ({sum(len(v) for v in per_dir.values())} "
          f"tuples for {args.model}/{args.setting}, {total_attempts} attempts)")


if __name__ == "__main__":
    sys.exit(main() or 0)
