"""Generate the (clean, cf) 3-element tuple pool for ONE (model, setting)
of the §5.3.2 experiment, filtered **under the ordering ablation**.

Applies the `behavioral_analysis_wo_ve_oi` ablation hook (Sec 5.3.1) to
both the clean and cf forwards before checking top-1 correctness — the
same hook the downstream patching sweep applies — so a surviving tuple is
guaranteed to start from a clean/cf pair the *ablated* model already
predicts correctly (c0 on clean, c2 on cf). Patching tuples that fail
this precondition adds noise (no correct baseline for the sweep to flip).

Usage:
    python -m criss_cross_wo_ve_oi.build_cf_pairs \\
        --model qwen2vl --setting shapes --target 50

Settings: squares / shapes / objects (whatsup has no paper-defined
ablation — see README). Output merged into ``cf_pairs/<setting>.json``.
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
    CF_PAIRS_KEY, build_vl_prompts, candidate_pool,
)
from last_token_exp.src.filter import norm    # noqa: E402
from behavioral_analysis_wo_ve_oi.src.hook import (    # noqa: E402
    ResidualCache, run_ablated,
)
from behavioral_analysis_wo_ve_oi.src.references import empty_image  # noqa: E402
from criss_cross_wo_ve_oi.src.refs import (    # noqa: E402
    clean_cf_entities, hashable, iso_reference_images,
)


HERE = Path(__file__).resolve().parent
SETTINGS = ("squares", "shapes", "objects")


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ablated_passes(model, processor, *, model_key, direction, vlp,
                    clean_ents, cf_ents, cache, iso_imgs, empty_img) -> bool:
    """True iff the ablated model's top-1 matches BOTH clean and cf answers."""
    prompt = vlp.prompt  # clean prompt == cf prompt (same middle entity)
    if ("empty", prompt) not in cache._cache:
        cache.populate(("empty", prompt), prompt, empty_img)
    for e in set(clean_ents) | set(cf_ents):
        if (e, prompt) not in cache._cache:
            cache.populate((e, prompt), prompt, iso_imgs[e])

    pred_clean, _ = run_ablated(
        model, processor, model_key=model_key,
        prompt=vlp.prompt, image=vlp.images[0], direction=direction,
        entity_ref_keys=[(e, prompt) for e in clean_ents],
        empty_ref_key=("empty", prompt), cache=cache,
    )
    if norm(pred_clean) != norm(vlp.answer):
        return False
    pred_cf, _ = run_ablated(
        model, processor, model_key=model_key,
        prompt=vlp.cf_prompt, image=vlp.cf_images[0], direction=direction,
        entity_ref_keys=[(e, prompt) for e in cf_ents],
        empty_ref_key=("empty", prompt), cache=cache,
    )
    return norm(pred_cf) == norm(vlp.cf_answer)


def collect_survivors(model, processor, direction, geom, *, setting, model_key,
                      pool, cache, iso_imgs, empty_img,
                      target, max_attempts, rng):
    survivors: list[list] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(survivors) < target and attempts < max_attempts:
        attempts += 1
        candidate = list(rng.sample(pool, 3))
        key = tuple(hashable(e) for e in candidate)
        if key in seen:
            continue
        seen.add(key)
        vlp = build_vl_prompts(processor, [candidate], direction, geom,
                               setting=setting, model_key=model_key)[0]
        clean_ents, cf_ents = clean_cf_entities(candidate, direction)
        if _ablated_passes(model, processor, model_key=model_key, direction=direction,
                           vlp=vlp, clean_ents=clean_ents, cf_ents=cf_ents,
                           cache=cache, iso_imgs=iso_imgs, empty_img=empty_img):
            survivors.append(candidate)
            if len(survivors) % 10 == 0 or len(survivors) == target:
                stamp(f"    {direction}: {len(survivors)}/{target} "
                      f"(attempts={attempts}, hit_rate={len(survivors)/attempts:.2f})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return survivors, attempts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(GEOM))
    p.add_argument("--setting", required=True, choices=SETTINGS)
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
    stamp(f"candidate pool ({len(pool)}): {pool}")
    cache = ResidualCache(model, processor, args.model)
    iso_imgs = iso_reference_images(args.setting, args.model)
    empty_img = empty_image(args.model)

    per_dir: dict[str, list[list]] = {}
    total_attempts = 0
    for direction in ("left", "right", "above", "below"):
        stamp(f"\n== {direction} ==")
        survivors, attempts = collect_survivors(
            model, processor, direction, geom,
            setting=args.setting, model_key=args.model, pool=pool,
            cache=cache, iso_imgs=iso_imgs, empty_img=empty_img,
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
