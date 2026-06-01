"""Run criss-cross object patching under the ordering ablation (paper §5.3.2).

For each layer slot, patch the clean run's residuals at the (left, right)
or (top, bottom) square-token positions with the cf run's residuals at the
*swapped* positions — but every forward (clean and cf alike) is first
ablated at ``blocks.0.hook_resid_pre`` so the vision encoder's ordering
signal is removed before any LM layer sees it.

Reuses the canonical color tuples from ``criss_cross_patching/color_tuples.json``
(same per-model palette and direction-symmetric `(c0, c1, c2)` construction
as the §5.2.2 criss-cross experiment).

Usage::

    python -m criss_cross_wo_ve_oi.run_experiment \\
        --model qwen2vl --direction left
    python -m criss_cross_wo_ve_oi.run_experiment \\
        --model pixtral --all-directions

Output: ``results/<model>/squares_object_patching_under_ablation.npy``,
a pickled ``dict[direction -> ndarray]`` of shape
``(n_samples, n_layers + 1, 1, 3)``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shared.experiment_config import BATCH_SIZES, GEOM
from shared import models as _models
from shared.palette import DIRECTIONS
from shared.paths import setup as _setup_paths

_setup_paths()

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

# Avoid CUDA fragmentation on 12B models.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from criss_cross_patching.src.data import build_vl_prompts  # noqa: E402
from behavioral_analysis_wo_ve_oi.src.hook import ResidualCache  # noqa: E402
from behavioral_analysis_wo_ve_oi.src.references import empty_image  # noqa: E402
from criss_cross_wo_ve_oi.src.refs import (  # noqa: E402
    clean_cf_entities, iso_reference_images,
)
from criss_cross_wo_ve_oi.src.patch import (  # noqa: E402
    criss_cross_sweep_under_ablation,
    object_swap_positions,
    resolve_image_starts,
)


HERE = Path(__file__).resolve().parent

_SLUG_TO_JSON = {
    "qwen2vl": "qwen2-vl-7b",
    "gemma3":  "gemma-3-4b-it",
    "pixtral": "pixtral-12b",
}

# Ablation is paper-defined for synthetic settings only (Sec 5.3.1).
ABLATION_SETTINGS = ("squares", "shapes", "objects")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=list(GEOM))
    p.add_argument("--setting", default="squares", choices=ABLATION_SETTINGS,
                   help="task setting (squares/shapes/objects; whatsup has no "
                        "paper-defined ablation and is not supported here)")
    p.add_argument("--direction", choices=DIRECTIONS, default=None)
    p.add_argument("--all-directions", action="store_true")
    p.add_argument("--n-samples", type=int, default=None,
                   help="cap samples per direction (default: use all from cf_pairs.json)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override per-model default batch size from "
                        "BATCH_SIZES['criss_cross_patching']")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-list", type=int, nargs="*", default=None)
    p.add_argument("--torch-dtype", default=None,
                   choices=("float32", "bfloat16", "float16"),
                   help="torch dtype (default: each model's `models.DEFAULT_DTYPE`)")
    p.add_argument("--output-name", default=None,
                   help="basename (no .npy) for the output file under "
                        "`results/<model>/`. Override for parallel per-direction "
                        "runs. Default: <setting>_object_patching_under_ablation.")
    p.add_argument("--cf-pairs", type=Path, default=None,
                   help="path to the (ablated-filter) cf-pairs pool "
                        "(default: cf_pairs/<setting>.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    directions = list(DIRECTIONS) if args.all_directions else [args.direction]
    if not directions or directions == [None]:
        raise SystemExit("pick --direction or --all-directions")

    geom = GEOM[args.model]
    batch_size = args.batch_size or BATCH_SIZES["criss_cross_patching"][args.model]

    cf_pairs_path = args.cf_pairs or HERE / "cf_pairs" / f"{args.setting}.json"
    if not cf_pairs_path.exists():
        raise SystemExit(
            f"cf_pairs file not found at {cf_pairs_path}. Build it first:\n"
            f"    python -m criss_cross_wo_ve_oi.build_cf_pairs "
            f"--model {args.model} --setting {args.setting}"
        )
    cf_pairs_all = json.loads(cf_pairs_path.read_text())
    if _SLUG_TO_JSON[args.model] not in cf_pairs_all:
        raise SystemExit(
            f"{_SLUG_TO_JSON[args.model]!r} not found in {cf_pairs_path}. Build it:\n"
            f"    python -m criss_cross_wo_ve_oi.build_cf_pairs "
            f"--model {args.model} --setting {args.setting}"
        )
    color_tuples = cf_pairs_all[_SLUG_TO_JSON[args.model]]

    print(f"loading {args.model} (setting={args.setting}, "
          f"dtype={args.torch_dtype or 'default'}) ...")
    model, processor = _models.load(
        args.model,
        device=args.device, device_list=args.device_list,
        cache_dir=args.cache_dir,
        torch_dtype=getattr(torch, args.torch_dtype) if args.torch_dtype else None,
    )

    cache = ResidualCache(model, processor, args.model)
    iso_imgs = iso_reference_images(args.setting, args.model)
    empty_img = empty_image(args.model)

    out_dir = HERE / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, np.ndarray] = {}
    for direction in directions:
        tups = color_tuples[direction]
        if args.n_samples is not None:
            tups = tups[: args.n_samples]
        print(f"\n  {direction}: {len(tups)} samples, batch_size={batch_size}")

        vl_prompts = build_vl_prompts(processor, tups, direction, geom,
                                      setting=args.setting, model_key=args.model)
        clean_image_start, cf_image_start = resolve_image_starts(model, processor, vl_prompts)
        clean_positions, cf_positions = object_swap_positions(
            args.model, direction,
            clean_image_start=clean_image_start, cf_image_start=cf_image_start,
        )
        print(f"    swapping {len(clean_positions)} positions per sample (object patching)")

        # Populate the reference cache for every (entity, prompt) pair. Prompts
        # only differ when the middle entity differs, so populate lazily.
        clean_ents_per_item: list[tuple] = []
        cf_ents_per_item:    list[tuple] = []
        prompts_per_item: list[str] = []
        seen_prompts: set[str] = set()
        for tup, vlp in zip(tups, vl_prompts):
            clean_ents, cf_ents = clean_cf_entities(tup, direction)
            prompt = vlp.prompt  # clean prompt == cf prompt (same middle entity)
            clean_ents_per_item.append(clean_ents)
            cf_ents_per_item.append(cf_ents)
            prompts_per_item.append(prompt)
            if prompt not in seen_prompts:
                cache.populate(("empty", prompt), prompt, empty_img)
                seen_prompts.add(prompt)
            for e in set(clean_ents) | set(cf_ents):
                cache.populate((e, prompt), prompt, iso_imgs[e])

        results = criss_cross_sweep_under_ablation(
            model, vl_prompts,
            model_key=args.model, direction=direction, cache=cache,
            clean_image_start=clean_image_start, cf_image_start=cf_image_start,
            clean_colors_per_item=clean_ents_per_item,
            cf_colors_per_item=cf_ents_per_item,
            prompts_per_item=prompts_per_item,
            clean_positions=clean_positions, cf_positions=cf_positions,
            batch_size=batch_size,
        )
        all_results[direction] = results.numpy()

    name = args.output_name or f"{args.setting}_object_patching_under_ablation"
    out_path = out_dir / f"{name}.npy"
    np.save(out_path, all_results)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
