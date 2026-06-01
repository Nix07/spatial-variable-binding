"""Run criss-cross vision-token patching on the Squares dataset.

For each LM layer ``l``, replace the clean run's residual-stream activations
at the **left and right object regions** with the cf run's activations at
the *swapped* regions:

    clean[left]  <- cf[right]
    clean[right] <- cf[left]

(or top/bottom for vertical queries). The cf image is the clean image with
its colors reversed left↔right (above↔below for vertical), so the colors at
each region match — only ordering changes.

The runner sweeps every LM hook point (``resid_pre`` layer 0 + ``resid_post``
layer 0..L-1) for one **patching kind** (``object`` = Fig 6, ``strip`` = Fig
7) at a time. Logit-prob of the three candidate colors at the final-token
position is recorded.

Usage::

    python -m criss_cross_patching.run_experiment \\
        --model qwen2vl --kind strip --direction left
    python -m criss_cross_patching.run_experiment \\
        --model qwen2vl --kind object --all-directions

Output: ``results/<model>/squares_<kind>_patching.npy``, a pickled
``dict[direction → ndarray]`` of shape
``(n_samples, n_layers + 1, 1, 3)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from shared.experiment_config import BATCH_SIZES, GEOM
from shared.palette import DIRECTIONS, is_horizontal
from shared import models as _models
from shared.paths import setup as _setup_paths

_setup_paths()  # parent repo + vendored TLens on sys.path

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

from shared.vision_language_prompts import VLPrompt    # noqa: E402

from criss_cross_patching.src import patch as _patch  # noqa: E402
from criss_cross_patching.src import data as _data    # noqa: E402
from criss_cross_patching.src.regions import (         # noqa: E402
    regions_for, whatsup_regions,
)
from criss_cross_patching.src.tokens import (             # noqa: E402
    grid_to_lm_positions, image_start_offset,
)


HERE = Path(__file__).resolve().parent


def swap_positions(model, processor, vl_prompts: List[VLPrompt],
                   *, model_slug: str, kind: str, direction: str
                   ) -> tuple[list[int], list[int]]:
    """Compute (clean_positions, cf_positions) for the chosen swap.

    Both lists have the same length; entry ``i`` says "patch clean's
    activation at position ``clean_positions[i]`` with cf's activation at
    position ``cf_positions[i]``".

    For horizontal queries the swap is L↔R; for vertical, T↔B.
    """
    sample_prompt = vl_prompts[0].prompt
    sample_image  = vl_prompts[0].images[0]
    image_start = image_start_offset(model, processor, sample_prompt, sample_image)

    regions = regions_for(model_slug, kind)
    if is_horizontal(direction):
        a_grid, b_grid = regions["left"], regions["right"]
    else:
        a_grid, b_grid = regions["top"],  regions["bottom"]

    a_lm = grid_to_lm_positions(a_grid, model_slug=model_slug, image_start=image_start)
    b_lm = grid_to_lm_positions(b_grid, model_slug=model_slug, image_start=image_start)

    clean_positions = a_lm + b_lm   # clean's A and B slots
    cf_positions    = b_lm + a_lm   # source from cf's B and A (swapped)
    return clean_positions, cf_positions


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=list(GEOM))
    p.add_argument("--setting", default="squares", choices=list(_data.SETTINGS),
                   help="task setting (default: squares). whatsup is "
                        "left/right only and Qwen/Gemma only.")
    p.add_argument("--kind", required=True, choices=("object", "strip"),
                   help="object → Fig 6, strip → Fig 7")
    p.add_argument("--direction", choices=DIRECTIONS, default=None)
    p.add_argument("--all-directions", action="store_true")
    p.add_argument("--n-samples", type=int, default=None,
                   help="cap samples per direction (default: use all from cf_pairs)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-list", type=int, nargs="*", default=None)
    p.add_argument("--torch-dtype", default=None,
                   choices=("float32", "bfloat16", "float16"),
                   help="torch dtype (default: each model's `models.DEFAULT_DTYPE`)")
    p.add_argument("--cf-pairs", type=Path, default=None,
                   help="cf-pairs JSON (default: cf_pairs/<setting>.json)")
    p.add_argument("--output-name", default=None,
                   help="basename (no .npy) for the output file under "
                        "`results/<model>/`. Override for parallel per-direction "
                        "runs. Default: <setting>_<kind>_patching.")
    args = p.parse_args()

    cfg = _data.SETTINGS[args.setting]
    valid_dirs = list(cfg.directions)
    if args.all_directions:
        directions = valid_dirs
    elif args.direction is not None:
        directions = [args.direction]
    else:
        p.error("pick --direction or --all-directions")
    bad = [d for d in directions if d not in valid_dirs]
    if bad:
        p.error(f"setting {args.setting!r} supports {valid_dirs}; got {bad}")

    geom = GEOM[args.model]
    batch_size = args.batch_size or BATCH_SIZES["criss_cross_patching"][args.model]
    model_slug = _models.canonical_slug(args.model)

    cf_pairs_path = args.cf_pairs or HERE / "cf_pairs" / f"{args.setting}.json"
    if not cf_pairs_path.exists():
        raise SystemExit(
            f"cf_pairs file not found at {cf_pairs_path}. Build it first:\n"
            f"    python -m criss_cross_patching.build_cf_pairs "
            f"--model {args.model} --setting {args.setting}"
        )
    cf_pairs_all = json.loads(cf_pairs_path.read_text())
    color_tuples = cf_pairs_all[_data.CF_PAIRS_KEY[args.model]]

    print(f"loading {args.model} (setting={args.setting}, "
          f"dtype={args.torch_dtype or 'default'}) ...")
    model, processor = _models.load(
        args.model,
        device=args.device, device_list=args.device_list, cache_dir=args.cache_dir,
        torch_dtype=getattr(torch, args.torch_dtype) if args.torch_dtype else None,
    )

    out_dir = HERE / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, np.ndarray] = {}
    for direction in directions:
        tups = color_tuples[direction]
        if args.n_samples is not None:
            tups = tups[: args.n_samples]
        print(f"\n  {direction}: {len(tups)} samples, batch_size={batch_size}")

        if cfg.grouped_by_middle:
            # whatsup: group by middle reference (tup[1]); regions are
            # per-(model, middle) hand-tuned maps. Concatenate per-middle
            # results along the sample axis.
            from collections import defaultdict
            groups: dict[str, list] = defaultdict(list)
            for t in tups:
                groups[t[1]].append(t)
            per_middle = []
            for middle, gtups in groups.items():
                vlp = _data.build_vl_prompts(processor, gtups, direction, geom,
                                             setting=args.setting, model_key=args.model)
                reg = whatsup_regions(model_slug, middle, args.kind)
                img_start = image_start_offset(model, processor,
                                               vlp[0].prompt, vlp[0].images[0])
                a = grid_to_lm_positions(reg["left"], model_slug=model_slug, image_start=img_start)
                b = grid_to_lm_positions(reg["right"], model_slug=model_slug, image_start=img_start)
                res = _patch.criss_cross_sweep(
                    model, vlp, clean_positions=a + b, cf_positions=b + a,
                    batch_size=batch_size,
                )
                per_middle.append(res.numpy())
                print(f"    middle={middle}: {len(gtups)} samples, "
                      f"{len(a + b)} positions")
            all_results[direction] = np.concatenate(per_middle, axis=0)
        else:
            vl_prompts = _data.build_vl_prompts(processor, tups, direction, geom,
                                                setting=args.setting, model_key=args.model)
            clean_pos, cf_pos = swap_positions(
                model, processor, vl_prompts,
                model_slug=model_slug, kind=args.kind, direction=direction,
            )
            print(f"  swapping {len(clean_pos)} positions per sample "
                  f"({args.kind} patching)")
            results = _patch.criss_cross_sweep(
                model, vl_prompts,
                clean_positions=clean_pos, cf_positions=cf_pos,
                batch_size=batch_size,
            )
            all_results[direction] = results.numpy()

    name = args.output_name or f"{args.setting}_{args.kind}_patching"
    out_path = out_dir / f"{name}.npy"
    np.save(out_path, all_results)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
