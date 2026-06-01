"""Run last-token residual-stream patching on the Squares dataset.

CLI orchestrator only; the actual machinery lives in `src/`:
  - `src/data.py`   — VLPrompt + image construction
  - `src/filter.py` — optional clean+cf top-1 correctness filter
  - `src/patch.py`  — `run_with_cache`/`run_with_hooks` patching loop

Output: `results/<versioned-slug>/squares_final_token.npy`, a pickled
`dict[direction → ndarray]` of shape `(n_samples, n_layers + 1, 1, 5)`
where index 0 along the layer axis is `resid_pre` of layer 0 (post-embed
residual stream) and indices `1..n_layers` are `resid_post` of each layer.

Usage::

    python -m last_token_exp.run_experiment --model qwen2vl
    python -m last_token_exp.run_experiment --model gemma3 --device cuda
    python -m last_token_exp.run_experiment --model pixtral \\
        --cache-dir /data/scratch/<you>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared import models as _models
from shared.experiment_config import BATCH_SIZES, GEOM
from shared.palette import DIRECTIONS
from shared.paths import setup as _setup_paths

_setup_paths()  # parent repo + vendored TLens on sys.path before TL imports

import numpy as np    # noqa: E402
import torch          # noqa: E402

from last_token_exp.src.data import (   # noqa: E402
    CF_PAIRS_KEY, SETTINGS, build_vl_prompts,
)
from last_token_exp.src.filter import filter_correct    # noqa: E402
from last_token_exp.src.patch import patch_all_layers   # noqa: E402

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=list(GEOM),
                   help="model slug (short form, matches the rest of the codebase)")
    p.add_argument("--setting", default="squares", choices=list(SETTINGS),
                   help="task setting (default: squares). whatsup is left/right only.")
    p.add_argument("--directions", nargs="+", default=None,
                   choices=list(DIRECTIONS),
                   help="directions to run (default: all valid for the setting)")
    p.add_argument("--cache-dir", default=None,
                   help="HF model cache directory (passed to load_model)")
    p.add_argument("--device", default="cuda",
                   help="primary device, e.g. cuda / cuda:0 / cpu")
    p.add_argument("--device-list", type=int, nargs="*", default=None,
                   help="GPU device ids for multi-GPU sharding (e.g. 0 1 2)")
    p.add_argument("--n-samples", type=int, default=None,
                   help="cap samples per direction BEFORE filtering "
                        "(default: use all from cf_pairs.json — 50 per direction)")
    p.add_argument("--filter", action="store_true",
                   help="re-validate every candidate at runtime by checking the "
                        "model's top-1 on both clean and cf prompts. Off by "
                        "default: the pool is expected to already be pre-filtered "
                        "(see build_cf_pairs.py). Defense-in-depth against drift.")
    p.add_argument("--cf-pairs", type=Path, default=None,
                   help="path to the JSON pool of (clean, cf) candidate tuples "
                        "(default: cf_pairs/<setting>.json)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override per-model default batch size")
    p.add_argument("--torch-dtype", default=None,
                   choices=("float32", "bfloat16", "float16"),
                   help="torch dtype for model weights (default: each "
                        "model's `models.DEFAULT_DTYPE`)")
    p.add_argument("--output-name", default=None,
                   help="basename (no .npy) for the output file under "
                        "`results/<versioned-slug>/`. Default: "
                        "<setting>_final_token. Override for parallel "
                        "per-direction runs to avoid collision.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    geom = GEOM[args.model]
    batch_size = args.batch_size or BATCH_SIZES["last_token_exp"][args.model]
    directions = args.directions or list(SETTINGS[args.setting].directions)
    cf_pairs_path = args.cf_pairs or HERE / "cf_pairs" / f"{args.setting}.json"

    cf_pairs_all = json.loads(cf_pairs_path.read_text())
    cf_pairs = cf_pairs_all[CF_PAIRS_KEY[args.model]]
    sizes = {d: len(cf_pairs[d]) for d in cf_pairs}
    print(f"loaded {cf_pairs_path.name} [{args.setting}]: per-direction sizes = {sizes}")

    print(f"loading {args.model} on {args.device}"
          f"{f' (devices {args.device_list})' if args.device_list else ''} ...")
    model, processor = _models.load(
        args.model,
        device=args.device,
        device_list=args.device_list,
        cache_dir=args.cache_dir,
        torch_dtype=getattr(torch, args.torch_dtype) if args.torch_dtype else None,
    )

    # Versioned slug for on-disk paths to stay compatible with committed
    # `results/qwen2-vl-7b/...` artifacts + `figures/<versioned>.{png,pdf}`.
    out_dir = HERE / "results" / CF_PAIRS_KEY[args.model]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    survivor_counts = {}
    for direction in directions:
        tups = cf_pairs[direction]
        if args.n_samples is not None:
            tups = tups[: args.n_samples]
        print(f"\n  {direction}: {len(tups)} candidate samples, batch_size={batch_size}")
        vl_prompts = build_vl_prompts(processor, tups, direction, geom,
                                      setting=args.setting, model_key=args.model)
        if args.filter:
            vl_prompts = filter_correct(model, processor, vl_prompts)
        survivor_counts[direction] = len(vl_prompts)
        if not vl_prompts:
            print(f"    no surviving pairs for {direction}; skipping patching")
            continue
        all_results[direction] = patch_all_layers(
            model, vl_prompts, batch_size=batch_size,
        )

    print(f"\nSurvivors per direction (clean+cf top-1 correct): {survivor_counts}")

    output_name = args.output_name or f"{args.setting}_final_token"
    out_path = out_dir / f"{output_name}.npy"
    np.save(out_path, all_results)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
