"""Train one linear position probe per (model × setting × orientation).

Reproduces section 5.2.1 of the paper. For each combination, builds
`P(|options|, 3)` synthetic images, splits 75/25 (90/30 for Squares/Shapes),
extracts post-projection patch embeddings, and fits a 3-class linear probe on
the object-region patches. Test-time predictions on **all** patches (not just
the object regions) are also saved so the strip-pattern heatmaps can be
rendered by ``plot.py``.

Usage::

    python -m probing.run_experiment --model qwen2vl --setting squares
    python -m probing.run_experiment --model gemma3 --setting shapes --orientation vertical
    python -m probing.run_experiment --model pixtral --setting objects --all-orientations

Use ``--all-settings`` to sweep squares + shapes + objects in one process load.

Output: ``results/<model>/<setting>_<orientation>.npz`` containing ``W``,
``b``, ``test_pred``, ``test_pred_all`` (shape `(n_test, N_patches, 3)`),
``test_acc``, ``train_loss``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from shared import models as _models
from shared.paths import setup as _setup_paths

_setup_paths()  # parent repo + vendored TLens on sys.path

from probing.src import data as _data       # noqa: E402
from probing.src import grids as _grids     # noqa: E402
from probing.src import probe as _probe     # noqa: E402
from probing.src import whatsup as _whatsup  # noqa: E402
from probing.src.config import GEOM         # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# Settings the runner can sweep. "synthetic" covers squares + shapes + objects.
SYNTHETIC = ("squares", "shapes", "objects")
EXTRA = ("whatsup", "grid2x2", "grid3x3")
SETTINGS = SYNTHETIC + EXTRA
ORIENTATIONS = ("horizontal", "vertical")


def _build_split(model, processor, *, model_slug, setting, orientation, seed):
    """Dispatch to the right setting-specific builder."""
    if setting in SYNTHETIC:
        return _data.build_split(
            model, processor,
            model_slug=model_slug,
            setting=setting,
            orientation=orientation,
            seed=seed,
        )
    if setting == "whatsup":
        if orientation != "horizontal":
            raise ValueError("whatsup only has horizontal layout")
        return _whatsup.build_split(
            model, processor,
            model_slug=model_slug,
            seed=seed,
        )
    if setting in ("grid2x2", "grid3x3"):
        k = int(setting[4])
        return _grids.build_split(
            model, processor,
            model_slug=model_slug,
            k=k,
            axis=orientation,
            seed=seed,
        )
    raise ValueError(f"unknown setting {setting!r}")


def run_one(model, processor, *,
            model_slug: str,
            setting: str,
            orientation: str,
            epochs: int,
            seed: int,
            device: str) -> _probe.ProbeFit:
    print(f"\n=== {model_slug} | {setting} | {orientation} ===")
    split = _build_split(
        model, processor,
        model_slug=model_slug, setting=setting,
        orientation=orientation, seed=seed,
    )
    # n_classes inferred from labels: synthetic+whatsup = 3, grids = k.
    n_classes = int(split.train_labels.max().item()) + 1
    print(f"  train obj tokens: {split.train_embeds.shape}  "
          f"test obj tokens: {split.test_embeds.shape}  "
          f"all-token test: {split.all_tokens_test_embeds.shape}  "
          f"n_classes={n_classes}")
    fit = _probe.train(
        split.train_embeds, split.train_labels,
        split.test_embeds,  split.test_labels,
        split.all_tokens_test_embeds,
        n_test_images=split.n_test_images,
        patches_per_image=split.patches_per_image,
        n_classes=n_classes,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    print(f"  acc={fit.test_acc:.4f}  loss={fit.train_loss:.4f}")
    return fit


def save(out_dir: Path, setting: str, orientation: str, fit: _probe.ProbeFit) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{setting}_{orientation}.npz"
    np.savez(
        path,
        W=fit.W, b=fit.b,
        test_pred=fit.test_pred,
        test_pred_all=fit.test_pred_all,
        test_acc=np.array(fit.test_acc),
        train_loss=np.array(fit.train_loss),
    )
    print(f"  saved {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="model slug — qwen2vl / gemma3 / pixtral, "
                        "or any alias accepted by shared.models")
    p.add_argument("--setting", choices=SETTINGS, default=None,
                   help="One of: " + ", ".join(SETTINGS))
    p.add_argument("--all-settings", action="store_true",
                   help="Sweep squares + shapes + objects (NOT whatsup/grids).")
    p.add_argument("--orientation", choices=ORIENTATIONS, default=None)
    p.add_argument("--all-orientations", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-list", type=int, nargs="*", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--torch-dtype", default=None,
                   choices=("float32", "bfloat16", "float16"),
                   help="torch dtype (default: each model's `models.DEFAULT_DTYPE`)")
    args = p.parse_args()

    # --all-settings expands to the synthetic three only; extras are opt-in.
    settings = list(SYNTHETIC) if args.all_settings else [args.setting]
    orientations = list(ORIENTATIONS) if args.all_orientations else [args.orientation]
    if not settings or settings == [None]:
        p.error("pick --setting or --all-settings")
    if not orientations or orientations == [None]:
        p.error("pick --orientation or --all-orientations")

    slug = _models.canonical_slug(args.model)
    if slug not in GEOM:
        p.error(f"no GEOM config for {slug!r}; add it to src/config.py")

    print(f"loading {slug} (dtype={args.torch_dtype or 'default'}) ...")
    model, processor = _models.load(
        args.model,
        device=args.device,
        device_list=args.device_list,
        cache_dir=args.cache_dir,
        torch_dtype=getattr(torch, args.torch_dtype) if args.torch_dtype else None,
    )

    out_dir = RESULTS_DIR / slug
    for setting in settings:
        for orientation in orientations:
            if setting == "whatsup" and orientation == "vertical":
                # WhatsUp composites are horizontal-only (Sec 3.1).
                continue
            fit = run_one(
                model, processor,
                model_slug=slug,
                setting=setting,
                orientation=orientation,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
            )
            save(out_dir, setting, orientation, fit)


if __name__ == "__main__":
    main()
