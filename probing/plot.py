"""Render strip-pattern heatmaps + an accuracy summary CSV from saved probes.

Reads every ``results/<model>/<setting>_<orientation>.npz`` and produces:

- ``figures/<model>/<setting>_<orientation>_heatmap.png`` — 3-panel
  per-class probability heatmap on the test-image patch grid (reproduces
  paper Fig 4 and Figs 23-40 in App C.5).
- ``figures/summary_table.csv`` and ``.md`` — argmax accuracy on
  object-region test tokens for every (model, setting, orientation).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from probing.src.config import GEOM, probing_geom


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

CLASS_LABELS_3 = {
    "horizontal": ("Left", "Middle", "Right"),
    "vertical":   ("Top",  "Middle", "Bottom"),
}


def _labels_for(orientation: str, n_classes: int) -> tuple[str, ...]:
    if n_classes == 3 and orientation in CLASS_LABELS_3:
        return CLASS_LABELS_3[orientation]
    if orientation == "horizontal":
        return tuple(f"Col {i}" for i in range(n_classes))
    return tuple(f"Row {i}" for i in range(n_classes))


def _grid_side(model_slug: str, n_patches: int) -> int:
    """Side length of the (square) patch grid encoded by a flat-N tensor.

    Cross-check against the probing geometry (`probing_geom`, which honors any
    probing-only override such as Phi-3.5-V's 24x24 CLIP grid) so a typo in
    either source is caught early.
    """
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        raise ValueError(f"non-square patch count {n_patches}")
    expected = probing_geom(model_slug) if model_slug in GEOM else None
    if expected is not None and expected.num_tokens != side:
        raise ValueError(
            f"{model_slug}: probe file has {side}x{side} patches but GEOM "
            f"says num_tokens={expected.num_tokens}"
        )
    return side


def render_heatmap(npz_path: Path, *, model_slug: str, setting: str,
                   orientation: str, out_path: Path) -> None:
    data = np.load(npz_path)
    all_pred = data["test_pred_all"]            # (n_test, N_patches, n_classes)
    avg = all_pred.mean(axis=0)                  # (N_patches, n_classes)
    n_classes = avg.shape[-1]
    side = _grid_side(model_slug, avg.shape[0])
    heatmaps = avg.reshape(side, side, n_classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5),
                             squeeze=False)
    axes = axes[0]
    fig.suptitle(
        f"Position probe — {model_slug} / {setting} / {orientation} "
        f"(acc={float(data['test_acc']):.3f})"
    )
    labels = _labels_for(orientation, n_classes)
    for i, lbl in enumerate(labels):
        im = axes[i].imshow(heatmaps[:, :, i], vmin=0, vmax=1, interpolation="nearest")
        axes[i].set_title(lbl)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        fig.colorbar(im, ax=axes[i])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def build_summary(rows: list[dict], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("model", "setting", "orientation",
                                               "test_acc", "train_loss"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {out_csv}")

    rows_sorted = sorted(rows, key=lambda r: (r["model"], r["setting"], r["orientation"]))
    lines = ["| model | setting | orientation | acc | loss |",
             "|-------|---------|-------------|-----|------|"]
    for r in rows_sorted:
        lines.append(
            f"| {r['model']} | {r['setting']} | {r['orientation']} | "
            f"{r['test_acc']:.4f} | {r['train_loss']:.4f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_md}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = p.parse_args()

    results_root = Path(args.results_dir)
    figures_root = Path(args.figures_dir)

    rows: list[dict] = []
    for model_dir in sorted(results_root.glob("*")):
        if not model_dir.is_dir():
            continue
        model_slug = model_dir.name
        for npz_path in sorted(model_dir.glob("*.npz")):
            stem = npz_path.stem  # e.g. "squares_horizontal"
            try:
                setting, orientation = stem.rsplit("_", 1)
            except ValueError:
                print(f"skipping unrecognized file {npz_path}")
                continue
            out_png = figures_root / model_slug / f"{stem}_heatmap.png"
            render_heatmap(npz_path,
                           model_slug=model_slug,
                           setting=setting,
                           orientation=orientation,
                           out_path=out_png)
            data = np.load(npz_path)
            rows.append({
                "model": model_slug,
                "setting": setting,
                "orientation": orientation,
                "test_acc": float(data["test_acc"]),
                "train_loss": float(data["train_loss"]),
            })

    if rows:
        build_summary(rows,
                      out_csv=figures_root / "summary_table.csv",
                      out_md=figures_root / "summary_table.md")
    else:
        print(f"no .npz files found under {results_root}")


if __name__ == "__main__":
    main()
