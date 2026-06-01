"""Reproduce Figure 9 from the paper.

Loads ``results/<model>/<setting>_object_patching_under_ablation.npy`` and
produces a 1×K grid (one panel per direction present) of per-layer
probability curves for the three positional slots (c0 = clean answer,
c1 = middle, c2 = cf answer), matching the criss_cross_patching plot
convention. Settings: squares / shapes / objects.

Run::

    python -m criss_cross_wo_ve_oi.plot --model qwen2vl --setting shapes
    python -m criss_cross_wo_ve_oi.plot --setting objects   # all models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"

DIRECTIONS = ("left", "right", "above", "below")

# Three lines per panel: c0 (clean answer), c1 (middle), c2 (cf answer).
LINE_COLORS = (
    (1.00, 0.00, 0.00),  # red    — c0
    (0.00, 0.65, 0.00),  # green  — c1
    (0.00, 0.00, 1.00),  # blue   — c2
)
LINE_LABELS = ("c0 (clean answer)", "c1 (middle)", "c2 (cf answer)")

MODELS = {
    "qwen2vl": "Qwen2-VL-7B-Instruct",
    "gemma3":  "Gemma-3-4b-it",
    "pixtral": "Pixtral-12B",
}


SETTINGS = ("squares", "shapes", "objects")


def _plot_one_model(slug: str, display: str, setting: str) -> None:
    in_path = RESULTS_DIR / slug / f"{setting}_object_patching_under_ablation.npy"
    if not in_path.exists():
        print(f"skip {slug}/{setting}: {in_path} not found")
        return
    data = np.load(in_path, allow_pickle=True).item()

    directions = [d for d in DIRECTIONS if d in data]
    fig, axes = plt.subplots(1, len(directions), figsize=(5 * len(directions), 4),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    n_samples = None
    for ax, direction in zip(axes, directions):
        arr = data[direction]                        # (n, L+1, 1, 3)
        n_samples = arr.shape[0]
        mean = arr.mean(axis=0).squeeze(axis=1)      # (L+1, 3)
        std  = arr.std(axis=0).squeeze(axis=1)
        x = np.arange(mean.shape[0])
        for j in range(3):
            ax.plot(x, mean[:, j], color=LINE_COLORS[j], label=LINE_LABELS[j], lw=1.6)
            ax.fill_between(x, mean[:, j] - std[:, j], mean[:, j] + std[:, j],
                            color=LINE_COLORS[j], alpha=0.15)
        ax.set_title(direction.capitalize())
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Logit probability")
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Object patching under ablation (Fig 9) — {setting.capitalize()} · "
                 f"{display} (n={n_samples})",
                 fontsize=13)
    fig.tight_layout()

    out_dir = FIG_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{setting}_object_patching_under_ablation.{ext}",
                    dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/{setting}_object_patching_under_ablation.{{png,pdf}}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODELS), default=None,
                   help="render only one model (default: all available)")
    p.add_argument("--setting", choices=SETTINGS, default="squares")
    args = p.parse_args()

    slugs = [args.model] if args.model else list(MODELS)
    for slug in slugs:
        _plot_one_model(slug, MODELS[slug], args.setting)


if __name__ == "__main__":
    main()
