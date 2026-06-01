"""Reproduce Figure 3 (Last-Token Position Patching Results) from the paper,
for any setting (squares / shapes / objects / whatsup).

Loads ``results/<model>/<setting>_final_token.npy`` and produces a 1×K grid
(one panel per direction present in the result) of logit-probability curves
with mean and shaded std bands across samples.

The 5 lines per panel are the positional slots of the candidate tuple:
``c0`` (clean answer), ``c1`` (middle/reference), ``c2`` (clean far side),
``c3`` (cf answer), ``c4`` (cf far side). They are drawn in 5 fixed display
colors regardless of setting — for objects/whatsup the tuple elements are
object names, not colors, so the labels are positional rather than literal.

Run::

    python plot.py                          # all models, squares
    python plot.py qwen2vl --setting shapes  # one model, one setting
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"

# Canonical ordering; only directions present in the result are plotted
# (whatsup is horizontal-only → left/right).
DIRECTIONS = ("left", "right", "above", "below")
SETTINGS = ("squares", "shapes", "objects", "whatsup")

# 5 fixed display colors for the 5 positional slots.
LINE_COLORS = (
    (1.00, 0.00, 0.00),  # c0 — clean answer
    (0.00, 0.65, 0.00),  # c1 — middle / reference
    (0.00, 0.00, 1.00),  # c2 — clean far side
    (0.00, 0.00, 0.00),  # c3 — cf answer
    (0.50, 0.50, 0.50),  # c4 — cf far side
)
LINE_LABELS = ("c0 (clean ans)", "c1 (middle)", "c2 (clean far)",
               "c3 (cf ans)", "c4 (cf far)")

_SHORT_TO_VERSIONED = {
    "qwen2vl": "qwen2-vl-7b",
    "gemma3":  "gemma-3-4b-it",
    "pixtral": "pixtral-12b",
}
_DISPLAY = {
    "qwen2vl": "Qwen2-VL-7B-Instruct",
    "gemma3":  "Gemma-3-4b-it",
    "pixtral": "Pixtral-12B",
}
MODELS = tuple(_SHORT_TO_VERSIONED)


def _xtick_labels(n_layers: int) -> tuple[list[int], list[str]]:
    """Layer ticks: 0 is the residual stream pre-layer-0 ("Embed")."""
    step = max(1, (n_layers - 1) // 8)
    ticks = [0] + list(range(step, n_layers, step))
    if ticks[-1] != n_layers - 1:
        ticks.append(n_layers - 1)
    labels = ["Embed" if t == 0 else str(t - 1) for t in ticks]
    return ticks, labels


def plot_one(model: str, setting: str) -> Path | None:
    versioned = _SHORT_TO_VERSIONED[model]
    data_path = RESULTS_DIR / versioned / f"{setting}_final_token.npy"
    if not data_path.exists():
        print(f"skip {model}/{setting}: {data_path} not found")
        return None
    results = np.load(data_path, allow_pickle=True).item()

    directions = [d for d in DIRECTIONS if d in results]
    first = results[directions[0]]
    n_samples, n_layers, _, n_colors = first.shape
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(1, len(directions), figsize=(5 * len(directions), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, direction in zip(axes, directions):
        arr = results[direction][:, :, 0, :]  # (samples, layers, colors)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        for c in range(n_colors):
            ax.plot(layers, mean[:, c], color=LINE_COLORS[c],
                    label=LINE_LABELS[c], linewidth=2.0)
            ax.fill_between(layers, mean[:, c] - std[:, c], mean[:, c] + std[:, c],
                            color=LINE_COLORS[c], alpha=0.18, linewidth=0)
        ax.set_title(direction.capitalize(), fontsize=14)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
        ticks, labels = _xtick_labels(n_layers)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Logit probability")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        f"Last Token Position Patching — {setting.capitalize()} · "
        f"{_DISPLAY[model]} (n={n_samples})",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    out_dir = FIG_DIR / versioned
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{setting}.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    fig.savefig(out_dir / f"{setting}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=list(MODELS),
                        help=f"which models to plot (choices: {list(MODELS)})")
    parser.add_argument("--setting", default="squares", choices=SETTINGS)
    args = parser.parse_args()
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {unknown}; choices are {list(MODELS)}")
    for m in args.models:
        path = plot_one(m, args.setting)
        if path is not None:
            print(f"wrote {path.relative_to(HERE)} (+ .pdf)")


if __name__ == "__main__":
    main()
