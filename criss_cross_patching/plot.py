"""Reproduce paper Figures 6 (object patching) and 7 (strip patching).

Loads ``results/<model>/squares_{object,strip}_patching.npy`` (dict of
``direction → ndarray(n_samples, n_layers + 1, 1, 3)``) and produces a
1×4 grid (Left / Right / Above / Below) of per-layer probability curves.

The 3 lines per panel are ``c0`` (clean answer, always), ``c1`` (middle,
unchanged) and ``c2`` (cf answer, always). ``build_vl_prompts`` reverses
the image tuple for right/below so that this assignment holds in every
direction — matching the paper's Figure 16 plot convention.

A Fig-6 (object) run shows c0 ≈ flat near 1.0 across layers (patching the
square tokens alone doesn't switch the output).
A Fig-7 (strip) run shows c2 rising up to layer ~24 then handing off to c0
again — the swapped ordering signal flips the output in early-to-mid layers.

Run::

    python -m criss_cross_patching.plot                       # squares, both kinds, all models
    python -m criss_cross_patching.plot --model qwen2vl       # one model
    python -m criss_cross_patching.plot --setting objects --kind strip
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
KINDS = ("object", "strip")
SETTINGS = ("squares", "shapes", "objects", "whatsup")

# 3 lines per panel: c0 (clean answer), c1 (middle, unchanged), c2 (cf answer).
LINE_COLORS = (
    (1.00, 0.00, 0.00),  # Red    — c0 (clean answer)
    (0.00, 0.65, 0.00),  # Green  — c1 (middle)
    (0.00, 0.00, 1.00),  # Blue   — c2 (cf answer)
)
LINE_LABELS = ("c0 (clean answer)", "c1 (middle)", "c2 (cf answer)")

MODELS = {
    "qwen2vl": "Qwen2-VL-7B-Instruct",
    "gemma3":  "Gemma-3-4b-it",
    "pixtral": "Pixtral-12B",
}

KIND_TITLE = {
    "object": "Object patching (Fig 6)",
    "strip":  "Strip patching (Fig 7)",
}


def _xtick_labels(n_slots: int) -> tuple[list[int], list[str]]:
    step = max(1, (n_slots - 1) // 8)
    ticks = [0] + list(range(step, n_slots, step))
    if ticks[-1] != n_slots - 1:
        ticks.append(n_slots - 1)
    labels = ["Embed" if t == 0 else str(t - 1) for t in ticks]
    return ticks, labels


def plot_one(model: str, setting: str, kind: str) -> Path | None:
    data_path = RESULTS_DIR / model / f"{setting}_{kind}_patching.npy"
    if not data_path.exists():
        print(f"skip {data_path} — missing")
        return None
    results = np.load(data_path, allow_pickle=True).item()

    directions = [d for d in DIRECTIONS if d in results]
    n_samples, n_slots, _, n_colors = results[directions[0]].shape
    layers = np.arange(n_slots)

    fig, axes = plt.subplots(1, len(directions), figsize=(5 * len(directions), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, direction in zip(axes, directions):
        arr = results[direction][:, :, 0, :]  # (samples, slots, colors)
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
        ticks, labels = _xtick_labels(n_slots)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Logit probability")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        f"{KIND_TITLE[kind]} — {setting.capitalize()} · {MODELS[model]} (n={n_samples})",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    out_dir = FIG_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{setting}_{kind}_patching.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    fig.savefig(out_dir / f"{setting}_{kind}_patching.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default=None,
                        help="render only one model (default: all)")
    parser.add_argument("--setting", choices=SETTINGS, default="squares")
    parser.add_argument("--kind", choices=KINDS, default=None,
                        help="render only one patching kind (default: both)")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODELS)
    kinds = [args.kind] if args.kind else list(KINDS)
    for m in models:
        for k in kinds:
            path = plot_one(m, args.setting, k)
            if path is not None:
                print(f"wrote {path.relative_to(HERE)} (+ .pdf)")


if __name__ == "__main__":
    main()
