"""Plot Table 1 reproduction: paper accuracy vs our measured accuracy.

Inputs:
- `results/table1_summary.csv` (produced by `analysis/aggregate.py`)

Outputs:
- `figures/table1_comparison.png` — 3×4 grid, one bar pair per cell (paper vs ours),
  with the residual annotated. Mismatches (|Δ|>0.04) are highlighted in red.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# Paper-reproduction models (have Table-1 targets) — used by the paper-vs-ours
# comparison figure. The grouped performance figure also includes the added
# models (no paper entry); see GROUPED_MODEL_ORDER.
MODEL_ORDER = ["qwen2vl", "gemma3", "pixtral"]
GROUPED_MODEL_ORDER = ["qwen2vl", "gemma3", "pixtral"]
TASK_ORDER = ["squares", "shapes", "objects", "whatsup"]
MODEL_LABELS = {
    "qwen2vl": "Qwen2-VL-7B",
    "gemma3":  "Gemma-3-4b",
    "pixtral": "Pixtral-12b",
}
MODEL_COLORS = {
    "qwen2vl": "#1f77b4",
    "gemma3":  "#2ca02c",
    "pixtral": "#d62728",
}
MATCH_BAND = 0.04


def _read_rows():
    with (RESULTS / "table1_summary.csv").open() as f:
        return {(r["model"], r["task"]): r for r in csv.DictReader(f)}


def _bar_pair(ax, row, color):
    paper = float(row["paper"])
    ours = float(row["ours"])
    delta = ours - paper
    is_match = abs(delta) <= MATCH_BAND
    edge = "black" if is_match else "#c00000"

    lw = 1.6 if not is_match else 0.5
    ax.bar([0], [paper], width=0.7, color=color, alpha=0.35,
           edgecolor=edge, linewidth=lw)
    ax.bar([1], [ours], width=0.7, color=color, alpha=1.0,
           edgecolor=edge, linewidth=lw)
    for xi, h in zip([0, 1], [paper, ours]):
        ax.text(xi, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    if not is_match:
        ax.text(0.5, -0.18, f"Δ={delta:+.2f}", ha="center", va="top",
                fontsize=9, color="#c00000", fontweight="bold",
                transform=ax.transData)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["paper", "ours"], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure():
    rows = _read_rows()
    fig, axes = plt.subplots(
        nrows=len(MODEL_ORDER), ncols=len(TASK_ORDER),
        figsize=(11, 7.5), sharey=True,
    )

    for i, model in enumerate(MODEL_ORDER):
        for j, task in enumerate(TASK_ORDER):
            ax = axes[i][j]
            row = rows.get((model, task))
            if row is None or row["ours"] in ("", None):
                ax.axis("off")
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                continue
            _bar_pair(ax, row, MODEL_COLORS[model])
            if i == 0:
                ax.set_title(task.capitalize(), fontsize=12, pad=10)
            if j == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=11)

    # Single y-label across the figure
    fig.supylabel("Accuracy", fontsize=11)
    fig.suptitle(
        "Table 1 reproduction: paper vs our measurement "
        "(red border = |Δ| > 0.04)",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    out = FIGURES / "table1_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def make_grouped_figure(model_order=GROUPED_MODEL_ORDER):
    """Grouped bar chart: x=task, one bar per model. Our measured accuracies
    only — no paper reference. Includes the added models (Qwen2-VL-2B,
    LLaVA-1.5-7B) alongside the three paper models."""
    rows = _read_rows()
    # Only plot models that actually have results on disk.
    model_order = [m for m in model_order
                   if any(rows.get((m, t), {}).get("ours") not in ("", None)
                          for t in TASK_ORDER)]
    n = len(model_order)
    # Width scales with both task count and model count so many bars stay legible.
    width = max(1.9 * len(TASK_ORDER) + 2.5, 0.55 * n * len(TASK_ORDER) + 2.0)
    fig, ax = plt.subplots(figsize=(width, 5.2))

    x = list(range(len(TASK_ORDER)))
    group_w = 0.82
    bar_w = group_w / max(n, 1)
    # center the n bars within each task group around the tick
    offsets = {m: (i - (n - 1) / 2) * bar_w for i, m in enumerate(model_order)}

    for model in model_order:
        ys = []
        for t in TASK_ORDER:
            r = rows.get((model, t))
            ys.append(float(r["ours"]) if r and r["ours"] not in ("", None) else 0.0)
        positions = [xi + offsets[model] for xi in x]
        ax.bar(positions, ys, width=bar_w * 0.92, color=MODEL_COLORS[model],
               label=MODEL_LABELS[model], edgecolor="white", linewidth=0.6)
        for xp, y in zip(positions, ys):
            ax.text(xp, y + 0.015, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TASK_ORDER], fontsize=11)
    ax.set_xlim(-0.5, len(TASK_ORDER) - 0.5)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(0.25, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Behavioral performance across spatial reasoning tasks",
                 fontsize=12, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=n,
              frameon=False, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = FIGURES / "behavioral_performance.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    make_grouped_figure()
