"""Run the §5.3.1 ordering-ablation experiment (paper Table 5).

For each clean (prompt, image) the ablation hook replaces every
entity-token slot with the middle-slot patches of an isolated-color
reference image (preserves color, erases position) and replaces every
other image-token position across the full ``num_tokens × num_tokens``
patch grid with the matching patch of an empty-image reference. This is
the full-grid ablation Sec 5.3.1's prose describes — the same hook
``criss_cross_wo_ve_oi`` reuses for the §5.3.2 patching sweep.

Loads one model and iterates Squares / Shapes / Objects in a single
process so the residual cache is shared across tasks. Use ``--aggregate``
to roll up existing per-cell pickles into the markdown summary.

Run::

    python -m behavioral_analysis_wo_ve_oi.run_experiment \\
        --model gemma3 --all-tasks
    # aggregate all per-cell artifacts already on disk:
    python -m behavioral_analysis_wo_ve_oi.run_experiment --aggregate

Outputs:
    results/<model>_<task>_ablated.{pkl,txt}
    results/table5_summary.{csv,md}  (after --aggregate)
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import torch

from shared import models, paths

paths.setup()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from behavioral_analysis_wo_ve_oi.src.tasks import REGISTRY  # noqa: E402


_PKG_ROOT = Path(__file__).resolve().parent

# Paper Table 5 targets (page 8). Used only by --aggregate to label cells.
_PAPER_TABLE5 = {
    "qwen2vl": {"squares": 0.60, "shapes": 0.62, "objects": 0.43},
    "gemma3":  {"squares": 0.64, "shapes": 0.82, "objects": 0.77},
    "pixtral": {"squares": 0.17, "shapes": 0.01, "objects": 0.39},
}

_DISPLAY_NAME = {
    "qwen2vl": "Qwen2-VL-7B-Instruct",
    "gemma3":  "Gemma-3-4b-it",
    "pixtral": "Pixtral-12b",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=False, choices=list(models.HF_ID))
    p.add_argument("--task", choices=list(REGISTRY), default=None)
    p.add_argument("--all-tasks", action="store_true",
                   help="run squares/shapes/objects in one process load")
    p.add_argument("--results-dir", type=Path,
                   default=_PKG_ROOT / "results",
                   help="output directory for per-cell .pkl/.txt artifacts")
    p.add_argument("--torch-dtype", default=None,
                   choices=("float32", "bfloat16", "float16"),
                   help="dtype for model load (default: each model's "
                        "`models.DEFAULT_DTYPE` — qwen/gemma fp32, pixtral bf16)")
    p.add_argument("--aggregate", action="store_true",
                   help="don't load any model; just aggregate existing per-cell "
                        "artifacts under --results-dir into the markdown table")
    return p.parse_args()


def _run_one_cell(model, processor, model_key: str, task: str,
                  results_dir: Path) -> dict:
    print("=" * 80)
    print(f"TABLE 5 (full-grid ablation, §5.3.1) — {model_key} / {task}")
    print("=" * 80)
    out = REGISTRY[task].run(model, processor, model_key)
    stem = results_dir / f"{model_key}_{task}_ablated"
    with stem.with_suffix(".pkl").open("wb") as f:
        pickle.dump(out, f)
    summary = (
        f"Model: {out['model']}\nTask: {out['task']}\n"
        f"Per-direction:\n"
        + "".join(
            f"  {d}: {out['per_direction_correct'][d]}/{out['per_direction_n'][d]} "
            f"= {out['per_direction_acc'][d]:.4f}\n"
            for d in out["per_direction_n"]
        )
        + f"OVERALL: {out['total_correct']}/{out['total_examples']} "
          f"= {out['overall_acc']:.4f}\n"
    )
    stem.with_suffix(".txt").write_text(summary)
    print(f"Wrote {stem.with_suffix('.pkl').name} + {stem.with_suffix('.txt').name}")
    return out


def _aggregate(results_dir: Path) -> None:
    """Read every ``<model>_<task>_ablated.pkl`` under ``results_dir`` and
    emit ``paper_table5_reproduction.{csv,md}`` alongside them."""
    rows: dict[str, dict[str, float]] = {}
    for model in ("qwen2vl", "gemma3", "pixtral"):
        rows[model] = {}
        for task in ("squares", "shapes", "objects"):
            p = results_dir / f"{model}_{task}_ablated.pkl"
            if p.exists():
                with p.open("rb") as f:
                    d = pickle.load(f)
                rows[model][task] = d["overall_acc"]

    csv_path = results_dir / "table5_summary.csv"
    csv_lines = ["model,squares,shapes,objects"]
    for m, accs in rows.items():
        csv_lines.append(",".join([m]
            + [f"{accs.get(t, ''):.4f}" if t in accs else "" for t in
               ("squares", "shapes", "objects")]))
    csv_path.write_text("\n".join(csv_lines) + "\n")

    md = [
        "# Table 5 — ordering-ablation behavioral accuracy",
        "",
        "Behavioral accuracy after the §5.3.1 vision-encoder "
        "position-information ablation (full-grid background ablation per "
        "Sec 5.3.1's prose). Precision: each model's `DEFAULT_DTYPE` — "
        "Qwen/Gemma fp32, Pixtral bf16.",
        "",
        "Each cell shows ``ours / paper`` and a status flag.",
        "``✓`` = within 0.05 of paper, ``⚠️`` = beyond that band.",
        "",
        "| Model | Squares | Shapes | Objects |",
        "|---|---|---|---|",
    ]
    for m in ("qwen2vl", "gemma3", "pixtral"):
        accs = rows[m]
        cells = []
        for t in ("squares", "shapes", "objects"):
            paper = _PAPER_TABLE5[m][t]
            if t not in accs:
                cells.append("—")
                continue
            ours = accs[t]
            delta = ours - paper
            flag = "✓" if abs(delta) <= 0.05 else (
                f"⚠️ {'+' if delta > 0 else ''}{delta:.2f}"
            )
            cells.append(f"{ours:.3f} / {paper:.2f} {flag}")
        md.append(f"| **{_DISPLAY_NAME[m]}** | " + " | ".join(cells) + " |")
    md.extend([
        "",
        "Chance is 1/3 (≈0.33) for Squares/Shapes/Objects.",
        "",
        "## Methodology note",
        "",
        "This is the full-grid ablation as described in Sec 5.3.1 — every "
        "background patch (across the whole `num_tokens × num_tokens` grid) "
        "is replaced with the matching patch of an empty-image reference. "
        "Some earlier implementations contained a `range(12 * 12) = 144` "
        "hardcode that under-ablates models with grids larger than 12×12; "
        "see `behavioral_analysis_wo_ve_oi/README.md` for context. We do "
        "**not** reproduce that bug here — the numbers in this table reflect "
        "the ablation actually described in the paper text.",
        "",
    ])
    md_path = results_dir / "table5_summary.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {csv_path.name} and {md_path.name}")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        _aggregate(args.results_dir)
        return

    if args.model is None:
        raise SystemExit("--model is required unless --aggregate is given")
    tasks = list(REGISTRY) if args.all_tasks else [args.task]
    if not tasks or tasks == [None]:
        raise SystemExit("pick --task or --all-tasks")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model} "
          f"(dtype={args.torch_dtype or 'default'}) ...", flush=True)
    model, processor = models.load(
        args.model,
        torch_dtype=getattr(torch, args.torch_dtype) if args.torch_dtype else None,
    )
    print(f"[{time.strftime('%H:%M:%S')}] loaded; n_layers={model.cfg.n_layers}",
          flush=True)

    for task in tasks:
        _run_one_cell(model, processor, args.model, task, args.results_dir)
        print(f"[{time.strftime('%H:%M:%S')}] elapsed {time.time()-t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
