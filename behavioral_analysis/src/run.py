"""CLI entry point: run one (model, task) cell of Table 1.

Usage:
    python -m behavioral_analysis.src.run --model gemma3 --task squares

Output paths (relative to behavioral_analysis/):
    results/<model>_<task>_baseline.pkl
    results/<model>_<task>_baseline.txt
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

from shared import models, paths

paths.setup()  # repo + vendored TLens on sys.path before TL imports below

# Avoid PyTorch CUDA fragmentation on 12B models.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from behavioral_analysis.src.tasks import REGISTRY  # noqa: E402

_PKG_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", required=True, choices=list(models.HF_ID))
    p.add_argument("--task", required=True, choices=list(REGISTRY))
    p.add_argument("--results-dir", type=Path,
                   default=_PKG_ROOT / "results",
                   help="Directory for output .pkl and .txt files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"TABLE 1 BEHAVIORAL ANALYSIS — {args.model} / {args.task}")
    print("=" * 80)

    model, processor = models.load(args.model)
    out = REGISTRY[args.task].run(model, processor, args.model)

    stem = args.results_dir / f"{args.model}_{args.task}_baseline"
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


if __name__ == "__main__":
    main()
