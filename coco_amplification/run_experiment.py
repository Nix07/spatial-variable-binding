"""Run COCO spatial-relation amplification.

Canonical usage from the repo root:

    python -m coco_amplification.run_experiment --model qwen
    python -m coco_amplification.run_experiment --model gemma
    python -m coco_amplification.run_experiment --model pixtral
    python -m coco_amplification.run_experiment --aggregate

The experiment intentionally uses the raw Hugging Face model paths from the
original notebooks to preserve the reported numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from shared.paths import setup as _setup_paths

_setup_paths()

from coco_amplification.src import common as cc  # noqa: E402
from coco_amplification.src import coco_cache  # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
EXPERIMENT_SLUG = "coco_spatial"
AXES = ("horizontal", "vertical")
MODEL_CONFIGS = {
    "qwen2-vl-2b-instruct": {
        "aliases": {
            "qwen2vl2b",
            "qwen",
            "qwen2-vl-2b",
            "qwen2-vl-2b-instruct",
            "qwen/qwen2-vl-2b-instruct",
        },
        "module": "qwen",
        "result_slug": "qwen2-vl-2b-instruct",
        "file_slug": "qwen2-vl-2b",
        "schema_version": "qwen2_coco_2obj_spatial_anchor_strict_v1",
        "prompt_mode": "spatial_anchor_reference",
        "artifact_type": "coco_2obj_spatial_qwen_baseline",
        "report_label": "Qwen2-VL-2B",
        "prompt_description": None,
        "eval_order": "natural",
    },
    "gemma-3-4b-it": {
        "aliases": {
            "gemma3",
            "gemma",
            "gemma-3-4b",
            "gemma-3-4b-it",
            "google/gemma-3-4b-it",
        },
        "module": "gemma",
        "result_slug": "gemma-3-4b-it",
        "file_slug": "gemma-3-4b",
        "schema_version": "gemma3_coco_2obj_spatial_direct_cleanprompt_strict_v1",
        "prompt_mode": "spatial_direct_cleanprompt",
        "artifact_type": "coco_2obj_spatial_gemma_baseline",
        "report_label": "Gemma3-4B",
        "prompt_description": (
            "Horizontal: Is the {subj} to the left or to the right of the {obj}? Answer with one word.\n"
            "Vertical:   Is the {subj} above or below the {obj}? Answer with one word.\n"
            "Note: prompts use cleaned object labels with leading articles stripped."
        ),
        "eval_order": "vertical_first",
    },
    "pixtral-12b": {
        "aliases": {
            "pixtral",
            "pixtral-12b",
            "mistral-community/pixtral-12b",
        },
        "module": "pixtral",
        "result_slug": "pixtral-12b",
        "file_slug": "pixtral-12b",
        "schema_version": "pixtral_coco_2obj_spatial_anchor_hf_strict_v1",
        "prompt_mode": "spatial_anchor_reference",
        "artifact_type": "pixtral-12b_coco_spatial_baseline",
        "report_label": "Pixtral-12B",
        "prompt_description": (
            "Horizontal: Using the {obj} as the reference point, which side is the {subj} on? "
            "Answer with one word: left or right.\n"
            "Vertical:   Using the {obj} as the reference point, where is the {subj} located? "
            "Answer with one word: above or below.\n"
            "Note: prompts use cleaned object labels with leading articles stripped."
        ),
        "eval_order": "natural",
    },
}


def canonical_config(name: str):
    key = name.lower()
    for canonical, cfg in MODEL_CONFIGS.items():
        if key in cfg["aliases"]:
            return canonical, cfg
    accepted = sorted(alias for cfg in MODEL_CONFIGS.values() for alias in cfg["aliases"])
    raise ValueError(f"unsupported COCO amplification model {name!r}; accepted: {accepted}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="qwen",
                   help="Supported: qwen, gemma, pixtral")
    p.add_argument("--torch-dtype", default="fp32", choices=("fp32", "bf16"))
    p.add_argument("--coco-dir", default=None,
                   help="COCO cache root (default: $COCO_DIR or shared/coco_2obj)")
    p.add_argument("--cache-dir", default=None,
                   help="HF model cache directory")
    p.add_argument("--alphas", type=int, nargs="*", default=list(range(16)),
                   help="Intervention strengths to test (default: 0..15)")
    p.add_argument("--recompute", action="store_true",
                   help="Ignore cached baseline/position-probe artifacts and recompute them")
    p.add_argument("--max-items", type=int, default=None,
                   help="Debug cap on spatial QA examples before baseline eval")
    p.add_argument("--aggregate", action="store_true",
                   help="Build summary CSV/MD from saved amp_results.pkl files")
    return p.parse_args()


def _artifact_dir(model_slug: str, dtype_name: str) -> Path:
    dtype_dir = cc.dtype_artifact_dir(dtype_name)
    return RESULTS_DIR / model_slug / dtype_dir


def _cap_spatial_data(spatial_data, max_items: int | None):
    if max_items is None or max_items >= len(spatial_data):
        return spatial_data
    if max_items < len(AXES):
        return spatial_data[:max_items]
    selected = []
    selected_ids = set()
    for axis in AXES:
        for idx, item in enumerate(spatial_data):
            if item["axis"] == axis:
                selected.append(item)
                selected_ids.add(idx)
                break
    for idx, item in enumerate(spatial_data):
        if len(selected) >= max_items:
            break
        if idx not in selected_ids:
            selected.append(item)
    return selected


def _probe_path(artifact_dir: Path, axis: str, param: str, artifact_suffix: str,
                file_slug: str) -> Path:
    return artifact_dir / f"{file_slug}_{EXPERIMENT_SLUG}_probe_{axis}_{param}_{artifact_suffix}.npy"


def load_or_train_probes(probe_data, artifact_dir: Path, *,
                         dtype_name: str, device, force: bool, file_slug: str):
    artifact_suffix = cc.dtype_artifact_dir(dtype_name)
    cached = (
        not force
        and all(_probe_path(artifact_dir, axis, "W", artifact_suffix, file_slug).exists()
                and _probe_path(artifact_dir, axis, "b", artifact_suffix, file_slug).exists()
                for axis in AXES)
    )
    if cached:
        print("Loading cached probe weights.")
        W_probe_by_axis, b_probe_by_axis, probe_train_stats = {}, {}, {}
        for axis in AXES:
            W_np = np.load(_probe_path(artifact_dir, axis, "W", artifact_suffix, file_slug))
            b_np = np.load(_probe_path(artifact_dir, axis, "b", artifact_suffix, file_slug))
            assert W_np.shape[0] == probe_data[axis]["X"].shape[1], (
                f"{axis}: cached probe dim {W_np.shape[0]} != live embedding dim "
                f"{probe_data[axis]['X'].shape[1]} - stale weights; delete the cached probe files."
            )
            W_probe_by_axis[axis] = torch.from_numpy(W_np).to(device)
            b_probe_by_axis[axis] = torch.from_numpy(b_np).to(device)
            probe_train_stats[axis] = {
                "train_acc": float("nan"),
                "test_acc": float("nan"),
                "D": int(W_np.shape[0]),
                "train_items": [],
                "test_items": [],
            }
            print(f"  {axis}: W={tuple(W_np.shape)} b={tuple(b_np.shape)}")
        return W_probe_by_axis, b_probe_by_axis, probe_train_stats

    W_probe_by_axis, b_probe_by_axis, probe_train_stats = cc.train_axis_probes(
        probe_data, axes=AXES, device=device,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for axis in AXES:
        np.save(_probe_path(artifact_dir, axis, "W", artifact_suffix, file_slug),
                W_probe_by_axis[axis].detach().cpu().numpy())
        np.save(_probe_path(artifact_dir, axis, "b", artifact_suffix, file_slug),
                b_probe_by_axis[axis].detach().cpu().numpy())
        print(f"  Saved {_probe_path(artifact_dir, axis, 'W', artifact_suffix, file_slug)}")
        print(f"  Saved {_probe_path(artifact_dir, axis, 'b', artifact_suffix, file_slug)}")
    return W_probe_by_axis, b_probe_by_axis, probe_train_stats


def run(args: argparse.Namespace) -> None:
    canonical_model, model_cfg = canonical_config(args.model)
    if model_cfg["module"] == "qwen":
        from coco_amplification.src import qwen as model_helpers

        runner_cls = model_helpers.QwenCocoRunner
    elif model_cfg["module"] == "gemma":
        from coco_amplification.src import gemma as model_helpers

        runner_cls = model_helpers.GemmaCocoRunner
    elif model_cfg["module"] == "pixtral":
        from coco_amplification.src import pixtral as model_helpers

        runner_cls = model_helpers.PixtralCocoRunner
    else:
        raise AssertionError(f"bad model module {model_cfg['module']!r}")

    model_slug = model_helpers.canonical_model_name(canonical_model)
    artifact_suffix = cc.dtype_artifact_dir(args.torch_dtype)
    artifact_dir = _artifact_dir(model_cfg["result_slug"], args.torch_dtype)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    coco_dir = Path(args.coco_dir).expanduser() if args.coco_dir else coco_cache.default_coco_dir()
    qa, img_dir = cc.download_coco_data(str(coco_dir))
    spatial_data = cc.build_spatial_data(qa)
    spatial_data = _cap_spatial_data(spatial_data, args.max_items)

    from collections import Counter
    axis_counts = Counter(e["axis"] for e in spatial_data)
    truth_counts = Counter(e["truth"] for e in spatial_data)
    print(f"COCO_DIR: {coco_dir}")
    print(f"Filtered to spatial left/right/above/below: {len(spatial_data)} / {len(qa)}")
    print(f'  horizontal:   {axis_counts["horizontal"]}')
    print(f'  vertical:     {axis_counts["vertical"]}')
    print(f"  truth counts: {dict(truth_counts)}")
    assert axis_counts["horizontal"] > 0, "No horizontal examples found."
    assert axis_counts["vertical"] > 0, "No vertical examples found."

    if model_cfg["eval_order"] == "vertical_first":
        eval_items = [e for e in spatial_data if e["axis"] == "vertical"]
        eval_items += [e for e in spatial_data if e["axis"] == "horizontal"]
    else:
        eval_items = spatial_data

    runner = runner_cls(
        model_name=model_slug,
        dtype_name=args.torch_dtype,
        img_dir=img_dir,
        cache_dir=args.cache_dir,
    )

    cache_path = artifact_dir / f"{model_cfg['file_slug']}_{EXPERIMENT_SLUG}_baseline_{artifact_suffix}.pkl"
    baseline_metadata = cc.coco_artifact_metadata(
        model_name=model_slug,
        dtype=args.torch_dtype,
        prompt_mode=model_cfg["prompt_mode"],
        artifact_type=model_cfg["artifact_type"],
    )
    results = cc.run_coco_baseline_eval(
        eval_items,
        predict_fn=lambda item, listing: runner.predict_with_amp(item, listing, W=None, alpha=0),
        cache_path=str(cache_path),
        schema_version=model_cfg["schema_version"],
        model_name=runner.model_id,
        force_recompute=args.recompute,
        metadata=baseline_metadata,
    )
    clean = cc.clean_results(results)
    partition = cc.partition_results(clean)

    coco_bboxes, coco_img_size = cc.load_coco_gt_bboxes(str(coco_dir))
    find_coco_bbox = cc.make_find_coco_bbox(coco_bboxes)
    get_bbox = cc.make_get_bbox(find_coco_bbox, coco_img_size)
    print("Loaded COCO ground-truth bboxes.")

    probe_data = runner.build_probe_data(
        partition,
        get_bbox,
        artifact_dir,
        model_slug=model_cfg["file_slug"],
        experiment_slug=EXPERIMENT_SLUG,
        artifact_suffix=artifact_suffix,
        axes=AXES,
    )
    W_probe_by_axis, b_probe_by_axis, probe_train_stats = load_or_train_probes(
        probe_data,
        artifact_dir,
        dtype_name=args.torch_dtype,
        device=runner.model_device,
        force=args.recompute,
        file_slug=model_cfg["file_slug"],
    )

    alpha_range = list(args.alphas)
    amp_alphas = [a for a in alpha_range if a > 0]
    if not amp_alphas:
        raise SystemExit("At least one alpha > 0 is required for amplification summaries.")
    n_per_cat = {c: len(partition[c]) for c in cc.FAIL_CATS}
    n_fail_total = sum(n_per_cat.values())
    print(f"Fail items: {n_per_cat}  (total {n_fail_total})")

    for axis in AXES:
        W_probe = W_probe_by_axis[axis]
        assert W_probe.ndim == 2 and W_probe.shape[1] == 2, (
            f"{axis}: bad W_probe shape: {tuple(W_probe.shape)}"
        )
        assert W_probe.shape[0] == probe_data[axis]["X"].shape[1], (
            f"{axis}: W dim={W_probe.shape[0]} does not match X dim={probe_data[axis]['X'].shape[1]}"
        )
    assert len(runner.lm_layers[0]._forward_pre_hooks) == 0, "stale layer-0 pre-hooks before sweep"
    assert len(runner.lm_layers[0]._forward_hooks) == 0, "stale layer-0 forward hooks before sweep"

    print("\n" + "=" * 60)
    print("SWEEP: PROBE direction")
    print("=" * 60)
    probe_per_alpha, probe_per_alpha_axis, probe_log = cc.run_coco_amp_sweep(
        partition,
        {axis: W_probe_by_axis[axis].to(dtype=runner.torch_dtype) for axis in AXES},
        predict_with_amp_fn=runner.predict_with_amp,
        clear_hooks_fn=runner.clear_layer0_hooks,
        axes=AXES,
        alphas=alpha_range,
    )
    probe_summary = cc.print_amp_summary(
        "PROBE AMPLIFICATION",
        partition=partition,
        per_alpha=probe_per_alpha,
        per_alpha_axis=probe_per_alpha_axis,
        per_item_log=probe_log,
        amp_alphas=amp_alphas,
        axes=AXES,
    )
    best_probe_a = probe_summary["best_alpha"]
    probe_oracle, probe_oracle_axis, _ = cc.compute_oracle(
        probe_log, amp_alphas=amp_alphas, axes=AXES,
    )
    alpha0_fixed = sum(1 for entry in probe_log if entry["fixed_at"].get(0))
    print(f"Alpha=0 replay check: {alpha0_fixed}/{n_fail_total} fixed")
    assert alpha0_fixed == 0, "Some failed items are fixed at alpha=0; partition/pipeline mismatch."

    W_random_by_axis = cc.make_random_W(
        W_probe_by_axis,
        axes=AXES,
        device=runner.model_device,
        dtype=runner.torch_dtype,
    )
    print("\n" + "=" * 60)
    print("SWEEP: RANDOM direction")
    print("=" * 60)
    random_per_alpha, random_per_alpha_axis, random_log = cc.run_coco_amp_sweep(
        partition,
        W_random_by_axis,
        predict_with_amp_fn=runner.predict_with_amp,
        clear_hooks_fn=runner.clear_layer0_hooks,
        axes=AXES,
        alphas=alpha_range,
    )
    random_summary = cc.print_amp_summary(
        "RANDOM CONTROL",
        partition=partition,
        per_alpha=random_per_alpha,
        per_alpha_axis=random_per_alpha_axis,
        per_item_log=random_log,
        amp_alphas=amp_alphas,
        axes=AXES,
    )
    best_random_a = random_summary["best_alpha"]
    random_oracle, random_oracle_axis, _ = cc.compute_oracle(
        random_log, amp_alphas=amp_alphas, axes=AXES,
    )

    cc.write_coco_amp_report(
        artifact_dir,
        model_cfg["report_label"],
        results=results,
        clean=clean,
        partition=partition,
        probe_per_alpha=probe_per_alpha,
        probe_per_alpha_axis=probe_per_alpha_axis,
        random_per_alpha=random_per_alpha,
        random_per_alpha_axis=random_per_alpha_axis,
        probe_oracle=probe_oracle,
        random_oracle=random_oracle,
        probe_oracle_axis=probe_oracle_axis,
        random_oracle_axis=random_oracle_axis,
        best_probe_a=best_probe_a,
        best_random_a=best_random_a,
        alpha_range=alpha_range,
        axes=AXES,
        probe_train_stats=probe_train_stats,
        output_prefix="",
        prompt_description=model_cfg["prompt_description"],
    )


def aggregate() -> None:
    rows = []
    for path in RESULTS_DIR.glob("*/*/amp_results.pkl"):
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        n_golden = int(artifact["n_golden"])
        n_fail = sum(artifact["partition_counts"].values())
        n_total = n_golden + n_fail
        best_probe = artifact["best_probe_alpha"]
        best_random = artifact["best_random_alpha"]
        probe_fixed = sum(artifact["probe_per_alpha"][best_probe].values())
        random_fixed = sum(artifact["random_per_alpha"][best_random].values())
        probe_star = sum(artifact["probe_oracle"].values())
        random_star = sum(artifact["random_oracle"].values())
        rows.append({
            "model": path.parents[1].name,
            "dtype": path.parent.name,
            "n_total": n_total,
            "n_golden": n_golden,
            "none_acc": n_golden / max(n_total, 1),
            "random_acc": (n_golden + random_fixed) / max(n_total, 1),
            "random_star_acc": (n_golden + random_star) / max(n_total, 1),
            "probe_acc": (n_golden + probe_fixed) / max(n_total, 1),
            "probe_star_acc": (n_golden + probe_star) / max(n_total, 1),
            "best_random_alpha": best_random,
            "best_probe_alpha": best_probe,
        })
    if not rows:
        raise SystemExit(f"No amp_results.pkl files found under {RESULTS_DIR}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGURES_DIR / "summary_table.csv"
    md_path = FIGURES_DIR / "summary_table.md"
    fieldnames = list(rows[0])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join("---" for _ in fieldnames) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[k]) for k in fieldnames) + " |\n")
    print(f"saved {csv_path}")
    print(f"saved {md_path}")


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate()
    else:
        run(args)


if __name__ == "__main__":
    main()
