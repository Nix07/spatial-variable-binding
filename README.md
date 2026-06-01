# Dual Mechanisms of Spatial Reasoning in Vision-Language Models

Research code for ["The Dual Mechanisms of Spatial Reasoning in Vision-Language
Models"](https://arxiv.org/abs/2603.22278). The paper shows that VLMs associate
objects with spatial relations through **two concurrent mechanisms**:

1. **Language-model backbone** — intermediate layers encode *content-independent*
   spatial relations on top of visual tokens at object positions (a **secondary**
   contribution to predictions).
2. **Vision encoder** — encodes object layout in representations spread *globally*
   across visual tokens, including background tokens beyond the object regions
   (the **dominant** source of spatial signal, directly exploited by the LM).

The central practical claim: amplifying the vision-encoder-derived spatial
representation **globally across all image tokens** improves spatial reasoning on
naturalistic images. This repo contains the mechanistic-interpretability tooling
(activation patching, linear probing on visual-token embeddings) and the
amplification testbed used to substantiate both findings.

**Models:** Qwen2-VL-7B-Instruct, Gemma-3-4b-it, Pixtral-12B (the COCO
amplification testbed also uses Qwen2-VL-2B-Instruct).

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the repo root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
./setup.sh                                         # creates .venv and installs deps
source .venv/bin/activate
```

For gated models (Qwen2-VL, Pixtral, …) authenticate with Hugging Face, either
interactively (`uv run huggingface-cli login`) or by exporting a token
(`export HF_TOKEN=hf_...`). On the 12B model, set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid CUDA fragmentation.

> The experiments use the **vendored** TransformerLens in `third_party/`
> (it adds `HookedVLTransformer`, which the PyPI build lacks). `shared.paths.setup()`
> puts it ahead of any pip-installed copy on `sys.path`, so no extra steps are needed.

## Quickstart

Run any experiment as a module **from the repo root**, one model at a time
(`--model` accepts `qwen2vl`, `gemma3`, `pixtral`):

```bash
# Table 1 — top-1 next-token accuracy on a synthetic task
python -m behavioral_analysis.src.run --model qwen2vl --task squares

# §5.2.1 — linear position probe on visual-token embeddings
python -m probing.run_experiment --model gemma3 --setting squares --orientation horizontal
```

Each experiment writes its `results/` and `figures/` under its own package
directory; an `analysis/` module or `plot.py` aggregates them. The per-experiment
`scripts/*.sh` submit the same runs as SLURM jobs.

## The experiments

| Package | What it does |
|---|---|
| `behavioral_analysis/` | Table 1 — next-token accuracy on Squares / Shapes / Objects / What'sUp |
| `probing/` | §5.2.1 — linear position probes on post-projection visual tokens (the "strip" pattern) |
| `last_token_exp/` | Fig 3 — last-token residual-stream patching, clean ↔ reversed counterfactual |
| `criss_cross_patching/` | Figs 6/7 — interchange (swap) patching at left/right image-token regions |
| `coco_amplification/` | Table 2 — amplify the ordering probe at LM layer 0 on COCO-spatial |
| `behavioral_analysis_wo_ve_oi/` | Table 5 — full-grid vision-encoder ablation |
| `criss_cross_wo_ve_oi/` | §5.3.2 — swap patching under the ablation |
| `absolute_vs_relative/` | shifted-squares absolute-vs-relative control |

Some experiments score **golden (clean, counterfactual) pairs** the model already
answers correctly. Those provide a `build_cf_pairs.py` that writes
`cf_pairs/<setting>.json` (keyed by model); build it once before running, e.g.:

```bash
python -m last_token_exp.build_cf_pairs   --model qwen2vl --setting squares
python -m last_token_exp.run_experiment   --model qwen2vl --setting squares --all-directions

python -m criss_cross_patching.build_cf_pairs --model qwen2vl --setting squares
python -m criss_cross_patching.run_experiment --model qwen2vl --kind strip --all-directions

# COCO-spatial amplification (Table 2); --aggregate builds the cross-model summary
python -m coco_amplification.run_experiment --model qwen2vl
python -m coco_amplification.run_experiment --aggregate
```

COCO images are read from `$COCO_DIR` (default `~/coco_2obj`); the What'sUp images
ship in `shared/whatsup_images/controlled_images/`.

## Layout

```
shared/                  cross-experiment code + data
  models.py prompts.py experiment_config.py palette.py images.py
  whatsup.py objects.py paths.py                     (helpers)
  analysis_utils.py general_utils.py component.py
  metrics.py vision_language_prompts.py              (mechinterp core)
  assets/  whatsup_images/                           (object thumbnails, What'sUp JPEGs)
<experiment>/            one package per experiment (see table above)
  src/  run_experiment.py  cf_pairs/  results/  figures/  scripts/
third_party/TransformerLens/   vendored TransformerLens with HookedVLTransformer
setup.sh
```

Patch-grid geometry per model lives in `shared/experiment_config.py::GEOM`;
see `shared/GEOMETRY.md` for the values and the paper Table-4 alignment.
