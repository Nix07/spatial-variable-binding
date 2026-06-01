# Probing Visual Embeddings (Paper Section 5.2.1)

The position-probing analysis (§5.2.1) — training linear position probes on
**post-projection visual token embeddings** to show that ordering information
is linearly decodable from the vision encoder's output, and is **diffused
across background tokens** (the "strip" pattern in paper Fig 4).

For each (model, setting, orientation), the pipeline:

1. Renders `P(|options|, 3)` synthetic images (or stitches WhatsUp pairs).
2. Extracts post-projection patch embeddings: shape `(N_patches, D)`.
3. 75/25 train/test split. Trains a 3-class linear probe
   `\hat y = Wh + b` on the object-region patches (AdamW lr=1e-3
   wd=1e-4, 200 epochs).
4. Saves `W`, `b`, test predictions on the object tokens, **and** softmax
   predictions on every patch of every test image — the latter is what the
   strip-pattern heatmap renders.

```
probing/
├── README.md
├── __init__.py
├── run_experiment.py          # CLI: train one probe per (model, setting, orient)
├── plot.py                    # render heatmaps + summary CSV/MD
├── scripts/                   # SLURM helpers
├── src/
│   ├── config.py              # per-model patch-grid geometry + palette
│   ├── images.py              # probing wrappers over shared.images + token IDs
│   ├── extract.py             # post-projection patch embeddings (per-model)
│   ├── probe.py               # linear-probe trainer
│   ├── data.py                # synthetic-setting data assembly
│   ├── whatsup.py             # WhatsUp bbox-based variant
│   └── grids.py               # 2x2 / 3x3 grid extension (App C.6)
├── results/<model>/<setting>_<orientation>.npz
└── figures/<model>/<setting>_<orientation>_heatmap.png
```

## Quickstart

```bash
# From the repo root (so the packages are importable):
cd /path/to/vlm-spatial-reasoning

# Train one probe
python -m probing.run_experiment \
    --model qwen2vl --setting squares --orientation horizontal

# Sweep all three synthetic settings × both orientations for one model
python -m probing.run_experiment \
    --model gemma3 --all-settings --all-orientations

# WhatsUp (horizontal-only by construction)
python -m probing.run_experiment \
    --model gemma3 --setting whatsup --orientation horizontal

# Grid extension (App C.6 — 2x2 or 3x3 random-shape arrangements)
python -m probing.run_experiment \
    --model qwen2vl --setting grid2x2 --orientation horizontal
```

After training, render the heatmaps + summary:

```bash
python -m probing.plot
```

## Settings

| `--setting`  | Source                                              | Classes | Notes |
|--------------|-----------------------------------------------------|---------|-------|
| `squares`    | 6 colors → P(6,3)=120 images, 90/30 split           | 3       | Main paper Fig 4 |
| `shapes`     | 6 (shape, color) pairs → 120 images, 90/30 split    | 3       | App C.5 |
| `objects`    | 8 stickpng PNGs → P(8,3)=336 images, 75/25 split    | 3       | App C.5 |
| `whatsup`    | Stitched controlled-images, ≤120 images             | 3       | App C.7, bbox-based |
| `grid2x2`    | Random 2×2 shape grids                              | 2       | App C.6, Fig 41 |
| `grid3x3`    | Random 3×3 shape grids                              | 3       | App C.6, Fig 42 |

For `whatsup`, the controlled images must already be present at
`shared/whatsup_images/controlled_images/*.jpeg` (the default path resolved
by `shared.whatsup.DEFAULT_IMAGES_DIR`).

## Models

`--model` accepts any slug supported by `shared.models`:

| Slug         | HF id                              | `num_tokens` | `image_px` |
|--------------|------------------------------------|--------------|------------|
| `qwen2vl`    | Qwen/Qwen2-VL-7B-Instruct          | 12           | 336        |
| `gemma3`     | google/gemma-3-4b-it               | 16           | 896        |
| `pixtral`    | mistral-community/pixtral-12b      | 15           | 240        |

The post-projection patch grid is a square of side `num_tokens`; the rendered
PIL canvas is `num_tokens × token_size` pixels (see `src/config.py`). Changing
these will invalidate saved probe artifacts.

Embedding extraction (`src/extract.py`):

- **Qwen2-VL** — `model.vision_model(pixel_values, grid_thw=grid_thw)` already
  returns the projected `(N_merged, D_text)` tensor.
- **Gemma3 / Pixtral** — `model.vision_model(pixel_values).last_hidden_state`
  then `model.multi_modal_projector(.)[0]`. Pixtral's `pixel_values` are
  unwrapped from the processor's list-of-tensors form and cast to the vision
  tower's weight dtype.

## Outputs

Each `.npz` contains:

| key             | shape                          | meaning                                |
|-----------------|--------------------------------|----------------------------------------|
| `W`             | `(D, n_classes)`               | probe weight matrix                    |
| `b`             | `(n_classes,)`                 | probe bias                             |
| `test_pred`     | `(n_obj_tokens, n_classes)`    | softmax on object-region test tokens   |
| `test_pred_all` | `(n_test, N_patches, n_classes)` | softmax on every patch of every test image — used for the strip heatmap |
| `test_acc`      | scalar                         | argmax accuracy on object-region tokens |
| `train_loss`    | scalar                         | final training cross-entropy           |

`plot.py` writes `figures/summary_table.{csv,md}` summarizing test accuracy
across every saved `.npz`.

## Hyperparameters

All hard-coded to match section 5.2.1 / App A.4:

- Optimizer: AdamW
- Learning rate: 1e-3
- Weight decay: 1e-4
- Epochs: 200 (override with `--epochs`)
- Random seed: 0 (override with `--seed`)
