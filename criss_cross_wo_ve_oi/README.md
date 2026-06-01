# Criss-cross under ablation — Section 5.3.2 (Figure 9)

Reproduces **Figure 9** of *The Dual Mechanisms of Spatial Reasoning in
Vision-Language Models* (arXiv [2603.22278](https://arxiv.org/abs/2603.22278)).

The paper's Section 5.3.2 asks whether the **LM backbone reconstructs
ordering information** after the vision encoder's ordering channel has
been ablated. The experiment composes two interventions:

1. **Layer-0 ordering ablation** (Sec 5.3.1 / `behavioral_analysis_wo_ve_oi/`):
   replace each entity-token slot with the middle-slot patches of an
   isolated-middle reference image (preserves color, erases position),
   and replace every background patch with patches of an empty image.
   Applied to **both** the clean and counterfactual forwards at
   `blocks.0.hook_resid_pre`.
2. **Criss-cross object patching** (Sec 5.2.2 /
   `criss_cross_patching/`): at each LM layer slot
   (`resid_pre` layer 0 + `resid_post` layers 0..L-1), copy cf's
   *ablated* residuals at the **swapped** square positions into clean's
   *ablated* residuals at the (unswapped) square positions.

If the LM backbone independently reconstructs the order signal in some
intermediate layer, that swap should flip the model's final prediction.
The paper observes the flip at layers 11–20.

Only the **object** patching variant is implemented here — the paper
text explicitly narrows the §5.3.2 sweep to entity tokens, since the
ablation has already overwritten every background patch with the same
empty-image residual, so a strip swap would shuffle identical values.

## Layout

```
criss_cross_wo_ve_oi/
├── README.md                — this file
├── __init__.py
├── build_cf_pairs.py        — filter random triples under the ablation
├── cf_pairs.json            — surviving triples per (model, direction)
├── run_experiment.py        — `python -m ... --model X --direction Y` CLI
├── plot.py                  — 1×4 per-direction probability traces
├── src/
│   └── patch.py             — `criss_cross_sweep_under_ablation`
├── scripts/
│   ├── run_cell.sh          — submit one (model, direction) cell
│   └── submit_all.sh        — submit four directions in parallel + merge
└── results/<model>/squares_object_patching_under_ablation.npy
```

## Reuse — no copy

- `behavioral_analysis_wo_ve_oi.src.hook`: `ResidualCache`, `make_ablation_hook`,
  grid-id helpers, `HOOK_NAME`. The ablation hook is built per batch
  from the cached reference residuals.
- `behavioral_analysis_wo_ve_oi.src.references`: `empty_image`, `isolated_middle_square`.
- `criss_cross_patching.src.regions.object_regions`: square grid IDs.
- `criss_cross_patching.src.tokens`: `image_start_offset`,
  `grid_to_lm_positions` (handles Pixtral's `[IMG_BREAK]` separators).
- `criss_cross_patching.src.patch._prob_label_tensor`: 4-spelling-variant
  per-color token lookup.
- `criss_cross_patching.run_experiment.build_vl_prompts`: direction-
  symmetric `(c0, c1, c2)` VLPrompt construction. The same
  `color_tuples.json` pool is reused as the §5.2.2 experiment.

## Reproduction

Single (model, direction):
```bash
python -m criss_cross_wo_ve_oi.run_experiment \
    --model qwen2vl --direction left
```

All four directions for one model, in parallel via SLURM, with a
dependent merge job:
```bash
criss_cross_wo_ve_oi/scripts/submit_all.sh qwen2vl
```

Render the figure from cached results:
```bash
python -m criss_cross_wo_ve_oi.plot --model qwen2vl
```

## Notes on `dtype`, palette, geometry

- Precision defaults to each model's `models.DEFAULT_DTYPE` (currently
  bf16 for all three). Pass `--torch-dtype float32` to override.
- Palette is `palette_for(model)` — Pixtral uses Orange (no Brown).
- Geometry is `GEOM[model]`. The grid → LM position mapping in
  `criss_cross_patching.src.tokens` handles Pixtral's
  `[IMG_BREAK]`/`[IMG_END]` row separators automatically.

## Why our Gemma cf_pairs pool is smaller than paper Fig 22's

For Gemma the paper's §5.3.2 pool is `{left: 14, right: 18, above: 62→50,
below: 64→50}`; our `build_cf_pairs.py` produces `{12, 4, 50, 34}`. The
difference is **not** a bug in this module — it's downstream of a paper-side
bug in the underlying §5.3.1 ablation. See
`behavioral_analysis_wo_ve_oi/README.md` ("Why our Gemma-3 numbers are
lower than the paper") for details. Qwen pool sizes are unaffected
because Qwen's 12×12 grid matches the notebook's hardcoded constant.
