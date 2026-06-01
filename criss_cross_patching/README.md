# Criss-Cross Vision-Token Patching (Paper Figures 6 & 7)

Reproduces section 5.2.2 of the spatial-reasoning analysis — the
**interchange-intervention** experiments on visual token embeddings that
test the causal role of ordering information.

For every (clean, counterfactual) image pair we **symmetrically swap** the
left/right (or top/bottom) image-token regions between the runs:

```
clean's left  ← cf's right
clean's right ← cf's left
```

The cf image is the clean image with its colors *reversed along the queried
axis* (so the color values at each swapped position match — only the
ordering signal changes). At each LM hook point (resid_pre of layer 0 +
resid_post of every layer) we run the swap once and record the final-token
probability over the three candidate colors.

Two swap variants:

| `--kind`   | Region swapped                                | Reproduces |
|------------|-----------------------------------------------|-----------|
| `object`   | Only the patches inside each object square    | Fig 6     |
| `strip`    | The full column (horizontal) / row (vertical) strip of patches containing the object — width = `entity_size + spacing` | Fig 7 |

**Paper finding**: object patching alone does NOT flip the model's output
at any layer (Fig 6: the clean-correct color stays high), but strip patching
DOES flip the output up to layer ~24 (Fig 7: the counterfactual-correct
color rises). This is the causal complement to section 5.2.1: ordering
information is encoded across the strip-aligned background tokens, not
inside the object patches alone.

```
criss_cross_patching/
├── README.md
├── __init__.py
├── color_tuples.json                 # 50 three-color tuples per model per direction
├── run_experiment.py                 # CLI runner: one (model, kind, direction) per call
├── plot.py                           # Render Fig 6 / Fig 7 grids
├── scripts/
│   ├── run_cell.sh                   # one (model, kind, direction) SLURM job
│   ├── run_all_directions.sh         # one (model, kind) job, all 4 directions inline
│   └── submit_all.sh                 # full 6-job sweep
├── results/<model>/
│   ├── squares_object_patching.npy   # Fig 6 data
│   └── squares_strip_patching.npy    # Fig 7 data
└── src/
    ├── tokens.py                     # grid index → LM-input position per model
    ├── regions.py                    # object slots + full-strip slots per model
    └── patch.py                      # the per-layer swap-patching loop
```

## Quickstart

```bash
# From the repo root
cd /path/to/vlm-spatial-reasoning

# One direction
python -m criss_cross_patching.run_experiment \
    --model qwen2vl --kind strip --direction left

# All four directions in one process (model loads once)
python -m criss_cross_patching.run_experiment \
    --model qwen2vl --kind strip --all-directions
python -m criss_cross_patching.run_experiment \
    --model qwen2vl --kind object --all-directions

# Then render Fig 6 + Fig 7
python -m criss_cross_patching.plot
```

SLURM helpers under `scripts/` mirror the layout used by `last_token_exp`
and `behavioral_analysis`. `scripts/submit_all.sh` queues all 6 jobs at once
(3 models × 2 kinds, each job sweeps all 4 directions).

## How `--kind` translates to the swap

For each (model, kind, direction) the runner computes:

1. **Region grid indices** — `regions.regions_for(slug, kind)` returns flat
   patch indices on the model's post-projection grid. For `object` this is
   the `entity_size × entity_size` square block centered at each slot. For
   `strip` it's the full column (horizontal) or row (vertical) of width
   `entity_size + spacing`, centered on the slot.
2. **LM-input positions** — `tokens.grid_to_lm_positions` lifts each grid
   index to its absolute LM-input position. For Qwen / Gemma this is just
   `image_start + grid_idx`. For Pixtral the patches are interleaved with
   `[IMG_BREAK]` separators, so the offset is
   `image_start + r * (N + 1) + c`.
3. **Swap arrays** — `clean_positions = A_lm + B_lm`,
   `cf_positions = B_lm + A_lm` (where A/B are left/right for horizontal
   queries or top/bottom for vertical). The patch hook then writes
   `value[:, clean_positions, :] = cf_cache[:, cf_positions, :]`.

`image_start` is computed at runtime by tokenizing the (prompt, image) pair
and finding the first model-specific image-token id in the tokenized input,
so changes to the chat template don't silently move the offset.

## Hyperparameters

All hard-coded to match section 5.2.2:

- 50 (clean, cf) pairs per direction, drawn from a 6-color palette.
- Clean image = `(c0, c1, c2)`; cf image = `(c2, c1, c0)`.
- Query direction is the **same** in clean and cf (`prep == prep_cf`); the
  intervention is what changes the predicted color, not the prompt.
- `prob_strs = [c0, c1, c2]`, summed token-prob over
  `{lower, Capitalize, ' '+lower, ' '+Capitalize}`.
- Per-model batch size: qwen2vl 13, gemma3 5, pixtral 1.

## Saved-data format

Each `.npy` is a pickled `dict[direction → ndarray]` with keys
`left`, `right`, `above`, `below`. Each array has shape
`(n_samples, n_layers + 1, 1, 3)` of `float32` token-summed probabilities,
where the layer axis has `n_lm_layers + 1` slots:

| slot | hook                       | semantics                  |
|------|----------------------------|----------------------------|
| 0    | `blocks.0.hook_resid_pre`  | post-embed (vision tokens) |
| 1..L | `blocks.{l-1}.hook_resid_post` | `resid_post` of LM layer `l-1` |

And the last axis is the three candidate colors `[c0, c1, c2]`. For a
left/above query, the clean-correct answer is c0 and the cf-correct
("swapped") answer is c2; for right/below, the roles flip.

## Reading the figure

- **Object patching (Fig 6)**: c0 (clean-correct) stays high across all
  layers — patching just the square tokens doesn't transfer ordering info.
- **Strip patching (Fig 7)**: c2 (the swapped, cf-correct color) rises
  sharply at the embedding slot and stays high through the early/mid
  layers; c0 overtakes again at layer ~23-24, which is where the LM
  finalises the answer at the last token (matches Fig 3's transition).
