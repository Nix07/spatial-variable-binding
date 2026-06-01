# Ordering Ablation — Table 5 reproduction

Reproduces **Table 5** (Appendix C.2) of *The Dual Mechanisms of Spatial
Reasoning in Vision-Language Models* (arXiv [2603.22278](https://arxiv.org/abs/2603.22278)):

> Behavioral performance after removing spatial information from vision
> embeddings.

The intervention (paper Sec 5.3.1) ablates the vision encoder's ordering
channel at the LM input residual stream:

1. **Entity slots** → replaced with the *middle-slot* patches of an
   isolated, equally-identified reference image (same color / shape /
   object placed alone at the canvas centre, rest white). Identity
   preserved; position erased.
2. **Background** → replaced with the matching patches of an all-white
   reference image, eliminating any ordering signal encoded in
   background regions.

Both substitutions happen at the same hook point as
`criss_cross_patching`: ``blocks.0.hook_resid_pre`` — the deepest layer
at which the substitution is still "purely vision-derived". Anything the
LM does downstream operates on the ablated representation.

The eval metric (top-1 next-token decode) is unchanged from
`behavioral_analysis`, so this directory's results sit directly next to
the Table 1 numbers for comparison.

WhatsUp is **excluded** by paper text (Sec 5.3.1, last sentence): the
ablation is precisely defined only for the synthetic three-entity
layouts where the entity slots are known a priori.

## Layout

```
behavioral_analysis_wo_ve_oi/
├── README.md                — this file
├── run_experiment.py        — `python -m ... --model X --all-tasks` CLI
├── src/
│   ├── references.py        — build empty + isolated-middle reference images
│   ├── hook.py              — ResidualCache + substitution hook at layer-0 resid_pre
│   └── tasks/
│       ├── squares.py       — same 480 items as behavioral_analysis squares
│       ├── shapes.py        — same 480 items
│       └── objects.py       — same 1344 items
└── results/
    ├── <model>_<task>_ablated.pkl/.txt   — per-cell artifacts
    ├── table5_summary.csv                — produced by `run_experiment.py --aggregate`
    └── table5_summary.md
```

The reference-image builders and the substitution hook are this
directory's only new code. Everything else (image compositors, palettes,
prompt templating, model loader, grid-to-LM position lifting, eval
scoring) is reused from `shared/`, `probing/`,
`criss_cross_patching/`, and `behavioral_analysis/`.

The ablation hook (`src/hook.py`) and reference factories
(`src/references.py`) are also imported by
`criss_cross_wo_ve_oi/` for the §5.3.2 patching sweep.

## Reproduction

One model, all 3 tasks in a single process:
```bash
python -m behavioral_analysis_wo_ve_oi.run_experiment \
    --model gemma3 --all-tasks --torch-dtype float32
```

Refresh the summary from cached `.pkl` artifacts on disk:
```bash
python -m behavioral_analysis_wo_ve_oi.run_experiment --aggregate
```

## Implementation notes

**Reference caching.** Each task pre-builds the reference PIL images
(one empty + one per entity identity), but the per-prompt residual cache
is populated lazily. The squares task has 24 unique prompts × 7 unique
references = 168 reference forwards maximum; objects has 32 × 9 = 288.
Reference residuals are cached on CPU (a few MB each), looked up on every
clean forward.

**Model-uniform substitution.** The hook reads the cached image-token
slice using `grid_to_lm_positions` from `criss_cross_patching/src/tokens`,
so Pixtral's interleaved `[IMG_BREAK]` separators are handled exactly the
same way as in the criss-cross patching experiments — no Pixtral-specific
branch in this directory. Qwen2-VL and Gemma3 take the contiguous-block
path automatically.

**Position embeddings.** All three models use RoPE inside the LM, so
``blocks.0.hook_resid_pre`` carries no absolute positional contribution
at image-token positions — only the projected vision-encoder output.
That's what makes the substitution well-defined: writing residuals from
a reference forward into the clean forward doesn't bring along a
mismatched positional signal.

**Why the hook generalizes across models.** A naive implementation calls
``model.vision_model(pixel_values)`` directly and hardcodes
``IMG_START_TOK_ID = 21``. That binding is fragile across models and
across changes to the chat template. This implementation runs a full
forward through ``model.run_with_cache`` with ``names_filter=
"blocks.0.hook_resid_pre"`` (so only one tensor is kept) and looks up
``image_start_offset`` per prompt — both already in
``criss_cross_patching/src/tokens.py``.

## Paper Table 5 (target numbers)

|                          | Squares | Shapes | Objects |
|--------------------------|---------|--------|---------|
| Qwen2-VL-7B-Instruct     | 0.60    | 0.62   | 0.43    |
| Gemma-3-4b-it            | 0.64    | 0.82   | 0.77    |
| Pixtral-12b              | 0.17    | 0.01   | 0.39    |

Chance is 1/3 (≈ 0.33) for Squares/Shapes/Objects. Pixtral Shapes at
0.01 indicates a near-complete collapse: the model not only loses
ordering, it falls **below chance**, suggesting the ablation
distribution drives it into a degenerate output mode.

## Our reproduction

`results/table5_summary.md` is regenerated by
`run_experiment.py --aggregate` after each batch of runs (all rows
below in fp32):

| Model | Squares | Shapes | Objects |
|---|---|---|---|
| **Qwen2-VL-7B-Instruct** | 0.596 / 0.60 ✓ | 0.613 / 0.62 ✓ | 0.427 / 0.43 ✓ |
| **Gemma-3-4b-it** | 0.567 / 0.64 ⚠ −0.07 | 0.635 / 0.82 ⚠ −0.18 | 0.541 / 0.77 ⚠ −0.23 |

Qwen reproduces every cell within ±0.01 of the paper. Pixtral is not
included in this session's run.

## Why our Gemma-3 numbers are lower than the paper

The paper's Gemma-3 Table 5 row was produced with a **model-grid hardcode
bug** that under-ablates Gemma's canvas — a `square_patching_hook` of the
form:

```python
whitespace_token_ids = [
    tok for tok in range(12 * 12) if tok not in sum(square_token_ids, [])
]
```

The constant `12 * 12 = 144` is **Qwen2-VL's patch grid** (12×12). For
Gemma the grid is **16×16 = 256**, so only patches 0..143 are ablated.
For horizontal squares, that means rows 0..8 of the 16-row patch grid get
the ordering signal removed, while **rows 9..15 retain their original
3-square-image (contextualized) vision-encoder embeddings**. The LM can
still read substantial residual ordering signal off those un-ablated
patches — exactly what the paper's stated procedure (Sec 5.3.1) says it
removes.

We confirmed empirically: applying the notebook's `range(144)` bg-cap to
our pipeline (in a one-off verification script, now deleted) reproduces
paper Table 5 for Gemma to within ±0.02 (Squares: 0.6229 vs paper 0.64).
Our canonical pipeline keeps the full-grid ablation
(`background_grid_ids(model_key, direction)` covers all
`num_tokens × num_tokens` patches) because that's what Sec 5.3.1's prose
describes — and the resulting accuracy drop is the *actual* effect of
the intervention as written in the paper.

**Why Qwen still matches**: 12×12 = 144 happens to equal Qwen's grid
size, so the notebook's hardcode is accidentally correct for Qwen.

**Downstream effect** — this same bug propagates into the §5.3.2
patching pool size (`criss_cross_wo_ve_oi/`): the
notebook's pool of 14/18/50/50 (left/right/above/below) for Gemma comes
from filtering tuples through this partial ablation; our full ablation
yields a smaller pool (12/4/50/34) because fewer triples remain top-1
correct once the bottom half of the canvas is also ablated. We keep our
smaller-but-correct pool for §5.3.2 patching.

**Pixtral implication**: the same `range(12*12)` hardcode would also
under-ablate Pixtral's 15×15 grid; if Pixtral results from Table 5 /
Figure 22 are reproduced here, expect a similar gap vs paper.
