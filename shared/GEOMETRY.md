# Patch-grid geometry across experiments

This file documents the **active geometry** (in `experiment_config.GEOM`)
and the **paper-stated geometry** (Table 4, Appendix A.3) that the active
values are aligned against.

## Active geometry (`experiment_config.GEOM`)

| Model | # Image tokens | Pixels / Token | Tokens / Object | Object spacing | Source |
|---|---|---|---|---|---|
| Qwen2-VL-7B-Instruct | 12×12 | 28×28 | 2×2 | 2 | paper Table 4 |
| Gemma-3-4b-it       | 16×16 | 56×56 | 3×3 | 3 | paper Table 4 |
| Pixtral-12b         | **15×15** | **16×16** | **2×2** | **2** | matches Pixtral's 16-px patch stride (axes transposed vs paper Table 4 — see below) |

### Per-(model, task) overrides (`experiment_config.GEOM_OVERRIDES`)

Empty. `GEOM["pixtral"]` already provides the uniform `(15, 16, 2, 2)`
that behavioral_analysis / probing / criss_cross / last_token /
behavioral_analysis_wo_ve_oi all read. The override mechanism (`geom_for`) is kept
as an extension point in case a future task needs to deviate from its
model's default.

### Behavioral_analysis Pixtral results (under the uniform geometry)

| Task | Geometry (N, P, S, sp) | n items | Result |
|---|---|---:|---|
| Squares | (15, 16, 2, 2) | 480 | 264/480 = 0.5500 (paper 0.60, Δ −0.05) |
| Shapes  | (15, 16, 2, 2) | 480 | 262/480 = 0.5458 (paper 0.55, Δ −0.004) ✓ |
| Objects | (15, 16, 2, 2) | **840** (7 objects, bomb excluded) | 529/840 = 0.6298 (paper 0.63, Δ 0.000) ✓ |

Pixtral Objects also drops "bomb" from the object set (in
`tasks/objects.py::object_names_for`), so it filters `bomb` out when
listing `objects/`. Qwen and Gemma keep the full 8-object set.

**System text**: all three Pixtral tasks use the short
`DEFAULT_SYSTEM_TEXT` ("You are a helpful assistant. You respond in one
token."), uniform with Qwen and Gemma. Appending " Do not add any
additional text." to the Squares system message would land Squares
at 286/480 = 0.5958 ≈ paper 0.60. We drop that clause for consistency
with Shapes / Objects and accept Squares at 0.55 (Δ −0.05 vs paper).

Ablation note: adding the "Do not add any additional text." clause to
Shapes lifts Pixtral from 0.55 to **0.93** — strong evidence that the
paper's Pixtral Shapes baseline is an underestimate of the model's
capability, driven by a system-prompt asymmetry.

Callers should use `geom_for(model_key, task)`; overrides fall back to
`GEOM[model_key]` when absent.

### Per-model precision (`models.DEFAULT_DTYPE`)

| Model | dtype | Reason |
|---|---|---|
| Qwen2-VL-7B-Instruct | **fp32** | `analysis_utils.load_model` defaults to fp32. Table 5 (ordering ablation) is precision-sensitive at borderline logits, so the canonical results are fp32. |
| Gemma-3-4b-it | **fp32** | Same. |
| Pixtral-12b | bf16 | bf16-vs-fp32 ablation on all four behavioral tasks (`pixtral_<task>_baseline_bf16.{pkl,txt}` at the time of the switch) showed Δ≈0 on Squares/Objects/WhatsUp and Δ −0.023 on Shapes (0.546 → 0.523), still within ±0.04 of paper 0.55. The Shapes drift comes from ~11/480 borderline-logit flips — per-direction tallies (30, 68, 81, 83) → (26, 64, 80, 81). Pixtral keeps bf16 for the memory/speed savings on the 12B model. |

Override programmatically via `models.load(..., torch_dtype=...)` when
needed for precision ablations.

## Paper Table 4 (Appendix A.3, p. 14) — for reference

| Model | # Image tokens | Pixels / Token | Tokens / Object | Object spacing | Strip width¹ |
|---|---|---|---|---|---|
| Qwen2-VL-7B-Instruct | 12×12 | 28×28 | 2×2 | 2 | 4 |
| Gemma-3-4b-it       | 16×16 | 56×56 | 3×3 | 3 | 3 |
| Pixtral-12b         | 16×16 | 15×15 | 2×2 | 2 | 2 |

## Why Pixtral deviates from Table 4

The paper's Table 4 entry for Pixtral (`(16, 15, 2, 2)`) is inconsistent
with the paper's own Table 1 behavioral numbers for Pixtral
(Squares 0.60, Shapes 0.55):

- Pixtral's vision encoder uses a fixed 16-px patch stride. Setting
  `num_tokens=16, token_size=15` produces a 240-px canvas, which the
  encoder retokenizes into a 15×15 patch grid regardless of how we
  label the rendering grid. So Table 4's "16×16" claim is a
  rendering-grid convention, not the actual post-encoder grid.
- A configuration of `num_tokens=15, token_size=16, h=w=3, spacing=2`
  — matching Pixtral's native 16-px patch size exactly — gives a
  baseline of 268/480 = 0.558, close to paper Table 1's 0.60.
- An empirical sweep with our runner (with the palette using Orange
  `(255, 165, 0)` instead of Brown `(150, 75, 0)` for Pixtral, to break
  the Brown/Red Cross confusion) found the following bf16 sweep results:

  | Geometry (N, P, S, sp) | Palette | Squares | Δsq | Shapes | Δsh |
  |---|---|---:|---:|---:|---:|
  | `(15, 16, 3, 3)` | Orange | 0.585 | −0.015 ✓ | 0.519 | −0.031 ✓ |
  | `(15, 16, 2, 2)` | Orange | 0.550 | −0.050 | 0.523 | −0.027 ✓ |
  | `(15, 16, 3, 2)` | Orange | 0.583 | −0.017 ✓ | 0.481 | −0.069 |
  | `(16, 15, 3, 3)` | Orange | 0.704 | +0.104 | 0.488 | −0.063 |
  | `(16, 15, 3, 2)` | Orange | 0.581 | −0.019 ✓ | 0.442 | −0.108 |
  | `(16, 15, 2, 3)` | Orange | 0.569 | −0.031 ✓ | 0.427 | −0.123 |
  | `(16, 15, 2, 2)` (Table 4) | Orange | 0.508 | −0.092 | 0.417 | −0.133 |
  | `(16, 15, 2, 2)` (Table 4) | Brown | 0.458 | −0.142 | 0.358 | −0.192 |
  | `(24, 14, 2, 2)` | Brown | 0.344 | −0.256 | 0.288 | −0.262 |

  Paper Table 1 target: Squares 0.60, Shapes 0.55. ✓ = within ±0.04.

  The previous choice was `(15, 16, 3, 3)` (top row; highest Δ-product).
  It was retired in favor of uniform `(15, 16, 2, 2)` (entity_size=2,
  giving Shapes 0.55, Objects 0.63). Squares drops from 0.585 → 0.550
  (Δ −0.05 vs paper 0.60) as the accepted trade-off.

The Pixtral palette also diverges from Qwen/Gemma: Brown `(150, 75, 0)`
is replaced by Orange `(255, 165, 0)` because Pixtral classifies the
thin Cross sprite at small entity sizes as red when Brown is the
ground-truth color. Lives in `shared/palette.py` as
`COLOR_RGB_PIXTRAL`, selected by `palette_for(model_key)`.

¹ paper's Table 4 calls this column "whitespace width"; the prose
(§A.2, p. 13) defines it as the strip width used by criss-cross-patching
interventions in Fig. 7.

Mapping the column names to our `ModelGeom` schema:

| paper column | `ModelGeom` field |
|---|---|
| `# Image tokens` (NxN) | `num_tokens` (N) |
| `Pixels / Token` (PxP) | `token_size` (P) |
| `Tokens / Object` (SxS) | `entity_size` (S) |
| `Object spacing` | `spacing` |

Resulting canvas size = `num_tokens × token_size`:
- Qwen 336 px, Gemma 896 px, Pixtral 240 px.
