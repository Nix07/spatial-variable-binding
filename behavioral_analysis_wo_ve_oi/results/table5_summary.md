# Table 5 — ordering-ablation behavioral accuracy

Behavioral accuracy after the §5.3.1 vision-encoder position-information ablation (full-grid background ablation per Sec 5.3.1's prose). Precision: each model's `DEFAULT_DTYPE` — Qwen/Gemma fp32, Pixtral bf16.

Each cell shows ``ours / paper`` and a status flag.
``✓`` = within 0.05 of paper, ``⚠️`` = beyond that band.

| Model | Squares | Shapes | Objects |
|---|---|---|---|
| **Qwen2-VL-7B-Instruct** | 0.596 / 0.60 ✓ | 0.613 / 0.62 ✓ | 0.427 / 0.43 ✓ |
| **Gemma-3-4b-it** | 0.567 / 0.64 ⚠️ -0.07 | 0.635 / 0.82 ⚠️ -0.18 | 0.541 / 0.77 ⚠️ -0.23 |
| **Pixtral-12b** | 0.408 / 0.17 ⚠️ +0.24 | 0.758 / 0.01 ⚠️ +0.75 | 0.418 / 0.39 ✓ |

Chance is 1/3 (≈0.33) for Squares/Shapes/Objects.

## Methodology note

This is the full-grid ablation as described in Sec 5.3.1 — every background patch (across the whole `num_tokens × num_tokens` grid) is replaced with the matching patch of an empty-image reference. Some earlier implementations contained a `range(12 * 12) = 144` hardcode that under-ablates models with grids larger than 12×12; see `behavioral_analysis_wo_ve_oi/README.md` for context. We do **not** reproduce that bug here — the numbers in this table reflect the ablation actually described in the paper text.
