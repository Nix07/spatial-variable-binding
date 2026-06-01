"""Per-model patch-grid geometry for the synthetic Squares/Shapes/Objects
settings, plus per-(experiment, model) SLURM batch sizes.

Qwen2-VL and Gemma3 match paper Table 4 (Appendix A.3, p. 14) bit-exactly.
**Pixtral deviates** from Table 4's `(16, 15, 2, 2)` and uses
`(15, 16, 2, 2)` (axes transposed: 15×15 token grid at 16×16 px each,
matching Pixtral's native 16-px vision-encoder patch stride). The canvas
size is the same 240 px either way. See `GEOMETRY.md` for the column
mapping and the pre-alignment values.

Public symbols:
  - `ModelGeom`: frozen dataclass; `.image_px` returns the canvas side.
  - `GEOM`: per-model geometry, used by every experiment.
  - `GEOM_OVERRIDES`: per-(model, task) overrides; consulted before
    `GEOM` by `geom_for`. Currently empty — kept as the extension point
    if a task ever needs to deviate from its model's default.
  - `geom_for(model_key, task)`: resolve the effective `ModelGeom` for
    a given (model, task) — overrides first, model default otherwise.
  - `STRIP_WIDTH`: criss-cross-patching only — paper's "whitespace
    width" column, in token units.
  - `BATCH_SIZES`: per-(experiment, model) batch sizes tuned to fit a
    single H100 80 GB at fp32.

`num_tokens` is the side length (in patch units) of the post-projection
visual-token grid each VLM produces for the synthetic image. `token_size`
is the pixel size of one patch unit, so `num_tokens * token_size` is the
side length of the rendered PIL canvas. `entity_size` is the rectangular
side length (in patch units) of one entity; `spacing` is the gap (patch
units) between adjacent entities along the laid-out axis.

**Changing these invalidates every committed `.npy`/`.pkl`/figure
artifact across all four experiments. They must be re-run after any
change here.** The current Pixtral entry was changed from
`(15, 16, 3, 3)` to `(15, 16, 2, 2)`; the committed artifacts under
`probing/`, `criss_cross_patching/`, `last_token_exp/`, and
`behavioral_analysis_wo_ve_oi/` for Pixtral were generated under the older value
and need re-running. `behavioral_analysis` Pixtral results are already
under `(15, 16, 2, 2)`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelGeom:
    num_tokens: int        # patches per side (post-projection grid)
    token_size: int        # pixels per patch unit
    entity_size: int       # one entity spans entity_size × entity_size patches
    spacing: int           # gap (patch units) between adjacent entities

    @property
    def image_px(self) -> int:
        return self.num_tokens * self.token_size


# Per-model default geometry. Qwen2-VL and Gemma3 follow paper Table 4
# (Appendix A.3, p. 14). Pixtral uses `(15, 16, 2, 2)` (axes transposed
# from paper Table 4's `(16, 15, 2, 2)` to match the encoder's native
# 16-px patch stride). See `GEOMETRY.md` for the column mapping.
GEOM: dict[str, ModelGeom] = {
    "qwen2vl":  ModelGeom(num_tokens=12, token_size=28, entity_size=2, spacing=2),
    "gemma3":   ModelGeom(num_tokens=16, token_size=56, entity_size=3, spacing=3),
    "pixtral":  ModelGeom(num_tokens=15, token_size=16, entity_size=2, spacing=2),
}


# Per-(model, task) geometry overrides, consulted before `GEOM` by
# `geom_for`. Currently empty — Pixtral's per-task config now matches
# its `GEOM` default, so no override is needed. Kept as the extension
# point for future per-task deviations.
GEOM_OVERRIDES: dict[tuple[str, str], ModelGeom] = {}


def geom_for(model_key: str, task: str) -> ModelGeom:
    """Return geometry for (model_key, task), falling back to GEOM[model_key]."""
    return GEOM_OVERRIDES.get((model_key, task), GEOM[model_key])


# Criss-cross-patching's strip-width parameter (paper Table 4's
# "whitespace width" column). In token units.
STRIP_WIDTH: dict[str, int] = {
    "qwen2vl":  4,
    "gemma3":   3,
    "pixtral":  2,
}


# Per-(experiment, model) SLURM batch sizes. Indexed by experiment slug.
BATCH_SIZES: dict[str, dict[str, int]] = {
    "last_token_exp": {
        "qwen2vl": 13,
        "gemma3":  5,
        "pixtral": 1,
    },
    "criss_cross_patching": {
        "qwen2vl": 13,
        "gemma3":  5,
        "pixtral": 1,
    },
}
