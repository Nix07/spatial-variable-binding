"""Probing-specific constants. Patch geometry is hoisted to
`shared.experiment_config` so it stays in sync across
the four experiments that share it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from shared.experiment_config import GEOM as GEOM, ModelGeom  # noqa: F401


# Probing-only geometry overrides, consulted before the shared `GEOM` by
# `probing_geom`. Currently empty — kept as the extension point.
_PROBING_GEOM_OVERRIDES: Mapping[str, ModelGeom] = {}


def probing_geom(model_slug: str) -> ModelGeom:
    """Geometry for probing: a per-model override if present, else shared GEOM."""
    return _PROBING_GEOM_OVERRIDES.get(model_slug, GEOM[model_slug])


# Canonical color palette for Squares / Shapes / Objects backgrounds.
# 6 colors: matches `len = 6 ⇒ P(6,3) = 120` permutations ⇒ 75/25 split
# yields the paper's 90/30 train/test image counts.
COLOR_RGB: Mapping[str, tuple[int, int, int]] = {
    "Red":   (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue":  (0, 0, 255),
    "Black": (0, 0, 0),
    "Brown": (150, 75, 0),
    "Gray":  (128, 128, 128),
}

# White is added only for the "missing-object" slot used by the Shapes / Objects
# blanking helpers — never used as a label.
COLOR_RGB_WHITE = {**COLOR_RGB, "White": (255, 255, 255)}


# Per shape, its drawing function name in `images.SHAPE_DRAW_FNS`.
SHAPE_NAMES = ("Square", "Circle", "Triangle", "Star", "Cross", "Heart")

# Fixed (shape, color) pairs used by the Shapes setting — same as the original
# notebook, 6 pairs ⇒ P(6,3) = 120 tuples.
SHAPE_COLOR_PAIRS = (
    ("Square",  "Red"),
    ("Circle",  "Green"),
    ("Triangle", "Blue"),
    ("Star",    "Black"),
    ("Cross",   "Brown"),
    ("Heart",   "Gray"),
)


# Default on-disk cache for the stickpng object PNGs (see
# ``shared.objects``). Lives under `shared/assets/` so the
# behavioral_analysis Objects task can share the same cache instead of
# re-downloading.
DEFAULT_OBJECT_ASSETS_DIR: Path = (
    Path(__file__).resolve().parents[2] / "shared" / "assets" / "objects"
)
