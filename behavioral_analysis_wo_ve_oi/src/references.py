"""Reference-image factories for the Table 5 ordering ablation.

Paper Sec 5.3.1 prescribes two reference signals that together ablate the
vision encoder's ordering channel:

1. **Per-entity isolated-middle** — the entity (square / shape / object)
   placed alone at the canvas's middle slot, every other slot left blank
   (white). Used to *re-encode* each entity in the clean image as if it
   were viewed in isolation: color/identity preserved, position erased.
2. **Empty image** — a uniform white canvas at the same size. Used to
   re-encode every non-entity (background) image-token position.

The canvas geometry follows ``shared.experiment_config.GEOM``,
so reference images are bit-for-bit aligned with the clean images they
will substitute against — same patch grid, same slot offsets, same
``num_tokens × token_size`` canvas.

This module just produces PIL images. ``hook.py`` runs them through the
model to cache image-token residuals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image

from shared.experiment_config import GEOM
from shared.images import (
    three_object_image,
    three_rect_image,
    three_shape_image,
)
from shared.objects import load_thumbnail


# ---------------------------------------------------------------------------
# Empty (all-white) reference
# ---------------------------------------------------------------------------

def empty_image(model_key: str) -> Image.Image:
    """All-white canvas at the model's task-image size — paper's "empty image"
    reference for background tokens."""
    geom = GEOM[model_key]
    return Image.new("RGB", (geom.image_px, geom.image_px), (255, 255, 255))


# ---------------------------------------------------------------------------
# Squares: one color at middle, rest white
# ---------------------------------------------------------------------------

# The compositor needs every slot color in the palette; we add White for the
# two empty slots in the isolated-middle reference. Paint-only — no entity
# referenced by name as "White" in any task.
_WHITE = (255, 255, 255)


def isolated_middle_square(
    color_name: str,
    *,
    model_key: str,
    palette: dict[str, Tuple[int, int, int]],
) -> Image.Image:
    """Reference image for a single colored square at the middle slot.

    The horizontal-layout middle is the same set of patches as the
    vertical-layout middle (square entity, centered canvas), so one image
    serves every direction.
    """
    geom = GEOM[model_key]
    palette_with_white = {**palette, "_white": _WHITE}
    return three_rect_image(
        ("_white", color_name, "_white"),
        horizontal=True,
        num_tokens=geom.num_tokens, token_size=geom.token_size,
        entity_height=geom.entity_size, entity_width=geom.entity_size,
        spacing=geom.spacing,
        color_rgb=palette_with_white,
    )


# ---------------------------------------------------------------------------
# Shapes: one (shape, color) at middle, rest blank (white canvas)
# ---------------------------------------------------------------------------

def isolated_middle_shape(
    shape_name: str,
    color_name: str,
    *,
    model_key: str,
    palette: dict[str, Tuple[int, int, int]],
) -> Image.Image:
    """Reference image with one (shape, color) at the middle slot."""
    geom = GEOM[model_key]
    return three_shape_image(
        (None, (shape_name, color_name), None),
        horizontal=True,
        num_tokens=geom.num_tokens, token_size=geom.token_size,
        entity_size=geom.entity_size, spacing=geom.spacing,
        color_rgb=palette,
    )


# ---------------------------------------------------------------------------
# Objects: one thumbnail at middle, rest blank
# ---------------------------------------------------------------------------

def isolated_middle_object(
    object_name: str,
    *,
    model_key: str,
    asset_dir: Path,
) -> Image.Image:
    """Reference image with one object thumbnail at the middle slot.

    Uses ``shared.objects.load_thumbnail`` for the cached PNG
    (same path the behavioral_analysis Objects task uses).
    """
    geom = GEOM[model_key]
    thumb_px = geom.entity_size * geom.token_size
    thumb = load_thumbnail(object_name, cache_dir=asset_dir).resize((thumb_px, thumb_px))
    return three_object_image(
        (None, thumb, None),
        horizontal=True,
        num_tokens=geom.num_tokens, token_size=geom.token_size,
        entity_size=geom.entity_size, spacing=geom.spacing,
    )
