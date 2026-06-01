"""Per-setting image factories and token-id helpers, on top of `shared.images`.

This module is a thin probing-specific layer over the shared renderers in
``shared.images``:

- ``make_image`` dispatches Squares/Shapes/Objects to the right
  ``three_*_image`` helper.
- ``get_position_token_ids`` returns the flat patch indices each entity slot
  covers on the centered three-entity layout used by every renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from shared.images import (
    SHAPE_DRAW_FNS,
    three_entity_offsets,
    three_object_image,
    three_rect_image,
    three_shape_image,
)
from shared.objects import load_thumbnail

from .config import COLOR_RGB_WHITE


def _is_horizontal(orientation: str) -> bool:
    if orientation == "horizontal":
        return True
    if orientation == "vertical":
        return False
    raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")


def make_three_rectangles(colors: Sequence[str],
                          *,
                          orientation: str,
                          num_tokens: int,
                          token_size: int,
                          entity_size: int,
                          spacing: int) -> Image.Image:
    """Three colored rectangles — squares setting."""
    return three_rect_image(
        colors,
        horizontal=_is_horizontal(orientation),
        num_tokens=num_tokens, token_size=token_size,
        entity_height=entity_size, entity_width=entity_size,
        spacing=spacing,
        color_rgb=COLOR_RGB_WHITE,
    )


def make_three_shapes(shape_color_tuples: Sequence[tuple[str, str] | None],
                      *,
                      orientation: str,
                      num_tokens: int,
                      token_size: int,
                      entity_size: int,
                      spacing: int) -> Image.Image:
    """Three drawn shapes — shapes setting."""
    return three_shape_image(
        shape_color_tuples,
        horizontal=_is_horizontal(orientation),
        num_tokens=num_tokens, token_size=token_size,
        entity_size=entity_size, spacing=spacing,
        color_rgb=COLOR_RGB_WHITE,
    )


def make_three_objects(object_names: Sequence[str],
                       *,
                       orientation: str,
                       num_tokens: int,
                       token_size: int,
                       entity_size: int,
                       spacing: int,
                       object_assets_dir: Path) -> Image.Image:
    """Three PNG-thumbnail objects — objects setting."""
    side_px = entity_size * token_size
    thumbs = [load_thumbnail(name, cache_dir=object_assets_dir).resize((side_px, side_px))
              for name in object_names]
    return three_object_image(
        thumbs,
        horizontal=_is_horizontal(orientation),
        num_tokens=num_tokens, token_size=token_size,
        entity_size=entity_size, spacing=spacing,
    )


# ---------------------------------------------------------------------------
# Position → patch-token IDs (the indices the probe trains positives at)
# ---------------------------------------------------------------------------

def get_position_token_ids(*,
                           num_tokens: int,
                           entity_size: int,
                           spacing: int) -> dict[str, list[int]]:
    """Flat patch indices (`row * num_tokens + col`) covered by each slot
    in the fixed centered three-entity layout used by all renderers.

    Returns keys ``left``, ``middle``, ``right`` (horizontal layout) and
    ``top``, ``bottom`` (vertical layout). The vertical "middle" is the same
    set of patches as the horizontal "middle" since the entity is square
    (entity_size shared for H and W).
    """
    ids: dict[str, list[int]] = {
        "left": [], "middle": [], "right": [], "top": [], "bottom": [],
    }

    # Horizontal slot origins → left, middle, right.
    horiz = three_entity_offsets(
        horizontal=True, num_tokens=num_tokens,
        entity_height=entity_size, entity_width=entity_size,
        spacing=spacing,
    )
    for slot, (c0, r0) in zip(("left", "middle", "right"), horiz):
        for i in range(entity_size):
            for j in range(entity_size):
                ids[slot].append((r0 + i) * num_tokens + (c0 + j))

    # Vertical slot origins → top, (middle reuses horizontal), bottom.
    vert = three_entity_offsets(
        horizontal=False, num_tokens=num_tokens,
        entity_height=entity_size, entity_width=entity_size,
        spacing=spacing,
    )
    for slot, (c0, r0) in zip(("top", None, "bottom"), vert):
        if slot is None:
            continue
        for i in range(entity_size):
            for j in range(entity_size):
                ids[slot].append((r0 + i) * num_tokens + (c0 + j))
    return ids


def horizontal_label_indices(token_ids: dict[str, list[int]]) -> tuple[list[int], list[int]]:
    """Concatenated (token_ids, labels) for the horizontal three-class probe.

    Labels: 0=left, 1=middle, 2=right.
    """
    ids = token_ids["left"] + token_ids["middle"] + token_ids["right"]
    labels = ([0] * len(token_ids["left"])
              + [1] * len(token_ids["middle"])
              + [2] * len(token_ids["right"]))
    return ids, labels


def vertical_label_indices(token_ids: dict[str, list[int]]) -> tuple[list[int], list[int]]:
    """Concatenated (token_ids, labels) for the vertical three-class probe.

    Labels: 0=top, 1=middle, 2=bottom.
    """
    ids = token_ids["top"] + token_ids["middle"] + token_ids["bottom"]
    labels = ([0] * len(token_ids["top"])
              + [1] * len(token_ids["middle"])
              + [2] * len(token_ids["bottom"]))
    return ids, labels


# Re-export SHAPE_DRAW_FNS so grids.py (and any future ablation) can build
# arbitrary shape layouts without re-importing from `shared`.
__all__ = [
    "SHAPE_DRAW_FNS",
    "make_three_rectangles",
    "make_three_shapes",
    "make_three_objects",
    "get_position_token_ids",
    "horizontal_label_indices",
    "vertical_label_indices",
]
