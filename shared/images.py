"""Compositing utilities for Squares/Shapes/Objects synthetic images.

Each image is a `num_tokens × num_tokens` grid of `token_size`-pixel cells.
Three same-sized entities (colored rectangles, drawn shapes, or PNG thumbnails)
are placed at fixed equal-distance positions along a row (horizontal:
left/right relations) or a column (vertical: above/below relations), centered
on the grid.

Cells outside the rectangles are white. The grid units make it easy to
match each VLM's native patch size (e.g. Pixtral's 16, Gemma3's 56 px).

Public helpers:
- ``three_rect_image`` — three colored rectangles (squares setting).
- ``three_shape_image`` — three drawn shapes (shapes setting).
- ``three_object_image`` — three PNG thumbnails (objects setting).
- ``SHAPE_DRAW_FNS`` — the six shape renderers keyed by name.
- ``three_entity_offsets`` — token-unit (col, row) origins for the three
  entity slots, shared by every renderer + by probe token-id helpers.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

from PIL import Image, ImageDraw


def three_entity_offsets(
    *,
    horizontal: bool,
    num_tokens: int,
    entity_height: int,
    entity_width: int,
    spacing: int,
) -> list[tuple[int, int]]:
    """Token-unit ``(col, row)`` origin for each of the three entity slots,
    centered on the canvas.

    The same formula every renderer and probe-token-id helper uses, so
    callers stay aligned.
    """
    if horizontal:
        y_tok = (num_tokens - entity_height) // 2
        x0_tok = (num_tokens - (3 * entity_width + 2 * spacing)) // 2
        return [(x0_tok + i * (entity_width + spacing), y_tok)
                for i in range(3)]
    x_tok = (num_tokens - entity_width) // 2
    y0_tok = (num_tokens - (3 * entity_height + 2 * spacing)) // 2
    return [(x_tok, y0_tok + i * (entity_height + spacing))
            for i in range(3)]


def _blank_canvas(num_tokens: int, token_size: int) -> Image.Image:
    side = num_tokens * token_size
    return Image.new("RGB", (side, side), (255, 255, 255))


def three_rect_image(
    colors: Sequence[str],
    *,
    horizontal: bool,
    num_tokens: int,
    token_size: int,
    entity_height: int,
    entity_width: int,
    spacing: int,
    color_rgb: Mapping[str, tuple[int, int, int]],
) -> Image.Image:
    """Three colored rectangles in a row (horizontal) or column (vertical).

    Args:
        colors: 3 color names that key into `color_rgb`.
        horizontal: True → arrange left-to-right; False → top-to-bottom.
        num_tokens: edge length of the square canvas in token units.
        token_size: pixels per token unit.
        entity_height / entity_width: rectangle size in token units.
        spacing: gap between adjacent rectangles in token units.
        color_rgb: mapping color-name → RGB triplet. Per-experiment so
            each caller picks the saturation/brightness it was built with.

    Returns:
        An RGB PIL image of side `num_tokens * token_size`.
    """
    if len(colors) != 3:
        raise ValueError(f"expected exactly 3 colors, got {len(colors)}")
    canvas = _blank_canvas(num_tokens, token_size)
    positions = three_entity_offsets(
        horizontal=horizontal, num_tokens=num_tokens,
        entity_height=entity_height, entity_width=entity_width, spacing=spacing,
    )
    rect_w_px = entity_width * token_size
    rect_h_px = entity_height * token_size
    for (col_tok, row_tok), name in zip(positions, colors):
        rect = Image.new("RGB", (rect_w_px, rect_h_px), color_rgb[name])
        canvas.paste(rect, (col_tok * token_size, row_tok * token_size))
    return canvas


# ---------------------------------------------------------------------------
# Shape draw functions (Shapes setting + probing's Shapes/grid extensions)
# ---------------------------------------------------------------------------

def _draw_square(draw: ImageDraw.ImageDraw, size: int, color):
    s = 0.8 * size
    off = (size - s) / 2
    draw.rectangle([off, off, off + s - 1, off + s - 1], fill=color)


def _draw_circle(draw: ImageDraw.ImageDraw, size: int, color):
    s = 0.9 * size
    off = (size - s) / 2
    draw.ellipse([off, off, off + s - 1, off + s - 1], fill=color)


def _draw_triangle(draw: ImageDraw.ImageDraw, size: int, color):
    h = size * math.sqrt(3) / 2
    draw.polygon([(size / 2, 0), (0, h), (size - 1, h)], fill=color)


def _draw_star(draw: ImageDraw.ImageDraw, size: int, color):
    c = (size - 1) / 2
    r = (size - 1) / 2
    pts = []
    for i in range(10):
        a = i * math.pi / 5 - math.pi / 2
        rad = 1.0 if i % 2 == 0 else 0.381966
        pts.append((rad * math.cos(a), rad * math.sin(a)))
    s = 1.0 / max(abs(x) for x, _ in pts)
    draw.polygon([(c + x * r * s, c + y * r) for x, y in pts], fill=color)


def _draw_cross(draw: ImageDraw.ImageDraw, size: int, color):
    w = size // 3
    off = (size - w) / 2
    draw.rectangle([off, 0, off + w - 1, size - 1], fill=color)
    draw.rectangle([0, off, size - 1, off + w - 1], fill=color)


def _draw_heart(draw: ImageDraw.ImageDraw, size: int, color, samples: int = 100):
    xs, ys = [], []
    for i in range(samples):
        t = 2 * math.pi * i / samples
        xs.append(16 * math.sin(t) ** 3)
        ys.append(13 * math.cos(t) - 5 * math.cos(2 * t)
                  - 2 * math.cos(3 * t) - math.cos(4 * t))
    mn_x, mx_x = min(xs), max(xs)
    mn_y, mx_y = min(ys), max(ys)
    pts = []
    for x, y in zip(xs, ys):
        xn = (x - mn_x) / (mx_x - mn_x)
        yn = (y - mn_y) / (mx_y - mn_y)
        pts.append((xn * (size - 1), (1 - yn) * (size - 1)))
    draw.polygon(pts, fill=color)


SHAPE_DRAW_FNS: Mapping[str, Callable] = {
    "Square":   _draw_square,
    "Circle":   _draw_circle,
    "Triangle": _draw_triangle,
    "Star":     _draw_star,
    "Cross":    _draw_cross,
    "Heart":    _draw_heart,
}


def three_shape_image(
    shape_color_tuples: Sequence[tuple[str, str] | None],
    *,
    horizontal: bool,
    num_tokens: int,
    token_size: int,
    entity_size: int,
    spacing: int,
    color_rgb: Mapping[str, tuple[int, int, int]],
) -> Image.Image:
    """Three shape sprites at the same fixed positions as `three_rect_image`.

    Each entry is ``(shape_name, color_name)``; pass ``None`` for an empty slot
    (used by some ablation single-entity baselines).
    """
    if len(shape_color_tuples) != 3:
        raise ValueError(f"expected exactly 3 (shape, color) tuples, got {len(shape_color_tuples)}")
    canvas = _blank_canvas(num_tokens, token_size)
    positions = three_entity_offsets(
        horizontal=horizontal, num_tokens=num_tokens,
        entity_height=entity_size, entity_width=entity_size, spacing=spacing,
    )
    side_px = entity_size * token_size
    for (col_t, row_t), sc in zip(positions, shape_color_tuples):
        if sc is None:
            continue
        shape_name, color_name = sc
        sprite = Image.new("RGBA", (side_px, side_px), (0, 0, 0, 0))
        SHAPE_DRAW_FNS[shape_name](
            ImageDraw.Draw(sprite), side_px, color_rgb[color_name],
        )
        canvas.paste(sprite, (col_t * token_size, row_t * token_size),
                     mask=sprite.split()[3])
    return canvas


def three_object_image(
    object_thumbs: Sequence[Image.Image | None],
    *,
    horizontal: bool,
    num_tokens: int,
    token_size: int,
    entity_size: int,
    spacing: int,
) -> Image.Image:
    """Three pre-loaded PNG thumbnails at the fixed three-slot positions.

    Pass pre-resized RGBA `PIL.Image` instances (or `None` for an empty slot);
    use `shared.objects.load_thumbnail` to obtain them.
    """
    if len(object_thumbs) != 3:
        raise ValueError(f"expected exactly 3 thumbnails, got {len(object_thumbs)}")
    canvas = _blank_canvas(num_tokens, token_size)
    positions = three_entity_offsets(
        horizontal=horizontal, num_tokens=num_tokens,
        entity_height=entity_size, entity_width=entity_size, spacing=spacing,
    )
    side_px = entity_size * token_size
    for (col_t, row_t), thumb in zip(positions, object_thumbs):
        if thumb is None:
            continue
        if thumb.size != (side_px, side_px):
            thumb = thumb.resize((side_px, side_px))
        mask = thumb if thumb.mode == "RGBA" else None
        canvas.paste(thumb, (col_t * token_size, row_t * token_size), mask)
    return canvas
