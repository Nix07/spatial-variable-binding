"""Patch-region helpers: object slots (Fig 6) and full strip slots (Fig 7).

Both setups expose the same five named regions over the
``num_tokens × num_tokens`` post-projection grid:

- ``left`` / ``middle`` / ``right`` (horizontal layout)
- ``top``  / ``bottom``           (vertical layout; middle reuses horizontal)

For **object** patching, each region is the entity_size × entity_size patch
block where the object sits — i.e. exactly the indices the probing module's
``get_position_token_ids`` returns.

For **strip** patching, each region is the full column (horizontal) or row
(vertical) of patches that contains the object. Width is
``entity_size + spacing`` patches centered on the object, clipped to the
canvas — this matches the reference layout
``strip_ids[..] = [N*r + c for r in range(N) for c in <strip_cols>]`` math.
"""

from __future__ import annotations

from typing import Mapping

from probing.src.config import GEOM
from probing.src.images import get_position_token_ids


def object_regions(model_slug: str) -> Mapping[str, list[int]]:
    """Per-slot grid indices for the entity_size × entity_size square patches."""
    geom = GEOM[model_slug]
    return get_position_token_ids(
        num_tokens=geom.num_tokens,
        entity_size=geom.entity_size,
        spacing=geom.spacing,
    )


def _column_strip(num_tokens: int, c_start: int, c_end: int) -> list[int]:
    return [r * num_tokens + c
            for r in range(num_tokens)
            for c in range(c_start, c_end)]


def _row_strip(num_tokens: int, r_start: int, r_end: int) -> list[int]:
    return [r * num_tokens + c
            for r in range(r_start, r_end)
            for c in range(num_tokens)]


# Per-model strip layout (reference values).
# Tuple is (left_width, middle_width, right_width); left=[0,L), middle=[L,L+M),
# right=[L+M, L+M+R). The sum equals num_tokens so strips tile the grid.
# Left and right MUST have equal width — the swap_hook pairs positions 1:1.
_STRIP_LAYOUT: Mapping[str, tuple[int, int, int]] = {
    "qwen2vl":  (4, 4, 4),     # N=12: 4+4+4
    "gemma3":   (5, 6, 5),     # N=16: 5+6+5 (outer = (N - middle) / 2 with middle = entity_size+spacing)
    "pixtral":  (5, 5, 5),     # N=15: 5+5+5
}


def strip_regions(model_slug: str) -> Mapping[str, list[int]]:
    """Per-slot grid indices for the full column (horizontal) / row (vertical)
    strip that contains the object.

    Strip widths come from `_STRIP_LAYOUT` (per-model, tuned to match the
    the per-model reference values; outer widths are kept equal so the
    `swap_hook` left↔right pairing stays balanced).

    Returned keys: ``left``, ``middle``, ``right``, ``top``, ``bottom``
    (vertical reuses the horizontal layout transposed).
    """
    geom = GEOM[model_slug]
    N = geom.num_tokens
    L, M, R = _STRIP_LAYOUT[model_slug]
    assert L + M + R == N, f"{model_slug} strip widths {L}+{M}+{R} != N={N}"
    assert L == R, f"{model_slug} outer strips must be equal width (got L={L}, R={R})"

    cols = [(0, L), (L, L + M), (L + M, N)]
    rows = list(cols)  # symmetric for vertical layout

    return {
        "left":   _column_strip(N, *cols[0]),
        "middle": _column_strip(N, *cols[1]),
        "right":  _column_strip(N, *cols[2]),
        "top":    _row_strip(N, *rows[0]),
        "bottom": _row_strip(N, *rows[2]),
    }


def regions_for(model_slug: str, kind: str) -> Mapping[str, list[int]]:
    """Dispatch: ``object`` → square-localized slots; ``strip`` → full-strip slots."""
    if kind == "object":
        return object_regions(model_slug)
    if kind == "strip":
        return strip_regions(model_slug)
    raise ValueError(f"kind must be 'object' or 'strip', got {kind!r}")


# ---------------------------------------------------------------------------
# WhatsUp region maps — hand-tuned per (model, middle reference object).
#
# Unlike the synthetic settings, WhatsUp regions are NOT geometry-derived:
# objects sit at middle-specific locations in the real stitched photos, so
# the left/right grid boxes are hand-labelled per reference object on each
# model's grid (Qwen2-VL 12×12, Gemma3 16×16). Only the ``left`` / ``right``
# regions are used by the criss-cross swap.
#
# No Pixtral entry: whatsup criss-cross is supported for Qwen2-VL and Gemma3
# only (no hand-labelled Pixtral region map).
# ---------------------------------------------------------------------------

def _box(n: int, rows: range, cols: range) -> list[int]:
    return [n * r + c for r in rows for c in cols]


WHATSUP_OBJECT_REGIONS: Mapping[str, Mapping[str, Mapping[str, list[int]]]] = {
    "qwen2vl": {  # 12×12
        "chair":    {"left": _box(12, range(8, 11), range(1, 4)),
                     "right": _box(12, range(8, 11), range(8, 11))},
        "table":    {"left": _box(12, range(6, 11), range(0, 3)),
                     "right": _box(12, range(6, 11), range(9, 12))},
        "armchair": {"left": _box(12, range(9, 11), range(1, 4)),
                     "right": _box(12, range(9, 11), range(8, 11))},
    },
    "gemma3": {  # 16×16
        "chair":    {"left": _box(16, range(12, 15), range(3, 6)),
                     "right": _box(16, range(12, 15), range(10, 13))},
        "table":    {"left": _box(16, range(10, 15), range(2, 4)),
                     "right": _box(16, range(10, 15), range(12, 14))},
        "armchair": {"left": _box(16, range(13, 15), range(2, 5)),
                     "right": _box(16, range(13, 15), range(11, 14))},
    },
}

WHATSUP_STRIP_REGIONS: Mapping[str, Mapping[str, Mapping[str, list[int]]]] = {
    "qwen2vl": {  # 12×12
        "chair":    {"left": _box(12, range(12), range(0, 4)),
                     "right": _box(12, range(12), range(8, 12))},
        "table":    {"left": _box(12, range(12), range(0, 3)),
                     "right": _box(12, range(12), range(9, 12))},
        "armchair": {"left": _box(12, range(12), range(0, 4)),
                     "right": _box(12, range(12), range(8, 12))},
    },
    "gemma3": {  # 16×16
        "chair":    {"left": _box(16, range(16), range(0, 6)),
                     "right": _box(16, range(16), range(10, 16))},
        "table":    {"left": _box(16, range(16), range(0, 6)),
                     "right": _box(16, range(16), range(12, 16))},
        "armchair": {"left": _box(16, range(16), range(0, 5)),
                     "right": _box(16, range(16), range(11, 16))},
    },
}


def whatsup_regions(model_slug: str, middle: str, kind: str) -> Mapping[str, list[int]]:
    """Per-(model, middle) ``{left, right}`` grid indices for whatsup.

    Raises if the model has no ported whatsup maps (e.g. Pixtral).
    """
    table = WHATSUP_OBJECT_REGIONS if kind == "object" else WHATSUP_STRIP_REGIONS
    if model_slug not in table:
        raise ValueError(
            f"no whatsup {kind} region map for {model_slug!r} "
            f"(supported: {sorted(table)})"
        )
    return table[model_slug][middle]
