"""2×2 / 3×3 grid extension (paper App C.6).

The synthetic Squares/Shapes/Objects setting arranges three objects along one
axis. App C.6 extends this to a grid of identical shapes filling a `k × k`
window and asks the probe to predict both horizontal **and** vertical position
labels jointly: class = `(row_label, col_label)` over {top, middle, bottom}
× {left, middle, right} = 9 classes for 3×3, or 4 classes for 2×2.

For simplicity (and to keep the linear probe at the same complexity as the
main pipeline), this module exposes a flat **row-label-only** and
**column-label-only** probe option — equivalent to running the standard 3-class
probe on shapes arranged in a grid. That reproduces the Fig 41/42 strip
patterns separately for the horizontal and vertical axes.

`k=2` covers App C.6 Fig 41 (Qwen + Gemma); `k=3` covers Fig 42.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image, ImageDraw

from shared.images import SHAPE_DRAW_FNS

from .config import COLOR_RGB_WHITE, SHAPE_COLOR_PAIRS, probing_geom
from .extract import patch_embeddings


@dataclass
class GridGeom:
    """Geometry of a `k × k` grid of identically-sized shapes on the model's
    patch canvas."""
    k: int                    # 2 or 3
    cell_size: int            # patches per cell (entity_size in synthetic)
    cell_spacing: int         # patches between adjacent cells

    def offsets(self, num_tokens: int) -> list[int]:
        """Patch-unit start offset of each row (or column)."""
        total = self.k * self.cell_size + (self.k - 1) * self.cell_spacing
        first = (num_tokens - total) // 2
        return [first + i * (self.cell_size + self.cell_spacing) for i in range(self.k)]


def render_grid(shape_color_grid: Sequence[Sequence[tuple[str, str]]],
                *,
                num_tokens: int,
                token_size: int,
                grid_geom: GridGeom) -> Image.Image:
    """`shape_color_grid[r][c] = (shape_name, color_name)` arranged on a
    `k × k` grid."""
    side = num_tokens * token_size
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    rows = grid_geom.offsets(num_tokens)
    cols = grid_geom.offsets(num_tokens)
    side_px = grid_geom.cell_size * token_size
    for r, row_offset in enumerate(rows):
        for c, col_offset in enumerate(cols):
            shape_name, color_name = shape_color_grid[r][c]
            sprite = Image.new("RGBA", (side_px, side_px), (0, 0, 0, 0))
            SHAPE_DRAW_FNS[shape_name](
                ImageDraw.Draw(sprite), side_px, COLOR_RGB_WHITE[color_name],
            )
            canvas.paste(sprite,
                         (col_offset * token_size, row_offset * token_size),
                         mask=sprite.split()[3])
    return canvas


def cell_token_ids(*, num_tokens: int, grid_geom: GridGeom) -> dict[tuple[int, int], list[int]]:
    """`(row_idx, col_idx) → flat patch indices for that cell`."""
    rows = grid_geom.offsets(num_tokens)
    cols = grid_geom.offsets(num_tokens)
    out: dict[tuple[int, int], list[int]] = {}
    for ri, r0 in enumerate(rows):
        for ci, c0 in enumerate(cols):
            ids = []
            for i in range(grid_geom.cell_size):
                for j in range(grid_geom.cell_size):
                    ids.append((r0 + i) * num_tokens + (c0 + j))
            out[(ri, ci)] = ids
    return out


def _grid_geom_for(model_slug: str, k: int) -> GridGeom:
    """Per-model defaults: 2x2 patches per cell for Qwen (cell_size=2), 3x3
    for Gemma/Pixtral (cell_size=3). Spacing tuned so the grid fits in the
    canvas with reasonable margins."""
    geom = probing_geom(model_slug)
    return GridGeom(k=k, cell_size=geom.entity_size, cell_spacing=geom.spacing)


def build_split(model, processor, *,
                model_slug: str,
                k: int,
                axis: str,            # "horizontal" or "vertical"
                n_samples: int = 90,
                test_fraction: float = 0.25,
                seed: int = 0):
    """Build (train, test) splits for the grid axis probe.

    `axis="horizontal"` trains a `k`-class probe on column index (0..k-1);
    `axis="vertical"` trains on row index. The probe size grows with k, so
    this is no longer strictly a 3-class probe; consumers should pass the
    returned label set size to `probe.train` via reshape if needed.

    Returns a `data.Split` with `n_classes` accessible as `max(labels)+1`.
    """
    from .data import Split  # local import to avoid cycle

    if axis not in ("horizontal", "vertical"):
        raise ValueError(axis)
    geom = probing_geom(model_slug)
    grid_geom = _grid_geom_for(model_slug, k)

    # Sample grids by drawing (shape, color) pairs from SHAPE_COLOR_PAIRS.
    rng = random.Random(seed)
    pairs = list(SHAPE_COLOR_PAIRS)
    all_grids = []
    for _ in range(n_samples):
        grid = [[rng.choice(pairs) for _ in range(k)] for _ in range(k)]
        all_grids.append(grid)
    split_idx = int((1.0 - test_fraction) * n_samples)
    train_grids = all_grids[:split_idx]
    test_grids = all_grids[split_idx:]

    patches_per_image = geom.num_tokens * geom.num_tokens
    cells = cell_token_ids(num_tokens=geom.num_tokens, grid_geom=grid_geom)

    def collect(grids, *, also_all_tokens: bool):
        obj_embeds, obj_labels = [], []
        all_embeds = [] if also_all_tokens else None
        for i, grid in enumerate(grids):
            img = render_grid(grid,
                              num_tokens=geom.num_tokens,
                              token_size=geom.token_size,
                              grid_geom=grid_geom)
            emb = patch_embeddings(model, processor, img)
            if emb.shape[0] != patches_per_image:
                raise RuntimeError(
                    f"{model_slug}: expected {patches_per_image} patches but got {emb.shape[0]}"
                )
            for (ri, ci), ids in cells.items():
                label = ci if axis == "horizontal" else ri
                obj_embeds.append(emb[ids, :])
                obj_labels.extend([label] * len(ids))
            if also_all_tokens:
                all_embeds.append(emb)
            if (i + 1) % 25 == 0 or i == len(grids) - 1:
                print(f"    image {i + 1}/{len(grids)}")
        return (torch.cat(obj_embeds, dim=0),
                torch.tensor(obj_labels, dtype=torch.long),
                torch.cat(all_embeds, dim=0) if also_all_tokens else None)

    print(f"  building grid{k}x{k} train split ({len(train_grids)} images)")
    train_e, train_l, _ = collect(train_grids, also_all_tokens=False)
    print(f"  building grid{k}x{k} test split  ({len(test_grids)} images)")
    test_e, test_l, all_tokens_test = collect(test_grids, also_all_tokens=True)

    return Split(
        train_embeds=train_e,
        train_labels=train_l,
        test_embeds=test_e,
        test_labels=test_l,
        all_tokens_test_embeds=all_tokens_test,
        n_test_images=len(test_grids),
        patches_per_image=patches_per_image,
        train_tups=train_grids,
        test_tups=test_grids,
    )
