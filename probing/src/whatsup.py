"""What'sUp setting — bounding-box-based probing on real-world images
(paper App C.7 / Fig 43-45).

Section 5.2.1 says: *"To handle [variable object sizes in What'sUp], we
preprocess What'sUp images by extracting bounding boxes for each object and
use their spatial coordinates to identify the corresponding visual tokens."*

We mirror that for **post-projection** patch embeddings (the same extraction
the synthetic settings use). The bboxes are defined per reference object on a
**normalized [0, 1] image grid**, then mapped per-model to the model's post-
projection patch grid (`config.GEOM[slug].num_tokens` per side).

Source bboxes were defined on a 64x64 raw patch grid for Gemma at 896×896
input, converted to [0,1] coords here.

Each composite image stitches two controlled-images JPEGs (taken from
`shared/whatsup_images/controlled_images/`) into one side-by-side scene
sharing the middle reference object.
"""

from __future__ import annotations

import os
import random
from glob import glob
from itertools import permutations
from pathlib import Path
from typing import Sequence

import torch

from shared.whatsup import DEFAULT_IMAGES_DIR, stitch_horizontal

from .config import probing_geom
from .extract import patch_embeddings


# Normalized [0,1] bboxes per reference object (rows = y-extent, cols = x-extent).
# Derived from 64x64-grid bboxes (divide both endpoints by 64).
WHATSUP_BBOX_NORM = {
    "chair": {
        "left":   {"rows": (48 / 64, 60 / 64), "cols": (12 / 64, 24 / 64)},
        "middle": {"rows": (16 / 64, 60 / 64), "cols": (24 / 64, 40 / 64)},
        "right":  {"rows": (48 / 64, 60 / 64), "cols": (40 / 64, 52 / 64)},
    },
    "table": {
        "left":   {"rows": (40 / 64, 60 / 64), "cols": ( 8 / 64, 16 / 64)},
        "middle": {"rows": (20 / 64, 60 / 64), "cols": (16 / 64, 48 / 64)},
        "right":  {"rows": (40 / 64, 60 / 64), "cols": (48 / 64, 56 / 64)},
    },
    "armchair": {
        "left":   {"rows": (52 / 64, 60 / 64), "cols": ( 8 / 64, 20 / 64)},
        "middle": {"rows": (24 / 64, 60 / 64), "cols": (20 / 64, 44 / 64)},
        "right":  {"rows": (52 / 64, 60 / 64), "cols": (44 / 64, 56 / 64)},
    },
}

REFS = ("chair", "table", "armchair")
POSITIONS = ("left", "middle", "right")
POS_TO_LABEL = {"left": 0, "middle": 1, "right": 2}

def bbox_norm_to_patches(bbox_norm: dict, num_tokens: int) -> list[int]:
    """Normalized bbox → flat patch indices on a `num_tokens × num_tokens` grid."""
    r0, r1 = bbox_norm["rows"]
    c0, c1 = bbox_norm["cols"]
    pr0 = max(0, int(r0 * num_tokens))
    pr1 = max(pr0 + 1, min(num_tokens, int(round(r1 * num_tokens))))
    pc0 = max(0, int(c0 * num_tokens))
    pc1 = max(pc0 + 1, min(num_tokens, int(round(c1 * num_tokens))))
    return [r * num_tokens + c
            for r in range(pr0, pr1)
            for c in range(pc0, pc1)]


def list_tuples(controlled_dir: Path,
                references: Sequence[str] = REFS) -> list[tuple[str, str, str]]:
    """Scan disk for valid (left_obj, ref, right_obj) WhatsUp triples."""
    out: list[tuple[str, str, str]] = []
    for ref in references:
        left = {os.path.basename(p).replace(f"_left_of_{ref}.jpeg", "")
                for p in glob(str(controlled_dir / f"*_left_of_{ref}.jpeg"))}
        right = {os.path.basename(p).replace(f"_right_of_{ref}.jpeg", "")
                 for p in glob(str(controlled_dir / f"*_right_of_{ref}.jpeg"))}
        valid = sorted(left & right)
        out.extend((o1, ref, o2) for o1, o2 in permutations(valid, 2))
    return out


def build_split(model, processor, *,
                model_slug: str,
                controlled_dir: Path = DEFAULT_IMAGES_DIR,
                max_tuples: int = 120,
                train_fraction: float = 0.75,
                seed: int = 42):
    """Build (train, test) WhatsUp embeddings indexed by per-image bbox tokens.

    Returns the same fields as `data.Split`, with `train_tups`/`test_tups`
    populated by `(left_obj, ref, right_obj)` triples.
    """
    from .data import Split  # local import to avoid cycle at module load

    geom = probing_geom(model_slug)
    side = geom.image_px

    all_tups = list_tuples(controlled_dir)
    rng = random.Random(seed)
    rng.shuffle(all_tups)
    if max_tuples is not None:
        all_tups = all_tups[:max_tuples]
    split_idx = int(train_fraction * len(all_tups))
    train_tups = all_tups[:split_idx]
    test_tups  = all_tups[split_idx:]

    patches_per_image = geom.num_tokens * geom.num_tokens

    # Pre-compute bbox token indices per reference on this model's grid.
    bbox_ids = {
        ref: {pos: bbox_norm_to_patches(WHATSUP_BBOX_NORM[ref][pos], geom.num_tokens)
              for pos in POSITIONS}
        for ref in REFS
    }

    def collect(tups, *, also_all_tokens: bool):
        obj_embeds, obj_labels = [], []
        all_embeds = [] if also_all_tokens else None
        for i, tup in enumerate(tups):
            left_obj, ref, right_obj = tup
            img = stitch_horizontal(left_obj, ref, right_obj,
                                    size=side, images_dir=controlled_dir)
            emb = patch_embeddings(model, processor, img)
            if emb.shape[0] != patches_per_image:
                raise RuntimeError(
                    f"{model_slug}: expected {patches_per_image} patches "
                    f"but got {emb.shape[0]} (image stitched to {side}px)"
                )
            for pos in POSITIONS:
                ids = bbox_ids[ref][pos]
                obj_embeds.append(emb[ids, :])
                obj_labels.extend([POS_TO_LABEL[pos]] * len(ids))
            if also_all_tokens:
                all_embeds.append(emb)
            if (i + 1) % 25 == 0 or i == len(tups) - 1:
                print(f"    image {i + 1}/{len(tups)}")
        return (torch.cat(obj_embeds, dim=0),
                torch.tensor(obj_labels, dtype=torch.long),
                torch.cat(all_embeds, dim=0) if also_all_tokens else None)

    print(f"  building whatsup train split ({len(train_tups)} images)")
    train_e, train_l, _ = collect(train_tups, also_all_tokens=False)
    print(f"  building whatsup test split  ({len(test_tups)} images)")
    test_e, test_l, all_tokens_test = collect(test_tups, also_all_tokens=True)

    return Split(
        train_embeds=train_e,
        train_labels=train_l,
        test_embeds=test_e,
        test_labels=test_l,
        all_tokens_test_embeds=all_tokens_test,
        n_test_images=len(test_tups),
        patches_per_image=patches_per_image,
        train_tups=train_tups,
        test_tups=test_tups,
    )
