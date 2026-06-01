"""Build (train_embeds, test_embeds, all_tokens_test_embeds) for one
(setting, orientation, model) configuration.

Parameterized by `ModelGeom`, with the object cut-outs read from the local
asset cache instead of fetched fresh.

`setting` is one of:
- ``squares`` — `config.COLOR_RGB` keys
- ``shapes``  — `config.SHAPE_COLOR_PAIRS` (fixed shape↔color)
- ``objects`` — `shared.objects.OBJECT_NAMES` (PNGs cached locally)
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from shared.objects import OBJECT_NAMES, ensure_all_cached

from . import images as _img
from .config import (
    COLOR_RGB,
    DEFAULT_OBJECT_ASSETS_DIR,
    SHAPE_COLOR_PAIRS,
    ModelGeom,
    probing_geom,
)
from .extract import patch_embeddings


SettingName = str  # "squares" | "shapes" | "objects"
Orientation = str  # "horizontal" | "vertical"


def options_for(setting: SettingName) -> Sequence:
    if setting == "squares":
        return list(COLOR_RGB.keys())
    if setting == "shapes":
        return list(SHAPE_COLOR_PAIRS)
    if setting == "objects":
        return list(OBJECT_NAMES)
    raise ValueError(f"unknown setting {setting!r}")


def build_image(setting: SettingName,
                tup: Sequence,
                *,
                orientation: Orientation,
                geom: ModelGeom,
                object_assets_dir: Path) -> Image.Image:
    """Render one image for one (setting, orientation, geom) tuple."""
    kw = dict(orientation=orientation, num_tokens=geom.num_tokens,
              token_size=geom.token_size, entity_size=geom.entity_size,
              spacing=geom.spacing)
    if setting == "squares":
        return _img.make_three_rectangles(tup, **kw)
    if setting == "shapes":
        return _img.make_three_shapes(tup, **kw)
    if setting == "objects":
        return _img.make_three_objects(tup, **kw,
                                       object_assets_dir=object_assets_dir)
    raise ValueError(setting)


@dataclass
class Split:
    train_embeds: torch.Tensor          # (N_train_obj, D)
    train_labels: torch.Tensor          # (N_train_obj,)
    test_embeds: torch.Tensor           # (N_test_obj, D)
    test_labels: torch.Tensor           # (N_test_obj,)
    all_tokens_test_embeds: torch.Tensor  # (n_test_imgs * N_patches, D)
    n_test_images: int
    patches_per_image: int
    train_tups: list
    test_tups: list


def build_split(model,
                processor,
                *,
                model_slug: str,
                setting: SettingName,
                orientation: Orientation,
                train_fraction: float = 0.75,
                seed: int = 0,
                object_assets_dir: Path | None = None) -> Split:
    """Render images, extract patch embeddings, slice into the probe train/test
    splits. 75/25 random split."""
    geom = probing_geom(model_slug)
    object_assets_dir = object_assets_dir or DEFAULT_OBJECT_ASSETS_DIR
    if setting == "objects":
        ensure_all_cached(object_assets_dir)

    options = options_for(setting)
    all_tups = list(itertools.permutations(options, 3))
    rng = random.Random(seed)
    rng.shuffle(all_tups)
    split_idx = int(train_fraction * len(all_tups))
    train_tups = all_tups[:split_idx]
    test_tups = all_tups[split_idx:]

    token_ids = _img.get_position_token_ids(
        num_tokens=geom.num_tokens,
        entity_size=geom.entity_size,
        spacing=geom.spacing,
    )
    if orientation == "horizontal":
        obj_idx, labels_per_image = _img.horizontal_label_indices(token_ids)
    else:
        obj_idx, labels_per_image = _img.vertical_label_indices(token_ids)

    patches_per_image = geom.num_tokens * geom.num_tokens

    def collect(tups, *, also_all_tokens: bool):
        obj_embeds, obj_labels = [], []
        all_embeds = [] if also_all_tokens else None
        for i, tup in enumerate(tups):
            img = build_image(setting, tup,
                              orientation=orientation,
                              geom=geom,
                              object_assets_dir=object_assets_dir)
            emb = patch_embeddings(model, processor, img)
            if emb.shape[0] != patches_per_image:
                raise RuntimeError(
                    f"{model_slug}: expected {patches_per_image} patches "
                    f"({geom.num_tokens}x{geom.num_tokens}) but got {emb.shape[0]}"
                )
            obj_embeds.append(emb[obj_idx, :])
            obj_labels.extend(labels_per_image)
            if also_all_tokens:
                all_embeds.append(emb)
            if (i + 1) % 25 == 0 or i == len(tups) - 1:
                print(f"    image {i + 1}/{len(tups)}")
        return (torch.cat(obj_embeds, dim=0),
                torch.tensor(obj_labels, dtype=torch.long),
                torch.cat(all_embeds, dim=0) if also_all_tokens else None)

    print(f"  building train split ({len(train_tups)} images)")
    train_e, train_l, _ = collect(train_tups, also_all_tokens=False)
    print(f"  building test split  ({len(test_tups)} images)")
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
