"""Per-setting isolated-middle reference images for the §5.3.2 ablation.

The ordering ablation (Sec 5.3.1) replaces each entity slot with the
*middle-slot* patches of an isolated reference image of the same entity
(color/shape/object), and the background with an empty image. This module
maps each setting's entities to those reference images, so the
criss-cross-under-ablation runner and the cf-pair builder share one
entity → reference factory.

WhatsUp is intentionally absent: the paper does not define the ablation
for natural images (Sec 5.3.1), and its under-ablation figures (Figs
21/22) cover squares/shapes/objects only.
"""

from __future__ import annotations

from pathlib import Path

from shared.images import SHAPE_DRAW_FNS
from shared.objects import OBJECT_NAMES
from shared.palette import palette_for
from behavioral_analysis_wo_ve_oi.src.references import (
    isolated_middle_object, isolated_middle_shape, isolated_middle_square,
)

# Objects thumbnail cache (shared with behavioral_analysis / probing).
# parents[2] is `` (refs.py lives at <exp>/src/refs.py).
_OBJ_CACHE = Path(__file__).resolve().parents[2] / "shared" / "assets" / "objects"

SETTINGS = ("squares", "shapes", "objects")


def hashable(entity):
    """Cache/JSON entities round-trip as lists for shapes; make them hashable."""
    return tuple(entity) if isinstance(entity, list) else entity


def clean_cf_entities(tup, direction):
    """Slot entities (hashable) in the clean and cf images for one 3-tuple.

    Mirrors `criss_cross_patching.src.data.build_vl_prompts`: for
    left/above the clean image is ``(t0, t1, t2)``; for right/below it's
    reversed so the clean answer still sits at the queried slot. The cf
    image is the reverse of the clean one.
    """
    t0, t1, t2 = (hashable(e) for e in tup)
    if direction in ("left", "above"):
        return (t0, t1, t2), (t2, t1, t0)
    return (t2, t1, t0), (t0, t1, t2)


def object_names_for(model_key: str) -> tuple[str, ...]:
    """8 objects, or 7 (bomb excluded) for Pixtral — matches behavioral_analysis."""
    if model_key == "pixtral":
        return tuple(n for n in OBJECT_NAMES if n != "bomb")
    return OBJECT_NAMES


def iso_reference_images(setting: str, model_key: str) -> dict:
    """Map each entity (hashable) → its isolated-middle reference PIL image.

    Entity keys match what `criss_cross_patching.src.data.build_vl_prompts`
    puts in the tuples: color name (squares), ``(shape, color)`` tuple
    (shapes), object name (objects).
    """
    palette = palette_for(model_key)
    if setting == "squares":
        return {c: isolated_middle_square(c, model_key=model_key, palette=palette)
                for c in palette}
    if setting == "shapes":
        entities = list(zip(SHAPE_DRAW_FNS.keys(), palette.keys()))
        return {(sh, co): isolated_middle_shape(sh, co, model_key=model_key, palette=palette)
                for sh, co in entities}
    if setting == "objects":
        return {n: isolated_middle_object(n, model_key=model_key, asset_dir=_OBJ_CACHE)
                for n in object_names_for(model_key)}
    raise ValueError(f"unsupported setting for ablation: {setting!r}")
