"""Image + prompt + VLPrompt construction for the last-token patching
experiment, across all four settings (squares / shapes / objects / whatsup).

Shared by `run_experiment.py` (patching) and `build_cf_pairs.py` (pool gen).

Each candidate is a 5-element tuple ``(t0, t1, t2, t3, t4)`` encoding a
(clean, counterfactual) pair for the "final token" patching:

- ``t1`` is the **middle** entity (referenced in the prompt), shared by
  clean and cf.
- For ``direction in {left, above}``: clean image entities are
  ``(t0, t1, t2)``; cf image entities are ``(t4, t1, t3)``; the cf prompt
  uses the *opposite* direction.
- For ``direction in {right, below}``: clean is ``(t2, t1, t0)``; cf is
  ``(t3, t1, t4)``.

In both cases the **clean answer is derived from t0** and the **cf answer
from t3**; all 5 elements enter ``prob_strs`` so the logit_prob metric
covers every candidate appearing in either run.

Per-setting differences (image renderer, prompt noun, how the answer
string is read off an entity) are captured by the `Setting` registry at
the bottom of this module. Everything else is shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

from shared import images as _img
from shared.experiment_config import ModelGeom
from shared.objects import OBJECT_NAMES, load_thumbnail
from shared.palette import (
    OPPOSITE, PREPOSITION, is_horizontal, palette_for,
)
from shared.images import SHAPE_DRAW_FNS
from shared.paths import setup as _setup_paths
from shared.prompts import chat_prompt
from shared.whatsup import stitch_horizontal, DEFAULT_IMAGES_DIR

_setup_paths()  # parent repo on sys.path so VLPrompt is importable

from shared.vision_language_prompts import VLPrompt    # noqa: E402


# Color palette unique to the squares/shapes settings. Saturations match
# the Figure-3 reference; do NOT unify with other experiments'
# palettes without re-running.
COLOR_RGB = {
    "Red":    (255, 0, 0),
    "Green":  (0, 255, 0),
    "Blue":   (0, 0, 255),
    "Black":  (0, 0, 0),
    "Brown":  (150, 75, 0),
    "Gray":   (128, 128, 128),
    "Orange": (255, 165, 0),
    "White":  (255, 255, 255),
}

# `cf_pairs.json` uses versioned slug keys; CLIs accept the short slugs.
CF_PAIRS_KEY = {
    "qwen2vl": "qwen2-vl-7b",
    "gemma3":  "gemma-3-4b-it",
    "pixtral": "pixtral-12b",
}

# Objects thumbnail cache (shared with behavioral_analysis / probing).
_OBJ_CACHE = Path(__file__).resolve().parents[2] / "shared" / "assets" / "objects"

# WhatsUp reference (middle) objects. Side-object pools per middle are
# derived at runtime from the controlled-images directory.
WHATSUP_MIDDLES = ("chair", "table", "armchair")
WHATSUP_IMAGE_SIZE = {"qwen2vl": 336, "gemma3": 896, "pixtral": 512}


def geom_to_image_kwargs(geom: ModelGeom) -> dict:
    """ModelGeom → kwargs the synthetic `three_*_image` builders expect."""
    return dict(
        num_tokens=geom.num_tokens,
        token_size=geom.token_size,
        entity_size=geom.entity_size,
        spacing=geom.spacing,
    )


# ---------------------------------------------------------------------------
# Per-setting configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
    """Everything that differs between settings; the (clean, cf) tuple
    algebra is shared in `build_vl_prompts`."""
    name: str
    directions: tuple[str, ...]
    # noun in the prompt ("square"/"shape"/"object") and whether the answer
    # is the entity's color (shapes) or the entity itself (squares/objects).
    prompt: Callable[[object, object, str], str]      # (processor, middle_entity, direction) -> str
    render: Callable[..., object]                     # (entities3, direction, geom, model_key) -> PIL.Image
    answer_of: Callable[[object], str]                # entity -> answer string
    prob_str_of: Callable[[object], str]              # entity -> prob_strs entry


# ---- squares ----

def _squares_prompt(processor, middle, direction):
    return chat_prompt(
        processor,
        f"The color of the square {PREPOSITION[direction]} the "
        f"{middle.lower()} square is ",
    )


def _squares_render(entities3, direction, geom, model_key):
    return _img.three_rect_image(
        entities3, horizontal=is_horizontal(direction), color_rgb=COLOR_RGB,
        num_tokens=geom.num_tokens, token_size=geom.token_size,
        entity_height=geom.entity_size, entity_width=geom.entity_size,
        spacing=geom.spacing,
    )


# ---- shapes ----  (entity = (shape_name, color_name); answer = color)

def _shapes_prompt(processor, middle, direction):
    shape, color = middle
    return chat_prompt(
        processor,
        f"The color of the shape {PREPOSITION[direction]} the "
        f"{color.lower()} {shape.lower()} is ",
    )


def _shapes_render(entities3, direction, geom, model_key):
    return _img.three_shape_image(
        entities3, horizontal=is_horizontal(direction),
        color_rgb=palette_for(model_key),
        num_tokens=geom.num_tokens, token_size=geom.token_size,
        entity_size=geom.entity_size, spacing=geom.spacing,
    )


# ---- objects ----  (entity = object name; answer = name)

def _objects_prompt(processor, middle, direction):
    return chat_prompt(
        processor,
        f"The object {PREPOSITION[direction]} the {middle} is a(n) ",
    )


def _objects_render(entities3, direction, geom, model_key):
    side_px = geom.entity_size * geom.token_size
    thumbs = [load_thumbnail(n, cache_dir=_OBJ_CACHE).resize((side_px, side_px))
              for n in entities3]
    return _img.three_object_image(
        thumbs, horizontal=is_horizontal(direction),
        num_tokens=geom.num_tokens, token_size=geom.token_size,
        entity_size=geom.entity_size, spacing=geom.spacing,
    )


# ---- whatsup ----  (entity = object name; answer = name; horizontal only)

def _whatsup_prompt(processor, middle, direction):
    return chat_prompt(
        processor,
        f"The object on the floor {PREPOSITION[direction]} the "
        f"{middle.lower()} is a(n) ",
    )


def _whatsup_render(entities3, direction, geom, model_key):
    # entities3 = (side_l, middle, side_r) already laid out for `direction`.
    size = WHATSUP_IMAGE_SIZE[model_key]
    return stitch_horizontal(entities3[0], entities3[1], entities3[2], size=size)


_IDENTITY = lambda e: e            # noqa: E731
_COLOR_OF = lambda e: e[1]         # noqa: E731  (shapes: entity is (shape, color))

SETTINGS: dict[str, Setting] = {
    "squares": Setting("squares", ("left", "right", "above", "below"),
                       _squares_prompt, _squares_render, _IDENTITY, _IDENTITY),
    "shapes":  Setting("shapes", ("left", "right", "above", "below"),
                       _shapes_prompt, _shapes_render, _COLOR_OF, _COLOR_OF),
    "objects": Setting("objects", ("left", "right", "above", "below"),
                       _objects_prompt, _objects_render, _IDENTITY, _IDENTITY),
    "whatsup": Setting("whatsup", ("left", "right"),
                       _whatsup_prompt, _whatsup_render, _IDENTITY, _IDENTITY),
}


# ---------------------------------------------------------------------------
# Shared (clean, cf) tuple algebra → VLPrompt
# ---------------------------------------------------------------------------

def build_vl_prompts(processor,
                     tuples: Sequence[Sequence],
                     direction: str,
                     geom: ModelGeom,
                     *,
                     setting: str,
                     model_key: str) -> List[VLPrompt]:
    """Build (clean, cf) VLPrompt pairs for one (setting, direction).

    Each input is a 5-element tuple; see the module docstring for the
    clean/cf ordering. Works for every setting via the `SETTINGS` registry.
    """
    cfg = SETTINGS[setting]
    if direction in ("left", "above"):
        order_clean, order_cf = (0, 1, 2), (4, 1, 3)
    else:
        order_clean, order_cf = (2, 1, 0), (3, 1, 4)

    out: List[VLPrompt] = []
    for tup in tuples:
        clean_entities = tuple(tup[i] for i in order_clean)
        cf_entities = tuple(tup[i] for i in order_cf)
        out.append(VLPrompt(
            prompt=cfg.prompt(processor, tup[1], direction),
            images=[cfg.render(clean_entities, direction, geom, model_key)],
            answer=cfg.answer_of(tup[0]).lower(),
            cf_prompt=cfg.prompt(processor, tup[1], OPPOSITE[direction]),
            cf_images=[cfg.render(cf_entities, OPPOSITE[direction], geom, model_key)],
            cf_answer=cfg.answer_of(tup[3]).lower(),
            prob_strs=[cfg.prob_str_of(t).lower() for t in tup],
        ))
    return out


# ---------------------------------------------------------------------------
# Candidate pools for build_cf_pairs (per setting / model)
# ---------------------------------------------------------------------------

def candidate_pool(setting: str, model_key: str, processor):
    """Return the sampling pool for build_cf_pairs.

    - squares: single-token color names (from COLOR_RGB minus White).
    - shapes:  the 6 fixed (shape, color) entities for this model's palette.
    - objects: single-token object names (model's object set).
    - whatsup: dict ``{middle: [single-token side objects with both
      left_of/right_of controlled images]}``.
    """
    tok = processor.tokenizer

    def single_token(s: str) -> bool:
        return (len(tok.tokenize(s)) == 1 and len(tok.tokenize(s.capitalize())) == 1)

    if setting == "squares":
        return [c for c in COLOR_RGB if c != "White" and single_token(c)]

    if setting == "shapes":
        palette = palette_for(model_key)
        # entity i = shape i paired with color i (same zip as behavioral shapes)
        return list(zip(SHAPE_DRAW_FNS.keys(), palette.keys()))

    if setting == "objects":
        names = OBJECT_NAMES if model_key != "pixtral" else tuple(
            n for n in OBJECT_NAMES if n != "bomb")
        return [n for n in names if single_token(n)]

    if setting == "whatsup":
        pools: dict[str, list[str]] = {}
        for middle in WHATSUP_MIDDLES:
            sides = []
            for p in DEFAULT_IMAGES_DIR.glob(f"*_left_of_{middle}.jpeg"):
                obj = p.name[: -len(f"_left_of_{middle}.jpeg")]
                if (DEFAULT_IMAGES_DIR / f"{obj}_right_of_{middle}.jpeg").exists() \
                        and single_token(obj):
                    sides.append(obj)
            if single_token(middle) and len(sides) >= 5:
                pools[middle] = sorted(sides)
        return pools

    raise ValueError(f"unknown setting {setting!r}")
