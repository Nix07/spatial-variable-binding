"""Image + prompt + VLPrompt construction for criss-cross patching, across
settings (squares / shapes / objects / whatsup).

Shared by `run_experiment.py` (patching) and `build_cf_pairs.py` (pool gen).

Each candidate is a 3-element tuple ``(t0, t1, t2)`` encoding a (clean, cf)
pair. ``t1`` is the **middle** entity (referenced in the prompt). The
clean image lays the entities out so the queried answer is ``t0``; the cf
image is the **left↔right (or top↔bottom) reverse**, so the cf answer is
``t2``. The criss-cross intervention then swaps the two side regions
between the clean and cf runs (see `run_experiment.swap_positions` /
`src.regions`).

Per-setting differences (image renderer, prompt noun, answer extraction)
live in the `SETTINGS` registry; the (clean, cf) tuple algebra is shared.

- squares / shapes / objects: synthetic three-entity grid; region maps
  are geometry-derived (`src.regions.regions_for`); 4 directions.
- whatsup: real stitched WhatsUp photos; region maps are hand-tuned
  per (model, middle reference object) in `src.regions`; **horizontal
  (left/right) only**; candidate tuples are grouped per middle object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

from shared import images as _img
from shared.experiment_config import ModelGeom
from shared.images import SHAPE_DRAW_FNS
from shared.objects import OBJECT_NAMES, load_thumbnail
from shared.palette import PREPOSITION, is_horizontal, palette_for
from shared.paths import setup as _setup_paths
from shared.prompts import chat_prompt
from shared.whatsup import DEFAULT_IMAGES_DIR, stitch_horizontal

_setup_paths()  # parent repo on sys.path so VLPrompt is importable

from shared.vision_language_prompts import VLPrompt    # noqa: E402


# Color palette for squares/shapes — same saturations as the Figure-6/7
# notebooks; do NOT unify with other experiments without re-running.
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

CF_PAIRS_KEY = {
    "qwen2vl": "qwen2-vl-7b",
    "gemma3":  "gemma-3-4b-it",
    "pixtral": "pixtral-12b",
}

_OBJ_CACHE = Path(__file__).resolve().parents[2] / "shared" / "assets" / "objects"

WHATSUP_MIDDLES = ("chair", "table", "armchair")
WHATSUP_IMAGE_SIZE = {"qwen2vl": 336, "gemma3": 896, "pixtral": 512}


def geom_to_image_kwargs(geom: ModelGeom) -> dict:
    return dict(
        num_tokens=geom.num_tokens, token_size=geom.token_size,
        entity_height=geom.entity_size, entity_width=geom.entity_size,
        spacing=geom.spacing,
    )


# ---------------------------------------------------------------------------
# Per-setting configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
    name: str
    directions: tuple[str, ...]
    grouped_by_middle: bool                            # whatsup: tuples grouped per middle ref
    prompt: Callable[[object, object, str], str]       # (processor, middle_entity, direction) -> str
    render: Callable[..., object]                      # (entities3, direction, geom, model_key) -> image
    answer_of: Callable[[object], str]
    prob_str_of: Callable[[object], str]


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


# ---- shapes ----  (entity = (shape, color); answer = color)

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


# ---- objects ----  (entity = object name)

def _objects_prompt(processor, middle, direction):
    return chat_prompt(
        processor, f"The object {PREPOSITION[direction]} the {middle} is a(n) ",
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


# ---- whatsup ----  (entity = object name; horizontal only; real photos)

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
_COLOR_OF = lambda e: e[1]         # noqa: E731

SETTINGS: dict[str, Setting] = {
    "squares": Setting("squares", ("left", "right", "above", "below"), False,
                       _squares_prompt, _squares_render, _IDENTITY, _IDENTITY),
    "shapes":  Setting("shapes", ("left", "right", "above", "below"), False,
                       _shapes_prompt, _shapes_render, _COLOR_OF, _COLOR_OF),
    "objects": Setting("objects", ("left", "right", "above", "below"), False,
                       _objects_prompt, _objects_render, _IDENTITY, _IDENTITY),
    "whatsup": Setting("whatsup", ("left", "right"), True,
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

    Clean lays entities as ``(t0, t1, t2)`` for left/above and
    ``(t2, t1, t0)`` for right/below (so the queried answer is always
    ``t0``); the cf image is the reverse (so the cf answer is ``t2``).
    """
    cfg = SETTINGS[setting]
    out: List[VLPrompt] = []
    for tup in tuples:
        t0, t1, t2 = tup
        if direction in ("left", "above"):
            clean_e, cf_e = (t0, t1, t2), (t2, t1, t0)
        else:
            clean_e, cf_e = (t2, t1, t0), (t0, t1, t2)
        out.append(VLPrompt(
            prompt=cfg.prompt(processor, t1, direction),
            images=[cfg.render(clean_e, direction, geom, model_key)],
            answer=cfg.answer_of(t0).lower(),
            cf_prompt=cfg.prompt(processor, t1, direction),
            cf_images=[cfg.render(cf_e, direction, geom, model_key)],
            cf_answer=cfg.answer_of(t2).lower(),
            prob_strs=[cfg.prob_str_of(t0).lower(),
                       cfg.prob_str_of(t1).lower(),
                       cfg.prob_str_of(t2).lower()],
        ))
    return out


# ---------------------------------------------------------------------------
# Candidate pools for build_cf_pairs
# ---------------------------------------------------------------------------

def candidate_pool(setting: str, model_key: str, processor):
    """Sampling pool for build_cf_pairs.

    - squares: single-token color names.
    - shapes:  the 6 fixed (shape, color) entities for the model palette.
    - objects: single-token object names (model's object set).
    - whatsup: ``{middle: [single-token sides with both L/R photos]}``.
    """
    tok = processor.tokenizer

    def single_token(s: str) -> bool:
        return len(tok.tokenize(s)) == 1 and len(tok.tokenize(s.capitalize())) == 1

    if setting == "squares":
        return [c for c in COLOR_RGB if c != "White" and single_token(c)]
    if setting == "shapes":
        palette = palette_for(model_key)
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
            if single_token(middle) and len(sides) >= 3:
                pools[middle] = sorted(sides)
        return pools
    raise ValueError(f"unknown setting {setting!r}")
