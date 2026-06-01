"""Objects task: 3 real-world object thumbnails per image, predict the
object name at a specified spatial position relative to the middle.

Object set:
  Qwen2-VL / Gemma3: 8 objects (orange, apple, pot, bell, hat, rose, key, bomb)
                    → P(8, 3) · 4 = 1344 items
  Pixtral:           7 objects (bomb excluded; accuracy 529/840 = 0.6298 ≈
                    paper Table 1's 0.63)
                    → P(7, 3) · 4 = 840 items

Prompt template:
  "The object {prep} the {middle_object} is a(n) "
"""

from __future__ import annotations

import itertools
from pathlib import Path

from shared.experiment_config import geom_for
from shared.images import three_object_image
from shared.objects import OBJECT_NAMES, load_thumbnail
from shared.palette import DIRECTIONS, PREPOSITION, is_horizontal
from shared.prompts import chat_prompt

from ..eval import EvalResult, norm_token, top1_token


# Shared with probing — see `shared/assets/objects/`.
_THIS = Path(__file__).resolve().parent
_CACHE = _THIS.parents[2] / "shared" / "assets" / "objects"


def object_names_for(model_key: str) -> tuple[str, ...]:
    if model_key == "pixtral":
        return tuple(n for n in OBJECT_NAMES if n != "bomb")
    return OBJECT_NAMES


def iter_items(names: tuple[str, ...]):
    for tup in itertools.permutations(names, 3):
        middle = tup[1]
        for direction in DIRECTIONS:
            target = tup[0] if direction in ("left", "above") else tup[2]
            user_text = f"The object {PREPOSITION[direction]} the {middle} is a(n) "
            yield tup, direction, target, user_text


def run(model, processor, model_key: str) -> dict:
    geom = geom_for(model_key, "objects")
    names = object_names_for(model_key)
    result = EvalResult(model=model.model_name, task="objects")
    items = list(iter_items(names))
    thumb_px = geom.entity_size * geom.token_size
    print(f"Objects: {len(items)} items (P({len(names)},3) × 4 directions); "
          f"canvas {geom.image_px}px; geom={geom}; objects={names}")

    for i, (tup, direction, target, user_text) in enumerate(items):
        prompt = chat_prompt(processor, user_text)
        thumbs = [load_thumbnail(name, cache_dir=_CACHE).resize((thumb_px, thumb_px))
                  for name in tup]
        image = three_object_image(
            thumbs, horizontal=is_horizontal(direction),
            num_tokens=geom.num_tokens, token_size=geom.token_size,
            entity_size=geom.entity_size, spacing=geom.spacing,
        )
        pred, prob = top1_token(model, processor, prompt, image)
        ok = norm_token(pred) == norm_token(target)
        result.record(direction, ok,
                      tup=tup, target=target, pred=pred.strip(), prob=prob)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)}")

    print(result.summary_text())
    return result.to_dict()
