"""Shapes task: 3 (shape, color) entities per image, predict the color of
the entity at a specified spatial position relative to the middle.

The 6 entities are a fixed 1-to-1 pairing of (shape, color):
  (Square, Red), (Circle, Green), (Triangle, Blue),
  (Star, Black), (Cross, Brown), (Heart, Gray)
The middle entity is referenced by *both* its color and shape, e.g.
  "The color of the shape to the left of the red square is "

P(6, 3) = 120 ordered triples × 4 directions = 480 items per model.
"""

from __future__ import annotations

import itertools

from shared.experiment_config import geom_for
from shared.images import SHAPE_DRAW_FNS, three_shape_image
from shared.palette import (
    DIRECTIONS,
    PREPOSITION,
    is_horizontal,
    palette_for,
)
from shared.prompts import chat_prompt

from ..eval import EvalResult, norm_token, top1_token


def iter_items(entities):
    for tup in itertools.permutations(entities, 3):
        middle_shape, middle_color = tup[1]
        for direction in DIRECTIONS:
            target_idx = 0 if direction in ("left", "above") else 2
            target_color = tup[target_idx][1]
            user_text = (f"The color of the shape {PREPOSITION[direction]} "
                         f"the {middle_color.lower()} {middle_shape.lower()} is ")
            yield tup, direction, target_color, user_text


def run(model, processor, model_key: str) -> dict:
    geom = geom_for(model_key, "shapes")
    palette = palette_for(model_key)
    # Fixed 1-to-1 zip — entity i is shape i with color i (from this palette).
    entities = list(zip(SHAPE_DRAW_FNS.keys(), palette.keys()))
    result = EvalResult(model=model.model_name, task="shapes")
    items = list(iter_items(entities))
    print(f"Shapes: {len(items)} items (P(6,3) × 4 directions); "
          f"canvas {geom.image_px}px; entities={entities}")

    for i, (tup, direction, target, user_text) in enumerate(items):
        prompt = chat_prompt(processor, user_text)
        image = three_shape_image(
            tup, horizontal=is_horizontal(direction),
            num_tokens=geom.num_tokens, token_size=geom.token_size,
            entity_size=geom.entity_size, spacing=geom.spacing,
            color_rgb=palette,
        )
        pred, prob = top1_token(model, processor, prompt, image)
        ok = norm_token(pred) == norm_token(target)
        result.record(direction, ok,
                      tup=tup, target=target, pred=pred.strip(), prob=prob)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)}")

    print(result.summary_text())
    return result.to_dict()
