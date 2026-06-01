"""Squares task: 3 colored squares per image, predict the color of the square
at a specified spatial position relative to the middle.

Image canvas size differs per model to match each model's native vision input:

  qwen2vl: 336 px (12 tokens × 28 px)
  gemma3:  896 px (16 tokens × 56 px)
  pixtral: 240 px (15 tokens × 16 px) — small to fit TL's precomputed rotary cache
"""

from __future__ import annotations

import itertools

from shared.experiment_config import geom_for
from shared.images import three_rect_image
from shared.palette import (
    DIRECTIONS,
    PREPOSITION,
    is_horizontal,
    palette_for,
)
from shared.prompts import chat_prompt

from ..eval import EvalResult, norm_token, top1_token


def iter_items(color_names):
    """Yield (tup, direction, target, user_text) over all 6P3 × 4 directions = 480."""
    for tup in itertools.permutations(color_names, 3):
        middle = tup[1]
        for direction in DIRECTIONS:
            target = tup[0] if direction in ("left", "above") else tup[2]
            user_text = (f"The color of the square {PREPOSITION[direction]} "
                         f"the {middle.lower()} square is ")
            yield tup, direction, target, user_text


def run(model, processor, model_key: str) -> dict:
    geom = geom_for(model_key, "squares")
    palette = palette_for(model_key)
    result = EvalResult(model=model.model_name, task="squares")
    items = list(iter_items(tuple(palette.keys())))
    print(f"Squares: {len(items)} items (P(6,3) × 4 directions); "
          f"canvas {geom.image_px}px; geom={geom}; "
          f"palette={list(palette.keys())}")

    for i, (tup, direction, target, user_text) in enumerate(items):
        prompt = chat_prompt(processor, user_text)
        image = three_rect_image(
            tup, horizontal=is_horizontal(direction),
            num_tokens=geom.num_tokens, token_size=geom.token_size,
            entity_height=geom.entity_size, entity_width=geom.entity_size,
            spacing=geom.spacing, color_rgb=palette,
        )
        pred, prob = top1_token(model, processor, prompt, image)
        ok = norm_token(pred) == norm_token(target)
        result.record(direction, ok,
                      tup=tup, target=target, pred=pred.strip(), prob=prob)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)}")

    print(result.summary_text())
    return result.to_dict()
