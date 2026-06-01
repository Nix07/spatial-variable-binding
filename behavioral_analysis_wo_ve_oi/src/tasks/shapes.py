"""Shapes task under the Table 5 ordering ablation.

Same items as ``behavioral_analysis.src.tasks.shapes`` — the
reference entity per slot is the (shape, color) pair, placed alone at the
canvas middle. Each entity's identity (which shape, which color) is
preserved; only the position signal is destroyed.
"""

from __future__ import annotations

import itertools

from shared.experiment_config import GEOM
from shared.images import SHAPE_DRAW_FNS, three_shape_image
from shared.palette import (
    DIRECTIONS,
    PREPOSITION,
    is_horizontal,
    palette_for,
)
from shared.prompts import chat_prompt
from behavioral_analysis.src.eval import EvalResult, norm_token

from ..hook import ResidualCache, run_ablated
from ..references import empty_image, isolated_middle_shape


def _items(entities):
    for tup in itertools.permutations(entities, 3):
        middle_shape, middle_color = tup[1]
        for direction in DIRECTIONS:
            target_idx = 0 if direction in ("left", "above") else 2
            target_color = tup[target_idx][1]
            user_text = (f"The color of the shape {PREPOSITION[direction]} "
                         f"the {middle_color.lower()} {middle_shape.lower()} is ")
            yield tup, direction, target_color, user_text


def run(model, processor, model_key: str) -> dict:
    geom = GEOM[model_key]
    palette = palette_for(model_key)
    entities = list(zip(SHAPE_DRAW_FNS.keys(), palette.keys()))
    result = EvalResult(model=model.model_name, task="shapes")
    items = list(_items(entities))
    print(f"Shapes (ablated): {len(items)} items; canvas {geom.image_px}px; "
          f"entities={entities}")

    cache = ResidualCache(model, processor, model_key)

    empty_img = empty_image(model_key)
    iso_imgs = {ent: isolated_middle_shape(ent[0], ent[1],
                                            model_key=model_key, palette=palette)
                for ent in entities}

    seen_prompts: set[str] = set()
    for i, (tup, direction, target, user_text) in enumerate(items):
        prompt = chat_prompt(processor, user_text)
        image = three_shape_image(
            tup, horizontal=is_horizontal(direction),
            num_tokens=geom.num_tokens, token_size=geom.token_size,
            entity_size=geom.entity_size, spacing=geom.spacing,
            color_rgb=palette,
        )

        if prompt not in seen_prompts:
            cache.populate(("empty", prompt), prompt, empty_img)
            seen_prompts.add(prompt)
        for ent in tup:
            cache.populate((ent, prompt), prompt, iso_imgs[ent])

        pred, prob = run_ablated(
            model, processor,
            model_key=model_key,
            prompt=prompt, image=image, direction=direction,
            entity_ref_keys=[(ent, prompt) for ent in tup],
            empty_ref_key=("empty", prompt),
            cache=cache,
        )
        ok = norm_token(pred) == norm_token(target)
        result.record(direction, ok,
                      tup=tup, target=target, pred=pred.strip(), prob=prob)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)}  cache size={len(cache._cache)}")

    print(result.summary_text())
    return result.to_dict()
