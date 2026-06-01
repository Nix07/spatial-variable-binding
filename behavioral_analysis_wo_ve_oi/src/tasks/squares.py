"""Squares task under the Table 5 ordering ablation.

Same items + prompts as ``behavioral_analysis.src.tasks.squares``,
but every clean forward is run through the substitution hook that re-encodes
each colored square as if it had been shown alone at the middle of the
canvas, and every background patch as if it came from an all-white image.

The eval metric (top-1 next-token decode) is unchanged.
"""

from __future__ import annotations

import itertools

from shared.experiment_config import GEOM
from shared.images import three_rect_image
from shared.palette import (
    DIRECTIONS,
    PREPOSITION,
    is_horizontal,
    palette_for,
)
from shared.prompts import chat_prompt
from behavioral_analysis.src.eval import EvalResult, norm_token

from ..hook import ResidualCache, run_ablated
from ..references import empty_image, isolated_middle_square


def _items(color_names):
    """Same iteration order as behavioral_analysis.squares — 480 items."""
    for tup in itertools.permutations(color_names, 3):
        middle = tup[1]
        for direction in DIRECTIONS:
            target = tup[0] if direction in ("left", "above") else tup[2]
            user_text = (f"The color of the square {PREPOSITION[direction]} "
                         f"the {middle.lower()} square is ")
            yield tup, direction, target, user_text


def run(model, processor, model_key: str) -> dict:
    geom = GEOM[model_key]
    palette = palette_for(model_key)
    result = EvalResult(model=model.model_name, task="squares")
    items = list(_items(tuple(palette.keys())))
    print(f"Squares (ablated): {len(items)} items; canvas {geom.image_px}px; "
          f"palette={list(palette.keys())}")

    cache = ResidualCache(model, processor, model_key)

    # Pre-build reference PIL images once.
    empty_img = empty_image(model_key)
    iso_imgs = {c: isolated_middle_square(c, model_key=model_key, palette=palette)
                for c in palette}

    seen_prompts: set[str] = set()
    for i, (tup, direction, target, user_text) in enumerate(items):
        prompt = chat_prompt(processor, user_text)
        image = three_rect_image(
            tup, horizontal=is_horizontal(direction),
            num_tokens=geom.num_tokens, token_size=geom.token_size,
            entity_height=geom.entity_size, entity_width=geom.entity_size,
            spacing=geom.spacing, color_rgb=palette,
        )

        # Populate cache for this prompt the first time we see it.
        # Each unique prompt needs (empty, color0, color1, color2) cached.
        if prompt not in seen_prompts:
            cache.populate(("empty", prompt), prompt, empty_img)
            for c in tup:
                cache.populate((c, prompt), prompt, iso_imgs[c])
            seen_prompts.add(prompt)
        else:
            # The three entities are colors from `tup`; the prompt's middle
            # color always recurs across items, but tup[0]/tup[2] vary, so
            # populate any new (color, prompt) pairs we haven't seen yet.
            for c in tup:
                cache.populate((c, prompt), prompt, iso_imgs[c])

        pred, prob = run_ablated(
            model, processor,
            model_key=model_key,
            prompt=prompt, image=image, direction=direction,
            entity_ref_keys=[(c, prompt) for c in tup],
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
