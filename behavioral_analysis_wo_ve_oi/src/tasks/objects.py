"""Objects task under the Table 5 ordering ablation.

Same items as ``behavioral_analysis.src.tasks.objects``. The
reference entity per slot is one stickpng thumbnail placed alone at the
canvas middle.
"""

from __future__ import annotations

import itertools
from pathlib import Path

from shared.experiment_config import GEOM
from shared.images import three_object_image
from shared.objects import OBJECT_NAMES, load_thumbnail
from shared.palette import DIRECTIONS, PREPOSITION, is_horizontal
from shared.prompts import chat_prompt
from behavioral_analysis.src.eval import EvalResult, norm_token

from ..hook import ResidualCache, run_ablated
from ..references import empty_image, isolated_middle_object


_THIS = Path(__file__).resolve().parent
_CACHE = _THIS.parents[2] / "shared" / "assets" / "objects"


def _items():
    for tup in itertools.permutations(OBJECT_NAMES, 3):
        middle = tup[1]
        for direction in DIRECTIONS:
            target = tup[0] if direction in ("left", "above") else tup[2]
            user_text = f"The object {PREPOSITION[direction]} the {middle} is a(n) "
            yield tup, direction, target, user_text


def run(model, processor, model_key: str) -> dict:
    geom = GEOM[model_key]
    result = EvalResult(model=model.model_name, task="objects")
    items = list(_items())
    thumb_px = geom.entity_size * geom.token_size
    print(f"Objects (ablated): {len(items)} items; canvas {geom.image_px}px")

    cache = ResidualCache(model, processor, model_key)

    empty_img = empty_image(model_key)
    iso_imgs = {name: isolated_middle_object(name, model_key=model_key, asset_dir=_CACHE)
                for name in OBJECT_NAMES}

    seen_prompts: set[str] = set()
    for i, (tup, direction, target, user_text) in enumerate(items):
        prompt = chat_prompt(processor, user_text)
        thumbs = [load_thumbnail(name, cache_dir=_CACHE).resize((thumb_px, thumb_px))
                  for name in tup]
        image = three_object_image(
            thumbs, horizontal=is_horizontal(direction),
            num_tokens=geom.num_tokens, token_size=geom.token_size,
            entity_size=geom.entity_size, spacing=geom.spacing,
        )

        if prompt not in seen_prompts:
            cache.populate(("empty", prompt), prompt, empty_img)
            seen_prompts.add(prompt)
        for name in tup:
            cache.populate((name, prompt), prompt, iso_imgs[name])

        pred, prob = run_ablated(
            model, processor,
            model_key=model_key,
            prompt=prompt, image=image, direction=direction,
            entity_ref_keys=[(name, prompt) for name in tup],
            empty_ref_key=("empty", prompt),
            cache=cache,
        )
        ok = norm_token(pred) == norm_token(target)
        result.record(direction, ok,
                      tup=tup, target=target, pred=pred.strip(), prob=prob)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)}  cache size={len(cache._cache)}")

    print(result.summary_text())
    return result.to_dict()
