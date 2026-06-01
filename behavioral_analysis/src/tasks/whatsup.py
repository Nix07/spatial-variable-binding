"""What'sUp task: stitched composite images from the What'sUp control subset.

Each stitched image places a target object on one side of a reference (chair /
table / armchair) and a distractor on the other side. We score whether the
model's top-1 next-token after the floor-prompt matches the target object name.

Filtering: we restrict to the subset of side-objects whose name tokenizes to a
single token under the chosen model's tokenizer (so the next-token target is
unambiguous). 3 reference objects × ~P(N,2) side-pairs × 2 directions.

Prompt template (matches the paper's Fig. 1 family):
  "The object on the floor {prep} the {middle_object} is a(n) "

Image data lives at `shared/whatsup_images/controlled_images/` — pre-rendered
JPEGs from the What'sUp benchmark.

Per-model image size is each model's native vision input size, used for the
stitched composite.
"""

from __future__ import annotations

import itertools

from shared.palette import PREPOSITION
from shared.prompts import chat_prompt
from shared.whatsup import stitch_horizontal

from ..eval import EvalResult, norm_token, top1_token

# Per-model stitched image size (each model's native vision input size).
WHATSUP_IMAGE_SIZE = {
    "qwen2vl": 336,
    "gemma3":  896,
    "pixtral": 512,   # pixtral_whatsup_floor_baseline_runner.py
}

# 3 reference (middle) objects + the candidate side-object pool for each.
# (Side objects are filtered down to single-token strings under each tokenizer
# at runtime.) Same lists as the floor-baseline reference.
MIDDLE_OBJECTS = {
    "chair": [
        "bottle-of-orange-juice", "oven-mitt", "toy-cactus", "plate", "headphones", "pot",
        "water-bottle", "box", "plant", "banana", "mug", "blanket", "beer-bottle", "book",
        "cap", "orange", "bottle", "laptop", "knife", "scissors", "cup", "pan",
        "bottle-of-mustard", "pillow", "vase", "kettle", "bowl", "sunglasses", "water-filter",
        "lemon", "hat", "toilet-roll", "dragonfruit", "remote", "spatula", "painting",
        "helmet", "photograph", "shoes", "microphone", "scarf", "wineglass", "can",
        "ball-of-yarn",
    ],
    "table": [
        "kettle", "bowl", "dog", "sunglasses", "lamp", "chair", "toilet-roll", "dragonfruit",
        "spatula", "lemon", "helmet", "painting", "scarf", "wineglass", "ball-of-yarn",
        "plate", "plant", "banana", "chair2", "mug", "book", "phone", "orange", "cap",
        "bottle", "scissors", "cup", "pillow", "knife",
    ],
    "armchair": [
        "yarn", "water-bottle", "pan", "orange", "scarf", "cap", "bowl", "apple", "mug",
        "book", "soda-can", "beer-bottle", "glass", "cup", "glove", "tape", "bread",
        "bottle", "wineglass", "glasses", "can", "rose", "phone", "notepad", "plate",
        "headphones", "banjo", "remote", "sunglasses", "pillow", "laptop",
    ],
}


def _one_token_subset(processor, names) -> list[str]:
    """Keep only names that tokenize to a single token (in both lowercase and capitalized form)."""
    keep = []
    tok = processor.tokenizer
    for n in names:
        if len(tok.tokenize(n)) == 1 and len(tok.tokenize(n.capitalize())) == 1:
            keep.append(n)
    return keep


def run(model, processor, model_key: str, size: int | None = None) -> dict:
    if size is None:
        size = WHATSUP_IMAGE_SIZE[model_key]
    result = EvalResult(model=model.model_name, task="whatsup")
    one_token = {m: _one_token_subset(processor, names) for m, names in MIDDLE_OBJECTS.items()}
    print(f"What'sUp: stitch_size={size}px (model {model_key} native)")
    for middle, names in one_token.items():
        print(f"  {middle}: {len(names)} single-token side objects")

    total_planned = sum(len(list(itertools.permutations(names, 2))) * 2
                        for names in one_token.values())
    print(f"What'sUp: ~{total_planned} forward passes")

    i = 0
    for middle, names in one_token.items():
        user_text_by_dir = {
            d: f"The object on the floor {PREPOSITION[d]} the {middle.lower()} is a(n) "
            for d in ("left", "right")
        }
        for tup in itertools.permutations(names, 2):
            target, distractor = tup[0], tup[1]
            for direction in ("left", "right"):
                # Floor-baseline matches the Pixtral floor-baseline reference
                # (`pixtral_whatsup_floor_baseline_runner.py` @ main, the
                # "You respond in one token." system message). Without it,
                # Qwen / Gemma emit verbose preambles ("The…", "Based…") and
                # the top-1 next-token compare always fails.
                prompt = chat_prompt(processor, user_text_by_dir[direction])
                if direction == "left":
                    image = stitch_horizontal(target, middle, distractor, size=size)
                else:
                    image = stitch_horizontal(distractor, middle, target, size=size)
                pred, prob = top1_token(model, processor, prompt, image)
                ok = norm_token(pred) == norm_token(target)
                result.record(direction, ok,
                              middle=middle, target=target, distractor=distractor,
                              pred=pred.strip(), prob=prob)
                i += 1
                if i % 100 == 0:
                    print(f"  {i}/{total_planned}")

    print(result.summary_text())
    return result.to_dict()
