"""Map "grid-relative patch index" → "absolute LM-input position".

A probe defines patches in terms of where they sit on the model's
``num_tokens × num_tokens`` post-projection grid (flat index ``r * N + c``).
Patching at LM hook points needs absolute positions in the LM token stream,
i.e. grid index + the count of pre-image text tokens that the chat template
prepended. This module turns the former into the latter for each model:

- **Qwen2-VL** & **Gemma3** — image tokens form one contiguous block in the
  LM stream, so the mapping is simply ``grid_idx + image_start_offset``.
- **Pixtral-12B** — patches are interleaved with ``[IMG_BREAK]`` separators
  (one per row), so the LM-position for grid row ``r`` is
  ``image_start + r * (num_cols + 1) + c``.

Geometry per model is reused from ``probing.src.config.GEOM``.
"""

from __future__ import annotations

from typing import Sequence

from probing.src.config import GEOM


# Per-model image-token marker(s). The first match in the LM input is the
# start of the image-token block.
IMAGE_TOKEN_IDS = {
    "qwen2vl": [151655],            # <|image_pad|>
    "gemma3":  [262144],            # <image_soft_token>
    "pixtral": [10, 12],            # [IMG], [IMG_BREAK]
}


def _flat_input_ids(model, processor, prompt: str, image):
    """One-image processor batch's `input_ids` flattened to 1D LongTensor."""
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return inputs["input_ids"][0]


def image_token_count(model_slug: str, processor, prompt: str, image) -> int:
    """LM-stream span of all image-related tokens (incl. Pixtral separators).

    Pixtral interleaves ``[IMG_BREAK]`` between rows and terminates with
    ``[IMG_END]``; Qwen2-VL and Gemma3 emit one contiguous block. Counts
    the maximal stretch of tokens whose IDs match the model's image-token
    markers from the first such token onward.
    """
    if model_slug not in IMAGE_TOKEN_IDS:
        raise ValueError(f"image-token IDs not configured for {model_slug!r}")
    targets = set(IMAGE_TOKEN_IDS[model_slug])
    if model_slug == "pixtral":
        targets |= {13}  # [IMG_END] terminator counts as part of the span
    ids = _flat_input_ids(None, processor, prompt, image)
    in_block = False
    start = end = -1
    for i, t in enumerate(ids.tolist()):
        if t in targets:
            if not in_block:
                in_block = True
                start = i
            end = i
        elif in_block:
            break
    if start < 0:
        raise RuntimeError(f"no image tokens found in {len(ids)}-token input for {model_slug!r}")
    return end - start + 1


def image_start_offset(model, processor, prompt: str, image) -> int:
    """Find the LM-input index of the first image token for one (prompt, image).

    Tokenizes via the model's processor (so chat-template prefixes are
    accounted for) and returns the position of the first occurrence of any
    of the model's image-token IDs.
    """
    from shared import models as _models  # lazy: avoids pulling torch
    slug = _models.canonical_slug(model.model_name)
    if slug not in IMAGE_TOKEN_IDS:
        raise ValueError(f"image-token IDs not configured for {slug!r}")
    targets = set(IMAGE_TOKEN_IDS[slug])
    ids = _flat_input_ids(model, processor, prompt, image)
    for pos, t in enumerate(ids.tolist()):
        if t in targets:
            return pos
    raise RuntimeError(
        f"no image token ({targets}) found in the {len(ids)}-token input "
        f"for model {slug!r} — chat template may have changed"
    )


def grid_to_lm_positions(
    grid_indices: Sequence[int],
    *,
    model_slug: str,
    image_start: int,
) -> list[int]:
    """Lift each grid-relative patch index to its LM-input position.

    For Qwen/Gemma this is just ``grid_idx + image_start``. For Pixtral, each
    grid row of N patches is followed by one ``[IMG_BREAK]`` token, so a
    patch at grid position ``(r, c)`` lives at
    ``image_start + r * (N + 1) + c``.
    """
    geom = GEOM[model_slug]
    N = geom.num_tokens

    if model_slug in ("qwen2vl", "gemma3"):
        return [image_start + i for i in grid_indices]

    if model_slug == "pixtral":
        out = []
        for i in grid_indices:
            r, c = divmod(i, N)
            out.append(image_start + r * (N + 1) + c)
        return out

    raise ValueError(f"unsupported model slug {model_slug!r}")
