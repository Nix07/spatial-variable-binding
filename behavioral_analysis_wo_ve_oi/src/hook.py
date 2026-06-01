"""The Table 5 substitution: replace image-token residuals at layer-0 entry.

For one ``(prompt, image)`` clean run, every image-token position in the
LM residual stream is overwritten with one of two reference signals:

- **Entity slots** (left / middle / right or top / middle / bottom on the
  centered three-entity grid) — replaced with the *middle-slot* patches
  from a per-entity isolated-middle reference image. This re-encodes each
  entity as if it had been shown alone at the middle of the canvas: color
  / shape / identity preserved, ordering signal destroyed.
- **Background** — every other image-token position is replaced with the
  matching patch from an empty (all-white) reference image.

Hook point is ``blocks.0.hook_resid_pre`` — the LM's input residual after
the multimodal projector has folded the post-vision-encoder image tokens
into the LM token stream. This is the deepest point at which the
substitution is still purely "vision-derived" — any LM-side processing
that follows must build off the ablated representation.

Reference residuals are cached by `(prompt_id, reference_image_id)` so a
task with K unique prompts × R unique references runs at most K·R cache
forwards regardless of the number of items.
"""

from __future__ import annotations

from typing import Hashable, Sequence

import torch

from shared.experiment_config import GEOM
from criss_cross_patching.src.tokens import (
    IMAGE_TOKEN_IDS,
    grid_to_lm_positions,
    image_start_offset,
)
from probing.src.images import get_position_token_ids

HOOK_NAME = "blocks.0.hook_resid_pre"


# ---------------------------------------------------------------------------
# Grid-index helpers
# ---------------------------------------------------------------------------

def slot_grid_ids(model_key: str, direction: str) -> list[list[int]]:
    """Per-slot grid indices for the three entity slots, in tuple order.

    Returns ``[slot0, slot1, slot2]`` where slot[k] holds the grid indices
    covered by the k-th entity in the clean image (k=0 leftmost / topmost).
    """
    geom = GEOM[model_key]
    ids = get_position_token_ids(
        num_tokens=geom.num_tokens,
        entity_size=geom.entity_size,
        spacing=geom.spacing,
    )
    if direction in ("left", "right"):
        return [ids["left"], ids["middle"], ids["right"]]
    if direction in ("above", "below"):
        return [ids["top"], ids["middle"], ids["bottom"]]
    raise ValueError(f"unsupported direction {direction!r}")


def middle_grid_ids(model_key: str) -> list[int]:
    """Grid indices the middle slot covers — same for both orientations."""
    geom = GEOM[model_key]
    ids = get_position_token_ids(
        num_tokens=geom.num_tokens,
        entity_size=geom.entity_size,
        spacing=geom.spacing,
    )
    return ids["middle"]


def background_grid_ids(model_key: str, direction: str) -> list[int]:
    """All grid indices NOT covered by any entity slot for the given direction."""
    geom = GEOM[model_key]
    occupied = set().union(*slot_grid_ids(model_key, direction))
    return [i for i in range(geom.num_tokens * geom.num_tokens) if i not in occupied]


def image_token_count(model_key: str, processor, prompt: str, image) -> int:
    """LM-stream span of all image-related tokens (incl. Pixtral separators).

    Pixtral interleaves ``[IMG_BREAK]`` between rows and terminates with
    ``[IMG_END]``; Qwen2-VL and Gemma3 emit one contiguous block. We count
    the maximal stretch of tokens whose IDs match the model's image-token
    markers from the first such token onward.
    """
    if model_key not in IMAGE_TOKEN_IDS:
        raise ValueError(f"image-token IDs not configured for {model_key!r}")
    targets = set(IMAGE_TOKEN_IDS[model_key])
    if model_key == "pixtral":
        targets |= {13}  # [IMG_END] terminator counts as part of the span
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    in_block = False
    start = end = -1
    for i, t in enumerate(ids):
        if t in targets:
            if not in_block:
                in_block = True
                start = i
            end = i
        elif in_block:
            break
    if start < 0:
        raise RuntimeError(f"no image tokens found in {len(ids)}-token input for {model_key!r}")
    return end - start + 1


# ---------------------------------------------------------------------------
# Reference-residual cache
# ---------------------------------------------------------------------------

class ResidualCache:
    """Per-task cache of layer-0 image-token residuals keyed by reference id.

    A reference is a ``(prompt, image)`` pair we forward through the model
    once and slice ``blocks.0.hook_resid_pre`` at the image-token span.
    The result is a tensor of shape ``(N_image_tokens, d_model)``, kept on
    CPU until needed.

    Cache key is provided by the caller (e.g. ``("empty", prompt_hash)`` or
    ``("Red", prompt_hash)``) — we don't try to hash images ourselves.
    """

    def __init__(self, model, processor, model_key: str):
        self.model = model
        self.processor = processor
        self.model_key = model_key
        self._cache: dict[Hashable, torch.Tensor] = {}
        self._meta: dict[Hashable, dict] = {}

    def populate(self, key: Hashable, prompt: str, image) -> torch.Tensor:
        """Cache the image-token residual slice for ``(prompt, image)`` under ``key``.

        Returns the slice (shape ``(n_image_tokens, d_model)``, CPU). Re-runs
        only on miss; idempotent across repeated calls with the same key.
        """
        if key in self._cache:
            return self._cache[key]

        img_start = image_start_offset(self.model, self.processor, prompt, image)
        n_img = image_token_count(self.model_key, self.processor, prompt, image)

        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                prompt, [image],
                names_filter=HOOK_NAME,
                return_type="logits",
            )
        slice_ = cache[HOOK_NAME][0, img_start : img_start + n_img].detach().to("cpu")
        del cache

        self._cache[key] = slice_
        self._meta[key] = {"image_start": img_start, "n_img": n_img}
        return slice_

    def get(self, key: Hashable) -> torch.Tensor:
        """Return the cached slice for ``key``. Caller must have populated it first."""
        return self._cache[key]


# ---------------------------------------------------------------------------
# The ablation hook
# ---------------------------------------------------------------------------

def _grid_to_lm(model_key: str, image_start: int, grid_ids: Sequence[int]) -> list[int]:
    """Thin alias so call sites read more clearly."""
    return grid_to_lm_positions(grid_ids, model_slug=model_key, image_start=image_start)


def _ref_lookup_indices(model_key: str, grid_ids: Sequence[int]) -> list[int]:
    """Reference-side row indices into a cached `(n_image_tokens, d_model)` slice.

    `n_image_tokens` is the full image span (incl. Pixtral [IMG_BREAK] /
    [IMG_END]); we lift grid indices through `grid_to_lm_positions` with
    `image_start=0` so the resulting indices live in `[0, n_image_tokens)`.
    """
    return grid_to_lm_positions(grid_ids, model_slug=model_key, image_start=0)


def make_ablation_hook(
    *,
    model_key: str,
    direction: str,
    clean_image_start: int,
    entity_ref_keys: Sequence[Hashable],
    empty_ref_key: Hashable,
    cache: ResidualCache,
):
    """Build the substitution hook for one clean item.

    Args:
        model_key: canonical model slug (``qwen2vl`` / ``gemma3`` / ``pixtral``).
        direction: ``left`` / ``right`` / ``above`` / ``below``.
        clean_image_start: LM-input position of the first image token in
            the clean (prompt, image) pair.
        entity_ref_keys: 3 cache keys, one per slot in tuple order — each
            keys into the per-entity isolated-middle reference residual.
        empty_ref_key: cache key for the empty-image reference residual.
        cache: pre-populated ``ResidualCache`` for this model.

    Returns:
        A function ``hook(value, hook)`` suitable for ``run_with_hooks``.
    """
    mid_grid = middle_grid_ids(model_key)
    bg_grid = background_grid_ids(model_key, direction)
    slot_grids = slot_grid_ids(model_key, direction)

    # Pull reference slices (caller must have populated them).
    empty_slice = cache.get(empty_ref_key)
    entity_slices = [cache.get(k) for k in entity_ref_keys]
    # Reference-side rows index into the cached `(n_image_tokens, d_model)` slice.
    mid_ref_rows = _ref_lookup_indices(model_key, mid_grid)
    bg_ref_rows = _ref_lookup_indices(model_key, bg_grid)
    # Clean-side LM positions we'll write to.
    bg_clean_lm = _grid_to_lm(model_key, clean_image_start, bg_grid)
    slot_clean_lm = [_grid_to_lm(model_key, clean_image_start, ids) for ids in slot_grids]

    def hook(value: torch.Tensor, hook):
        # 1) Each entity slot ← middle-slot patches from its isolated reference.
        for clean_pos, ref_slice in zip(slot_clean_lm, entity_slices):
            value[:, clean_pos, :] = ref_slice[mid_ref_rows, :].to(value.device, dtype=value.dtype)
        # 2) Background ← matching patches from the empty reference.
        value[:, bg_clean_lm, :] = empty_slice[bg_ref_rows, :].to(value.device, dtype=value.dtype)
        return value

    return hook


def run_ablated(
    model,
    processor,
    *,
    model_key: str,
    prompt: str,
    image,
    direction: str,
    entity_ref_keys: Sequence[Hashable],
    empty_ref_key: Hashable,
    cache: ResidualCache,
) -> tuple[str, float]:
    """One ablated forward, returning ``(decoded_top1, prob)``.

    Caller must have pre-populated ``cache`` with the four reference slices
    referenced by ``entity_ref_keys`` and ``empty_ref_key`` (passing the
    matching ``(prompt, image)`` pairs to ``cache.get`` ahead of time).
    """
    clean_start = image_start_offset(model, processor, prompt, image)
    hook = make_ablation_hook(
        model_key=model_key, direction=direction,
        clean_image_start=clean_start,
        entity_ref_keys=entity_ref_keys, empty_ref_key=empty_ref_key,
        cache=cache,
    )
    with torch.no_grad():
        logits = model.run_with_hooks(
            prompt, [image],
            fwd_hooks=[(HOOK_NAME, hook)],
            return_type="logits",
        )
    probs = torch.softmax(logits[:, -1], dim=-1)
    tok = int(torch.argmax(probs, dim=-1).item())
    return processor.tokenizer.decode([tok]), float(probs[0, tok].item())


# Re-export for convenience.
__all__ = [
    "HOOK_NAME",
    "ResidualCache",
    "background_grid_ids",
    "image_token_count",
    "make_ablation_hook",
    "middle_grid_ids",
    "run_ablated",
    "slot_grid_ids",
]
