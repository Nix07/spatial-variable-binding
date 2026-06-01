"""Criss-cross object patching *under the ordering ablation* (paper §5.3.2).

Combines two existing interventions, applied per layer slot:

1. **Layer-0 ablation** (paper §5.3.1, ``behavioral_analysis_wo_ve_oi.src.hook``) on
   **both** the clean and the cf forwards: every square-token residual is
   replaced by the middle-slot patches of an isolated-middle reference
   image with the same color (so color is preserved, position is erased),
   and every background residual is replaced by the matching patches of
   an all-white reference. Hook point: ``blocks.0.hook_resid_pre``.

2. **Criss-cross object patching** (paper §5.2.2, ``criss_cross_patching``)
   on the *ablated* activations: at each layer slot ``l`` (``resid_pre``
   layer 0 + ``resid_post`` layers 0..L-1), copy cf's residuals at the
   *swapped* square positions over clean's residuals at the (unswapped)
   square positions.

If the LM backbone reconstructs ordering information after the
vision-derived signal has been ablated, swapping clean's square tokens
with cf's swapped square tokens at the right layer should flip the model's
prediction. Figure 9 shows this happens at intermediate layers (11-20).

The runner only handles **object** patching here — the paper text rules
out strip patching for this experiment because the ablation has already
overwritten background tokens, so a strip swap would just shuffle
identical ablated values.
"""

from __future__ import annotations

import gc
from functools import partial
from typing import List, Sequence

import torch

from shared.component import Component               # parent repo
from shared.metrics import logit_prob                 # parent repo
from shared.vision_language_prompts import VLPrompt   # parent repo

from criss_cross_patching.src.patch import _prob_label_tensor
from criss_cross_patching.src.regions import object_regions
from criss_cross_patching.src.tokens import (
    grid_to_lm_positions,
    image_start_offset,
)
from behavioral_analysis_wo_ve_oi.src.hook import (
    HOOK_NAME as ABL_HOOK,
    ResidualCache,
    background_grid_ids,
    middle_grid_ids,
    slot_grid_ids,
)


# ---------------------------------------------------------------------------
# Per-batch ablation hook
# ---------------------------------------------------------------------------

def _ref_lookup_indices(model_key: str, grid_ids: Sequence[int]) -> list[int]:
    """Reference-side rows into a cached ``(n_image_tokens, d_model)`` slice.
    (Pixtral [IMG_BREAK] interleaving is folded in by `grid_to_lm_positions`.)"""
    return grid_to_lm_positions(grid_ids, model_slug=model_key, image_start=0)


def make_batch_ablation_hook(
    *,
    model_key: str,
    direction: str,
    image_start: int,
    item_colors: Sequence[Sequence[str]],
    item_prompts: Sequence[str],
    cache: ResidualCache,
):
    """Build the ordering ablation hook for one batch of (prompt, image) pairs.

    ``item_colors[i]`` is the 3-color tuple that occupies slots 0/1/2 in
    the *i*-th item's image (entry 0 = leftmost / topmost). ``item_prompts[i]``
    is the prompt used to key reference slices in ``cache``.
    """
    bg_grid = background_grid_ids(model_key, direction)
    slot_grids = slot_grid_ids(model_key, direction)
    mid_grid = middle_grid_ids(model_key)

    bg_lm   = grid_to_lm_positions(bg_grid,   model_slug=model_key, image_start=image_start)
    slot_lm = [grid_to_lm_positions(g, model_slug=model_key, image_start=image_start)
               for g in slot_grids]
    mid_ref_rows = _ref_lookup_indices(model_key, mid_grid)
    bg_ref_rows  = _ref_lookup_indices(model_key, bg_grid)

    def hook(value: torch.Tensor, hook):
        for i, (colors, prompt) in enumerate(zip(item_colors, item_prompts)):
            empty_slice = cache.get(("empty", prompt))
            for slot_idx, color in enumerate(colors):
                ref = cache.get((color, prompt))
                value[i, slot_lm[slot_idx], :] = ref[mid_ref_rows, :].to(
                    value.device, dtype=value.dtype,
                )
            value[i, bg_lm, :] = empty_slice[bg_ref_rows, :].to(
                value.device, dtype=value.dtype,
            )
        return value

    return hook


# ---------------------------------------------------------------------------
# Patching sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def _patch_single_component_ablated(
    model,
    vl_prompts: List[VLPrompt],
    component: Component,
    *,
    model_key: str,
    direction: str,
    cache: ResidualCache,
    clean_image_start: int,
    cf_image_start: int,
    clean_colors_per_item: Sequence[Sequence[str]],
    cf_colors_per_item: Sequence[Sequence[str]],
    prompts_per_item: Sequence[str],
    clean_positions: Sequence[int],
    cf_positions: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    """Patch one component once per sample under the ordering ablation.

    Mirrors ``criss_cross_patching.src.patch._patch_single_component`` but
    composes the ablation hook into both the cf cache forward and the
    clean patched forward.
    """
    n_prompts = len(vl_prompts)
    n_colors = len(vl_prompts[0].prob_strs)
    out = torch.zeros(n_prompts, n_colors, dtype=torch.float32)
    swap_hook_name = component.valid_hook_name()

    prob_labels = torch.stack(
        [_prob_label_tensor(model, p.prob_strs) for p in vl_prompts],
        dim=0,
    )

    for start in range(0, n_prompts, batch_size):
        end = min(start + batch_size, n_prompts)
        batch = vl_prompts[start:end]
        clean_prompts = [p.prompt for p in batch]
        clean_images  = [p.images for p in batch]
        cf_prompts    = [p.cf_prompt for p in batch]
        cf_images     = [p.cf_images for p in batch]

        clean_abl = make_batch_ablation_hook(
            model_key=model_key, direction=direction,
            image_start=clean_image_start,
            item_colors=clean_colors_per_item[start:end],
            item_prompts=prompts_per_item[start:end],
            cache=cache,
        )
        cf_abl = make_batch_ablation_hook(
            model_key=model_key, direction=direction,
            image_start=cf_image_start,
            item_colors=cf_colors_per_item[start:end],
            item_prompts=prompts_per_item[start:end],
            cache=cache,
        )

        # cf forward with ablation, caching only the swap hook
        with model.hooks(fwd_hooks=[(ABL_HOOK, cf_abl)]):
            _, cf_cache = model.run_with_cache(
                cf_prompts, cf_images, return_type="logits",
                names_filter=[swap_hook_name],
            )

        def swap_hook(value, hook, *, cache):
            value[:, clean_positions, :] = cache[hook.name][:, cf_positions, :].to(value.device)
            return value

        # clean forward: ablation at layer-0, swap at swap_hook_name. When the
        # swap target IS layer-0 resid_pre, both hooks attach to the same name
        # and TLens applies them in the listed order (ablation first, then swap).
        fwd_hooks = [(ABL_HOOK, clean_abl)]
        fwd_hooks.append((swap_hook_name, partial(swap_hook, cache=cf_cache)))

        patched = model.run_with_hooks(
            clean_prompts, clean_images,
            fwd_hooks=fwd_hooks,
            return_type="logits",
        )
        out[start:end] = logit_prob(
            patched[:, -1].cpu(),
            prob_labels[start:end].cpu(),
        )

        del cf_cache, patched
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def criss_cross_sweep_under_ablation(
    model,
    vl_prompts: List[VLPrompt],
    *,
    model_key: str,
    direction: str,
    cache: ResidualCache,
    clean_image_start: int,
    cf_image_start: int,
    clean_colors_per_item: Sequence[Sequence[str]],
    cf_colors_per_item: Sequence[Sequence[str]],
    prompts_per_item: Sequence[str],
    clean_positions: Sequence[int],
    cf_positions: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    """Layer-by-layer object-patching sweep under ordering ablation.

    Returns shape ``(n_samples, n_layers + 1, 1, n_colors)``. Slot 0 is
    ``resid_pre`` of layer 0; slots ``1..L`` are ``resid_post`` of layer
    ``slot - 1``.
    """
    n_layers = model.cfg.n_layers
    n_slots = n_layers + 1
    n_prompts = len(vl_prompts)
    n_colors = len(vl_prompts[0].prob_strs)

    results = torch.zeros(n_prompts, n_slots, 1, n_colors, dtype=torch.float32)
    for slot in range(n_slots):
        comp = (Component("resid_pre", layer=0) if slot == 0
                else Component("resid_post", layer=slot - 1))
        results[:, slot, 0, :] = _patch_single_component_ablated(
            model, vl_prompts, comp,
            model_key=model_key, direction=direction, cache=cache,
            clean_image_start=clean_image_start, cf_image_start=cf_image_start,
            clean_colors_per_item=clean_colors_per_item,
            cf_colors_per_item=cf_colors_per_item,
            prompts_per_item=prompts_per_item,
            clean_positions=clean_positions, cf_positions=cf_positions,
            batch_size=batch_size,
        )
        print(f"    layer slot {slot}/{n_slots - 1} done")
    return results


# ---------------------------------------------------------------------------
# Position resolution
# ---------------------------------------------------------------------------

def object_swap_positions(
    model_key: str, direction: str,
    *, clean_image_start: int, cf_image_start: int,
) -> tuple[list[int], list[int]]:
    """LM-stream positions for the criss-cross object swap.

    Returns ``(clean_positions, cf_positions)`` such that index ``i`` says:
    "replace clean's residual at ``clean_positions[i]`` with cf's residual
    at ``cf_positions[i]``." For horizontal queries the swap is L↔R; for
    vertical, T↔B.
    """
    regions = object_regions(model_key)
    if direction in ("left", "right"):
        a_grid, b_grid = regions["left"], regions["right"]
    else:
        a_grid, b_grid = regions["top"], regions["bottom"]

    a_lm_clean = grid_to_lm_positions(a_grid, model_slug=model_key, image_start=clean_image_start)
    b_lm_clean = grid_to_lm_positions(b_grid, model_slug=model_key, image_start=clean_image_start)
    a_lm_cf    = grid_to_lm_positions(a_grid, model_slug=model_key, image_start=cf_image_start)
    b_lm_cf    = grid_to_lm_positions(b_grid, model_slug=model_key, image_start=cf_image_start)

    # clean's A and B slots receive cf's B and A (swapped).
    clean_positions = a_lm_clean + b_lm_clean
    cf_positions    = b_lm_cf    + a_lm_cf
    return clean_positions, cf_positions


def resolve_image_starts(model, processor, vl_prompts: List[VLPrompt]
                         ) -> tuple[int, int]:
    """Find the LM-input image-start offset for the clean and cf prompts.

    Within a direction all clean prompts tokenize to the same length (the
    middle color is the only varying token in the canonical palette), so
    one sample suffices for both clean and cf.
    """
    sample = vl_prompts[0]
    clean_start = image_start_offset(model, processor, sample.prompt, sample.images[0])
    cf_start    = image_start_offset(model, processor, sample.cf_prompt, sample.cf_images[0])
    return clean_start, cf_start
