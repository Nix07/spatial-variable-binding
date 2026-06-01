"""The criss-cross swap-patching loop.

For one ``(prompt, image)`` clean–counterfactual pair, sweep every LM-layer
hook point (resid_pre of layer 0 + resid_post of every layer) and patch the
clean run's residual stream from the cf run's cache at *swapped* image-token
positions:

    clean's left-region positions  ←  cf cache @ right-region positions
    clean's right-region positions ←  cf cache @ left-region positions

(For ``above``/``below`` queries, swap top↔bottom instead.)

Each layer slot returns probabilities over the three candidate colors
``[c0, c1, c2]`` at the final-token position; ``logit_prob`` sums the four
spelling variants per color, exactly like ``last_token_exp``.

This module is direction-agnostic — the caller passes the swap pair
(``clean_a / clean_b`` and matching ``cf_b / cf_a``) so the same routine
handles object patching (Fig 6) and strip patching (Fig 7), as well as both
horizontal and vertical orientations.
"""

from __future__ import annotations

import gc
from functools import partial
from typing import List, Sequence

import torch

from shared.component import Component               # parent repo
from shared.metrics import logit_prob                 # parent repo
from shared.vision_language_prompts import VLPrompt   # parent repo


def _prob_label_tensor(model, prob_strs: List[str]) -> torch.Tensor:
    """For each color, gather token IDs for {lower, Capitalize, ' '+lower,
    ' '+Capitalize}. Shape: ``(n_colors, n_token_variants)``."""
    rows = [
        model.to_tokens(
            [p.lower(), p.capitalize(), " " + p.lower(), " " + p.capitalize()],
            prepend_bos=False,
        ).reshape(1, -1)
        for p in prob_strs
    ]
    return torch.cat(rows, dim=0)


@torch.no_grad()
def _patch_single_component(model,
                            vl_prompts: List[VLPrompt],
                            component: Component,
                            clean_positions: Sequence[int],
                            cf_positions: Sequence[int],
                            batch_size: int) -> torch.Tensor:
    """Patch one component once per sample. Returns ``(n_prompts, n_colors)``
    of per-color final-token probability."""
    n_prompts = len(vl_prompts)
    n_colors = len(vl_prompts[0].prob_strs)
    out = torch.zeros(n_prompts, n_colors, dtype=torch.float32)
    hook_name = component.valid_hook_name()

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

        # Only need the one hook we're about to patch — caching every hook
        # blows up memory (162+ tensors per batch for a 27-layer model).
        _, cf_cache = model.run_with_cache(
            cf_prompts, cf_images, return_type="logits",
            names_filter=[hook_name],
        )

        def swap_hook(value, hook, *, cache):
            # clean@clean_positions <- cf_cache@cf_positions
            value[:, clean_positions, :] = cache[hook.name][:, cf_positions, :].to(value.device)
            return value

        patched = model.run_with_hooks(
            clean_prompts, clean_images,
            fwd_hooks=[(hook_name, partial(swap_hook, cache=cf_cache))],
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


def criss_cross_sweep(model,
                      vl_prompts: List[VLPrompt],
                      *,
                      clean_positions: Sequence[int],
                      cf_positions: Sequence[int],
                      batch_size: int) -> torch.Tensor:
    """Per-layer slot sweep. Returns ``(n_samples, n_layers + 1, 1, n_colors)``.

    Slot 0 = ``resid_pre`` of layer 0 (post-embed, where vision tokens enter
    the LM); slots ``1..L`` = ``resid_post`` of LM layer ``slot - 1``.
    """
    n_layers = model.cfg.n_layers
    n_slots = n_layers + 1
    n_prompts = len(vl_prompts)
    n_colors = len(vl_prompts[0].prob_strs)

    results = torch.zeros(n_prompts, n_slots, 1, n_colors, dtype=torch.float32)
    for slot in range(n_slots):
        comp = (Component("resid_pre", layer=0) if slot == 0
                else Component("resid_post", layer=slot - 1))
        results[:, slot, 0, :] = _patch_single_component(
            model, vl_prompts, comp,
            clean_positions=clean_positions,
            cf_positions=cf_positions,
            batch_size=batch_size,
        )
        print(f"    layer slot {slot}/{n_slots - 1} done")
    return results
