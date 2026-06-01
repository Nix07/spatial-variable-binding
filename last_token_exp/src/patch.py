"""Last-token residual-stream patching machinery for paper Figure 3.

For each LM hook point (resid_pre layer 0 + resid_post layer 0..L-1):

  1. Run the counterfactual prompt+image with `run_with_cache` and pull
     the activation at the **final ("is") token** of the chosen hook
     point. `names_filter` restricts the cache to just that one hook —
     the difference between OOM and not on 80 GB H100s for 12B Pixtral.
  2. Run the clean prompt+image with `run_with_hooks` and overwrite the
     same final-token slice with the cf activation captured above.
  3. Score the logits at the final position by `logit_prob` over the
     4 spelling variants of each candidate color.

Returns a `(n_samples, n_layers + 1, 1, n_colors)` array; slot 0 along
the layer axis is `resid_pre` of layer 0 (post-embed), slots `1..L` are
`resid_post` of each LM layer.
"""

from __future__ import annotations

import gc
from functools import partial
from typing import List

from shared.paths import setup as _setup_paths

_setup_paths()  # parent repo on sys.path

import numpy as np    # noqa: E402
import torch          # noqa: E402

from shared.component import Component               # noqa: E402
from shared.metrics import logit_prob                # noqa: E402
from shared.vision_language_prompts import VLPrompt  # noqa: E402


def _prob_label_tensor(model, prob_strs: List[str]) -> torch.Tensor:
    """Per color, gather token IDs for the 4 spelling variants
    `{lower, Capitalize, ' '+lower, ' '+Capitalize}`. Output shape
    `(n_colors, 4)`. logit_prob sums probability across those tokens.

    Every color's variants must tokenize to exactly 1 token (uniform row
    length), or `torch.cat` will raise; `build_cf_pairs.single_token_palette`
    filters the palette to guarantee this.
    """
    rows = [
        model.to_tokens(
            [p.lower(), p.capitalize(), " " + p.lower(), " " + p.capitalize()],
            prepend_bos=False,
        ).reshape(1, -1)
        for p in prob_strs
    ]
    return torch.cat(rows, dim=0)


def _patch_single_component(
    model, vl_prompts: List[VLPrompt],
    component: Component, *, batch_size: int,
) -> torch.Tensor:
    """Patch one component at the last token from cf → clean; return
    per-color summed probability at the final position.
    Shape: `(n_prompts, n_colors)`."""
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
        clean_images = [p.images for p in batch]
        cf_prompts = [p.cf_prompt for p in batch]
        cf_images = [p.cf_images for p in batch]

        # `torch.no_grad` drops the autograd graph (huge activations otherwise
        # retained for backward). `names_filter` restricts the cache to just
        # the one hook actually read by patch_last_token. Together these are
        # what makes the whole thing fit on a single 80 GB H100 for 12B Pixtral.
        with torch.no_grad():
            _, cf_cache = model.run_with_cache(
                cf_prompts, cf_images, return_type="logits",
                names_filter=[hook_name],
            )

            def patch_last_token(value, hook, *, cache):
                value[:, -1, :] = cache[hook.name][:, -1, :].to(value.device)
                return value

            patched = model.run_with_hooks(
                clean_prompts, clean_images,
                fwd_hooks=[(hook_name, partial(patch_last_token, cache=cf_cache))],
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


def patch_all_layers(
    model, vl_prompts: List[VLPrompt], *, batch_size: int,
) -> np.ndarray:
    """Sweep every LM layer (plus the post-embed slot). Returns ndarray
    of shape `(n_samples, n_layers + 1, 1, n_colors)`."""
    n_layers = model.cfg.n_layers
    n_slots = n_layers + 1   # idx 0 = resid_pre layer 0; idx l = resid_post (l-1)
    n_prompts = len(vl_prompts)
    n_colors = len(vl_prompts[0].prob_strs)

    results = torch.zeros(n_prompts, n_slots, 1, n_colors, dtype=torch.float32)
    for idx in range(n_slots):
        component = (
            Component("resid_pre", layer=0) if idx == 0
            else Component("resid_post", layer=idx - 1)
        )
        results[:, idx, 0, :] = _patch_single_component(
            model, vl_prompts, component, batch_size=batch_size,
        )
        print(f"    layer slot {idx}/{n_slots - 1} done")
    return results.numpy()
