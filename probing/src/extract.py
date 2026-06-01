"""Post-projection patch embeddings: one (N_patches, D) flat tensor per image.

The probing experiment trains on embeddings extracted **immediately after the
multimodal projector** — the same place the LM backbone consumes them. Each
model exposes this slightly differently:

- **Qwen2-VL** — its `vision_model` is the merged vision+projector tower and
  already returns the flat (N_patches_merged, D_text) tensor.
- **Gemma-3** and **Pixtral** — `vision_model` returns
  `(1, N_patches, D_vision)` and `multi_modal_projector` does the projection
  to `D_text`.

The flat result is laid out in row-major patch order on a
`(num_tokens × num_tokens)` grid (see `config.GEOM`), so flat index
`r * num_tokens + c` is the patch at (row=r, col=c).

For Pixtral specifically: the vision encoder's flat output is contiguous on
its own grid; the non-contiguous `[IMG_BREAK]` interleaving only happens later
when those embeddings are spliced into the LM token stream. We extract before
that splice, so probing is contiguous-grid for all three models.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from shared import models as _models
from shared.prompts import chat_prompt


_DUMMY_USER_TEXT = "The color of the square to the left of the red square is "


def _build_dummy_inputs(processor, image: Image.Image) -> dict[str, torch.Tensor]:
    """Build a one-image processor batch. The text is irrelevant for vision
    feature extraction, but the processor needs *some* text to render."""
    text = chat_prompt(processor, _DUMMY_USER_TEXT)
    return processor(images=image, text=text, return_tensors="pt")


@torch.no_grad()
def patch_embeddings(model: Any, processor: Any, image: Image.Image) -> torch.Tensor:
    """Return the projected patch embeddings for one image.

    Shape: `(N_patches, D)` on CPU. N_patches must equal `num_tokens**2` for
    the model under test (see `config.GEOM`); the caller asserts this.
    """
    slug = _models.canonical_slug(model.model_name)
    inputs = _build_dummy_inputs(processor, image)

    if slug == "qwen2vl":
        pv = inputs["pixel_values"].to(model.vision_model.device)
        grid_thw = inputs["image_grid_thw"].to(model.vision_model.device)
        out = model.vision_model(pv, grid_thw=grid_thw)
        # Qwen's vision tower already returns (N_merged, D_text) flat.
        return out.detach().to(torch.float32).cpu()

    # Gemma3 + Pixtral share the same Llava-style two-step extraction.
    pv = inputs["pixel_values"]
    while isinstance(pv, (list, tuple)):
        pv = pv[0]                    # Pixtral wraps in list-of-list
    if pv.ndim == 3:
        pv = pv.unsqueeze(0)
    pv = pv.to(model.vision_model.device,
               dtype=next(model.vision_model.parameters()).dtype)
    hs = model.vision_model(pv).last_hidden_state         # (1, N, D_v)
    proj = model.multi_modal_projector(hs)                # (1, N, D_t)
    return proj[0].detach().to(torch.float32).cpu()
