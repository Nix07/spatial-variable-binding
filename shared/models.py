"""Unified VLM loader for the three Table-1 / Figure-3 models.

Accepts any of the slug naming conventions used across the codebase:

  - short:       "qwen2vl"           "gemma3"          "pixtral"
  - versioned:   "qwen2-vl-7b"       "gemma-3-4b-it"   "pixtral-12b"
  - full HF id:  "Qwen/Qwen2-VL-7B-Instruct" / "google/gemma-3-4b-it" /
                 "mistral-community/pixtral-12b"

and dispatches to `analysis_utils.load_model()`, plus the post-load patches
required on `transformers>=4.55`:

  - Gemma3:  alias `inner_model.language_model.lm_head = inner_model.lm_head`
             so the gemma3 weight converter finds it (handled inside
             `analysis_utils.load_model`).
  - Pixtral: disable processor truncation (default ~2k joint tokens) and
             raise `model.cfg.n_ctx`. Inert on short sequences, so always
             safe to apply.
"""

from __future__ import annotations

from typing import Any

import torch

from . import paths

paths.setup()  # ensure analysis_utils + TLens are importable


HF_ID = {
    "qwen2vl": "Qwen/Qwen2-VL-7B-Instruct",
    "gemma3":  "google/gemma-3-4b-it",
    "pixtral": "mistral-community/pixtral-12b",
}


# Per-model default precision.
#
# Qwen2-VL and Gemma3 use fp32 (`analysis_utils.load_model` defaults to
# fp32) — the §5.3.1 / Table 5 evaluation under the vision-encoder ordering
# ablation is noticeably precision-sensitive at Gemma's borderline logits,
# so fp32 is what the canonical results were generated under.
#
# Pixtral stays bf16. The bf16-vs-fp32 ablation on Pixtral behavioral
# baselines (Squares/Shapes/Objects/WhatsUp) was essentially noise on
# three tasks and a 0.023 gap on Shapes:
#     Squares 0.550 → 0.550 (Δ 0)
#     Shapes  0.546 → 0.523 (Δ −0.023; both within ±0.04 of paper 0.55)
#     Objects 0.630 → 0.627 (Δ −0.003)
#     WhatsUp 0.436 → 0.438 (Δ +0.002)
# Shapes per-direction under bf16: (26, 64, 80, 81) vs fp32 (30, 68, 81, 83) —
# ~11/480 borderline-logit drift, fully reproducible. Pixtral keeps bf16
# for the memory/speed savings on the 12B model. Override per-call with
# the `torch_dtype` kwarg on `load()` for precision ablations.
DEFAULT_DTYPE: dict[str, torch.dtype] = {
    "qwen2vl": torch.float32,
    "gemma3":  torch.float32,
    "pixtral": torch.bfloat16,
}

# Accepted slug aliases → canonical short slug.
_SLUG_ALIASES = {
    "qwen2vl":                       "qwen2vl",
    "qwen2-vl-7b":                   "qwen2vl",
    "qwen2-vl-7b-instruct":          "qwen2vl",
    "qwen/qwen2-vl-7b-instruct":     "qwen2vl",
    "gemma3":                        "gemma3",
    "gemma-3-4b-it":                 "gemma3",
    "google/gemma-3-4b-it":          "gemma3",
    "pixtral":                       "pixtral",
    "pixtral-12b":                   "pixtral",
    "mistral-community/pixtral-12b": "pixtral",
}


def canonical_slug(name: str) -> str:
    """Map any accepted form of a model name to the canonical short slug.

    Raises ValueError on an unknown name.
    """
    key = name.lower()
    if key not in _SLUG_ALIASES:
        raise ValueError(
            f"unknown model {name!r}; accepted: {sorted(set(_SLUG_ALIASES))}"
        )
    return _SLUG_ALIASES[key]


def load(
    model_name: str,
    *,
    device: str = "cuda",
    device_list: list[int] | None = None,
    cache_dir: str | None = None,
    torch_dtype: torch.dtype | None = None,
    extra_hooks: bool = False,
) -> tuple[Any, Any]:
    """Return `(model, processor)` with all known patches applied.

    `model` is a `HookedVLTransformer`; `processor` is the HF processor.
    `torch_dtype=None` (default) selects each model's `DEFAULT_DTYPE`
    (Qwen/Gemma → fp32, Pixtral → bf16); pass an explicit dtype to override.
    """
    from .analysis_utils import load_model

    slug = canonical_slug(model_name)
    hf_id = HF_ID[slug]
    if torch_dtype is None:
        torch_dtype = DEFAULT_DTYPE[slug]
    model, processor = load_model(
        hf_id, hf_id,
        device=device,
        device_list=device_list,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        extra_hooks=extra_hooks,
    )

    if slug == "pixtral":
        if processor.tokenizer is not None:
            processor.tokenizer.model_max_length = 1_000_000
        model.cfg.n_ctx = max(model.cfg.n_ctx, 100_000)
        _patch_pixtral(model, torch_dtype)

    model.eval()
    model.model_name = hf_id
    return model, processor


def _patch_pixtral(model, dtype: torch.dtype) -> None:
    """Two patches required for Pixtral at non-fp32:

    1. `input_to_embed` returns nested list/tuple pixel_values; unwrap them
       and add a batch dim if missing so the vision encoder can consume them.
    2. `PixtralVisionModel.forward` receives fp32 pixel_values from the
       processor even when the rest of the model is bf16; cast on entry.
    """
    if dtype == torch.float32:
        return  # patches are inert at fp32; skip to keep things minimal

    def _normalize(pixel_values):
        while isinstance(pixel_values, (list, tuple)):
            pixel_values = pixel_values[0]
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        return pixel_values

    orig_input_to_embed = model.input_to_embed

    def patched_input_to_embed(*args, **kwargs):
        out = orig_input_to_embed(*args, **kwargs)
        if isinstance(out, tuple) and len(out) == 9:
            head, pixel_values, tail = out[:2], out[2], out[3:]
            return (*head, _normalize(pixel_values), *tail)
        return out

    model.input_to_embed = patched_input_to_embed

    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel
    orig_forward = PixtralVisionModel.forward

    def patched_forward(self, pixel_values, *args, **kwargs):
        target = self.patch_conv.weight.dtype
        if pixel_values.dtype != target:
            pixel_values = pixel_values.to(dtype=target)
        return orig_forward(self, pixel_values, *args, **kwargs)

    PixtralVisionModel.forward = patched_forward
