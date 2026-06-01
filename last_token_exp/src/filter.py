"""Pre-patching clean+cf top-1 correctness filter.

Patching a baseline the model already gets wrong adds noise without
informing the layer-by-layer story. This module gives `run_experiment.py`
an optional defense-in-depth filter (via `--filter`) and lets
`build_cf_pairs.py` re-use the same per-pair check.
"""

from __future__ import annotations

import re
from typing import List

from shared.paths import setup as _setup_paths

_setup_paths()  # parent repo on sys.path

import torch    # noqa: E402

from shared.vision_language_prompts import VLPrompt    # noqa: E402

_NORM_RE = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    """Lowercase + strip + drop non-alphanumeric (matches behavioral analysis's
    `norm_token`). Two strings compare equal after `norm` iff they're the same
    word modulo casing and adjacent whitespace/punctuation."""
    return _NORM_RE.sub("", s.strip().lower())


def top1_norm(model, prompt: str, images, processor) -> str:
    """Decoded top-1 next-token prediction after `prompt+images`, normalized.

    Uses the `HookedVLTransformer` `model(prompt, images)` path."""
    with torch.no_grad():
        logits = model(prompt, images, return_type="logits")
    tok = int(torch.argmax(logits[:, -1], dim=-1).item())
    return norm(processor.tokenizer.decode([tok]))


def passes_filter(model, processor, vlp: VLPrompt) -> bool:
    """True iff the model's top-1 matches BOTH the clean and the cf answer."""
    clean_ok = top1_norm(model, vlp.prompt, vlp.images, processor) == norm(vlp.answer)
    cf_ok = top1_norm(model, vlp.cf_prompt, vlp.cf_images, processor) == norm(vlp.cf_answer)
    return clean_ok and cf_ok


def filter_correct(model, processor, vl_prompts: List[VLPrompt]) -> List[VLPrompt]:
    """Drop any (clean, cf) pair where the model's top-1 doesn't already match
    the expected answer. Reports the surviving count."""
    kept: List[VLPrompt] = []
    for vlp in vl_prompts:
        if passes_filter(model, processor, vlp):
            kept.append(vlp)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"    pre-filter: {len(kept)}/{len(vl_prompts)} pairs pass clean+cf top-1 check")
    return kept
