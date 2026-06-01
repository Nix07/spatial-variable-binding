"""Section 5.2.2 — Causal Intervention on Vision Tokens (paper Figs 6 & 7).

Per (clean, counterfactual) pair, swap the post-projection visual token
embeddings (and per-LM-layer residual stream) between the **left** and
**right** object regions in opposite directions:

    clean's left  ← cf's right
    clean's right ← cf's left

The cf image is the clean image with its left/right colors reversed (so the
colors at each patched region match — only the ordering signal swaps).
Two variants of the swap are run:

- **Object patching** (Fig 6): swap only the patches inside the squares
  themselves. Does NOT reliably flip the model's output.
- **Strip patching** (Fig 7): swap entire vertical strips (the column of
  patches containing each square, plus all background patches above and
  below). DOES flip the output up to layer ~24.

Together these isolate the result from section 5.2.1: ordering information
is distributed across strip-aligned visual tokens, not localized to objects.

Entry points:
- `python -m criss_cross_patching.run_experiment` — train probes
- `python -m criss_cross_patching.plot` — render Fig 6 / Fig 7
"""
