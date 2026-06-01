"""Section 5.2.1 — Probing visual embeddings for ordering information.

Reproduces the linear-probe pipeline from the paper: per (model, setting,
orientation), train a 3-class linear probe `\\hat y = Wh + b` on patch
embeddings extracted immediately after the vision encoder's projection, with
labels = ordinal position (left/middle/right or top/middle/bottom). At test
time the probe is also applied to every patch in the image to produce the
"strip" heatmap plots.

Entry points:
- `python -m probing.run_experiment` — train probes
- `python -m probing.plot` — build heatmaps + accuracy table
"""
