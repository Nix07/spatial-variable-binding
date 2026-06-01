"""Per-task modules for Table 1 behavioral analysis.

Each task exposes:
    run(model, processor, model_key: str) -> dict

returning a result dict with at least `overall_acc`, `per_direction_acc`,
`total_correct`, `total_examples`, and a per-tuple `records` list.
"""

from . import squares, shapes, objects, whatsup  # noqa: F401

REGISTRY = {
    "squares": squares,
    "shapes":  shapes,
    "objects": objects,
    "whatsup": whatsup,
}
