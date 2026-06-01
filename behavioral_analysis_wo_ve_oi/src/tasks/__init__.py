"""Per-task drivers for the Table 5 ordering-ablation behavioral eval.

Each task module exposes ``run(model, processor, model_key) -> dict`` with
the same result schema as ``behavioral_analysis``.
"""

from . import squares, shapes, objects  # noqa: F401

REGISTRY = {
    "squares": squares,
    "shapes":  shapes,
    "objects": objects,
}
