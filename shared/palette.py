"""Shared color / direction / preposition vocab for Squares-style experiments."""

from __future__ import annotations

from typing import Mapping

# Canonical color names used across Squares / Shapes evals.
COLOR_NAMES_SIX = ("Red", "Green", "Blue", "Black", "Brown", "Gray")

# RGB palette for the Squares / Shapes settings (Qwen and Gemma).
COLOR_RGB_DEFAULT: Mapping[str, tuple[int, int, int]] = {
    "Red":   (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue":  (0, 0, 255),
    "Black": (0, 0, 0),
    "Brown": (150, 75, 0),
    "Gray":  (128, 128, 128),
}

# Pixtral substitutes Orange for Brown — at small entity sizes Pixtral
# classifies the thin Cross sprite as red when the ground-truth color
# is Brown, so we use a more saturated warm hue.
COLOR_RGB_PIXTRAL: Mapping[str, tuple[int, int, int]] = {
    "Red":    (255, 0, 0),
    "Green":  (0, 255, 0),
    "Blue":   (0, 0, 255),
    "Black":  (0, 0, 0),
    "Orange": (255, 165, 0),
    "Gray":   (128, 128, 128),
}


def palette_for(model_key: str) -> Mapping[str, tuple[int, int, int]]:
    """Return the per-model color palette used by Squares + Shapes."""
    return COLOR_RGB_PIXTRAL if model_key == "pixtral" else COLOR_RGB_DEFAULT

# Cardinal spatial relations the paper evaluates.
DIRECTIONS = ("left", "right", "above", "below")

# Direction → preposition-phrase used inside prompt templates.
PREPOSITION = {
    "left":  "to the left of",
    "right": "to the right of",
    "above": "above",
    "below": "below",
}

# Direction → its 180° flip (used in counterfactual constructions).
OPPOSITE = {
    "left":  "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


def is_horizontal(direction: str) -> bool:
    """Whether `direction` describes a left/right (horizontal) relation."""
    return direction in ("left", "right")
