"""Shared helpers for the WhatsUp controlled-images subset.

Used by both ``behavioral_analysis/src/tasks/whatsup.py`` (floor-prompt eval)
and ``probing/src/whatsup.py`` (bbox-based position probe).

Public:
- ``DEFAULT_IMAGES_DIR`` — ``shared/whatsup_images/controlled_images``.
- ``stitch_horizontal(side_l, middle, side_r, *, size, images_dir)`` — build
  one composite image: left half of ``<side_l>_left_of_<middle>.jpeg`` +
  right half of ``<side_r>_right_of_<middle>.jpeg``, both resized to ``size×size``.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# The controlled-images subset lives next to this module.
DEFAULT_IMAGES_DIR = Path(__file__).resolve().parent / "whatsup_images" / "controlled_images"


def _load_resized(path: Path, size: int) -> Image.Image:
    """Resample with BILINEAR to match the floor-baseline reference
    (which used `torchvision.transforms.Resize` whose default mode is bilinear)."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return img


def stitch_horizontal(side_l: str, middle: str, side_r: str,
                      *, size: int,
                      images_dir: Path = DEFAULT_IMAGES_DIR) -> Image.Image:
    """One composite image: left half of ``<side_l>_left_of_<middle>.jpeg`` +
    right half of ``<side_r>_right_of_<middle>.jpeg``, both resized to
    ``size × size`` first, then stitched at the midline."""
    left_full = _load_resized(images_dir / f"{side_l}_left_of_{middle}.jpeg", size)
    right_full = _load_resized(images_dir / f"{side_r}_right_of_{middle}.jpeg", size)
    out = Image.new("RGB", (size, size))
    out.paste(left_full.crop((0, 0, size // 2, size)), (0, 0))
    out.paste(right_full.crop((size // 2, 0, size, size)), (size // 2, 0))
    return out
