"""Stickpng object thumbnail catalog + local cache.

Shared by the Objects setting in both ``behavioral_analysis`` (top-1 prediction
task) and ``probing`` (linear position probe). Each thumbnail is fetched once
to the caller-chosen cache directory and reused on subsequent runs.

Public:
- ``OBJECT_NAMES`` — canonical 8-object list (paper Sec 3.1, App A).
- ``OBJECT_URLS`` — name → stickpng URL.
- ``load_thumbnail(name, *, cache_dir)`` — return a cached RGBA `PIL.Image`.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Mapping
from urllib.request import Request, urlopen

from PIL import Image

OBJECT_URLS: Mapping[str, str] = {
    "orange": "https://assets.stickpng.com/thumbs/580b57fcd9996e24bc43c16d.png",
    "apple":  "https://assets.stickpng.com/thumbs/5a02121818e87004f1ca4379.png",
    "pot":    "https://assets.stickpng.com/thumbs/580b57fcd9996e24bc43c546.png",
    "bell":   "https://assets.stickpng.com/thumbs/589c8acb64b351149f22a866.png",
    "hat":    "https://assets.stickpng.com/thumbs/580b57fbd9996e24bc43bf14.png",
    "rose":   "https://assets.stickpng.com/thumbs/580b585b2edbce24c47b2694.png",
    "key":    "https://assets.stickpng.com/thumbs/580b585b2edbce24c47b2850.png",
    "bomb":   "https://assets.stickpng.com/thumbs/588713cbd27829db9cf6da6a.png",
}

OBJECT_NAMES = tuple(OBJECT_URLS.keys())

_UA = "Mozilla/5.0 (shared object cache)"
_MEMO: dict[tuple[str, str], Image.Image] = {}


def load_thumbnail(name: str, *, cache_dir: Path) -> Image.Image:
    """Return the RGBA thumbnail for `name`, fetching + caching on miss.

    Each process-level call caches the decoded `PIL.Image` in memory keyed by
    `(cache_dir, name)`, so repeated calls in the same run don't re-decode.
    """
    cache_dir = Path(cache_dir)
    memo_key = (str(cache_dir), name)
    if memo_key in _MEMO:
        return _MEMO[memo_key]

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{name}.png"
    if not out.exists():
        if name not in OBJECT_URLS:
            raise KeyError(f"unknown object {name!r}; known: {OBJECT_NAMES}")
        req = Request(OBJECT_URLS[name], headers={"User-Agent": _UA})
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
        Image.open(BytesIO(data)).convert("RGBA").save(out)
    img = Image.open(out).convert("RGBA")
    _MEMO[memo_key] = img
    return img


def ensure_all_cached(cache_dir: Path) -> Path:
    """Download every thumbnail in `OBJECT_URLS` to `cache_dir`. Idempotent."""
    cache_dir = Path(cache_dir)
    for name in OBJECT_URLS:
        load_thumbnail(name, cache_dir=cache_dir)
    return cache_dir
