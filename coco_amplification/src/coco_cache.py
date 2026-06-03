"""Local COCO cache helpers for the COCO amplification experiment."""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

from shared.paths import REPO_ROOT


DEFAULT_COCO_DIR = REPO_ROOT / "shared" / "coco_2obj"
VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def default_coco_dir() -> Path:
    return Path(os.environ.get("COCO_DIR", DEFAULT_COCO_DIR)).expanduser()


def val2017_dir(coco_dir: str | os.PathLike | None = None) -> Path:
    return Path(coco_dir).expanduser() / "val2017" if coco_dir else default_coco_dir() / "val2017"


def annotations_path(coco_dir: str | os.PathLike | None = None) -> Path:
    root = Path(coco_dir).expanduser() if coco_dir else default_coco_dir()
    return root / "annotations" / "instances_val2017.json"


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {target}")
    urllib.request.urlretrieve(url, target)


def _extract(zip_path: Path, dest: Path) -> None:
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def ensure_val2017(coco_dir: str | os.PathLike | None = None) -> Path:
    root = Path(coco_dir).expanduser() if coco_dir else default_coco_dir()
    img_dir = root / "val2017"
    if img_dir.is_dir() and any(img_dir.glob("*.jpg")):
        return img_dir
    zip_path = root / "val2017.zip"
    if not zip_path.exists():
        _download(VAL2017_URL, zip_path)
    _extract(zip_path, root)
    return img_dir


def ensure_annotations(coco_dir: str | os.PathLike | None = None) -> Path:
    root = Path(coco_dir).expanduser() if coco_dir else default_coco_dir()
    ann_path = root / "annotations" / "instances_val2017.json"
    if ann_path.exists():
        return ann_path
    zip_path = root / "annotations_trainval2017.zip"
    if not zip_path.exists():
        _download(ANNOTATIONS_URL, zip_path)
    _extract(zip_path, root)
    return ann_path
