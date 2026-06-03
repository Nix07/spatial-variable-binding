"""COCO data loading and spatial-caption parsing."""

from __future__ import annotations

from .common import (
    AXIS_BY_DIRECTION,
    LABELS_BY_AXIS,
    VALID_BY_AXIS,
    build_spatial_data,
    download_coco_data,
    load_coco_gt_bboxes,
    load_grounding_dino,
    make_find_coco_bbox,
    make_get_bbox,
    parse_spatial_caption,
    strip_article,
)

__all__ = [
    "AXIS_BY_DIRECTION",
    "LABELS_BY_AXIS",
    "VALID_BY_AXIS",
    "build_spatial_data",
    "download_coco_data",
    "load_coco_gt_bboxes",
    "load_grounding_dino",
    "make_find_coco_bbox",
    "make_get_bbox",
    "parse_spatial_caption",
    "strip_article",
]
