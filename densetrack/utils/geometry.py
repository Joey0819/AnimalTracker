"""Geometry utilities for DenseTrack-AI."""

from __future__ import annotations

import numpy as np

__all__ = [
    "bbox_xyxy_to_xywh",
    "bbox_xywh_to_xyxy",
    "bbox_center",
    "bbox_area",
    "compute_iou",
    "angle_difference",
    "normalize_angle",
]


def bbox_xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
    bbox = np.asarray(bbox, dtype=np.float64)
    result = bbox.copy()
    if bbox.ndim == 1:
        result[2] = bbox[2] - bbox[0]
        result[3] = bbox[3] - bbox[1]
    else:
        result[:, 2] = bbox[:, 2] - bbox[:, 0]
        result[:, 3] = bbox[:, 3] - bbox[:, 1]
    return result


def bbox_xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """Convert (x, y, w, h) to (x1, y1, x2, y2)."""
    bbox = np.asarray(bbox, dtype=np.float64)
    result = bbox.copy()
    if bbox.ndim == 1:
        result[2] = bbox[0] + bbox[2]
        result[3] = bbox[1] + bbox[3]
    else:
        result[:, 2] = bbox[:, 0] + bbox[:, 2]
        result[:, 3] = bbox[:, 1] + bbox[:, 3]
    return result


def bbox_center(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Get center (cx, cy) from (x1, y1, x2, y2)."""
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.ndim == 1:
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    return np.stack([
        (bbox[:, 0] + bbox[:, 2]) / 2,
        (bbox[:, 1] + bbox[:, 3]) / 2,
    ], axis=1)


def bbox_area(bbox_xyxy: np.ndarray) -> np.ndarray | float:
    """Compute area from (x1, y1, x2, y2)."""
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.ndim == 1:
        return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
    a = np.asarray(bbox_a, dtype=np.float64)
    b = np.asarray(bbox_b, dtype=np.float64)

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def angle_difference(a: float, b: float) -> float:
    """Compute smallest signed angle difference in radians."""
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
