"""Deterministic hierarchical geometry tokens for long-term animal tracking.

The representation follows the geometry decomposition used by HieDG:
normalized position, normalized scale, and polar velocity. DenseTrack uses a
fixed two-stage residual quantizer so the method remains training-free and
reproducible on local research workstations. It is an adaptation of the paper's
representation principle, not a reproduction of its learned codebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from densetrack.models import Detection

__all__ = [
    "HierarchicalGeometryEncoder",
    "HierarchicalGeometryToken",
    "estimate_knn_velocity",
    "robust_detection_velocity",
]


@dataclass(frozen=True)
class _ResidualToken:
    coarse: tuple[int, int]
    fine: tuple[int, int]


@dataclass(frozen=True)
class HierarchicalGeometryToken:
    """Two-stage discrete codes for position, scale, and polar velocity."""

    position: _ResidualToken
    scale: _ResidualToken
    velocity: _ResidualToken
    low_speed: bool


class HierarchicalGeometryEncoder:
    """Encode bounding-box geometry with coarse-to-fine residual tokens."""

    def __init__(
        self,
        frame_size: tuple[int, int] | None,
        coarse_bins: int = 8,
        fine_bins: int = 8,
        max_speed: float | None = None,
    ):
        self.frame_size = frame_size
        self.coarse_bins = max(2, int(coarse_bins))
        self.fine_bins = max(2, int(fine_bins))
        diagonal = self.frame_diagonal
        self.max_speed = max(float(max_speed or diagonal * 0.08), 1.0)

    @property
    def frame_diagonal(self) -> float:
        if self.frame_size is None:
            return 1000.0
        width, height = self.frame_size
        return max(float(np.hypot(width, height)), 1.0)

    def encode(
        self,
        bbox_xyxy: Sequence[float],
        velocity_xy: Sequence[float] | np.ndarray,
    ) -> HierarchicalGeometryToken:
        """Encode normalized position/scale and polar velocity."""
        bbox = np.asarray(bbox_xyxy, dtype=np.float64)
        velocity = np.asarray(velocity_xy, dtype=np.float64)
        width = max(float(bbox[2] - bbox[0]), 1.0)
        height = max(float(bbox[3] - bbox[1]), 1.0)
        center = np.array(
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
            dtype=np.float64,
        )
        frame_width, frame_height = self._frame_dimensions(bbox)
        position = np.clip(
            np.array([center[0] / frame_width, center[1] / frame_height]),
            0.0,
            1.0,
        )
        scale = np.clip(
            np.array([width / frame_width, height / frame_height]),
            0.0,
            1.0,
        )
        speed = float(np.linalg.norm(velocity))
        low_speed = speed <= max(self.max_speed * 0.01, 0.25)
        angle = 0.0 if low_speed else float(np.arctan2(velocity[1], velocity[0]))
        polar_velocity = np.array(
            [
                min(speed / self.max_speed, 1.0),
                ((angle + np.pi) / (2.0 * np.pi)) % 1.0,
            ],
            dtype=np.float64,
        )
        return HierarchicalGeometryToken(
            position=self._quantize_pair(position),
            scale=self._quantize_pair(scale),
            velocity=self._quantize_pair(polar_velocity),
            low_speed=low_speed,
        )

    def distance(
        self,
        left: HierarchicalGeometryToken,
        right: HierarchicalGeometryToken,
        *,
        position_tolerance: float = 0.0,
    ) -> float:
        """Return a normalized discrete-token distance in ``[0, 1]``."""
        left_position = self._decode_pair(left.position)
        right_position = self._decode_pair(right.position)
        position = float(np.linalg.norm(left_position - right_position) / np.sqrt(2.0))
        position = max(0.0, position - max(position_tolerance, 0.0))

        left_scale = self._decode_pair(left.scale)
        right_scale = self._decode_pair(right.scale)
        scale = float(np.linalg.norm(left_scale - right_scale) / np.sqrt(2.0))

        left_velocity = self._decode_pair(left.velocity)
        right_velocity = self._decode_pair(right.velocity)
        speed = abs(float(left_velocity[0] - right_velocity[0]))
        if left.low_speed and right.low_speed:
            direction = 0.0
        else:
            angular_delta = abs(float(left_velocity[1] - right_velocity[1]))
            direction = 2.0 * min(angular_delta, 1.0 - angular_delta)

        weighted = 0.46 * position + 0.14 * scale + 0.16 * speed + 0.24 * direction
        return float(np.clip(weighted, 0.0, 1.0))

    def similarity(
        self,
        left: HierarchicalGeometryToken,
        right: HierarchicalGeometryToken,
        *,
        position_tolerance: float = 0.0,
    ) -> float:
        """Convert token distance to a smooth association score."""
        distance = self.distance(
            left,
            right,
            position_tolerance=position_tolerance,
        )
        return float(np.exp(-3.0 * distance))

    def _frame_dimensions(self, bbox: np.ndarray) -> tuple[float, float]:
        if self.frame_size is not None:
            width, height = self.frame_size
            return max(float(width), 1.0), max(float(height), 1.0)
        width = max(float(bbox[2]), 1.0)
        height = max(float(bbox[3]), 1.0)
        return width, height

    def _quantize_pair(self, values: np.ndarray) -> _ResidualToken:
        coarse: list[int] = []
        fine: list[int] = []
        for raw_value in values:
            value = float(np.clip(raw_value, 0.0, np.nextafter(1.0, 0.0)))
            coarse_index = min(int(value * self.coarse_bins), self.coarse_bins - 1)
            coarse_center = (coarse_index + 0.5) / self.coarse_bins
            residual = (value - coarse_center) * self.coarse_bins + 0.5
            residual = float(np.clip(residual, 0.0, np.nextafter(1.0, 0.0)))
            fine_index = min(int(residual * self.fine_bins), self.fine_bins - 1)
            coarse.append(coarse_index)
            fine.append(fine_index)
        return _ResidualToken(tuple(coarse), tuple(fine))

    def _decode_pair(self, token: _ResidualToken) -> np.ndarray:
        values = [
            (coarse + (fine + 0.5) / self.fine_bins) / self.coarse_bins
            for coarse, fine in zip(token.coarse, token.fine)
        ]
        return np.asarray(values, dtype=np.float64)


def robust_detection_velocity(
    detections: Sequence[Detection],
    *,
    endpoint: str = "end",
    window: int = 7,
) -> np.ndarray:
    """Estimate endpoint velocity with a median that rejects one-frame jitter."""
    ordered = sorted(
        (detection for detection in detections if detection.centroid is not None),
        key=lambda detection: detection.frame_idx,
    )
    if len(ordered) < 2:
        return np.zeros(2, dtype=np.float64)
    sample_count = max(2, int(window) + 1)
    selected = ordered[-sample_count:] if endpoint == "end" else ordered[:sample_count]
    velocities = []
    for left, right in zip(selected, selected[1:]):
        frame_delta = max(int(right.frame_idx - left.frame_idx), 1)
        displacement = (
            np.asarray(right.centroid, dtype=np.float64)
            - np.asarray(left.centroid, dtype=np.float64)
        )
        velocities.append(displacement / frame_delta)
    if not velocities:
        return np.zeros(2, dtype=np.float64)
    values = np.asarray(velocities, dtype=np.float64)
    median = np.median(values, axis=0)
    residuals = np.linalg.norm(values - median, axis=1)
    residual_median = float(np.median(residuals))
    if residual_median > 0:
        inliers = residuals <= residual_median * 3.5
        if np.any(inliers):
            values = values[inliers]
    return np.median(values, axis=0)


def estimate_knn_velocity(
    center: Sequence[float] | np.ndarray,
    direction_hint: Sequence[float] | np.ndarray,
    history: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    k: int = 5,
    spatial_scale: float = 90.0,
    direction_scale: float = 0.75,
) -> np.ndarray:
    """Estimate velocity from nearby historical states using HieDG-style kernels."""
    samples = list(history)
    if not samples:
        return np.asarray(direction_hint, dtype=np.float64)

    candidate_center = np.asarray(center, dtype=np.float64)
    hint = np.asarray(direction_hint, dtype=np.float64)
    centers = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    velocities = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    distances_sq = np.sum((centers - candidate_center) ** 2, axis=1)
    nearest = np.argsort(distances_sq)[: max(1, min(int(k), len(samples)))]

    nearest_velocities = velocities[nearest]
    spatial_weights = np.exp(
        -distances_sq[nearest] / max(float(spatial_scale) ** 2, 1.0)
    )
    hint_norm = float(np.linalg.norm(hint))
    velocity_norms = np.linalg.norm(nearest_velocities, axis=1)
    if hint_norm > 1e-6:
        cosine = (
            nearest_velocities @ hint
            / np.maximum(velocity_norms * hint_norm, 1e-6)
        )
        direction_weights = np.exp(cosine / max(float(direction_scale), 1e-3))
    else:
        direction_weights = np.ones_like(spatial_weights)
    weights = spatial_weights * direction_weights
    if float(np.sum(weights)) <= 1e-9:
        return np.median(nearest_velocities, axis=0)
    return np.average(nearest_velocities, axis=0, weights=weights)
