"""
Geometric pose estimation from segmentation masks.

Fully implemented — no external model dependencies, only OpenCV.
Extracts orientation, body length, aspect ratio, area, and centroid
from the binary mask contour using PCA and ellipse fitting.

This is the universal fallback that works for any species.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from densetrack.config import PoseConfig
from densetrack.models import Detection, PoseResult, PoseSource
from densetrack.pose.base import BasePoseEstimator

__all__ = ["GeometricPoseEstimator"]

logger = logging.getLogger(__name__)


class GeometricPoseEstimator(BasePoseEstimator):
    """
    Extract geometric pose features from segmentation masks.

    Works without any ML model — pure computer vision:
    - Body orientation (PCA main axis of contour points)
    - Body length (ellipse major axis)
    - Aspect ratio (major / minor axis)
    - Body area (contour area)
    - Centroid (image moments)

    This is always available as a fallback, even when DLC is not installed.
    """

    def __init__(self, config: PoseConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "Geometric (mask-based)"

    def estimate(
        self,
        frame: np.ndarray,
        detection: Detection,
        mask: np.ndarray | None = None,
    ) -> PoseResult:
        """
        Extract geometric features from mask or bbox region.

        If no mask is provided, creates a rough mask from the bbox
        region using Otsu thresholding on the grayscale crop.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in detection.bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if mask is None:
            # Create a rough mask from the bbox region
            mask = self._create_mask_from_bbox(frame, x1, y1, x2, y2)

        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # No contour found — return minimal result
            return PoseResult(
                frame_idx=detection.frame_idx,
                detection_id=detection.detection_id,
                body_area=detection.area,
                centroid=detection.centroid,
                source=PoseSource.GEOMETRY,
            )

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        # Centroid from moments
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)

        # Orientation via PCA
        orientation = self._compute_orientation(contour)

        # Ellipse fitting for body length and aspect ratio
        body_length, aspect_ratio = self._fit_ellipse(contour)

        return PoseResult(
            frame_idx=detection.frame_idx,
            detection_id=detection.detection_id,
            keypoints_xy=None,
            keypoint_conf=None,
            orientation=orientation,
            body_length=body_length,
            body_area=float(area),
            aspect_ratio=aspect_ratio,
            centroid=(cx, cy),
            source=PoseSource.GEOMETRY,
        )

    @staticmethod
    def _create_mask_from_bbox(
        frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """
        Create a rough binary mask from a bbox region using thresholding.

        Strategy:
        1. Crop the bbox region
        2. Convert to grayscale
        3. Apply Gaussian blur
        4. Otsu threshold
        5. Place back in full-frame mask
        """
        h, w = frame.shape[:2]
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros((h, w), dtype=np.uint8)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If more than 50% is foreground, invert (animal is probably darker)
        if binary.mean() > 127:
            binary = 255 - binary

        # Place crop mask into full-frame mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = binary

        return full_mask

    @staticmethod
    def _compute_orientation(contour: np.ndarray) -> float | None:
        """
        Compute body heading via PCA on contour points.

        Returns angle in radians [-π, π], or None if insufficient points.
        """
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            return None

        # PCA
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        # First eigenvector = main axis direction
        angle = float(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
        return angle

    @staticmethod
    def _fit_ellipse(contour: np.ndarray) -> tuple[float | None, float | None]:
        """
        Fit an ellipse to the contour to estimate body length and aspect ratio.

        Returns:
            (body_length, aspect_ratio) — body_length is major axis in pixels.
        """
        pts = contour.reshape(-1, 2)
        if len(pts) < 5:
            return None, None

        try:
            ellipse = cv2.fitEllipse(pts)
            # ellipse = ((cx, cy), (minor_axis, major_axis), angle)
            axes = ellipse[1]
            major = float(max(axes))
            minor = float(min(axes))
            aspect = major / max(minor, 1e-6)
            return major, aspect
        except cv2.error:
            return None, None
