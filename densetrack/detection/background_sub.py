"""
Background subtraction detector — fully implemented, no external model dependencies.

Uses OpenCV MOG2 background subtractor. This is the simplest baseline detector
and serves as the fallback when Grounding DINO or other model-based detectors
are not available.

Best suited for: static camera, consistent lighting, moderate animal count.
License: No restrictions (pure OpenCV).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from densetrack.config import DetectionConfig
from densetrack.detection.base import BaseDetector
from densetrack.models import Detection, DetectionSource

__all__ = ["BackgroundSubDetector"]

logger = logging.getLogger(__name__)


class BackgroundSubDetector(BaseDetector):
    """
    Background subtraction detector using OpenCV MOG2.

    Detects moving objects by comparing each frame against a learned
    background model. Works well for overhead camera setups with
    relatively static backgrounds.

    Advantages:
        - Zero external dependencies (just OpenCV)
        - No model weights to download
        - Fast inference
        - No licensing concerns

    Limitations:
        - Requires animals to be moving (misses stationary animals)
        - Sensitive to lighting changes
        - Cannot detect in the first few frames (learning period)
        - Does not provide instance segmentation masks
    """

    def __init__(self, config: DetectionConfig):
        super().__init__(config)

        # MOG2 background subtractor
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True,
        )

        # Morphological kernels
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        self._learning_rate = 0.005
        self._warmup_frames = 30
        self._frame_count = 0

    @property
    def name(self) -> str:
        return "Background Subtraction (MOG2)"

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Detect animals via background subtraction.

        Steps:
            1. Apply Gaussian blur to reduce noise
            2. Run MOG2 background subtraction
            3. Threshold to remove shadows
            4. Morphological open (remove small noise)
            5. Morphological close (fill holes)
            6. Find contours
            7. Filter by area and create Detection objects
        """
        self._frame_count += 1

        # Preprocessing: blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Background subtraction
        # Higher learning rate during warmup for faster convergence
        lr = 0.05 if self._frame_count <= self._warmup_frames else self._learning_rate
        fg_mask = self._bg_subtractor.apply(blurred, learningRate=lr)

        # During warmup, don't report detections (background model not stable)
        if self._frame_count <= self._warmup_frames:
            return []

        # Remove shadows (MOG2 marks shadows as 127, foreground as 255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel_close)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and create detections
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Area filter
            if area < self.config.min_mask_area or area > self.config.max_mask_area:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Aspect ratio filter (animals are roughly elliptical, not thin lines)
            aspect = w / max(h, 1)
            if aspect > 5.0 or aspect < 0.2:
                continue

            # Solidity filter (contour area / convex hull area — animals are fairly solid)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            if solidity < 0.3:
                continue

            # Confidence based on how well the detection matches expected animal shape
            # Factors: solidity, size relative to expected range, circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)

            conf_solidity = min(solidity / 0.7, 1.0)
            conf_circularity = min(circularity / 0.5, 1.0)
            conf_size = 1.0 - abs(
                area - (self.config.min_mask_area + self.config.max_mask_area) / 2
            ) / max(self.config.max_mask_area, 1)
            conf_size = max(0.0, min(conf_size, 1.0))

            confidence = 0.4 * conf_solidity + 0.3 * conf_circularity + 0.3 * conf_size

            # Compute centroid from moments
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2
                cy = y + h / 2

            # Extract mask polygon (simplified contour)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.reshape(-1, 2).tolist()

            det = Detection(
                frame_idx=frame_idx,
                bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                confidence=float(np.clip(confidence, 0.1, 0.95)),
                source=DetectionSource.BACKGROUND_SUB,
                centroid=(float(cx), float(cy)),
                area=float(area),
                mask_polygon=[coord for pt in polygon for coord in pt],
            )
            detections.append(det)

        return detections

    def reset(self) -> None:
        """Reset the background model (useful when processing a new video)."""
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True,
        )
        self._frame_count = 0
