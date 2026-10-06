"""
Dark-object detector for top-view black animals on light bedding.

Optimized for typical lab scenarios: dark mice on light sawdust/corn cob bedding.
Uses adaptive thresholding + morphology + watershed blob splitting.

Key improvements over MOG2 background subtraction:
- Detects STATIONARY animals (MOG2 misses them)
- No warmup period required
- More stable detection count
- Auto ROI to exclude cage walls
- Watershed split for merged blobs

License: No restrictions (pure OpenCV).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from densetrack.config import DetectionConfig
from densetrack.detection.base import BaseDetector
from densetrack.models import Detection, DetectionSource

__all__ = ["DarkObjectDetector"]

logger = logging.getLogger(__name__)


class DarkObjectDetector(BaseDetector):
    """Segment dark animal instances from a light static background."""

    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        # Auto ROI mask (computed on first frame)
        self._roi_mask: Optional[np.ndarray] = None

        # Running area statistics for adaptive split threshold
        self._area_history: list[float] = []
        self._median_area: float = 0.0

    @property
    def name(self) -> str:
        return "Dark Object Segmentation"

    def _compute_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Auto-detect the arena/cage floor to exclude walls.

        The arena is the large bright region. We find it by thresholding
        brightness and taking the largest contour, then eroding slightly
        to exclude edge artifacts.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, arena_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        arena_mask = cv2.morphologyEx(arena_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(arena_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 0.25 * h * w:
                roi = np.zeros_like(gray)
                cv2.drawContours(roi, [largest], -1, 255, -1)
                # Erode to exclude cage wall edges where mice touch
                erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                roi = cv2.erode(roi, erode_k)
                logger.info(f"Auto ROI: {np.sum(roi > 0)} pixels ({np.sum(roi > 0) * 100 / (h * w):.0f}% of frame)")
                return roi

        # Fallback: small margin
        margin = int(min(h, w) * 0.02)
        roi = np.zeros((h, w), dtype=np.uint8)
        roi[margin:h - margin, margin:w - margin] = 255
        return roi

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Detect dark objects via threshold + morphology + watershed.

        Steps:
        1. Auto ROI (first frame)
        2. Grayscale + blur
        3. Otsu threshold (dark → foreground)
        4. Apply ROI mask
        5. Morphological cleanup
        6. Contour detection + filtering
        7. Watershed split for oversized blobs
        """
        # Auto ROI on first frame
        if self._roi_mask is None:
            self._roi_mask = self._compute_roi_mask(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold: Otsu on inverted (dark objects → bright)
        otsu_val, _ = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Clamp threshold for stability
        threshold = min(max(float(otsu_val), 50.0), 120.0)
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

        # Apply ROI
        mask = cv2.bitwise_and(mask, self._roi_mask)

        # Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours and detect
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_mask_area or area > self.config.max_mask_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if aspect > 5.0 or aspect < 0.2:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            if solidity < 0.25:
                continue

            # Check if this blob should be split (merged animals)
            if self._should_split(area):
                split_dets = self._watershed_split(frame, mask, contour, frame_idx)
                if split_dets:
                    detections.extend(split_dets)
                    continue

            det = self._contour_to_detection(contour, area, solidity, frame_idx)
            detections.append(det)

        # Update area statistics (for adaptive splitting)
        for det in detections:
            if not det.area or det.area <= self.config.min_mask_area:
                continue
            if self._median_area > 100:
                ratio = float(det.area) / self._median_area
                if ratio < 0.40 or ratio > 1.75:
                    continue
            self._area_history.append(float(det.area))
            if len(self._area_history) > 200:
                self._area_history = self._area_history[-200:]
            if self._area_history:
                self._median_area = float(np.median(self._area_history))

        return sorted(detections, key=lambda d: d.area or 0, reverse=True)

    def _should_split(self, area: float) -> bool:
        """Check if blob is likely multiple merged animals."""
        if self._median_area < 100:
            expected = (self.config.min_mask_area + self.config.max_mask_area) / 3
            return area > expected * 1.5
        return area > self._median_area * 1.5

    def _estimate_n_animals(self, area: float) -> int:
        """Estimate how many animals are in a merged blob."""
        if self._median_area > 100:
            return max(2, round(area / self._median_area))
        expected = (self.config.min_mask_area + self.config.max_mask_area) / 3
        return max(2, round(area / expected))

    def _watershed_split(
        self,
        frame: np.ndarray,
        fg_mask: np.ndarray,
        contour: np.ndarray,
        frame_idx: int,
    ) -> list[Detection] | None:
        """
        Split a merged blob using distance transform + watershed.
        Falls back to progressive erosion if watershed can't find seeds.
        """
        x, y, w, h = cv2.boundingRect(contour)
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        roi_mask = fg_mask[y1:y2, x1:x2].copy()
        roi_frame = frame[y1:y2, x1:x2].copy()

        # Distance transform
        dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return None

        # Try progressively lower thresholds to find seeds
        for thresh_ratio in [0.4, 0.3, 0.25, 0.2]:
            _, sure_fg = cv2.threshold(
                dist, thresh_ratio * dist.max(), 255, cv2.THRESH_BINARY
            )
            sure_fg = sure_fg.astype(np.uint8)
            n_labels, markers = cv2.connectedComponents(sure_fg)
            if n_labels >= 3:  # At least 2 seeds (label 0 = background)
                break

        if n_labels <= 2:
            # Watershed failed — try erosion-based splitting
            return self._erosion_split(roi_mask, roi_frame, x1, y1, frame_idx)

        # Watershed
        markers = markers + 1
        markers[roi_mask == 0] = 0
        cv2.watershed(roi_frame, markers)

        split_dets = []
        for label in range(2, n_labels + 1):
            seg = (markers == label).astype(np.uint8) * 255
            seg_contours, _ = cv2.findContours(
                seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for sc in seg_contours:
                sa = cv2.contourArea(sc)
                if sa < self.config.min_mask_area * 0.3:
                    continue
                sc_off = sc.copy()
                sc_off[:, :, 0] += x1
                sc_off[:, :, 1] += y1
                hull = cv2.convexHull(sc_off)
                sol = sa / max(cv2.contourArea(hull), 1)
                split_dets.append(
                    self._contour_to_detection(sc_off, sa, sol, frame_idx)
                )

        if len(split_dets) >= 2:
            logger.debug(
                f"Frame {frame_idx}: watershed split → {len(split_dets)} detections"
            )
            return split_dets
        return None

    def _erosion_split(
        self,
        roi_mask: np.ndarray,
        roi_frame: np.ndarray,
        offset_x: int,
        offset_y: int,
        frame_idx: int,
    ) -> list[Detection] | None:
        """
        Fallback split: progressively erode the mask until it breaks
        into multiple components, then dilate back.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for iterations in range(1, 12):
            eroded = cv2.erode(roi_mask, kernel, iterations=iterations)
            n_labels, labels = cv2.connectedComponents(eroded)
            if n_labels >= 3:  # Found 2+ separate components
                # Dilate each component back separately
                split_dets = []
                for label in range(1, n_labels):
                    component = (labels == label).astype(np.uint8) * 255
                    # Dilate back (but constrained to original mask)
                    dilated = cv2.dilate(component, kernel, iterations=iterations)
                    dilated = cv2.bitwise_and(dilated, roi_mask)

                    contours, _ = cv2.findContours(
                        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area < self.config.min_mask_area * 0.3:
                            continue
                        c_off = c.copy()
                        c_off[:, :, 0] += offset_x
                        c_off[:, :, 1] += offset_y
                        hull = cv2.convexHull(c_off)
                        sol = area / max(cv2.contourArea(hull), 1)
                        split_dets.append(
                            self._contour_to_detection(c_off, area, sol, frame_idx)
                        )

                if len(split_dets) >= 2:
                    logger.debug(
                        f"Frame {frame_idx}: erosion split ({iterations} iters) "
                        f"→ {len(split_dets)} detections"
                    )
                    return split_dets
        return None

    def _contour_to_detection(
        self, contour: np.ndarray, area: float, solidity: float, frame_idx: int
    ) -> Detection:
        """Convert contour to Detection with confidence scoring."""
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter ** 2, 1)

        # Size confidence (adaptive)
        if self._median_area > 100:
            size_diff = abs(area - self._median_area) / self._median_area
            conf_size = max(0.0, 1.0 - size_diff)
        else:
            mid = (self.config.min_mask_area + self.config.max_mask_area) / 2
            conf_size = float(np.clip(1.0 - abs(area - mid) / max(mid, 1), 0, 1))

        conf_sol = float(np.clip(solidity / 0.75, 0, 1))
        conf_circ = float(np.clip(circularity / 0.35, 0, 1))
        confidence = 0.40 * conf_sol + 0.25 * conf_circ + 0.35 * conf_size

        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx, cy = x + w / 2, y + h / 2

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = approx.reshape(-1, 2).tolist()

        return Detection(
            frame_idx=frame_idx,
            bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
            confidence=float(np.clip(confidence, 0.2, 0.95)),
            source=DetectionSource.DARK_OBJECT,
            centroid=(float(cx), float(cy)),
            area=float(area),
            mask_polygon=[coord for pt in polygon for coord in pt],
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._roi_mask = None
        self._area_history.clear()
        self._median_area = 0.0
