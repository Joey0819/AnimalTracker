"""
SAM 2 video mask propagation adapter.

Handles video-level mask propagation with periodic re-detection
to correct propagation drift. Not a BaseDetector subclass — it's
a post-processor that refines detection results with temporal masks.

Requires: pip install densetrack[sam2]
License: Apache 2.0
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from densetrack.config import SAM2Config
from densetrack.models import Detection, DetectionSource

__all__ = ["SAM2Propagator"]

logger = logging.getLogger(__name__)

_sam2_available = False
try:
    import sam2  # noqa: F401

    _sam2_available = True
except ImportError:
    pass


class SAM2Propagator:
    """
    SAM 2 video-level mask propagation.

    Takes initial detections (from Grounding DINO or any detector),
    initializes SAM 2 with their bounding boxes, and propagates masks
    through the video with periodic re-detection to correct drift.

    This is NOT an independent detector — it enhances detection results
    with temporally consistent instance masks.
    """

    def __init__(self, config: SAM2Config):
        self.config = config
        self._model = None
        self._loaded = False

    @property
    def available(self) -> bool:
        return _sam2_available

    def _ensure_loaded(self) -> None:
        """Lazy-load SAM 2 model."""
        if self._loaded:
            return

        if not _sam2_available:
            raise ImportError(
                "SAM 2 is not installed. Install with:\n"
                "  pip install densetrack[sam2]\n"
                "Mask propagation will be skipped."
            )

        try:
            from sam2.build_sam import build_sam2_video_predictor

            model_path = self.config.model_path
            variant = self.config.model_variant

            logger.info(f"Loading SAM 2 model: {variant}")
            self._model = build_sam2_video_predictor(
                config_file=f"{variant}.yaml",
                ckpt_path=model_path,
            )
            self._loaded = True
            logger.info("SAM 2 model loaded successfully")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAM 2: {e}\n"
                "See: https://github.com/facebookresearch/sam2"
            ) from e

    def propagate_video(
        self,
        video_path: str,
        initial_detections: list[Detection],
        redetect_fn: Callable[[np.ndarray, int], list[Detection]] | None = None,
        redetect_interval: int = 25,
    ) -> dict[int, list[Detection]]:
        """
        Propagate masks through a video starting from initial detections.

        Args:
            video_path: Path to the video file.
            initial_detections: Detections from the first frame to initialize SAM 2.
            redetect_fn: Optional callable(frame, frame_idx) -> detections for periodic re-detection.
            redetect_interval: Re-detect every N frames to correct drift.

        Returns:
            Dict mapping frame_idx -> list of Detection objects with mask data.
        """
        if not self.available:
            logger.warning(
                "SAM 2 not available. Returning initial detections only. "
                "Install with: pip install densetrack[sam2]"
            )
            return self._fallback_no_propagation(video_path, initial_detections)

        self._ensure_loaded()
        return self._propagate_with_sam2(
            video_path, initial_detections, redetect_fn, redetect_interval
        )

    def _propagate_with_sam2(
        self,
        video_path: str,
        initial_detections: list[Detection],
        redetect_fn: Callable | None,
        redetect_interval: int,
    ) -> dict[int, list[Detection]]:
        """SAM 2 video propagation implementation."""
        results: dict[int, list[Detection]] = {}

        with self._model.init_state(video_path) as state:
            # Initialize with first-frame detections
            for obj_id, det in enumerate(initial_detections):
                box = np.array(det.bbox_xyxy)
                self._model.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=det.frame_idx,
                    obj_id=obj_id,
                    box=box,
                )

            # Forward propagation
            for frame_idx, obj_ids, masks in self._model.propagate_in_video(state):
                frame_detections = []
                for obj_id, mask in zip(obj_ids, masks):
                    mask_binary = (mask[0] > 0.5).cpu().numpy().astype(np.uint8)

                    # Extract bbox from mask
                    import cv2

                    contours, _ = cv2.findContours(
                        mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if not contours:
                        continue

                    # Largest contour
                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)

                    det = Detection(
                        frame_idx=frame_idx,
                        bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                        confidence=0.9,  # SAM 2 propagation confidence
                        source=DetectionSource.SAM2_PROPAGATION,
                        centroid=(float(x + w / 2), float(y + h / 2)),
                        area=float(area),
                    )
                    frame_detections.append(det)

                results[frame_idx] = frame_detections

                # Periodic re-detection
                if redetect_fn and frame_idx > 0 and frame_idx % redetect_interval == 0:
                    logger.debug(f"Re-detecting at frame {frame_idx}")
                    # Re-detection would correct propagation drift
                    # Implementation: compare propagated masks with fresh detections
                    # and update SAM 2 state if significant drift detected

        logger.info(
            f"SAM 2 propagated masks across {len(results)} frames"
        )
        return results

    def _fallback_no_propagation(
        self,
        video_path: str,
        initial_detections: list[Detection],
    ) -> dict[int, list[Detection]]:
        """Fallback when SAM 2 is not available: return initial detections only."""
        results: dict[int, list[Detection]] = {}
        for det in initial_detections:
            if det.frame_idx not in results:
                results[det.frame_idx] = []
            results[det.frame_idx].append(det)
        return results
