"""Abstract base class for pose estimation backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from densetrack.config import PoseConfig
from densetrack.models import Detection, PoseResult

__all__ = ["BasePoseEstimator"]

logger = logging.getLogger(__name__)


class BasePoseEstimator(ABC):
    """
    Abstract interface for pose estimation backends.

    Implementations: DLCSuperAnimalEstimator (keypoints), GeometricPoseEstimator (mask-based).
    """

    def __init__(self, config: PoseConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this pose estimator."""
        ...

    @abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        detection: Detection,
        mask: np.ndarray | None = None,
    ) -> PoseResult:
        """
        Estimate pose for a single detected animal.

        Args:
            frame: BGR image (H, W, 3).
            detection: Detection object with bbox.
            mask: Optional binary mask (H, W) for this detection.

        Returns:
            PoseResult with keypoints and/or geometric features.
        """
        ...

    def estimate_batch(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        masks: list[np.ndarray | None] | None = None,
    ) -> list[PoseResult]:
        """
        Estimate pose for multiple detections in one frame.

        Default: iterate and call estimate(). Subclasses may override
        for batched inference.
        """
        if masks is None:
            masks = [None] * len(detections)

        results = []
        for det, mask in zip(detections, masks):
            try:
                result = self.estimate(frame, det, mask)
                results.append(result)
            except Exception as e:
                logger.warning(f"Pose estimation failed for detection {det.detection_id}: {e}")
                # Return a minimal PoseResult with just geometric info
                results.append(PoseResult(
                    frame_idx=det.frame_idx,
                    detection_id=det.detection_id,
                    body_area=det.area,
                    centroid=det.centroid,
                ))
        return results
