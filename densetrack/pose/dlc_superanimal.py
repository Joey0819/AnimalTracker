"""
DeepLabCut SuperAnimal adapter for zero-shot pose estimation.

Requires: pip install densetrack[dlc]
License: Code is LGPL-3.0. SuperAnimal model weights have research/non-commercial
restrictions — do NOT package in commercial distributions without explicit license
confirmation from the DeepLabCut team.
"""

from __future__ import annotations

import logging

import numpy as np

from densetrack.config import PoseConfig
from densetrack.models import Detection, PoseResult, PoseSource
from densetrack.pose.base import BasePoseEstimator

__all__ = ["DLCSuperAnimalEstimator"]

logger = logging.getLogger(__name__)

_dlc_available = False
try:
    import deeplabcut  # noqa: F401

    _dlc_available = True
except ImportError:
    pass


class DLCSuperAnimalEstimator(BasePoseEstimator):
    """
    Zero-shot pose estimation using DLC SuperAnimal-TopViewMouse.

    The SuperAnimal models are pre-trained on diverse animal datasets
    and can perform zero-shot inference on new videos without any
    fine-tuning. For mouse top-view, use `superanimal_topviewmouse`.

    ⚠️  License warning: SuperAnimal model weights may have
    research/non-commercial restrictions. Check the DeepLabCut
    Model Zoo documentation before commercial use.

    Install:
        pip install densetrack[dlc]

    Reference:
        https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html
    """

    def __init__(self, config: PoseConfig):
        super().__init__(config)
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return f"DLC SuperAnimal ({self.config.superanimal_name})"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not _dlc_available:
            raise ImportError(
                "DeepLabCut is not installed. Install with:\n"
                "  pip install densetrack[dlc]\n"
                "Or use the 'geometry' backend instead (no keypoints, "
                "but orientation/body_length/area will still work)."
            )

        try:
            logger.info(
                f"Loading DLC SuperAnimal: {self.config.superanimal_name} "
                f"(model: {self.config.model_name})"
            )
            # DLC 3.0+ SuperAnimal API
            self._model = deeplabcut.create_pretrained_project(
                self.config.superanimal_name,
                model_name=self.config.model_name,
                videotype="mp4",
            )
            self._loaded = True
            logger.info("DLC SuperAnimal loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DLC SuperAnimal: {e}\n"
                "Make sure DeepLabCut >= 3.0 is installed."
            ) from e

    def estimate(
        self,
        frame: np.ndarray,
        detection: Detection,
        mask: np.ndarray | None = None,
    ) -> PoseResult:
        """
        Estimate keypoints using DLC SuperAnimal.

        Falls back to geometric features if DLC is unavailable and
        config.fallback_to_geometry is True.
        """
        if not _dlc_available:
            if self.config.fallback_to_geometry:
                from densetrack.pose.geometric import GeometricPoseEstimator

                logger.debug("DLC unavailable, falling back to geometric pose")
                fallback = GeometricPoseEstimator(self.config)
                return fallback.estimate(frame, detection, mask)
            raise ImportError("DLC not installed and fallback_to_geometry is False")

        self._ensure_loaded()

        # Crop the detection region
        x1, y1, x2, y2 = [int(v) for v in detection.bbox_xyxy]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return PoseResult(
                frame_idx=detection.frame_idx,
                detection_id=detection.detection_id,
                body_area=detection.area,
                centroid=detection.centroid,
                source=PoseSource.DLC_SUPERANIMAL,
            )

        try:
            # DLC inference on crop
            # API varies by DLC version; this is the expected pattern
            result = deeplabcut.analyze_image(
                self._model,
                crop,
            )

            # Extract keypoints (DLC returns DataFrame with x, y, likelihood columns)
            keypoints = result[["x", "y"]].values.astype(np.float32)
            confidences = result["likelihood"].values.astype(np.float32)

            # Convert keypoints from crop coordinates to full-frame coordinates
            keypoints[:, 0] += x1
            keypoints[:, 1] += y1

            # Compute orientation from keypoints (nose-to-tail vector)
            orientation = self._calc_orientation(keypoints, confidences)
            body_length = self._calc_body_length(keypoints, confidences)

            return PoseResult(
                frame_idx=detection.frame_idx,
                detection_id=detection.detection_id,
                keypoints_xy=keypoints,
                keypoint_conf=confidences,
                orientation=orientation,
                body_length=body_length,
                body_area=detection.area,
                centroid=detection.centroid,
                source=PoseSource.DLC_SUPERANIMAL,
            )

        except Exception as e:
            logger.warning(f"DLC inference failed: {e}")
            if self.config.fallback_to_geometry:
                from densetrack.pose.geometric import GeometricPoseEstimator

                fallback = GeometricPoseEstimator(self.config)
                return fallback.estimate(frame, detection, mask)
            raise

    @staticmethod
    def _calc_orientation(
        keypoints: np.ndarray, confidences: np.ndarray, min_conf: float = 0.3
    ) -> float | None:
        """Compute body orientation from keypoints (nose-to-tail direction)."""
        # Use high-confidence keypoints for PCA
        valid = confidences > min_conf
        if valid.sum() < 2:
            return None
        pts = keypoints[valid]
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Principal direction
        main_axis = eigenvectors[:, -1]
        return float(np.arctan2(main_axis[1], main_axis[0]))

    @staticmethod
    def _calc_body_length(
        keypoints: np.ndarray, confidences: np.ndarray, min_conf: float = 0.3
    ) -> float | None:
        """Compute body length as max pairwise distance of high-conf keypoints."""
        valid = confidences > min_conf
        if valid.sum() < 2:
            return None
        pts = keypoints[valid]
        from scipy.spatial.distance import pdist

        distances = pdist(pts)
        return float(distances.max()) if len(distances) > 0 else None
