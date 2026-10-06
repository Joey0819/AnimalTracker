"""Abstract base class for detection backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from densetrack.config import DetectionConfig
from densetrack.models import Detection

__all__ = ["BaseDetector"]

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    Abstract interface for animal detectors.

    All detection backends (Grounding DINO, background subtraction, RT-DETR, etc.)
    must implement this interface. This ensures the downstream pipeline
    (pose, tracking, export) is decoupled from the detection method.
    """

    def __init__(self, config: DetectionConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this detector."""
        ...

    def validate(self) -> None:
        """
        Validate backend readiness before expensive video loading.

        Lightweight backends can keep the default no-op. Model-backed
        detectors should override this to check imports/weights early.
        """
        return None

    @abstractmethod
    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Detect animals in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            frame_idx: Frame index in the video.

        Returns:
            List of Detection objects found in this frame.
        """
        ...

    def detect_video(
        self,
        frames: dict[int, np.ndarray],
        show_progress: bool = True,
    ) -> dict[int, list[Detection]]:
        """
        Detect animals across multiple frames.

        Default implementation: iterate frames and call detect_frame.
        Subclasses may override for batch/video-level processing.

        Args:
            frames: Dict mapping frame_idx -> BGR numpy array.
            show_progress: Show tqdm progress bar.

        Returns:
            Dict mapping frame_idx -> list of Detection objects.
        """
        results: dict[int, list[Detection]] = {}
        frame_iter = sorted(frames.items())

        if show_progress:
            frame_iter = tqdm(frame_iter, desc=f"[{self.name}] Detecting", unit="frame")

        for frame_idx, frame in frame_iter:
            detections = self.detect_frame(frame, frame_idx)

            # Filter by area constraints
            filtered = []
            for det in detections:
                area = det.area or 0
                if self.config.min_mask_area <= area <= self.config.max_mask_area:
                    filtered.append(det)
                else:
                    logger.debug(
                        f"Frame {frame_idx}: filtered detection with area {area:.0f} "
                        f"(bounds: [{self.config.min_mask_area}, {self.config.max_mask_area}])"
                    )

            results[frame_idx] = filtered

        total = sum(len(d) for d in results.values())
        logger.info(
            f"[{self.name}] Detected {total} objects across {len(results)} frames "
            f"(avg {total / max(len(results), 1):.1f}/frame)"
        )
        return results
