"""Detection module for DenseTrack-AI."""

from densetrack.detection.base import BaseDetector
from densetrack.detection.background_sub import BackgroundSubDetector
from densetrack.detection.dark_object import DarkObjectDetector
from densetrack.detection.grounding_dino import GroundingDINODetector
from densetrack.detection.sam2_propagator import SAM2Propagator


def create_detector(config) -> BaseDetector:
    """Factory function to create a detector from config."""
    from densetrack.config import DetectionConfig

    if not isinstance(config, DetectionConfig):
        raise TypeError(f"Expected DetectionConfig, got {type(config)}")

    backend = config.backend.lower()
    if backend == "grounding_dino":
        return GroundingDINODetector(config)
    elif backend == "background_sub":
        return BackgroundSubDetector(config)
    elif backend == "dark_object":
        return DarkObjectDetector(config)
    else:
        raise ValueError(f"Unknown detection backend: {backend}")


__all__ = [
    "BaseDetector",
    "GroundingDINODetector",
    "SAM2Propagator",
    "BackgroundSubDetector",
    "DarkObjectDetector",
    "create_detector",
]
