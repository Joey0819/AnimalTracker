"""Pose estimation module for DenseTrack-AI."""

from densetrack.pose.base import BasePoseEstimator
from densetrack.pose.dlc_superanimal import DLCSuperAnimalEstimator
from densetrack.pose.geometric import GeometricPoseEstimator


def create_pose_estimator(config) -> BasePoseEstimator:
    """Factory function to create a pose estimator from config."""
    from densetrack.config import PoseConfig

    if not isinstance(config, PoseConfig):
        raise TypeError(f"Expected PoseConfig, got {type(config)}")

    backend = config.backend.lower()
    if backend == "dlc_superanimal":
        return DLCSuperAnimalEstimator(config)
    elif backend == "geometry":
        return GeometricPoseEstimator(config)
    else:
        raise ValueError(f"Unknown pose backend: {backend}")


__all__ = [
    "BasePoseEstimator",
    "DLCSuperAnimalEstimator",
    "GeometricPoseEstimator",
    "create_pose_estimator",
]
