"""
Core data models for DenseTrack-AI.

These dataclasses define the contracts between pipeline modules.
All modules (detection, pose, tracking, reid, export) communicate
through these shared types.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConfidenceLevel(str, Enum):
    """Confidence tier for audit reporting."""

    HIGH = "high"          # 🟢 conf >= 0.8 — auto-accept
    MEDIUM = "medium"      # 🟡 0.5 <= conf < 0.8 — spot-check recommended
    LOW = "low"            # 🟠 0.2 <= conf < 0.5 — manual review needed
    UNCERTAIN = "uncertain"  # 🔴 conf < 0.2 — identity may be wrong

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        if score >= 0.8:
            return cls.HIGH
        elif score >= 0.5:
            return cls.MEDIUM
        elif score >= 0.2:
            return cls.LOW
        else:
            return cls.UNCERTAIN

    @property
    def emoji(self) -> str:
        return {
            self.HIGH: "🟢",
            self.MEDIUM: "🟡",
            self.LOW: "🟠",
            self.UNCERTAIN: "🔴",
        }[self]

    @property
    def description(self) -> str:
        return {
            self.HIGH: "可直接使用",
            self.MEDIUM: "建议抽查验证",
            self.LOW: "需要人工审查",
            self.UNCERTAIN: "身份可能错误",
        }[self]


class DetectionSource(str, Enum):
    """Which detector produced this detection."""

    GROUNDING_DINO = "grounding_dino"
    SAM2_PROPAGATION = "sam2_propagation"
    BACKGROUND_SUB = "background_sub"
    DARK_OBJECT = "dark_object"
    FIXED_COUNT_PREDICTION = "fixed_count_prediction"
    ADAPTIVE_PREDICTION = "adaptive_prediction"
    MANUAL = "manual"  # Ground-truth / manual annotation


class PoseSource(str, Enum):
    """Which pose estimator produced this result."""

    DLC_SUPERANIMAL = "dlc_superanimal"
    GEOMETRY = "geometry"  # Mask-based geometric features
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """
    Single-frame detection of one animal instance.

    Produced by: detection module
    Consumed by: pose estimation, tracklet generator, export
    """

    frame_idx: int
    bbox_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float  # Detection confidence [0, 1]
    source: DetectionSource

    # Segmentation mask (optional)
    mask_rle: dict[str, Any] | None = None  # RLE-encoded mask
    mask_polygon: list[list[float]] | None = None  # Polygon contour

    # Derived fields (computed from bbox or mask)
    centroid: tuple[float, float] | None = None
    area: float | None = None

    # Unique ID within this processing run
    detection_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # BGR crop image bytes (optional, JPEG-encoded, used for ReID)
    crop_bytes: bytes | None = None

    def __post_init__(self) -> None:
        """Compute derived fields if not provided."""
        if self.centroid is None:
            x1, y1, x2, y2 = self.bbox_xyxy
            self.centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        if self.area is None:
            x1, y1, x2, y2 = self.bbox_xyxy
            self.area = (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x, y, w, h) format."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1, y1, x2 - x1, y2 - y1)


@dataclass
class PoseResult:
    """
    Pose estimation result for one detected animal.

    Produced by: pose module
    Consumed by: tracklet confidence scoring, global optimizer, behavior analysis
    """

    frame_idx: int
    detection_id: str  # Links back to the Detection

    # Keypoint data (from DLC SuperAnimal or similar)
    keypoints_xy: np.ndarray | None = None  # [K, 2] — keypoint coordinates
    keypoint_conf: np.ndarray | None = None  # [K] — per-keypoint confidence

    # Geometric features (always available if mask exists)
    orientation: float | None = None  # Radians, body heading
    body_length: float | None = None  # Major axis length (pixels)
    body_area: float | None = None  # Mask area (pixels²)
    aspect_ratio: float | None = None  # Major / minor axis
    centroid: tuple[float, float] | None = None

    source: PoseSource = PoseSource.GEOMETRY

    @property
    def has_keypoints(self) -> bool:
        return self.keypoints_xy is not None and len(self.keypoints_xy) > 0

    @property
    def mean_keypoint_confidence(self) -> float:
        if self.keypoint_conf is None or len(self.keypoint_conf) == 0:
            return 0.0
        return float(np.mean(self.keypoint_conf))


@dataclass
class Tracklet:
    """
    A locally-reliable short trajectory fragment.

    Produced by: ByteTrack / OC-SORT tracklet generator
    Consumed by: self-supervised ReID, global optimizer

    Design principle: prefer breaking a tracklet at occlusion
    over risking a wrong association. Global optimizer will
    reconnect tracklets later.
    """

    tracklet_id: int
    start_frame: int
    end_frame: int
    detections: list[Detection] = field(default_factory=list)
    poses: list[PoseResult] = field(default_factory=list)

    # Quality metrics
    confidence: float = 0.0  # Overall tracklet reliability [0, 1]
    had_occlusion: bool = False
    num_id_conflicts: int = 0  # How many frames had ambiguous association

    # Cached aggregate features (computed after tracklet is complete)
    _avg_area: float | None = field(default=None, repr=False)
    _last_velocity: np.ndarray | None = field(default=None, repr=False)

    @property
    def duration(self) -> int:
        """Number of frames in this tracklet."""
        return self.end_frame - self.start_frame + 1

    @property
    def first_position(self) -> np.ndarray:
        if not self.detections:
            return np.array([0.0, 0.0])
        c = self.detections[0].centroid
        return np.array(c) if c else np.array([0.0, 0.0])

    @property
    def last_position(self) -> np.ndarray:
        if not self.detections:
            return np.array([0.0, 0.0])
        c = self.detections[-1].centroid
        return np.array(c) if c else np.array([0.0, 0.0])

    @property
    def avg_area(self) -> float:
        if self._avg_area is not None:
            return self._avg_area
        areas = [d.area for d in self.detections if d.area is not None]
        self._avg_area = float(np.mean(areas)) if areas else 0.0
        return self._avg_area

    @property
    def last_velocity(self) -> np.ndarray:
        """Estimated velocity at the end of this tracklet (pixels/frame)."""
        if self._last_velocity is not None:
            return self._last_velocity
        if len(self.detections) < 2:
            self._last_velocity = np.array([0.0, 0.0])
        else:
            positions = np.array([
                d.centroid for d in self.detections[-5:]
                if d.centroid is not None
            ])
            if len(positions) >= 2:
                velocities = np.diff(positions, axis=0)
                self._last_velocity = velocities[-1]
            else:
                self._last_velocity = np.array([0.0, 0.0])
        return self._last_velocity

    @property
    def last_orientation(self) -> float:
        """Orientation at the end of this tracklet."""
        for pose in reversed(self.poses):
            if pose.orientation is not None:
                return pose.orientation
        return 0.0

    @property
    def first_orientation(self) -> float:
        """Orientation at the start of this tracklet."""
        for pose in self.poses:
            if pose.orientation is not None:
                return pose.orientation
        return 0.0

    @property
    def has_pose(self) -> bool:
        return len(self.poses) > 0

    @property
    def last_pose(self) -> PoseResult | None:
        return self.poses[-1] if self.poses else None

    @property
    def first_pose(self) -> PoseResult | None:
        return self.poses[0] if self.poses else None

    def invalidate_cache(self) -> None:
        """Call after adding new detections/poses."""
        self._avg_area = None
        self._last_velocity = None


@dataclass
class TrajectorySegment:
    """One continuous segment within a global trajectory, linking to a tracklet."""

    tracklet: Tracklet
    link_confidence: float  # Confidence of the link FROM the previous segment
    confidence_reason: str = ""  # Human-readable explanation

    @property
    def start(self) -> int:
        return self.tracklet.start_frame

    @property
    def end(self) -> int:
        return self.tracklet.end_frame

    @property
    def confidence(self) -> float:
        return self.tracklet.confidence

    @property
    def duration(self) -> int:
        return self.tracklet.duration


@dataclass
class Trajectory:
    """
    A globally-assigned animal trajectory spanning the full video.

    Produced by: global optimizer
    Consumed by: confidence auditor, export, behavior analysis

    Each trajectory consists of multiple TrajectorySegments
    (each backed by a Tracklet), connected with link_confidence
    scores at junctions (typically occlusion recovery points).
    """

    global_id: int
    segments: list[TrajectorySegment] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        if not self.segments:
            return 0
        return self.segments[0].start

    @property
    def end_frame(self) -> int:
        if not self.segments:
            return 0
        return self.segments[-1].end

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def mean_confidence(self) -> float:
        if not self.segments:
            return 0.0
        return float(np.mean([s.confidence for s in self.segments]))

    @property
    def min_link_confidence(self) -> float:
        """Weakest link in the chain — most likely point of ID error."""
        if not self.segments:
            return 0.0
        return float(min(s.link_confidence for s in self.segments))

    @property
    def num_occlusions(self) -> int:
        """Number of occlusion recovery events."""
        return sum(
            1 for s in self.segments
            if s.link_confidence < 0.8 and s.link_confidence > 0.0
        )

    @property
    def risky_links(self) -> list[dict]:
        """Links with low confidence (potential ID switches)."""
        risky = []
        for i, seg in enumerate(self.segments):
            if seg.link_confidence < 0.5 and i > 0:
                risky.append({
                    "segment_index": i,
                    "frame": seg.start,
                    "link_confidence": seg.link_confidence,
                    "reason": seg.confidence_reason,
                })
        return risky

    def add_segment(
        self,
        tracklet: Tracklet,
        link_confidence: float = 1.0,
        confidence_reason: str = "",
    ) -> None:
        """Append a new tracklet segment to this trajectory."""
        self.segments.append(TrajectorySegment(
            tracklet=tracklet,
            link_confidence=link_confidence,
            confidence_reason=confidence_reason,
        ))

    def get_all_detections(self) -> list[Detection]:
        """Flatten all detections across all segments."""
        detections = []
        for seg in self.segments:
            detections.extend(seg.tracklet.detections)
        return detections

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return ConfidenceLevel.from_score(self.min_link_confidence)
