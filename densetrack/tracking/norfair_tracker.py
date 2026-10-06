"""Norfair-based tracking backend.

Norfair is a mature BSD-3-Clause multi-object tracking library. DenseTrack keeps
this adapter thin: convert local detections to Norfair detections, run Norfair's
tracker, and convert matched updates back to DenseTrack tracklets.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from densetrack.config import TrackletConfig
from densetrack.models import Detection, Tracklet

__all__ = ["NorfairTrackletGenerator"]

logger = logging.getLogger(__name__)


class NorfairTrackletGenerator:
    """Generate tracklets with the third-party Norfair tracker."""

    def __init__(self, config: TrackletConfig):
        self.config = config

    def generate(
        self,
        detections_per_frame: dict[int, list[Detection]],
    ) -> list[Tracklet]:
        norfair = self._import_norfair()
        tracker = norfair.Tracker(
            distance_function="euclidean",
            distance_threshold=self.config.fixed_max_assignment_distance,
            hit_counter_max=self.config.track_buffer,
            initialization_delay=0,
            detection_threshold=self.config.tau_low,
            past_detections_length=max(1, min(self.config.track_buffer, 8)),
        )
        records: dict[int, list[Detection]] = defaultdict(list)

        for frame_idx in sorted(detections_per_frame):
            norfair_detections = [
                norfair.Detection(
                    points=np.asarray([detection.centroid], dtype=np.float32),
                    scores=np.asarray([detection.confidence], dtype=np.float32),
                    data=detection,
                    label="animal",
                )
                for detection in detections_per_frame.get(frame_idx, [])
                if detection.confidence >= self.config.tau_low
            ]
            tracked_objects = tracker.update(detections=norfair_detections)
            for tracked in tracked_objects:
                track_id = getattr(tracked, "id", None)
                last_detection = getattr(tracked, "last_detection", None)
                source_detection = getattr(last_detection, "data", None)
                if track_id is None or not isinstance(source_detection, Detection):
                    continue
                if source_detection.frame_idx != frame_idx:
                    continue
                records[int(track_id)].append(source_detection)

        tracklets = [
            self._to_tracklet(track_id, detections)
            for track_id, detections in sorted(records.items())
            if len(detections) >= 2
        ]
        logger.info(
            "Norfair generated %s tracklets from %s frames",
            len(tracklets),
            len(detections_per_frame),
        )
        return tracklets

    def _to_tracklet(self, track_id: int, detections: list[Detection]) -> Tracklet:
        ordered = sorted(detections, key=lambda detection: detection.frame_idx)
        had_occlusion = any(
            ordered[idx + 1].frame_idx - ordered[idx].frame_idx > 1
            for idx in range(len(ordered) - 1)
        )
        tracklet = Tracklet(
            tracklet_id=track_id,
            start_frame=ordered[0].frame_idx,
            end_frame=ordered[-1].frame_idx,
            detections=ordered,
            had_occlusion=had_occlusion,
        )
        tracklet.confidence = self._calc_confidence(tracklet)
        return tracklet

    @staticmethod
    def _calc_confidence(tracklet: Tracklet) -> float:
        if not tracklet.detections:
            return 0.0
        detection_confidence = float(
            np.mean([detection.confidence for detection in tracklet.detections])
        )
        duration_factor = min(len(tracklet.detections) / 25.0, 1.0)
        occlusion_penalty = 0.85 if tracklet.had_occlusion else 1.0
        return float(np.clip(detection_confidence * duration_factor * occlusion_penalty, 0.0, 1.0))

    @staticmethod
    def _import_norfair() -> Any:
        try:
            import norfair
        except ImportError as exc:
            raise ImportError(
                "Norfair backend requested but norfair is not installed. "
                "Install with `pip install -e '.[norfair]'` or `pip install norfair`. "
                "/ 已选择 Norfair 后端，但当前环境未安装 norfair；请安装 `densetrack[norfair]`。"
            ) from exc
        return norfair
