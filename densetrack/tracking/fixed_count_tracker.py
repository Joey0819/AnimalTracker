"""
Fixed-count online tracker for known-animal-count top-view videos.

The tracker maintains exactly `n_animals` identities. It is intended for
visualization and auto-label bootstrapping when a stable camera sees a known
number of animals and detections may temporarily merge during close contact.
Observed detections update each ID; missing IDs are carried forward with a
constant-velocity prediction and low confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from densetrack.config import TrackletConfig
from densetrack.models import Detection, DetectionSource, Tracklet
from densetrack.tracking.matching import iou_batch, linear_assignment

__all__ = ["FixedCountTrackletGenerator", "FixedCountStats"]

logger = logging.getLogger(__name__)


@dataclass
class _TrackState:
    track_id: int
    bbox: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    area: float
    missing: int = 0


@dataclass
class FixedCountStats:
    """Summary of fixed-count tracking quality."""

    total_frames: int
    avg_detections_per_frame: float
    avg_observed_tracks_per_frame: float
    all_observed_ratio: float
    mostly_observed_ratio: float
    per_track_observed_ratio: dict[int, float]
    per_track_max_missing: dict[int, int]


class FixedCountTrackletGenerator:
    """Generate one full-length tracklet per expected animal."""

    def __init__(
        self,
        config: TrackletConfig,
        n_animals: int,
        frame_size: tuple[int, int] | None = None,
    ):
        self.config = config
        self.n_animals = n_animals
        self.frame_size = frame_size
        self.stats: FixedCountStats | None = None

    def generate(
        self, detections_per_frame: dict[int, list[Detection]]
    ) -> list[Tracklet]:
        frames = sorted(detections_per_frame)
        if not frames:
            self.stats = FixedCountStats(0, 0.0, 0.0, 0.0, 0.0, {}, {})
            return []

        init_frame = self._find_initialization_frame(detections_per_frame, frames)
        if init_frame is None:
            logger.warning(
                "Fixed-count tracker could not find a frame with enough detections; "
                "falling back to one track per available detection"
            )
            init_frame = frames[0]
        frames = [frame_idx for frame_idx in frames if frame_idx >= init_frame]

        states = self._initialize_tracks(detections_per_frame[init_frame])
        if not states:
            self.stats = FixedCountStats(len(frames), 0.0, 0.0, 0.0, 0.0, {}, {})
            return []

        records: dict[int, list[Detection]] = {state.track_id: [] for state in states}
        observed_flags: dict[int, list[bool]] = {state.track_id: [] for state in states}
        missing_history: dict[int, list[int]] = {state.track_id: [] for state in states}
        observed_counts: list[int] = []
        det_counts: list[int] = []

        for frame_idx in frames:
            detections = sorted(
                detections_per_frame.get(frame_idx, []),
                key=lambda det: det.area or 0.0,
                reverse=True,
            )
            det_counts.append(len(detections))

            predicted = [self._predict_state(state) for state in states]
            assignments = self._assign(states, predicted, detections)
            assigned_track_indices = {track_idx for track_idx, _ in assignments}
            observed_count = len(assigned_track_indices)
            observed_counts.append(observed_count)

            for track_idx, state in enumerate(states):
                detection = None
                for assigned_track_idx, det_idx in assignments:
                    if assigned_track_idx == track_idx:
                        detection = detections[det_idx]
                        break

                if detection is not None:
                    self._update_state(state, predicted[track_idx], detection)
                    record = detection
                    observed = True
                else:
                    self._mark_missing(state, predicted[track_idx], frame_idx)
                    record = self._prediction_detection(state, frame_idx)
                    observed = False

                records[state.track_id].append(record)
                observed_flags[state.track_id].append(observed)
                missing_history[state.track_id].append(state.missing)

        tracklets: list[Tracklet] = []
        for track_id in sorted(records):
            detections = records[track_id]
            if not detections:
                continue
            observed_ratio = float(np.mean(observed_flags[track_id]))
            tracklet = Tracklet(
                tracklet_id=track_id,
                start_frame=detections[0].frame_idx,
                end_frame=detections[-1].frame_idx,
                detections=detections,
                had_occlusion=observed_ratio < 0.98,
                confidence=observed_ratio,
            )
            tracklets.append(tracklet)

        self.stats = FixedCountStats(
            total_frames=len(frames),
            avg_detections_per_frame=float(np.mean(det_counts)) if det_counts else 0.0,
            avg_observed_tracks_per_frame=float(np.mean(observed_counts)) if observed_counts else 0.0,
            all_observed_ratio=float(np.mean([count == len(states) for count in observed_counts]))
            if observed_counts else 0.0,
            mostly_observed_ratio=float(np.mean([count >= max(1, len(states) - 1) for count in observed_counts]))
            if observed_counts else 0.0,
            per_track_observed_ratio={
                track_id: float(np.mean(flags)) for track_id, flags in observed_flags.items()
            },
            per_track_max_missing={
                track_id: int(max(values) if values else 0)
                for track_id, values in missing_history.items()
            },
        )

        logger.info(
            "Fixed-count tracker generated %s trajectories; avg observed %.2f/%s tracks/frame",
            len(tracklets),
            self.stats.avg_observed_tracks_per_frame if self.stats else 0.0,
            len(states),
        )
        return tracklets

    def _find_initialization_frame(
        self,
        detections_per_frame: dict[int, list[Detection]],
        frames: list[int],
    ) -> int | None:
        for frame_idx in frames:
            if len(detections_per_frame.get(frame_idx, [])) >= self.n_animals:
                return frame_idx
        return None

    def _initialize_tracks(self, detections: list[Detection]) -> list[_TrackState]:
        selected = sorted(detections, key=lambda det: det.area or 0.0, reverse=True)[
            : self.n_animals
        ]
        selected = sorted(selected, key=lambda det: (det.centroid[1], det.centroid[0]))

        states: list[_TrackState] = []
        for idx, det in enumerate(selected):
            bbox = np.asarray(det.bbox_xyxy, dtype=np.float64)
            center = np.asarray(det.centroid, dtype=np.float64)
            states.append(
                _TrackState(
                    track_id=idx,
                    bbox=bbox,
                    center=center,
                    velocity=np.zeros(2, dtype=np.float64),
                    area=float(det.area or self._bbox_area(bbox)),
                )
            )
        return states

    def _predict_state(self, state: _TrackState) -> tuple[np.ndarray, np.ndarray]:
        center = state.center + state.velocity
        width = max(state.bbox[2] - state.bbox[0], 1.0)
        height = max(state.bbox[3] - state.bbox[1], 1.0)
        bbox = np.array(
            [
                center[0] - width / 2,
                center[1] - height / 2,
                center[0] + width / 2,
                center[1] + height / 2,
            ],
            dtype=np.float64,
        )
        return center, bbox

    def _assign(
        self,
        states: list[_TrackState],
        predicted: list[tuple[np.ndarray, np.ndarray]],
        detections: list[Detection],
    ) -> list[tuple[int, int]]:
        if not states or not detections:
            return []

        pred_bboxes = np.asarray([bbox for _, bbox in predicted], dtype=np.float64)
        det_bboxes = np.asarray([det.bbox_xyxy for det in detections], dtype=np.float64)
        ious = iou_batch(pred_bboxes, det_bboxes)

        cost = np.full((len(states), len(detections)), 99.0, dtype=np.float64)
        for track_idx, state in enumerate(states):
            pred_center, _ = predicted[track_idx]
            max_distance = self.config.fixed_max_assignment_distance + 10.0 * state.missing
            for det_idx, det in enumerate(detections):
                det_center = np.asarray(det.centroid, dtype=np.float64)
                distance = float(np.linalg.norm(pred_center - det_center))
                distance_cost = distance / max(max_distance, 1.0)
                area_cost = abs(np.log((state.area + 1.0) / (float(det.area or 1.0) + 1.0)))
                cost[track_idx, det_idx] = (
                    0.65 * distance_cost
                    + 0.25 * (1.0 - ious[track_idx, det_idx])
                    + 0.10 * min(area_cost, 2.0)
                )

        matches, _, _ = linear_assignment(cost, thresh=1e6)

        accepted: list[tuple[int, int]] = []
        for track_idx, det_idx in matches:
            state = states[int(track_idx)]
            max_cost = 1.15 + 0.05 * state.missing
            if cost[track_idx, det_idx] <= max_cost:
                accepted.append((int(track_idx), int(det_idx)))
        return accepted

    def _update_state(
        self,
        state: _TrackState,
        prediction: tuple[np.ndarray, np.ndarray],
        detection: Detection,
    ) -> None:
        predicted_center, predicted_bbox = prediction
        measured_center = np.asarray(detection.centroid, dtype=np.float64)
        measured_bbox = np.asarray(detection.bbox_xyxy, dtype=np.float64)

        old_center = state.center.copy()
        new_center = 0.75 * measured_center + 0.25 * predicted_center
        instant_velocity = new_center - old_center

        state.velocity = 0.65 * state.velocity + 0.35 * instant_velocity
        state.center = new_center
        state.bbox = 0.80 * measured_bbox + 0.20 * predicted_bbox
        state.area = 0.80 * float(detection.area or self._bbox_area(measured_bbox)) + 0.20 * state.area
        state.missing = 0

    def _mark_missing(
        self,
        state: _TrackState,
        prediction: tuple[np.ndarray, np.ndarray],
        frame_idx: int,
    ) -> None:
        del frame_idx
        predicted_center, predicted_bbox = prediction
        if self.frame_size is not None:
            width, height = self.frame_size
            predicted_center = np.array(
                [
                    np.clip(predicted_center[0], 0, max(width - 1, 0)),
                    np.clip(predicted_center[1], 0, max(height - 1, 0)),
                ],
                dtype=np.float64,
            )
            bbox_width = max(predicted_bbox[2] - predicted_bbox[0], 1.0)
            bbox_height = max(predicted_bbox[3] - predicted_bbox[1], 1.0)
            predicted_bbox = np.array(
                [
                    predicted_center[0] - bbox_width / 2,
                    predicted_center[1] - bbox_height / 2,
                    predicted_center[0] + bbox_width / 2,
                    predicted_center[1] + bbox_height / 2,
                ],
                dtype=np.float64,
            )
        state.center = predicted_center
        state.bbox = predicted_bbox
        state.velocity *= 0.92
        state.missing += 1

    def _prediction_detection(self, state: _TrackState, frame_idx: int) -> Detection:
        confidence = max(
            0.15,
            self.config.fixed_prediction_confidence * float(np.exp(-state.missing / 25.0)),
        )
        bbox = tuple(float(v) for v in state.bbox)
        return Detection(
            frame_idx=frame_idx,
            bbox_xyxy=bbox,
            confidence=confidence,
            source=DetectionSource.FIXED_COUNT_PREDICTION,
            centroid=(float(state.center[0]), float(state.center[1])),
            area=float(self._bbox_area(state.bbox)),
        )

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        return float(max(1.0, bbox[2] - bbox[0]) * max(1.0, bbox[3] - bbox[1]))
