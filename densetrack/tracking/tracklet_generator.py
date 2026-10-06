"""
ByteTrack-style tracklet generator.

Generates short, locally-reliable trajectory fragments (tracklets).
Intentionally uses a SHORT track buffer — prefer breaking at occlusion
over wrong association. The global optimizer will reconnect later.

Fully implemented.
"""

from __future__ import annotations

import logging

import numpy as np

from densetrack.config import TrackletConfig
from densetrack.models import Detection, Tracklet
from densetrack.tracking.kalman_filter import KalmanBoxTracker
from densetrack.tracking.matching import iou_batch, linear_assignment

__all__ = ["TrackletGenerator"]

logger = logging.getLogger(__name__)


class _InternalTrack:
    """Internal track state used during generation."""

    def __init__(self, detection: Detection, tracker: KalmanBoxTracker):
        self.detections: list[Detection] = [detection]
        self.tracker = tracker
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.had_occlusion = False


class TrackletGenerator:
    """
    ByteTrack-style tracklet generator.

    Two-stage association:
    1. High-confidence detections matched to existing tracks via IoU
    2. Remaining tracks matched to low-confidence detections

    Design choice: track_buffer is intentionally SHORT (~0.4s at 25fps).
    Occlusion beyond this duration breaks the tracklet, and the global
    optimizer is responsible for reconnecting tracklets across gaps.
    """

    def __init__(self, config: TrackletConfig):
        self.config = config
        self._next_id = 0

    def generate(
        self, detections_per_frame: dict[int, list[Detection]]
    ) -> list[Tracklet]:
        """
        Generate tracklets from frame-by-frame detections.

        Args:
            detections_per_frame: Dict mapping frame_idx -> list of Detection.

        Returns:
            List of Tracklet objects (locally-reliable short trajectories).
        """
        active_tracks: list[_InternalTrack] = []
        finished_tracks: list[_InternalTrack] = []

        sorted_frames = sorted(detections_per_frame.keys())

        for frame_idx in sorted_frames:
            detections = detections_per_frame[frame_idx]

            # Predict all active tracks
            predicted_bboxes = []
            for track in active_tracks:
                pred_bbox = track.tracker.predict()
                predicted_bboxes.append(pred_bbox)

            # Split detections by confidence
            high_dets = [d for d in detections if d.confidence >= self.config.tau_high]
            low_dets = [
                d for d in detections
                if self.config.tau_low <= d.confidence < self.config.tau_high
            ]

            # ---- Stage 1: Match high-conf detections to tracks ----
            if active_tracks and high_dets:
                pred_arr = np.array(predicted_bboxes)
                det_arr = np.array([d.bbox_xyxy for d in high_dets])
                iou_matrix = iou_batch(pred_arr, det_arr)
                cost_matrix = 1 - iou_matrix

                matches, unmatched_tracks, unmatched_dets = linear_assignment(
                    cost_matrix, thresh=1 - self.config.match_thresh
                )

                # Update matched tracks
                for track_idx, det_idx in matches:
                    active_tracks[track_idx].detections.append(high_dets[det_idx])
                    active_tracks[track_idx].tracker.update(
                        np.array(high_dets[det_idx].bbox_xyxy)
                    )
                    active_tracks[track_idx].time_since_update = 0
                    active_tracks[track_idx].hits += 1
                    active_tracks[track_idx].hit_streak += 1

                remaining_tracks = [active_tracks[i] for i in unmatched_tracks]
                remaining_high_dets = [high_dets[i] for i in unmatched_dets]
            else:
                remaining_tracks = list(active_tracks)
                remaining_high_dets = list(high_dets)

            # ---- Stage 2: Match remaining tracks to low-conf detections ----
            if remaining_tracks and low_dets:
                pred_arr = np.array([t.tracker.get_state() for t in remaining_tracks])
                det_arr = np.array([d.bbox_xyxy for d in low_dets])
                iou_matrix = iou_batch(pred_arr, det_arr)
                cost_matrix = 1 - iou_matrix

                matches2, still_unmatched, _ = linear_assignment(
                    cost_matrix, thresh=1 - self.config.match_thresh
                )

                for track_local_idx, det_idx in matches2:
                    track = remaining_tracks[track_local_idx]
                    track.detections.append(low_dets[det_idx])
                    track.tracker.update(np.array(low_dets[det_idx].bbox_xyxy))
                    track.time_since_update = 0
                    track.hits += 1
                    track.hit_streak += 1

                # Mark tracks that weren't matched in either stage
                for i in still_unmatched:
                    remaining_tracks[i].time_since_update += 1
                    remaining_tracks[i].hit_streak = 0
                    remaining_tracks[i].had_occlusion = True
            else:
                for track in remaining_tracks:
                    track.time_since_update += 1
                    track.hit_streak = 0
                    if track.time_since_update > 0:
                        track.had_occlusion = True

            # ---- Create new tracks from unmatched high-conf detections ----
            for det in remaining_high_dets:
                tracker = KalmanBoxTracker(np.array(det.bbox_xyxy))
                new_track = _InternalTrack(det, tracker)
                active_tracks.append(new_track)

            # ---- Remove dead tracks ----
            still_active = []
            for track in active_tracks:
                if track.time_since_update <= self.config.track_buffer:
                    still_active.append(track)
                else:
                    # Only save tracks with minimum length
                    if track.hits >= 2:
                        finished_tracks.append(track)
            active_tracks = still_active

        # Flush remaining active tracks
        for track in active_tracks:
            if track.hits >= 2:
                finished_tracks.append(track)

        # Convert to Tracklet objects
        tracklets = []
        for internal in finished_tracks:
            tracklet = self._to_tracklet(internal)
            tracklets.append(tracklet)

        logger.info(
            f"Generated {len(tracklets)} tracklets from "
            f"{len(sorted_frames)} frames "
            f"(avg length: {np.mean([t.duration for t in tracklets]):.1f} frames)"
            if tracklets else f"Generated 0 tracklets from {len(sorted_frames)} frames"
        )

        return tracklets

    def _to_tracklet(self, internal: _InternalTrack) -> Tracklet:
        """Convert internal track to a Tracklet dataclass."""
        # A tracklet has occlusion only if there is an actual frame gap within its detections.
        # Gaps that happen after its last detection just represent its end, not occlusion.
        has_gap = False
        for idx in range(len(internal.detections) - 1):
            if internal.detections[idx + 1].frame_idx - internal.detections[idx].frame_idx > 1:
                has_gap = True
                break

        tracklet = Tracklet(
            tracklet_id=self._next_id,
            start_frame=internal.detections[0].frame_idx,
            end_frame=internal.detections[-1].frame_idx,
            detections=internal.detections,
            had_occlusion=has_gap,
        )
        self._next_id += 1

        tracklet.confidence = self._calc_confidence(tracklet)
        return tracklet

    @staticmethod
    def _calc_confidence(tracklet: Tracklet) -> float:
        """
        Compute multi-factor tracklet confidence.

        Factors:
        1. Detection confidence mean
        2. Trajectory smoothness (velocity std — lower is more reliable)
        3. Duration (longer = more reliable, capped at 1 second)
        4. Occlusion penalty
        """
        if not tracklet.detections:
            return 0.0

        # 1. Detection confidence
        det_conf = float(np.mean([d.confidence for d in tracklet.detections]))

        # 2. Smoothness
        positions = np.array([
            d.centroid for d in tracklet.detections if d.centroid is not None
        ])
        if len(positions) >= 3:
            velocities = np.diff(positions, axis=0)
            speed_std = float(np.std(np.linalg.norm(velocities, axis=1)))
            smoothness = 1.0 / (1.0 + speed_std)
        else:
            smoothness = 0.5

        # 3. Duration factor
        duration = len(tracklet.detections)
        duration_factor = min(duration / 25.0, 1.0)  # 1 second = full score

        # 4. Occlusion penalty
        occlusion_penalty = 0.8 if tracklet.had_occlusion else 1.0

        confidence = det_conf * smoothness * duration_factor * occlusion_penalty
        return float(np.clip(confidence, 0.0, 1.0))
