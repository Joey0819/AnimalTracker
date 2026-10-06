"""
Mac-native zero-click adaptive bidirectional tracker.

This tracker is an independent local algorithm for stable-camera videos with a
known animal count. It starts from detector-selected frames near both ends of
the video, or from optional first-frame clicks for the forward identity space,
tracks in both directions, adapts lightweight appearance templates, aligns the
two identity spaces, and fuses their frame-level assignments.

It intentionally uses NumPy, OpenCV, and Hungarian assignment only. Training
labels, CUDA runtime, and external research trackers are not needed. Optional
first-frame clicks are lightweight instance selection, not training annotation.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from densetrack.config import TrackletConfig
from densetrack.models import Detection, DetectionSource, Tracklet
from densetrack.tracking.discrete_geometry import (
    HierarchicalGeometryEncoder,
    estimate_knn_velocity,
)
from densetrack.tracking.matching import iou_batch, linear_assignment

__all__ = ["MacAdaptiveTrackletGenerator", "MacAdaptiveStats"]

logger = logging.getLogger(__name__)


@dataclass
class _AdaptiveState:
    track_id: int
    bbox: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    area: float
    descriptor: np.ndarray | None
    anchor_descriptor: np.ndarray | None
    descriptor_bank: list[np.ndarray]
    observed_positions: list[tuple[int, np.ndarray]]
    geometry_history: list[tuple[np.ndarray, np.ndarray]]
    missing: int = 0


@dataclass
class _DirectionalRecord:
    detection: Detection
    observed: bool
    assignment_cost: float


@dataclass
class _DirectionalPass:
    records: dict[int, dict[int, _DirectionalRecord]]
    init_frame: int
    descriptor_updates: int
    observed_assignments: int
    reacquisition_events: int
    deferred_ambiguities: int
    contact_deferred_assignments: int
    merged_detection_rejections: int
    geometry_comparisons: int
    geometry_consistent: int


@dataclass
class AdaptiveRiskFrame:
    """Frame-level identity risk produced by bidirectional consistency checks."""

    frame: int
    risk_score: float
    disagreement_count: int
    one_sided_count: int
    prediction_count: int
    observed_count: int
    reasons: str


@dataclass
class MacAdaptiveStats:
    """Summary of adaptive bidirectional tracking quality."""

    total_frames: int
    avg_detections_per_frame: float
    avg_observed_tracks_per_frame: float
    all_observed_ratio: float
    mostly_observed_ratio: float
    per_track_observed_ratio: dict[int, float]
    per_track_max_missing: dict[int, int]
    forward_init_frame: int
    backward_init_frame: int
    bidirectional_agreement_ratio: float
    appearance_descriptor_ratio: float
    fusion_conflicts: int
    disagreement_frames: int
    high_risk_frames: int
    identity_smoothing_changes: int = 0
    identity_stability_score: float = 0.0
    quantized_geometry_enabled: bool = False
    geometry_token_agreement_ratio: float = 0.0
    reacquisition_events: int = 0
    deferred_ambiguous_assignments: int = 0
    contact_deferred_assignments: int = 0
    merged_detection_rejections: int = 0
    motion_outlier_rejections: int = 0
    large_jump_frames: int = 0
    identity_switch_risk_events: int = 0
    max_smoothing_states: int = 0
    algorithm: str = "mac_adaptive_bidirectional"


class MacAdaptiveTrackletGenerator:
    """Generate full-length trajectories with zero-click bidirectional fusion."""

    def __init__(
        self,
        config: TrackletConfig,
        n_animals: int,
        frame_size: tuple[int, int] | None = None,
        initialization_points: list[dict] | None = None,
        initialization_frame: int | None = None,
    ):
        self.config = config
        self.n_animals = n_animals
        self.frame_size = frame_size
        self.initialization_points = initialization_points or []
        self.initialization_frame = initialization_frame
        self.stats: MacAdaptiveStats | None = None
        self.risk_frames: list[AdaptiveRiskFrame] = []
        self._descriptor_cache: dict[str, np.ndarray | None] = {}
        self._geometry_encoder = HierarchicalGeometryEncoder(
            frame_size=frame_size,
            coarse_bins=config.adaptive_geometry_coarse_bins,
            fine_bins=config.adaptive_geometry_fine_bins,
            max_speed=config.adaptive_consistency_distance,
        )
        self._max_smoothing_states = 0
        self._record_velocity_by_frame: dict[tuple[int, int], np.ndarray] = {}
        self._reference_detection_area = 0.0

    def generate(
        self,
        detections_per_frame: dict[int, list[Detection]],
        total_frames: int | None = None,
    ) -> list[Tracklet]:
        self._max_smoothing_states = 0
        self._record_velocity_by_frame = {}
        self._reference_detection_area = self._estimate_reference_detection_area(
            detections_per_frame
        )
        if not detections_per_frame:
            self.stats = self._empty_stats()
            return []

        detected_frames = sorted(detections_per_frame)
        n_frames = total_frames or detected_frames[-1] + 1
        valid_frames = [
            frame_idx
            for frame_idx in detected_frames
            if len(detections_per_frame.get(frame_idx, [])) >= self.n_animals
        ]
        if not valid_frames:
            logger.warning(
                "Mac-adaptive tracker could not find a frame with enough detections; "
                "falling back to the fixed-count tracker"
            )
            return self._fallback_fixed_count(detections_per_frame)
        if self.initialization_points:
            if len(self.initialization_points) != self.n_animals:
                raise ValueError(
                    "Click-assisted initialization must provide exactly one point per animal "
                    "/ 点击辅助初始化必须为每只动物提供一个点"
                )
            if self.initialization_frame not in valid_frames:
                raise ValueError(
                    "Click-assisted initialization frame does not have enough detections "
                    "/ 点击辅助初始化帧中的检测数量不足"
                )

        forward_init = self._select_initialization_frame(
            detections_per_frame,
            valid_frames,
            direction=1,
        )
        backward_init = self._select_initialization_frame(
            detections_per_frame,
            valid_frames,
            direction=-1,
        )

        forward = self._run_direction(
            detections_per_frame,
            init_frame=forward_init,
            end_frame=n_frames - 1,
            direction=1,
        )
        backward = self._run_direction(
            detections_per_frame,
            init_frame=backward_init,
            end_frame=0,
            direction=-1,
        )
        backward_to_forward = self._align_identity_spaces(forward, backward)
        raw_records, fusion_conflicts, raw_risk_frames = self._fuse_passes(
            detections_per_frame=detections_per_frame,
            forward=forward,
            backward=backward,
            backward_to_forward=backward_to_forward,
            total_frames=n_frames,
        )
        smoothing_changes = 0
        motion_outlier_rejections = 0
        records = raw_records
        if self.config.adaptive_identity_smoothing:
            records, smoothing_changes = self._smooth_identity_assignments(
                raw_records=raw_records,
                detections_per_frame=detections_per_frame,
                forward=forward,
                backward=backward,
                backward_to_forward=backward_to_forward,
                total_frames=n_frames,
            )
            records, motion_outlier_rejections = self._stabilize_motion_records(
                records,
                total_frames=n_frames,
            )
            self.risk_frames = self._stable_risk_frames(
                records=records,
                raw_records=raw_records,
                raw_risk_frames=raw_risk_frames,
                total_frames=n_frames,
            )
        else:
            self.risk_frames = raw_risk_frames
        tracklets = self._to_tracklets(records)
        self.stats = self._build_stats(
            detections_per_frame=detections_per_frame,
            records=records,
            total_frames=n_frames,
            forward=forward,
            backward=backward,
            backward_to_forward=backward_to_forward,
            fusion_conflicts=fusion_conflicts,
            identity_smoothing_changes=smoothing_changes,
            motion_outlier_rejections=motion_outlier_rejections,
        )
        logger.info(
            "Mac-adaptive tracker generated %s trajectories; observed %.2f/%s "
            "tracks/frame; bidirectional agreement %.1f%%",
            len(tracklets),
            self.stats.avg_observed_tracks_per_frame,
            self.n_animals,
            self.stats.bidirectional_agreement_ratio * 100,
        )
        return tracklets

    def write_identity_risk(self, output_dir: str | Path) -> dict[str, str]:
        """Write bilingual frame-level identity risk artifacts for experiment review."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "identity_risk.json"
        csv_path = output / "identity_risk.csv"
        high_risk = [frame for frame in self.risk_frames if frame.risk_score >= 0.50]
        payload = {
            "created_at": datetime.now().isoformat(),
            "format": "densetrack_identity_risk",
            "format_version": "1.0",
            "algorithm": (
                self.stats.algorithm
                if self.stats is not None
                else "mac_adaptive_bidirectional"
            ),
            "summary": {
                "total_frames": len(self.risk_frames),
                "disagreement_frames": sum(frame.disagreement_count > 0 for frame in self.risk_frames),
                "high_risk_frames": len(high_risk),
                "high_risk_ratio": len(high_risk) / max(len(self.risk_frames), 1),
                "mean_risk_score": float(
                    np.mean([frame.risk_score for frame in self.risk_frames])
                )
                if self.risk_frames
                else 0.0,
                "identity_smoothing_changes": (
                    self.stats.identity_smoothing_changes if self.stats is not None else 0
                ),
                "identity_stability_score": (
                    self.stats.identity_stability_score if self.stats is not None else 0.0
                ),
                "quantized_geometry_enabled": (
                    self.stats.quantized_geometry_enabled if self.stats is not None else False
                ),
                "geometry_token_agreement_ratio": (
                    self.stats.geometry_token_agreement_ratio if self.stats is not None else 0.0
                ),
                "reacquisition_events": (
                    self.stats.reacquisition_events if self.stats is not None else 0
                ),
                "deferred_ambiguous_assignments": (
                    self.stats.deferred_ambiguous_assignments if self.stats is not None else 0
                ),
                "contact_deferred_assignments": (
                    self.stats.contact_deferred_assignments if self.stats is not None else 0
                ),
                "merged_detection_rejections": (
                    self.stats.merged_detection_rejections if self.stats is not None else 0
                ),
                "motion_outlier_rejections": (
                    self.stats.motion_outlier_rejections if self.stats is not None else 0
                ),
                "large_jump_frames": (
                    self.stats.large_jump_frames if self.stats is not None else 0
                ),
                "identity_switch_risk_events": (
                    self.stats.identity_switch_risk_events if self.stats is not None else 0
                ),
                "max_smoothing_states": (
                    self.stats.max_smoothing_states if self.stats is not None else 0
                ),
                "warning": (
                    "Identity risk is an annotation-free audit signal, not a scientific accuracy metric. "
                    "/ 身份风险是无标注审计信号，不是科学准确率。"
                ),
            },
            "method": {
                "profile": "hierarchical_discrete_geometry_long_term",
                "geometry": (
                    "Fixed two-stage residual tokens for normalized position, scale, and "
                    "polar velocity; HieDG-inspired, not a reproduction of learned HieDG."
                ),
                "long_term": (
                    "Bidirectional closed-set smoothing, contact-aware ambiguity "
                    "deferral, merged-blob rejection, hard reacquisition gating, "
                    "motion-outlier rejection, and non-capped jump penalties."
                ),
                "references": [
                    "https://arxiv.org/abs/2607.00494",
                    "https://arxiv.org/abs/2607.08729",
                ],
            },
            "frames": [frame.__dict__ for frame in self.risk_frames],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as target:
            fieldnames = list(AdaptiveRiskFrame.__dataclass_fields__)
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for frame in self.risk_frames:
                writer.writerow(frame.__dict__)
        return {"identity_risk_json": str(json_path), "identity_risk_csv": str(csv_path)}

    def _run_direction(
        self,
        detections_per_frame: dict[int, list[Detection]],
        init_frame: int,
        end_frame: int,
        direction: int,
    ) -> _DirectionalPass:
        initialization_points = (
            self.initialization_points
            if direction > 0 and init_frame == self.initialization_frame
            else None
        )
        states, seed_detections = self._initialize_tracks(
            detections_per_frame[init_frame],
            initialization_points=initialization_points,
        )
        records = {state.track_id: {} for state in states}
        for state, detection in zip(states, seed_detections):
            records[state.track_id][init_frame] = _DirectionalRecord(
                detection=detection,
                observed=True,
                assignment_cost=0.0,
            )

        descriptor_updates = sum(state.descriptor is not None for state in states)
        observed_assignments = len(states)
        reacquisition_events = 0
        deferred_ambiguities = 0
        contact_deferred_assignments = 0
        merged_detection_rejections = 0
        geometry_comparisons = 0
        geometry_consistent = 0
        stop = end_frame + direction
        for frame_idx in range(init_frame + direction, stop, direction):
            detections = sorted(
                detections_per_frame.get(frame_idx, []),
                key=lambda detection: detection.area or 0.0,
                reverse=True,
            )
            predictions = [self._predict_state(state) for state in states]
            (
                assignments,
                deferred,
                contact_deferred,
                merged_rejections,
            ) = self._assign(states, predictions, detections)
            deferred_ambiguities += deferred
            contact_deferred_assignments += contact_deferred
            merged_detection_rejections += merged_rejections
            by_track = {
                track_idx: (
                    det_idx,
                    assignment_cost,
                    assignment_margin,
                    geometry_distance,
                )
                for (
                    track_idx,
                    det_idx,
                    assignment_cost,
                    assignment_margin,
                    geometry_distance,
                ) in assignments
            }
            for track_idx, state in enumerate(states):
                assignment = by_track.get(track_idx)
                if assignment is None:
                    self._mark_missing(state, predictions[track_idx])
                    record = _DirectionalRecord(
                        detection=self._prediction_detection(state, frame_idx),
                        observed=False,
                        assignment_cost=1.0 + min(state.missing / 25.0, 1.0),
                    )
                else:
                    (
                        det_idx,
                        assignment_cost,
                        assignment_margin,
                        geometry_distance,
                    ) = assignment
                    detection = detections[det_idx]
                    reacquisition_events += int(state.missing > 0)
                    updated_descriptor = self._update_state(
                        state,
                        predictions[track_idx],
                        detection,
                        assignment_cost,
                        assignment_margin,
                    )
                    descriptor_updates += int(updated_descriptor)
                    observed_assignments += 1
                    if self.config.adaptive_discrete_geometry:
                        geometry_comparisons += 1
                        geometry_consistent += int(geometry_distance <= 0.35)
                    record = _DirectionalRecord(
                        detection=detection,
                        observed=True,
                        assignment_cost=assignment_cost,
                    )
                records[state.track_id][frame_idx] = record

        return _DirectionalPass(
            records=records,
            init_frame=init_frame,
            descriptor_updates=descriptor_updates,
            observed_assignments=observed_assignments,
            reacquisition_events=reacquisition_events,
            deferred_ambiguities=deferred_ambiguities,
            contact_deferred_assignments=contact_deferred_assignments,
            merged_detection_rejections=merged_detection_rejections,
            geometry_comparisons=geometry_comparisons,
            geometry_consistent=geometry_consistent,
        )

    def _select_initialization_frame(
        self,
        detections_per_frame: dict[int, list[Detection]],
        valid_frames: list[int],
        direction: int,
    ) -> int:
        if (
            direction > 0
            and self.initialization_points
            and self.initialization_frame in valid_frames
        ):
            return int(self.initialization_frame)
        if direction > 0:
            edge = valid_frames[0]
            candidates = [
                frame_idx
                for frame_idx in valid_frames
                if frame_idx <= edge + self.config.adaptive_init_window
            ]
        else:
            edge = valid_frames[-1]
            candidates = [
                frame_idx
                for frame_idx in valid_frames
                if frame_idx >= edge - self.config.adaptive_init_window
            ]
        return max(
            candidates,
            key=lambda frame_idx: (
                self._initialization_score(detections_per_frame[frame_idx])
                - 0.15
                * abs(frame_idx - edge)
                / max(self.config.adaptive_init_window, 1),
                -frame_idx if direction > 0 else frame_idx,
            ),
        )

    def _initialization_score(self, detections: list[Detection]) -> float:
        selected = sorted(
            detections,
            key=lambda detection: detection.area or 0.0,
            reverse=True,
        )[: self.n_animals]
        centers = np.asarray([detection.centroid for detection in selected], dtype=np.float64)
        if len(centers) <= 1:
            separation = 1.0
        else:
            distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            distances += np.eye(len(centers)) * 1e9
            scale = self._frame_diagonal()
            separation = min(float(np.min(distances)) / scale, 1.0)
        areas = np.asarray([float(detection.area or 1.0) for detection in selected])
        area_consistency = float(1.0 / (1.0 + np.std(areas) / max(np.mean(areas), 1.0)))
        confidence = float(np.mean([detection.confidence for detection in selected]))
        return 0.55 * separation + 0.25 * area_consistency + 0.20 * confidence

    def _initialize_tracks(
        self,
        detections: list[Detection],
        initialization_points: list[dict] | None = None,
    ) -> tuple[list[_AdaptiveState], list[Detection]]:
        if initialization_points:
            selected = self._match_initialization_points(detections, initialization_points)
        else:
            selected = sorted(
                detections,
                key=lambda detection: detection.area or 0.0,
                reverse=True,
            )[: self.n_animals]
            selected = sorted(
                selected,
                key=lambda detection: (detection.centroid[1], detection.centroid[0]),
            )
        states = []
        for track_id, detection in enumerate(selected):
            bbox = np.asarray(detection.bbox_xyxy, dtype=np.float64)
            center = np.asarray(detection.centroid, dtype=np.float64)
            zero_velocity = np.zeros(2, dtype=np.float64)
            states.append(
                _AdaptiveState(
                    track_id=track_id,
                    bbox=bbox,
                    center=center,
                    velocity=zero_velocity,
                    area=float(detection.area or self._bbox_area(bbox)),
                    descriptor=self._cached_appearance_descriptor(detection),
                    anchor_descriptor=self._cached_appearance_descriptor(detection),
                    descriptor_bank=self._initial_descriptor_bank(detection),
                    observed_positions=[(detection.frame_idx, center.copy())],
                    geometry_history=[(center.copy(), zero_velocity.copy())],
                )
            )
        return states, selected

    def _match_initialization_points(
        self,
        detections: list[Detection],
        initialization_points: list[dict],
    ) -> list[Detection]:
        """Match click order to detected instances so click order becomes identity order."""
        if len(detections) < len(initialization_points):
            raise ValueError(
                "Not enough detected animals on the click frame / 点击帧中检测到的动物数量不足"
            )
        points = np.asarray(
            [[float(point["x"]), float(point["y"])] for point in initialization_points],
            dtype=np.float64,
        )
        centers = np.asarray([detection.centroid for detection in detections], dtype=np.float64)
        cost = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        matches, _, _ = linear_assignment(cost, thresh=self._frame_diagonal())
        by_point = {int(point_idx): int(detection_idx) for point_idx, detection_idx in matches}
        if len(by_point) != len(initialization_points):
            raise ValueError(
                "Could not match every click to a detected animal / 无法将每个点击匹配到检测到的动物"
            )
        return [detections[by_point[point_idx]] for point_idx in range(len(initialization_points))]

    def _assign(
        self,
        states: list[_AdaptiveState],
        predictions: list[tuple[np.ndarray, np.ndarray]],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int, float, float, float]], int, int, int]:
        if not states or not detections:
            return [], 0, 0, 0

        predicted_bboxes = np.asarray([bbox for _, bbox in predictions], dtype=np.float64)
        predicted_centers = np.asarray(
            [center for center, _ in predictions],
            dtype=np.float64,
        )
        detection_bboxes = np.asarray(
            [detection.bbox_xyxy for detection in detections],
            dtype=np.float64,
        )
        detection_centers = np.asarray(
            [detection.centroid for detection in detections],
            dtype=np.float64,
        )
        ious = iou_batch(predicted_bboxes, detection_bboxes)
        detection_ious = iou_batch(detection_bboxes, detection_bboxes)
        detection_distances = np.linalg.norm(
            detection_centers[:, None, :] - detection_centers[None, :, :],
            axis=2,
        )
        np.fill_diagonal(detection_distances, np.inf)
        np.fill_diagonal(detection_ious, 0.0)
        close_contact = (
            np.min(detection_distances, axis=1) < self.config.adaptive_contact_distance
        ) | (np.max(detection_ious, axis=1) > 0.08)
        reference_area = max(
            self._reference_detection_area,
            float(np.median([max(state.area, 1.0) for state in states])),
            1.0,
        )
        merged_detection = []
        for detection, bbox in zip(detections, detection_bboxes, strict=True):
            contained_predictions = sum(
                self._point_in_bbox(center, bbox, padding=10.0)
                for center in predicted_centers
            )
            merged_detection.append(
                contained_predictions >= 2
                and float(detection.area or 0.0)
                >= reference_area * self.config.adaptive_merged_area_ratio
            )
        descriptors = [self._cached_appearance_descriptor(detection) for detection in detections]
        cost = np.full((len(states), len(detections)), 99.0, dtype=np.float64)
        geometry_distance = np.ones_like(cost)
        appearance_distance = np.ones_like(cost)
        geometry_history = [
            sample
            for state in states
            for sample in state.geometry_history[-128:]
        ]

        for track_idx, state in enumerate(states):
            predicted_center, predicted_bbox = predictions[track_idx]
            max_distance = min(
                self.config.adaptive_max_reacquisition_distance,
                self.config.fixed_max_assignment_distance
                + self.config.adaptive_reacquisition_distance_growth
                * float(np.sqrt(state.missing)),
            )
            for detection_idx, detection in enumerate(detections):
                detection_center = np.asarray(detection.centroid, dtype=np.float64)
                distance = float(
                    np.linalg.norm(predicted_center - detection_center)
                )
                if distance > max_distance:
                    continue
                parts = [
                    (0.30, distance / max(max_distance, 1.0)),
                    (0.12, 1.0 - ious[track_idx, detection_idx]),
                    (
                        0.08,
                        min(
                            abs(
                                np.log(
                                    (state.area + 1.0)
                                    / (float(detection.area or 1.0) + 1.0)
                                )
                            ),
                            2.0,
                        )
                        / 2.0,
                    ),
                ]
                if self.config.adaptive_discrete_geometry:
                    measured_velocity = (
                        detection_center - state.center
                    ) / max(state.missing + 1, 1)
                    knn_velocity = estimate_knn_velocity(
                        detection_center,
                        measured_velocity,
                        geometry_history,
                        k=self.config.adaptive_geometry_knn,
                        spatial_scale=self.config.adaptive_consistency_distance,
                    )
                    candidate_velocity = 0.55 * measured_velocity + 0.45 * knn_velocity
                    expected_token = self._geometry_encoder.encode(
                        predicted_bbox,
                        state.velocity,
                    )
                    candidate_token = self._geometry_encoder.encode(
                        detection.bbox_xyxy,
                        candidate_velocity,
                    )
                    token_distance = self._geometry_encoder.distance(
                        expected_token,
                        candidate_token,
                        position_tolerance=min(state.missing * 0.006, 0.08),
                    )
                    geometry_distance[track_idx, detection_idx] = token_distance
                    parts.append((self.config.adaptive_geometry_weight, token_distance))
                descriptor = descriptors[detection_idx]
                if state.descriptor is not None and descriptor is not None:
                    descriptor_distance = self._state_descriptor_distance(state, descriptor)
                    appearance_distance[track_idx, detection_idx] = descriptor_distance
                    parts.append(
                        (
                            self.config.adaptive_appearance_weight,
                            descriptor_distance,
                        )
                    )
                weight_sum = sum(weight for weight, _ in parts)
                cost[track_idx, detection_idx] = sum(
                    weight * value for weight, value in parts
                ) / max(weight_sum, 1e-9)

        matches, _, _ = linear_assignment(cost, thresh=1e6)
        accepted = []
        deferred = 0
        contact_deferred = 0
        merged_rejections = 0
        for track_idx, detection_idx in matches:
            state = states[int(track_idx)]
            assignment_cost = float(cost[track_idx, detection_idx])
            max_cost = 0.88 + 0.025 * min(state.missing, 8)
            if assignment_cost <= max_cost:
                row_alternatives = np.delete(cost[int(track_idx)], int(detection_idx))
                col_alternatives = np.delete(cost[:, int(detection_idx)], int(track_idx))
                alternatives = np.concatenate([row_alternatives, col_alternatives])
                assignment_margin = (
                    float(np.min(alternatives) - assignment_cost)
                    if alternatives.size
                    else 1.0
                )
                token_distance = float(geometry_distance[track_idx, detection_idx])
                descriptor_distance = float(appearance_distance[track_idx, detection_idx])
                if merged_detection[int(detection_idx)]:
                    deferred += 1
                    merged_rejections += 1
                    continue
                if (
                    bool(close_contact[int(detection_idx)])
                    and assignment_margin < self.config.adaptive_contact_min_margin
                ):
                    deferred += 1
                    contact_deferred += 1
                    continue
                if (
                    self.config.adaptive_discrete_geometry
                    and state.missing > 0
                    and assignment_margin < self.config.adaptive_reacquisition_min_margin
                    and token_distance > 0.10
                    and descriptor_distance > 0.10
                ):
                    deferred += 1
                    continue
                accepted.append(
                    (
                        int(track_idx),
                        int(detection_idx),
                        assignment_cost,
                        assignment_margin,
                        token_distance,
                    )
                )
        return accepted, deferred, contact_deferred, merged_rejections

    def _update_state(
        self,
        state: _AdaptiveState,
        prediction: tuple[np.ndarray, np.ndarray],
        detection: Detection,
        assignment_cost: float,
        assignment_margin: float,
    ) -> bool:
        predicted_center, predicted_bbox = prediction
        measured_center = np.asarray(detection.centroid, dtype=np.float64)
        measured_bbox = np.asarray(detection.bbox_xyxy, dtype=np.float64)
        state.center = 0.78 * measured_center + 0.22 * predicted_center
        state.observed_positions.append((detection.frame_idx, measured_center.copy()))
        state.observed_positions = state.observed_positions[
            -(self.config.adaptive_velocity_window + 1):
        ]
        observed_velocities = []
        for (left_frame, left_center), (right_frame, right_center) in zip(
            state.observed_positions,
            state.observed_positions[1:],
        ):
            frame_delta = max(abs(int(right_frame - left_frame)), 1)
            observed_velocities.append((right_center - left_center) / frame_delta)
        if observed_velocities:
            robust_velocity = np.median(
                np.asarray(observed_velocities, dtype=np.float64),
                axis=0,
            )
            state.velocity = 0.45 * state.velocity + 0.55 * robust_velocity
        state.bbox = 0.82 * measured_bbox + 0.18 * predicted_bbox
        state.area = 0.82 * float(detection.area or self._bbox_area(measured_bbox)) + 0.18 * state.area
        state.missing = 0
        state.geometry_history.append((state.center.copy(), state.velocity.copy()))
        state.geometry_history = state.geometry_history[-256:]

        descriptor = self._cached_appearance_descriptor(detection)
        if (
            descriptor is None
            or assignment_cost > self.config.adaptive_template_update_cost
            or assignment_margin < self.config.adaptive_template_min_margin
            or detection.confidence < self.config.adaptive_template_min_confidence
        ):
            return False
        if state.descriptor is None:
            state.descriptor = descriptor
        else:
            momentum = self.config.adaptive_template_momentum
            state.descriptor = self._normalize((1.0 - momentum) * state.descriptor + momentum * descriptor)
        state.descriptor_bank.append(descriptor)
        state.descriptor_bank = state.descriptor_bank[-self.config.adaptive_template_bank_size :]
        return True

    def _mark_missing(
        self,
        state: _AdaptiveState,
        prediction: tuple[np.ndarray, np.ndarray],
    ) -> None:
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
            predicted_bbox = self._bbox_at_center(predicted_center, predicted_bbox)
        state.center = predicted_center
        state.bbox = predicted_bbox
        state.velocity *= 0.92
        state.missing += 1

    def _align_identity_spaces(
        self,
        forward: _DirectionalPass,
        backward: _DirectionalPass,
    ) -> dict[int, int]:
        cost = np.zeros((self.n_animals, self.n_animals), dtype=np.float64)
        diagonal = self._frame_diagonal()
        for forward_id in range(self.n_animals):
            for backward_id in range(self.n_animals):
                forward_records = forward.records[forward_id]
                backward_records = backward.records[backward_id]
                common = sorted(set(forward_records) & set(backward_records))
                distances = []
                shared_detections = 0
                both_observed = 0
                for frame_idx in common:
                    forward_record = forward_records[frame_idx]
                    backward_record = backward_records[frame_idx]
                    if forward_record.observed and backward_record.observed:
                        both_observed += 1
                        if forward_record.detection.detection_id == backward_record.detection.detection_id:
                            shared_detections += 1
                    distances.append(
                        np.linalg.norm(
                            np.asarray(forward_record.detection.centroid)
                            - np.asarray(backward_record.detection.centroid)
                        )
                    )
                median_distance = float(np.median(distances)) if distances else diagonal
                shared_ratio = shared_detections / max(both_observed, 1)
                cost[forward_id, backward_id] = median_distance / diagonal - 1.5 * shared_ratio
        matches, _, _ = linear_assignment(cost, thresh=1e6)
        return {int(backward_id): int(forward_id) for forward_id, backward_id in matches}

    def _fuse_passes(
        self,
        detections_per_frame: dict[int, list[Detection]],
        forward: _DirectionalPass,
        backward: _DirectionalPass,
        backward_to_forward: dict[int, int],
        total_frames: int,
    ) -> tuple[dict[int, dict[int, _DirectionalRecord]], int, list[AdaptiveRiskFrame]]:
        forward_to_backward = {
            forward_id: backward_id for backward_id, forward_id in backward_to_forward.items()
        }
        fused = {track_id: {} for track_id in range(self.n_animals)}
        conflicts = 0
        risk_frames = []
        for frame_idx in range(total_frames):
            detections = detections_per_frame.get(frame_idx, [])
            directional: dict[int, tuple[_DirectionalRecord | None, _DirectionalRecord | None]] = {}
            frame_conflicts = 0
            one_sided = 0
            for track_id in range(self.n_animals):
                backward_id = forward_to_backward.get(track_id)
                forward_record = forward.records.get(track_id, {}).get(frame_idx)
                backward_record = (
                    backward.records.get(backward_id, {}).get(frame_idx)
                    if backward_id is not None
                    else None
                )
                directional[track_id] = (forward_record, backward_record)
                if (
                    forward_record is not None
                    and backward_record is not None
                    and forward_record.observed
                    and backward_record.observed
                    and forward_record.detection.detection_id != backward_record.detection.detection_id
                ):
                    conflicts += 1
                    frame_conflicts += 1
                elif (
                    forward_record is not None
                    and backward_record is not None
                    and forward_record.observed != backward_record.observed
                ):
                    one_sided += 1

            assignments = self._fused_frame_assignments(directional, detections)
            assigned_by_track = {track_id: detection_idx for track_id, detection_idx in assignments}
            prediction_count = self.n_animals - len(assigned_by_track)
            for track_id in range(self.n_animals):
                detection_idx = assigned_by_track.get(track_id)
                if detection_idx is not None:
                    fused[track_id][frame_idx] = _DirectionalRecord(
                        detection=detections[detection_idx],
                        observed=True,
                        assignment_cost=0.0,
                    )
                else:
                    fused[track_id][frame_idx] = _DirectionalRecord(
                        detection=self._fused_prediction(directional[track_id], frame_idx),
                        observed=False,
                        assignment_cost=1.0,
                    )
            reasons = []
            if frame_conflicts:
                reasons.append(f"bidirectional_disagreement={frame_conflicts}")
            if one_sided:
                reasons.append(f"one_sided_observation={one_sided}")
            if prediction_count:
                reasons.append(f"prediction_carry_forward={prediction_count}")
            if not reasons:
                reasons.append("consistent_observation")
            risk_frames.append(
                AdaptiveRiskFrame(
                    frame=frame_idx,
                    risk_score=float(
                        min(
                            1.0,
                            0.55 * frame_conflicts / max(self.n_animals, 1)
                            + 0.20 * one_sided / max(self.n_animals, 1)
                            + 0.25 * prediction_count / max(self.n_animals, 1),
                        )
                    ),
                    disagreement_count=frame_conflicts,
                    one_sided_count=one_sided,
                    prediction_count=prediction_count,
                    observed_count=len(assigned_by_track),
                    reasons="; ".join(reasons),
                )
            )
        return fused, conflicts, risk_frames

    def _smooth_identity_assignments(
        self,
        raw_records: dict[int, dict[int, _DirectionalRecord]],
        detections_per_frame: dict[int, list[Detection]],
        forward: _DirectionalPass,
        backward: _DirectionalPass,
        backward_to_forward: dict[int, int],
        total_frames: int,
    ) -> tuple[dict[int, dict[int, _DirectionalRecord]], int]:
        """Choose a globally smooth identity path through per-frame detections.

        The bidirectional fusion step is intentionally local. This Viterbi-style
        pass adds a trajectory-level identity prior so a single ambiguous crossing
        does not permanently swap two IDs.
        """
        backpointers: list[dict[tuple[int, ...], tuple[int, ...] | None]] = []
        previous_scores: dict[tuple[int, ...], float] = {}
        anchors = self._forward_anchor_descriptors(forward)
        self._record_velocity_by_frame = self._precompute_record_velocities(
            raw_records,
            total_frames,
        )

        for frame_idx in range(total_frames):
            detections = detections_per_frame.get(frame_idx, [])
            detection_descriptors = [
                self._cached_appearance_descriptor(detection) for detection in detections
            ]
            candidates = self._candidate_identity_assignments(
                detections,
                frame_idx=frame_idx,
                raw_records=raw_records,
            )
            self._max_smoothing_states = max(self._max_smoothing_states, len(candidates))
            current_scores: dict[tuple[int, ...], float] = {}
            current_backpointers: dict[tuple[int, ...], tuple[int, ...] | None] = {}
            unary_costs = [
                self._smoothing_unary_cost(
                    frame_idx=frame_idx,
                    assignment=assignment,
                    detections=detections,
                    raw_records=raw_records,
                    forward=forward,
                    backward=backward,
                    backward_to_forward=backward_to_forward,
                    anchors=anchors,
                    detection_descriptors=detection_descriptors,
                )
                for assignment in candidates
            ]
            if not previous_scores:
                for assignment, unary in zip(candidates, unary_costs, strict=True):
                    current_scores[assignment] = unary
                    current_backpointers[assignment] = None
            else:
                previous_assignments = list(previous_scores)
                transition_matrix = self._smoothing_transition_matrix(
                    previous_frame=frame_idx - 1,
                    current_frame=frame_idx,
                    previous_assignments=previous_assignments,
                    current_assignments=candidates,
                    detections_per_frame=detections_per_frame,
                    raw_records=raw_records,
                )
                accumulated = (
                    np.asarray(
                        [previous_scores[assignment] for assignment in previous_assignments],
                        dtype=np.float64,
                    )[:, None]
                    + transition_matrix
                )
                best_previous_indices = np.argmin(accumulated, axis=0)
                for current_idx, (assignment, unary) in enumerate(
                    zip(candidates, unary_costs, strict=True)
                ):
                    previous_idx = int(best_previous_indices[current_idx])
                    current_scores[assignment] = (
                        float(accumulated[previous_idx, current_idx]) + unary
                    )
                    current_backpointers[assignment] = previous_assignments[previous_idx]
            previous_scores = current_scores
            backpointers.append(current_backpointers)

        if not previous_scores:
            return raw_records, 0
        best_assignment = min(previous_scores, key=previous_scores.get)
        path = [best_assignment]
        for frame_idx in range(total_frames - 1, 0, -1):
            best_assignment = backpointers[frame_idx][best_assignment]
            if best_assignment is None:
                break
            path.append(best_assignment)
        path.reverse()

        smoothed = {track_id: {} for track_id in range(self.n_animals)}
        changes = 0
        for frame_idx, assignment in enumerate(path):
            detections = detections_per_frame.get(frame_idx, [])
            for track_id, detection_idx in enumerate(assignment):
                raw_record = raw_records.get(track_id, {}).get(frame_idx)
                if detection_idx >= 0:
                    detection = detections[detection_idx]
                    if (
                        raw_record is not None
                        and raw_record.observed
                        and raw_record.detection.detection_id != detection.detection_id
                    ):
                        changes += 1
                    smoothed[track_id][frame_idx] = _DirectionalRecord(
                        detection=detection,
                        observed=True,
                        assignment_cost=0.0,
                    )
                elif raw_record is not None:
                    smoothed[track_id][frame_idx] = _DirectionalRecord(
                        detection=raw_record.detection,
                        observed=False,
                        assignment_cost=max(raw_record.assignment_cost, 1.0),
                    )
                else:
                    smoothed[track_id][frame_idx] = _DirectionalRecord(
                        detection=self._fused_prediction((None, None), frame_idx),
                        observed=False,
                        assignment_cost=1.0,
                    )
        return smoothed, changes

    def _stabilize_motion_records(
        self,
        records: dict[int, dict[int, _DirectionalRecord]],
        total_frames: int,
    ) -> tuple[dict[int, dict[int, _DirectionalRecord]], int]:
        """Reject physically implausible anchors and rebuild prediction gaps.

        Bidirectional predictions can disagree strongly during an occlusion.
        Those predictions must not become visible one-frame identity jumps.
        Observed anchors are therefore speed-gated, while missing intervals are
        rebuilt from the accepted anchors of the same identity only.
        """
        stabilized = {
            track_id: dict(track_records)
            for track_id, track_records in records.items()
        }
        max_observed_speed = max(
            self.config.adaptive_consistency_distance * 1.50,
            1.0,
        )
        rejected = 0

        for track_id, track_records in stabilized.items():
            last_frame: int | None = None
            last_center: np.ndarray | None = None
            for frame_idx in range(total_frames):
                record = track_records.get(frame_idx)
                if record is None or not record.observed:
                    continue
                center = self._bbox_center(record.detection.bbox_xyxy)
                if last_frame is not None and last_center is not None:
                    frame_gap = max(frame_idx - last_frame, 1)
                    implied_speed = float(
                        np.linalg.norm(center - last_center) / frame_gap
                    )
                    if implied_speed > max_observed_speed:
                        track_records[frame_idx] = _DirectionalRecord(
                            detection=record.detection,
                            observed=False,
                            assignment_cost=max(record.assignment_cost, 1.0),
                        )
                        rejected += 1
                        continue
                last_frame = frame_idx
                last_center = center

            observed_frames = [
                frame_idx
                for frame_idx in range(total_frames)
                if (
                    frame_idx in track_records
                    and track_records[frame_idx].observed
                )
            ]
            if not observed_frames:
                continue

            observed_set = set(observed_frames)
            next_observed_index = 0
            for frame_idx in range(total_frames):
                if frame_idx in observed_set:
                    while (
                        next_observed_index < len(observed_frames)
                        and observed_frames[next_observed_index] <= frame_idx
                    ):
                        next_observed_index += 1
                    continue

                previous_frame = (
                    observed_frames[next_observed_index - 1]
                    if next_observed_index > 0
                    else None
                )
                next_frame = (
                    observed_frames[next_observed_index]
                    if next_observed_index < len(observed_frames)
                    else None
                )
                if previous_frame is not None and next_frame is not None:
                    left_bbox = np.asarray(
                        track_records[previous_frame].detection.bbox_xyxy,
                        dtype=np.float64,
                    )
                    right_bbox = np.asarray(
                        track_records[next_frame].detection.bbox_xyxy,
                        dtype=np.float64,
                    )
                    alpha = (frame_idx - previous_frame) / max(
                        next_frame - previous_frame,
                        1,
                    )
                    bbox = (1.0 - alpha) * left_bbox + alpha * right_bbox
                    anchor_distance = min(
                        frame_idx - previous_frame,
                        next_frame - frame_idx,
                    )
                elif previous_frame is not None:
                    bbox = self._extrapolated_bbox(
                        track_records,
                        observed_frames,
                        previous_frame,
                        frame_idx,
                        direction=1,
                    )
                    anchor_distance = frame_idx - previous_frame
                else:
                    assert next_frame is not None
                    bbox = self._extrapolated_bbox(
                        track_records,
                        observed_frames,
                        next_frame,
                        frame_idx,
                        direction=-1,
                    )
                    anchor_distance = next_frame - frame_idx

                bbox = self._clip_bbox_to_frame(bbox)
                confidence = max(
                    0.15,
                    self.config.fixed_prediction_confidence
                    * float(np.exp(-anchor_distance / 25.0)),
                )
                center = (
                    float((bbox[0] + bbox[2]) / 2.0),
                    float((bbox[1] + bbox[3]) / 2.0),
                )
                track_records[frame_idx] = _DirectionalRecord(
                    detection=Detection(
                        frame_idx=frame_idx,
                        bbox_xyxy=tuple(float(value) for value in bbox),
                        confidence=confidence,
                        source=DetectionSource.ADAPTIVE_PREDICTION,
                        centroid=center,
                        area=self._bbox_area(bbox),
                    ),
                    observed=False,
                    assignment_cost=1.0 + min(anchor_distance / 25.0, 1.0),
                )
        return stabilized, rejected

    def _extrapolated_bbox(
        self,
        records: dict[int, _DirectionalRecord],
        observed_frames: list[int],
        anchor_frame: int,
        target_frame: int,
        direction: int,
    ) -> np.ndarray:
        anchor_bbox = np.asarray(
            records[anchor_frame].detection.bbox_xyxy,
            dtype=np.float64,
        )
        anchor_center = np.asarray(
            self._bbox_center(records[anchor_frame].detection.bbox_xyxy),
            dtype=np.float64,
        )
        anchor_index = observed_frames.index(anchor_frame)
        companion_index = anchor_index - direction
        if not 0 <= companion_index < len(observed_frames):
            return anchor_bbox
        companion_frame = observed_frames[companion_index]
        companion_center = np.asarray(
            self._bbox_center(records[companion_frame].detection.bbox_xyxy),
            dtype=np.float64,
        )
        frame_delta = anchor_frame - companion_frame
        if frame_delta == 0:
            return anchor_bbox
        velocity = (anchor_center - companion_center) / frame_delta
        speed = float(np.linalg.norm(velocity))
        speed_limit = max(
            self.config.adaptive_consistency_distance * 0.65,
            1.0,
        )
        if speed > speed_limit:
            velocity *= speed_limit / speed
        predicted_center = anchor_center + velocity * (target_frame - anchor_frame)
        return self._bbox_at_center(predicted_center, anchor_bbox)

    def _clip_bbox_to_frame(self, bbox: np.ndarray) -> np.ndarray:
        if self.frame_size is None:
            return bbox
        width, height = self.frame_size
        clipped = bbox.copy()
        clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0.0, float(width))
        clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0.0, float(height))
        return clipped

    @staticmethod
    def _bbox_center(bbox: Iterable[float]) -> np.ndarray:
        values = np.asarray(tuple(bbox), dtype=np.float64)
        return (values[:2] + values[2:]) / 2.0

    def _candidate_identity_assignments(
        self,
        detections: list[Detection],
        *,
        frame_idx: int | None = None,
        raw_records: dict[int, dict[int, _DirectionalRecord]] | None = None,
    ) -> list[tuple[int, ...]]:
        if not detections:
            return [tuple([-1] * self.n_animals)]
        eligible = list(range(len(detections)))
        if self._reference_detection_area > 0.0:
            lower = self._reference_detection_area * 0.30
            upper = self._reference_detection_area * max(
                self.config.adaptive_merged_area_ratio * 1.15,
                1.90,
            )
            eligible = [
                idx
                for idx, detection in enumerate(detections)
                if lower <= float(detection.area or 0.0) <= upper
            ]
        ranked = sorted(
            eligible,
            key=lambda idx: (
                (
                    abs(
                        np.log(
                            max(float(detections[idx].area or 1.0), 1.0)
                            / self._reference_detection_area
                        )
                    )
                    if self._reference_detection_area > 0.0
                    else -float(detections[idx].area or 0.0)
                ),
                1.0 - float(detections[idx].confidence),
            ),
        )[: self.n_animals]
        observed_count = min(len(ranked), self.n_animals)
        if self.n_animals <= 4:
            assignments_by_count: dict[int, list[tuple[int, ...]]] = {}
            for retained_count in range(observed_count, -1, -1):
                level: list[tuple[int, ...]] = []
                for retained_detections in combinations(ranked, retained_count):
                    for observed_tracks in combinations(
                        range(self.n_animals),
                        retained_count,
                    ):
                        for permuted_detections in permutations(
                            retained_detections,
                            retained_count,
                        ):
                            assignment = [-1] * self.n_animals
                            for track_id, detection_idx in zip(
                                observed_tracks,
                                permuted_detections,
                                strict=True,
                            ):
                                assignment[track_id] = int(detection_idx)
                            level.append(tuple(assignment))
                assignments_by_count[retained_count] = level

            # Preserve all complete and one-rejection hypotheses. A detector
            # candidate is evidence, not an obligation: sparse frames must be
            # allowed to reject a remote false positive instead of forcing it
            # onto an identity.
            prioritized: list[tuple[int, ...]] = []
            for retained_count in (
                observed_count,
                max(observed_count - 1, 0),
                0,
            ):
                for assignment in assignments_by_count.get(retained_count, []):
                    if assignment not in prioritized:
                        prioritized.append(assignment)

            remaining = [
                assignment
                for retained_count in range(max(observed_count - 2, 0), 0, -1)
                for assignment in assignments_by_count.get(retained_count, [])
                if assignment not in prioritized
            ]
            if raw_records is not None and frame_idx is not None:
                scale = max(self.config.adaptive_consistency_distance, 1.0)

                def continuity_cost(assignment: tuple[int, ...]) -> float:
                    total = 0.0
                    for track_id, detection_idx in enumerate(assignment):
                        if detection_idx < 0:
                            total += 0.70
                            continue
                        record = raw_records.get(track_id, {}).get(frame_idx)
                        if record is None:
                            total += 1.0
                            continue
                        total += float(
                            np.linalg.norm(
                                np.asarray(
                                    detections[detection_idx].centroid,
                                    dtype=np.float64,
                                )
                                - np.asarray(
                                    record.detection.centroid,
                                    dtype=np.float64,
                                )
                            )
                        ) / scale
                    return total

                remaining.sort(key=continuity_cost)
            prioritized.extend(remaining)
            return prioritized[: self.config.adaptive_smoothing_max_states]

        # Full permutation grows factorially. For larger groups, retain the raw
        # fused assignment, a geometry-nearest assignment, and local identity
        # swaps around both. This keeps full-video smoothing bounded.
        seeds: list[tuple[int, ...]] = []
        if raw_records is not None and frame_idx is not None:
            raw_assignment = [-1] * self.n_animals
            used: set[int] = set()
            by_detection_id = {
                detection.detection_id: detection_idx
                for detection_idx, detection in enumerate(detections)
            }
            for track_id in range(self.n_animals):
                record = raw_records.get(track_id, {}).get(frame_idx)
                if record is None or not record.observed:
                    continue
                detection_idx = by_detection_id.get(record.detection.detection_id)
                if detection_idx is not None and detection_idx not in used:
                    raw_assignment[track_id] = int(detection_idx)
                    used.add(int(detection_idx))
            seeds.append(tuple(raw_assignment))

            cost = np.full(
                (self.n_animals, len(ranked)),
                self._frame_diagonal(),
                dtype=np.float64,
            )
            for track_id in range(self.n_animals):
                record = raw_records.get(track_id, {}).get(frame_idx)
                if record is None:
                    continue
                center = np.asarray(record.detection.centroid, dtype=np.float64)
                for ranked_idx, detection_idx in enumerate(ranked):
                    cost[track_id, ranked_idx] = np.linalg.norm(
                        center
                        - np.asarray(
                            detections[detection_idx].centroid,
                            dtype=np.float64,
                        )
                    )
            matches, _, _ = linear_assignment(cost, thresh=self._frame_diagonal())
            nearest_assignment = [-1] * self.n_animals
            for track_id, ranked_idx in matches:
                nearest_assignment[int(track_id)] = int(ranked[int(ranked_idx)])
            seeds.append(tuple(nearest_assignment))

        if not seeds:
            base = [-1] * self.n_animals
            for track_id, detection_idx in zip(range(self.n_animals), ranked):
                base[track_id] = int(detection_idx)
            seeds.append(tuple(base))

        candidates: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()

        def add(candidate: tuple[int, ...]) -> None:
            if candidate not in seen and len(candidates) < self.config.adaptive_smoothing_max_states:
                seen.add(candidate)
                candidates.append(candidate)

        for seed in seeds:
            add(seed)
            assigned_tracks = [idx for idx, detection_idx in enumerate(seed) if detection_idx >= 0]
            for left, right in combinations(assigned_tracks, 2):
                swapped = list(seed)
                swapped[left], swapped[right] = swapped[right], swapped[left]
                add(tuple(swapped))
                if len(candidates) >= self.config.adaptive_smoothing_max_states:
                    break
        return candidates

    def _smoothing_unary_cost(
        self,
        frame_idx: int,
        assignment: tuple[int, ...],
        detections: list[Detection],
        raw_records: dict[int, dict[int, _DirectionalRecord]],
        forward: _DirectionalPass,
        backward: _DirectionalPass,
        backward_to_forward: dict[int, int],
        anchors: dict[int, np.ndarray | None],
        detection_descriptors: list[np.ndarray | None],
    ) -> float:
        forward_to_backward = {
            forward_id: backward_id for backward_id, forward_id in backward_to_forward.items()
        }
        total = 0.0
        evidence_weight = self.config.adaptive_smoothing_evidence_weight
        appearance_weight = self.config.adaptive_smoothing_appearance_weight
        scale = max(self.config.adaptive_consistency_distance, 1.0)
        for track_id, detection_idx in enumerate(assignment):
            raw_record = raw_records.get(track_id, {}).get(frame_idx)
            if detection_idx < 0:
                total += 0.70
                if raw_record is not None and not raw_record.observed:
                    total -= 0.10
                continue

            detection = detections[detection_idx]
            center = np.asarray(detection.centroid, dtype=np.float64)
            total += 0.08 * (1.0 - detection.confidence)

            if raw_record is not None:
                raw_center = np.asarray(raw_record.detection.centroid, dtype=np.float64)
                raw_distance = float(np.linalg.norm(center - raw_center))
                total += evidence_weight * 0.35 * min(raw_distance / scale, 2.0)
                if raw_record.observed and raw_record.detection.detection_id == detection.detection_id:
                    total -= evidence_weight * 0.42

            forward_record = forward.records.get(track_id, {}).get(frame_idx)
            backward_id = forward_to_backward.get(track_id)
            backward_record = (
                backward.records.get(backward_id, {}).get(frame_idx)
                if backward_id is not None
                else None
            )
            for record in (forward_record, backward_record):
                if record is None:
                    continue
                record_center = np.asarray(record.detection.centroid, dtype=np.float64)
                if record.observed and record.detection.detection_id == detection.detection_id:
                    total -= evidence_weight * 0.28
                else:
                    total += evidence_weight * 0.14 * min(
                        float(np.linalg.norm(center - record_center)) / scale,
                        2.0,
                    )

            anchor = anchors.get(track_id)
            descriptor = detection_descriptors[detection_idx]
            if anchor is not None and descriptor is not None:
                total += appearance_weight * self._descriptor_distance(anchor, descriptor)
        return total

    def _smoothing_transition_cost(
        self,
        previous_frame: int,
        current_frame: int,
        previous_assignment: tuple[int, ...],
        current_assignment: tuple[int, ...],
        detections_per_frame: dict[int, list[Detection]],
        raw_records: dict[int, dict[int, _DirectionalRecord]],
    ) -> float:
        previous_detections = detections_per_frame.get(previous_frame, [])
        current_detections = detections_per_frame.get(current_frame, [])
        return sum(
            self._smoothing_track_transition_cost(
                track_id=track_id,
                previous_frame=previous_frame,
                current_frame=current_frame,
                previous_detection_idx=previous_assignment[track_id],
                current_detection_idx=current_assignment[track_id],
                previous_detections=previous_detections,
                current_detections=current_detections,
                raw_records=raw_records,
            )
            for track_id in range(self.n_animals)
        )

    def _smoothing_transition_matrix(
        self,
        previous_frame: int,
        current_frame: int,
        previous_assignments: list[tuple[int, ...]],
        current_assignments: list[tuple[int, ...]],
        detections_per_frame: dict[int, list[Detection]],
        raw_records: dict[int, dict[int, _DirectionalRecord]],
    ) -> np.ndarray:
        """Vectorize Viterbi transitions after caching per-identity pair costs."""
        previous_detections = detections_per_frame.get(previous_frame, [])
        current_detections = detections_per_frame.get(current_frame, [])
        previous_array = np.asarray(previous_assignments, dtype=np.int16)
        current_array = np.asarray(current_assignments, dtype=np.int16)
        transition = np.zeros(
            (len(previous_assignments), len(current_assignments)),
            dtype=np.float64,
        )

        for track_id in range(self.n_animals):
            pair_costs = np.empty(
                (len(previous_detections) + 1, len(current_detections) + 1),
                dtype=np.float64,
            )
            for previous_idx in range(-1, len(previous_detections)):
                for current_idx in range(-1, len(current_detections)):
                    pair_costs[previous_idx + 1, current_idx + 1] = (
                        self._smoothing_track_transition_cost(
                            track_id=track_id,
                            previous_frame=previous_frame,
                            current_frame=current_frame,
                            previous_detection_idx=previous_idx,
                            current_detection_idx=current_idx,
                            previous_detections=previous_detections,
                            current_detections=current_detections,
                            raw_records=raw_records,
                        )
                    )
            transition += pair_costs[
                previous_array[:, track_id, None] + 1,
                current_array[None, :, track_id] + 1,
            ]
        return transition

    def _smoothing_track_transition_cost(
        self,
        track_id: int,
        previous_frame: int,
        current_frame: int,
        previous_detection_idx: int,
        current_detection_idx: int,
        previous_detections: list[Detection],
        current_detections: list[Detection],
        raw_records: dict[int, dict[int, _DirectionalRecord]],
    ) -> float:
        previous_center = self._assignment_center(
            track_id=track_id,
            frame_idx=previous_frame,
            detection_idx=previous_detection_idx,
            detections=previous_detections,
            raw_records=raw_records,
        )
        current_center = self._assignment_center(
            track_id=track_id,
            frame_idx=current_frame,
            detection_idx=current_detection_idx,
            detections=current_detections,
            raw_records=raw_records,
        )
        if previous_center is None or current_center is None:
            return 0.35

        scale = max(self.config.adaptive_consistency_distance, 1.0)
        distance = float(np.linalg.norm(current_center - previous_center))
        distance_ratio = distance / scale
        total = (
            self.config.adaptive_smoothing_motion_weight
            * 0.55
            * min(distance_ratio, 3.0)
        )
        jump_excess = max(0.0, distance_ratio - 1.0)
        total += (
            self.config.adaptive_smoothing_jump_penalty
            * jump_excess
            * jump_excess
        )
        expected_velocity = self._robust_record_velocity(
            raw_records,
            track_id,
            previous_frame,
        )
        expected_center = previous_center + expected_velocity
        prediction_error = float(np.linalg.norm(current_center - expected_center)) / scale
        total += (
            self.config.adaptive_smoothing_velocity_weight
            * min(prediction_error, 3.0)
            * min(prediction_error, 3.0)
        )
        if self.config.adaptive_discrete_geometry:
            previous_bbox = self._assignment_bbox(
                track_id=track_id,
                frame_idx=previous_frame,
                detection_idx=previous_detection_idx,
                detections=previous_detections,
                raw_records=raw_records,
            )
            current_bbox = self._assignment_bbox(
                track_id=track_id,
                frame_idx=current_frame,
                detection_idx=current_detection_idx,
                detections=current_detections,
                raw_records=raw_records,
            )
            if previous_bbox is not None and current_bbox is not None:
                candidate_velocity = current_center - previous_center
                expected_bbox = self._bbox_at_center(expected_center, previous_bbox)
                expected_token = self._geometry_encoder.encode(
                    expected_bbox,
                    expected_velocity,
                )
                candidate_token = self._geometry_encoder.encode(
                    current_bbox,
                    candidate_velocity,
                )
                total += (
                    self.config.adaptive_smoothing_geometry_weight
                    * self._geometry_encoder.distance(
                        expected_token,
                        candidate_token,
                    )
                )
        if current_detection_idx < 0:
            total += 0.35
        return total

    def _assignment_center(
        self,
        track_id: int,
        frame_idx: int,
        detection_idx: int,
        detections: list[Detection],
        raw_records: dict[int, dict[int, _DirectionalRecord]],
    ) -> np.ndarray | None:
        if detection_idx >= 0 and detection_idx < len(detections):
            return np.asarray(detections[detection_idx].centroid, dtype=np.float64)
        record = raw_records.get(track_id, {}).get(frame_idx)
        if record is None:
            return None
        return np.asarray(record.detection.centroid, dtype=np.float64)

    def _assignment_bbox(
        self,
        track_id: int,
        frame_idx: int,
        detection_idx: int,
        detections: list[Detection],
        raw_records: dict[int, dict[int, _DirectionalRecord]],
    ) -> np.ndarray | None:
        if detection_idx >= 0 and detection_idx < len(detections):
            return np.asarray(detections[detection_idx].bbox_xyxy, dtype=np.float64)
        record = raw_records.get(track_id, {}).get(frame_idx)
        if record is None:
            return None
        return np.asarray(record.detection.bbox_xyxy, dtype=np.float64)

    def _robust_record_velocity(
        self,
        raw_records: dict[int, dict[int, _DirectionalRecord]],
        track_id: int,
        frame_idx: int,
    ) -> np.ndarray:
        cached = self._record_velocity_by_frame.get((track_id, frame_idx))
        if cached is not None:
            return cached
        return np.zeros(2, dtype=np.float64)

    def _precompute_record_velocities(
        self,
        raw_records: dict[int, dict[int, _DirectionalRecord]],
        total_frames: int,
    ) -> dict[tuple[int, int], np.ndarray]:
        cache: dict[tuple[int, int], np.ndarray] = {}
        for track_id in range(self.n_animals):
            records = raw_records.get(track_id, {})
            recent_velocities: list[np.ndarray] = []
            previous_observation: tuple[int, np.ndarray] | None = None
            current_velocity = np.zeros(2, dtype=np.float64)
            for frame_idx in range(total_frames):
                record = records.get(frame_idx)
                if record is not None and record.observed:
                    center = np.asarray(record.detection.centroid, dtype=np.float64)
                    if previous_observation is not None:
                        previous_frame, previous_center = previous_observation
                        frame_delta = max(frame_idx - previous_frame, 1)
                        recent_velocities.append(
                            (center - previous_center) / frame_delta
                        )
                        recent_velocities = recent_velocities[
                            -self.config.adaptive_velocity_window:
                        ]
                        current_velocity = np.median(
                            np.asarray(recent_velocities, dtype=np.float64),
                            axis=0,
                        )
                    previous_observation = (frame_idx, center)
                cache[(track_id, frame_idx)] = current_velocity.copy()
        return cache

    def _forward_anchor_descriptors(
        self,
        forward: _DirectionalPass,
    ) -> dict[int, np.ndarray | None]:
        anchors: dict[int, np.ndarray | None] = {}
        for track_id in range(self.n_animals):
            record = forward.records.get(track_id, {}).get(forward.init_frame)
            anchors[track_id] = (
                self._cached_appearance_descriptor(record.detection)
                if record is not None and record.observed
                else None
            )
        return anchors

    def _stable_risk_frames(
        self,
        records: dict[int, dict[int, _DirectionalRecord]],
        raw_records: dict[int, dict[int, _DirectionalRecord]],
        raw_risk_frames: list[AdaptiveRiskFrame],
        total_frames: int,
    ) -> list[AdaptiveRiskFrame]:
        raw_by_frame = {frame.frame: frame for frame in raw_risk_frames}
        risk_frames: list[AdaptiveRiskFrame] = []
        previous_observations: dict[int, tuple[int, np.ndarray]] = {}
        for frame_idx in range(total_frames):
            raw_frame = raw_by_frame.get(frame_idx)
            prediction_count = 0
            changed_count = 0
            large_jump_count = 0
            observed_count = 0
            for track_id in range(self.n_animals):
                record = records.get(track_id, {}).get(frame_idx)
                raw_record = raw_records.get(track_id, {}).get(frame_idx)
                if record is None or not record.observed:
                    prediction_count += 1
                    continue
                observed_count += 1
                center = self._bbox_center(record.detection.bbox_xyxy)
                previous_observation = previous_observations.get(track_id)
                if previous_observation is not None:
                    previous_frame, previous_center = previous_observation
                    frame_gap = max(frame_idx - previous_frame, 1)
                    implied_speed = (
                        np.linalg.norm(center - previous_center) / frame_gap
                    )
                    if (
                        implied_speed
                        > self.config.adaptive_consistency_distance * 1.50
                    ):
                        large_jump_count += 1
                previous_observations[track_id] = (frame_idx, center)
                if (
                    raw_record is not None
                    and raw_record.observed
                    and raw_record.detection.detection_id != record.detection.detection_id
                ):
                    changed_count += 1

            raw_disagreement = raw_frame.disagreement_count if raw_frame is not None else 0
            reasons = []
            if raw_disagreement:
                reasons.append(f"bidirectional_disagreement={raw_disagreement}")
            if changed_count:
                reasons.append(f"identity_smoothing_override={changed_count}")
            if large_jump_count:
                reasons.append(f"stable_path_jump={large_jump_count}")
            if prediction_count:
                reasons.append(f"prediction_carry_forward={prediction_count}")
            if not reasons:
                reasons.append("stable_observation")
            risk = min(
                1.0,
                0.40 * large_jump_count / max(self.n_animals, 1)
                + 0.28 * prediction_count / max(self.n_animals, 1)
                + 0.20 * changed_count / max(self.n_animals, 1)
                + 0.12 * raw_disagreement / max(self.n_animals, 1),
            )
            risk_frames.append(
                AdaptiveRiskFrame(
                    frame=frame_idx,
                    risk_score=float(risk),
                    disagreement_count=raw_disagreement,
                    one_sided_count=changed_count,
                    prediction_count=prediction_count,
                    observed_count=observed_count,
                    reasons="; ".join(reasons),
                )
            )
        return risk_frames

    def _fused_frame_assignments(
        self,
        directional: dict[int, tuple[_DirectionalRecord | None, _DirectionalRecord | None]],
        detections: list[Detection],
    ) -> list[tuple[int, int]]:
        if not detections:
            return []
        cost = np.full((self.n_animals, len(detections)), 99.0, dtype=np.float64)
        for track_id, records in directional.items():
            usable = [record for record in records if record is not None]
            if not usable:
                continue
            for detection_idx, detection in enumerate(detections):
                voting_records = [
                    record
                    for record in usable
                    if record.observed and record.detection.detection_id == detection.detection_id
                ]
                if voting_records:
                    cost[track_id, detection_idx] = (
                        min(record.assignment_cost for record in voting_records)
                        - 0.42 * len(voting_records)
                    )
                else:
                    center = np.asarray(detection.centroid, dtype=np.float64)
                    distances = [
                        np.linalg.norm(
                            center - np.asarray(record.detection.centroid, dtype=np.float64)
                        )
                        for record in usable
                    ]
                    cost[track_id, detection_idx] = (
                        min(distances) / self.config.adaptive_consistency_distance + 0.30
                    )
        matches, _, _ = linear_assignment(cost, thresh=1.35)
        return [(int(track_id), int(detection_idx)) for track_id, detection_idx in matches]

    def _fused_prediction(
        self,
        records: tuple[_DirectionalRecord | None, _DirectionalRecord | None],
        frame_idx: int,
    ) -> Detection:
        usable = [record for record in records if record is not None]
        if not usable:
            bbox = np.zeros(4, dtype=np.float64)
            center = np.zeros(2, dtype=np.float64)
            missing = 25
        else:
            weights = np.asarray([2.0 if record.observed else 1.0 for record in usable])
            bboxes = np.asarray([record.detection.bbox_xyxy for record in usable], dtype=np.float64)
            centers = np.asarray([record.detection.centroid for record in usable], dtype=np.float64)
            bbox = np.average(bboxes, axis=0, weights=weights)
            center = np.average(centers, axis=0, weights=weights)
            missing = sum(not record.observed for record in usable)
        confidence = max(
            0.15,
            self.config.fixed_prediction_confidence * float(np.exp(-missing / 8.0)),
        )
        return Detection(
            frame_idx=frame_idx,
            bbox_xyxy=tuple(float(value) for value in bbox),
            confidence=confidence,
            source=DetectionSource.ADAPTIVE_PREDICTION,
            centroid=(float(center[0]), float(center[1])),
            area=self._bbox_area(bbox),
        )

    def _to_tracklets(
        self,
        records: dict[int, dict[int, _DirectionalRecord]],
    ) -> list[Tracklet]:
        tracklets = []
        for track_id in sorted(records):
            ordered = [records[track_id][frame_idx] for frame_idx in sorted(records[track_id])]
            if not ordered:
                continue
            observed_ratio = float(np.mean([record.observed for record in ordered]))
            detections = [record.detection for record in ordered]
            tracklets.append(
                Tracklet(
                    tracklet_id=track_id,
                    start_frame=detections[0].frame_idx,
                    end_frame=detections[-1].frame_idx,
                    detections=detections,
                    had_occlusion=observed_ratio < 0.98,
                    confidence=observed_ratio,
                )
            )
        return tracklets

    def _build_stats(
        self,
        detections_per_frame: dict[int, list[Detection]],
        records: dict[int, dict[int, _DirectionalRecord]],
        total_frames: int,
        forward: _DirectionalPass,
        backward: _DirectionalPass,
        backward_to_forward: dict[int, int],
        fusion_conflicts: int,
        identity_smoothing_changes: int = 0,
        motion_outlier_rejections: int = 0,
    ) -> MacAdaptiveStats:
        observed_flags = {
            track_id: [records[track_id][frame_idx].observed for frame_idx in range(total_frames)]
            for track_id in records
        }
        observed_counts = [
            sum(observed_flags[track_id][frame_idx] for track_id in observed_flags)
            for frame_idx in range(total_frames)
        ]
        max_missing = {}
        for track_id, flags in observed_flags.items():
            current = 0
            longest = 0
            for observed in flags:
                current = 0 if observed else current + 1
                longest = max(longest, current)
            max_missing[track_id] = longest
        agreement_ratio = self._agreement_ratio(forward, backward, backward_to_forward)
        descriptor_total = forward.observed_assignments + backward.observed_assignments
        descriptor_ratio = (
            (forward.descriptor_updates + backward.descriptor_updates) / max(descriptor_total, 1)
        )
        geometry_comparisons = forward.geometry_comparisons + backward.geometry_comparisons
        geometry_agreement = (
            (forward.geometry_consistent + backward.geometry_consistent)
            / max(geometry_comparisons, 1)
        )
        high_risk_frames = sum(frame.risk_score >= 0.50 for frame in self.risk_frames)
        large_jump_frames = sum(
            "stable_path_jump=" in frame.reasons
            for frame in self.risk_frames
        )
        identity_switch_risk_events = self._count_risk_episodes(
            frame.frame
            for frame in self.risk_frames
            if "stable_path_jump=" in frame.reasons
        )
        observed_support = float(np.mean(observed_counts)) / max(self.n_animals, 1)
        jump_free_ratio = 1.0 - large_jump_frames / max(total_frames, 1)
        stability_score = (
            0.30 * agreement_ratio
            + 0.30 * min(observed_support, 1.0)
            + 0.30 * jump_free_ratio
            + 0.10 * geometry_agreement
        )
        algorithm = (
            "mac_adaptive_bidirectional_click_assisted"
            if self.initialization_points
            else "mac_adaptive_bidirectional"
        )
        if self.config.adaptive_discrete_geometry:
            algorithm += "_hierarchical_geometry_long_term"
        if self.config.adaptive_identity_smoothing:
            algorithm += "_identity_stable"
        return MacAdaptiveStats(
            total_frames=total_frames,
            avg_detections_per_frame=float(
                np.mean([len(detections_per_frame.get(frame_idx, [])) for frame_idx in range(total_frames)])
            ),
            avg_observed_tracks_per_frame=float(np.mean(observed_counts)),
            all_observed_ratio=float(np.mean([count == self.n_animals for count in observed_counts])),
            mostly_observed_ratio=float(
                np.mean([count >= max(1, self.n_animals - 1) for count in observed_counts])
            ),
            per_track_observed_ratio={
                track_id: float(np.mean(flags)) for track_id, flags in observed_flags.items()
            },
            per_track_max_missing=max_missing,
            forward_init_frame=forward.init_frame,
            backward_init_frame=backward.init_frame,
            bidirectional_agreement_ratio=agreement_ratio,
            appearance_descriptor_ratio=float(min(descriptor_ratio, 1.0)),
            fusion_conflicts=fusion_conflicts,
            disagreement_frames=sum(frame.disagreement_count > 0 for frame in self.risk_frames),
            high_risk_frames=high_risk_frames,
            identity_smoothing_changes=identity_smoothing_changes,
            identity_stability_score=float(stability_score),
            quantized_geometry_enabled=self.config.adaptive_discrete_geometry,
            geometry_token_agreement_ratio=float(geometry_agreement),
            reacquisition_events=(
                forward.reacquisition_events + backward.reacquisition_events
            ),
            deferred_ambiguous_assignments=(
                forward.deferred_ambiguities + backward.deferred_ambiguities
            ),
            contact_deferred_assignments=(
                forward.contact_deferred_assignments
                + backward.contact_deferred_assignments
            ),
            merged_detection_rejections=(
                forward.merged_detection_rejections
                + backward.merged_detection_rejections
            ),
            motion_outlier_rejections=motion_outlier_rejections,
            large_jump_frames=int(large_jump_frames),
            identity_switch_risk_events=int(identity_switch_risk_events),
            max_smoothing_states=self._max_smoothing_states,
            algorithm=algorithm,
        )

    @staticmethod
    def _count_risk_episodes(frames: Iterable[int]) -> int:
        ordered = sorted(set(int(frame) for frame in frames))
        if not ordered:
            return 0
        return 1 + sum(
            current > previous + 1
            for previous, current in zip(ordered[:-1], ordered[1:], strict=True)
        )

    @staticmethod
    def _agreement_ratio(
        forward: _DirectionalPass,
        backward: _DirectionalPass,
        backward_to_forward: dict[int, int],
    ) -> float:
        forward_to_backward = {
            forward_id: backward_id for backward_id, forward_id in backward_to_forward.items()
        }
        same = 0
        compared = 0
        for forward_id, backward_id in forward_to_backward.items():
            for frame_idx in set(forward.records[forward_id]) & set(backward.records[backward_id]):
                forward_record = forward.records[forward_id][frame_idx]
                backward_record = backward.records[backward_id][frame_idx]
                if forward_record.observed and backward_record.observed:
                    compared += 1
                    same += (
                        forward_record.detection.detection_id
                        == backward_record.detection.detection_id
                    )
        return same / max(compared, 1)

    @staticmethod
    def _appearance_descriptor(detection: Detection) -> np.ndarray | None:
        if not detection.crop_bytes:
            return None
        encoded = np.frombuffer(detection.crop_bytes, dtype=np.uint8)
        crop = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if crop is None or crop.size == 0:
            return None
        blurred = cv2.GaussianBlur(crop, (5, 5), 0)
        _, foreground = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        occupancy = float(np.mean(foreground > 0))
        if occupancy < 0.05 or occupancy > 0.95:
            foreground = np.full_like(crop, 255)
        foreground_pixels = crop[foreground > 0]
        histogram = cv2.calcHist([crop], [0], foreground, [16], [0, 256]).reshape(-1)
        histogram = histogram / max(float(np.sum(histogram)), 1.0)
        silhouette = cv2.resize(foreground, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
        intensity = np.array(
            [
                float(np.mean(foreground_pixels)) / 255.0,
                float(np.std(foreground_pixels)) / 255.0,
                float(np.quantile(foreground_pixels, 0.25)) / 255.0,
                float(np.quantile(foreground_pixels, 0.75)) / 255.0,
                float(np.mean(foreground > 0)),
            ],
            dtype=np.float64,
        )
        moments = cv2.HuMoments(cv2.moments(foreground)).reshape(-1)
        moments = np.sign(moments) * np.log1p(np.abs(moments))
        return MacAdaptiveTrackletGenerator._normalize(
            np.concatenate([histogram.astype(np.float64), silhouette.reshape(-1), intensity, moments])
        )

    def _estimate_reference_detection_area(
        self,
        detections_per_frame: dict[int, list[Detection]],
    ) -> float:
        """Estimate one-animal mask area without using identity assignments."""
        areas = [
            float(detection.area)
            for detections in detections_per_frame.values()
            for detection in detections
            if detection.area is not None
            and float(detection.area) > 0.0
            and detection.confidence >= 0.45
        ]
        if not areas:
            return 0.0
        values = np.asarray(areas, dtype=np.float64)
        median = float(np.median(values))
        lower = median * 0.45
        upper = median * 1.85
        trimmed = values[(values >= lower) & (values <= upper)]
        return float(np.median(trimmed)) if trimmed.size else median

    def _initial_descriptor_bank(self, detection: Detection) -> list[np.ndarray]:
        descriptor = self._cached_appearance_descriptor(detection)
        return [descriptor] if descriptor is not None else []

    def _cached_appearance_descriptor(self, detection: Detection) -> np.ndarray | None:
        if not detection.crop_bytes:
            return None
        cached = self._descriptor_cache.get(detection.detection_id)
        if cached is not None or detection.detection_id in self._descriptor_cache:
            return cached
        descriptor = self._appearance_descriptor(detection)
        self._descriptor_cache[detection.detection_id] = descriptor
        return descriptor

    @staticmethod
    def _state_descriptor_distance(state: _AdaptiveState, descriptor: np.ndarray) -> float:
        candidates = [state.descriptor, state.anchor_descriptor, *state.descriptor_bank]
        distances = [
            MacAdaptiveTrackletGenerator._descriptor_distance(candidate, descriptor)
            for candidate in candidates
            if candidate is not None
        ]
        return min(distances) if distances else 1.0

    @staticmethod
    def _descriptor_distance(left: np.ndarray, right: np.ndarray) -> float:
        return float(min(np.linalg.norm(left - right) / 2.0, 1.0))

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        return values / max(float(np.linalg.norm(values)), 1e-9)

    def _predict_state(self, state: _AdaptiveState) -> tuple[np.ndarray, np.ndarray]:
        center = state.center + state.velocity
        return center, self._bbox_at_center(center, state.bbox)

    @staticmethod
    def _bbox_at_center(center: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        width = max(float(bbox[2] - bbox[0]), 1.0)
        height = max(float(bbox[3] - bbox[1]), 1.0)
        return np.array(
            [
                center[0] - width / 2,
                center[1] - height / 2,
                center[0] + width / 2,
                center[1] + height / 2,
            ],
            dtype=np.float64,
        )

    def _prediction_detection(self, state: _AdaptiveState, frame_idx: int) -> Detection:
        confidence = max(
            0.15,
            self.config.fixed_prediction_confidence * float(np.exp(-state.missing / 25.0)),
        )
        return Detection(
            frame_idx=frame_idx,
            bbox_xyxy=tuple(float(value) for value in state.bbox),
            confidence=confidence,
            source=DetectionSource.ADAPTIVE_PREDICTION,
            centroid=(float(state.center[0]), float(state.center[1])),
            area=self._bbox_area(state.bbox),
        )

    def _frame_diagonal(self) -> float:
        if self.frame_size is None:
            return 1000.0
        width, height = self.frame_size
        return max(float(np.hypot(width, height)), 1.0)

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        return float(max(1.0, bbox[2] - bbox[0]) * max(1.0, bbox[3] - bbox[1]))

    @staticmethod
    def _point_in_bbox(
        point: np.ndarray,
        bbox: np.ndarray,
        padding: float = 0.0,
    ) -> bool:
        return bool(
            float(bbox[0]) - padding <= float(point[0]) <= float(bbox[2]) + padding
            and float(bbox[1]) - padding <= float(point[1]) <= float(bbox[3]) + padding
        )

    def _fallback_fixed_count(
        self,
        detections_per_frame: dict[int, list[Detection]],
    ) -> list[Tracklet]:
        from densetrack.tracking.fixed_count_tracker import FixedCountTrackletGenerator

        fallback = FixedCountTrackletGenerator(
            config=self.config,
            n_animals=self.n_animals,
            frame_size=self.frame_size,
        )
        tracklets = fallback.generate(detections_per_frame)
        fixed_stats = fallback.stats
        self.stats = (
            MacAdaptiveStats(
                total_frames=fixed_stats.total_frames,
                avg_detections_per_frame=fixed_stats.avg_detections_per_frame,
                avg_observed_tracks_per_frame=fixed_stats.avg_observed_tracks_per_frame,
                all_observed_ratio=fixed_stats.all_observed_ratio,
                mostly_observed_ratio=fixed_stats.mostly_observed_ratio,
                per_track_observed_ratio=fixed_stats.per_track_observed_ratio,
                per_track_max_missing=fixed_stats.per_track_max_missing,
                forward_init_frame=-1,
                backward_init_frame=-1,
                bidirectional_agreement_ratio=0.0,
                appearance_descriptor_ratio=0.0,
                fusion_conflicts=0,
                disagreement_frames=0,
                high_risk_frames=0,
                algorithm="mac_adaptive_fallback_fixed_count",
            )
            if fixed_stats is not None
            else self._empty_stats()
        )
        return tracklets

    @staticmethod
    def _empty_stats() -> MacAdaptiveStats:
        return MacAdaptiveStats(
            total_frames=0,
            avg_detections_per_frame=0.0,
            avg_observed_tracks_per_frame=0.0,
            all_observed_ratio=0.0,
            mostly_observed_ratio=0.0,
            per_track_observed_ratio={},
            per_track_max_missing={},
            forward_init_frame=-1,
            backward_init_frame=-1,
            bidirectional_agreement_ratio=0.0,
            appearance_descriptor_ratio=0.0,
            fusion_conflicts=0,
            disagreement_frames=0,
            high_risk_frames=0,
        )
