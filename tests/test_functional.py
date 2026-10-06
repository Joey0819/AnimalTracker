#!/usr/bin/env python3
"""
Functional test: run core algorithms on synthetic data.

Tests the fully-implemented modules (no external model deps):
1. Kalman filter predict/update cycle
2. IoU batch computation
3. Linear assignment
4. ByteTrack tracklet generation from synthetic detections
5. Global optimizer greedy, Hungarian, and ILP assignment
6. Dark-object detector on synthetic top-view frame
7. Fixed-count tracker through a short occlusion
8. Geometric pose estimation on synthetic frame
9. Confidence auditor report generation
10. MOT/COCO export
11. SLEAP-style export
12. Re-export previous run outputs
13. Rule-based behavior analysis
14. Config load from YAML
15. UDMT external adapter contract
16. Automatic no-click initialization
17. Self-supervised ReID smoke test
18. MOT metrics standalone (parsing + IoU matrix)
19. UDMT config integration (YAML section loading)
20. Unsupervised behavior discovery
21. Static researcher dashboard
22. Zero-click complete local workflow
23. Local web workspace surface and artifact guard
24. Sparse GT validation review guard and confirmed evaluation
25. Mac-native adaptive bidirectional tracking through crossing and occlusion
26. BEAST3D-inspired visual SSL representation export
27. Dolph2Vec-inspired bioacoustic SSL export
28. Clean overlay preview generation from MOT with trail jump gating
29. Experiment template zone/time-bin metrics
30. Norfair open-source backend contract
31. HieDG-inspired hierarchical discrete geometry tokens
32. WaspMOT-style long-term stitching and ID-switch reduction
33. Auditable contact-event layer: merge/split records, top-K identity hypotheses, margins
"""

import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_kalman_filter():
    """Test Kalman filter predict/update."""
    from densetrack.tracking.kalman_filter import KalmanBoxTracker

    tracker = KalmanBoxTracker([100, 100, 200, 200])
    
    # Predict
    pred = tracker.predict()
    assert pred.shape == (4,), f"Bad shape: {pred.shape}"
    assert tracker.age == 1
    
    # Update with a moved bbox
    tracker.update([105, 103, 205, 203])
    state = tracker.get_state()
    assert state.shape == (4,)
    assert tracker.hits == 2
    
    # After several updates, velocity should be estimated
    for i in range(10):
        tracker.predict()
        tracker.update([100 + 5 * (i + 2), 100 + 3 * (i + 2), 200 + 5 * (i + 2), 200 + 3 * (i + 2)])
    
    vel = tracker.velocity
    assert abs(vel[0] - 5) < 2, f"Velocity X off: {vel[0]}"
    assert abs(vel[1] - 3) < 2, f"Velocity Y off: {vel[1]}"
    
    print("✅ Kalman filter: predict/update/velocity OK")


def test_iou_batch():
    """Test vectorized IoU computation."""
    from densetrack.tracking.matching import iou_batch

    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    b = np.array([[5, 5, 15, 15], [0, 0, 10, 10]])

    iou = iou_batch(a, b)
    assert iou.shape == (2, 2)
    
    # a[0] vs b[1] should be 1.0 (identical)
    assert abs(iou[0, 1] - 1.0) < 1e-6, f"Self-IoU wrong: {iou[0, 1]}"
    
    # a[0] vs b[0] should be 25/175 ≈ 0.143
    expected = 25.0 / 175.0
    assert abs(iou[0, 0] - expected) < 0.01, f"IoU wrong: {iou[0, 0]} vs {expected}"
    
    # a[1] vs b[0] and b[1] should be 0 (no overlap)
    assert iou[1, 1] == 0.0
    
    print("✅ IoU batch: vectorized computation OK")


def test_linear_assignment():
    """Test Hungarian assignment with threshold."""
    from densetrack.tracking.matching import linear_assignment

    cost = np.array([
        [0.1, 0.9, 0.9],
        [0.9, 0.2, 0.9],
        [0.9, 0.9, 0.3],
    ])
    
    matches, unmatched_a, unmatched_b = linear_assignment(cost, thresh=0.5)
    
    assert len(matches) == 3, f"Expected 3 matches, got {len(matches)}"
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 0
    
    # With strict threshold
    matches2, ua, ub = linear_assignment(cost, thresh=0.15)
    assert len(matches2) == 1, f"Expected 1 match at strict thresh, got {len(matches2)}"
    
    print("✅ Linear assignment: Hungarian + threshold OK")


def test_tracklet_generator():
    """Test ByteTrack tracklet generation with synthetic detections."""
    from densetrack.config import TrackletConfig
    from densetrack.models import Detection, DetectionSource
    from densetrack.tracking.tracklet_generator import TrackletGenerator

    config = TrackletConfig(tau_high=0.5, tau_low=0.1, match_thresh=0.3, track_buffer=5)
    gen = TrackletGenerator(config)

    # Simulate 2 animals moving linearly for 50 frames
    detections_per_frame = {}
    for f in range(50):
        dets = []
        # Animal A: moves right
        dets.append(Detection(
            frame_idx=f,
            bbox_xyxy=(100 + f * 3, 100, 150 + f * 3, 150),
            confidence=0.9,
            source=DetectionSource.BACKGROUND_SUB,
        ))
        # Animal B: moves down
        dets.append(Detection(
            frame_idx=f,
            bbox_xyxy=(300, 100 + f * 2, 350, 150 + f * 2),
            confidence=0.85,
            source=DetectionSource.BACKGROUND_SUB,
        ))
        detections_per_frame[f] = dets

    tracklets = gen.generate(detections_per_frame)
    
    assert len(tracklets) >= 2, f"Expected ≥2 tracklets, got {len(tracklets)}"
    
    # Total detections should be 100 (2 per frame × 50 frames)
    total = sum(len(t.detections) for t in tracklets)
    assert total == 100, f"Expected 100 detections in tracklets, got {total}"
    
    # Confidence should be positive
    for t in tracklets:
        assert t.confidence > 0, f"Tracklet {t.tracklet_id} has zero confidence"
    
    print(f"✅ Tracklet generator: {len(tracklets)} tracklets from 50 frames, {total} detections")


def test_global_optimizer():
    """Test global tracklet-to-trajectory optimization."""
    from densetrack.config import GlobalOptimizerConfig
    from densetrack.models import Detection, DetectionSource, Tracklet
    from densetrack.tracking.global_optimizer import GlobalTrackletOptimizer

    # Create 4 tracklets: 2 pairs that should be linked
    tracklets = []
    
    # Animal A, part 1 (frames 0-20)
    dets_a1 = [
        Detection(frame_idx=f, bbox_xyxy=(100 + f*3, 100, 150 + f*3, 150),
                  confidence=0.9, source=DetectionSource.BACKGROUND_SUB)
        for f in range(21)
    ]
    t_a1 = Tracklet(tracklet_id=0, start_frame=0, end_frame=20, detections=dets_a1, confidence=0.9)
    tracklets.append(t_a1)

    # Animal A, part 2 (frames 30-50) — gap of 10 frames (occlusion)
    dets_a2 = [
        Detection(frame_idx=f, bbox_xyxy=(100 + f*3, 100, 150 + f*3, 150),
                  confidence=0.85, source=DetectionSource.BACKGROUND_SUB)
        for f in range(30, 51)
    ]
    t_a2 = Tracklet(tracklet_id=1, start_frame=30, end_frame=50, detections=dets_a2, confidence=0.85)
    tracklets.append(t_a2)

    # Animal B, part 1 (frames 0-25)
    dets_b1 = [
        Detection(frame_idx=f, bbox_xyxy=(300, 100 + f*2, 350, 150 + f*2),
                  confidence=0.88, source=DetectionSource.BACKGROUND_SUB)
        for f in range(26)
    ]
    t_b1 = Tracklet(tracklet_id=2, start_frame=0, end_frame=25, detections=dets_b1, confidence=0.88)
    tracklets.append(t_b1)

    # Animal B, part 2 (frames 35-50)
    dets_b2 = [
        Detection(frame_idx=f, bbox_xyxy=(300, 100 + f*2, 350, 150 + f*2),
                  confidence=0.82, source=DetectionSource.BACKGROUND_SUB)
        for f in range(35, 51)
    ]
    t_b2 = Tracklet(tracklet_id=3, start_frame=35, end_frame=50, detections=dets_b2, confidence=0.82)
    tracklets.append(t_b2)

    # 1. Test Greedy
    config_greedy = GlobalOptimizerConfig(method="greedy", max_temporal_gap=50)
    opt_greedy = GlobalTrackletOptimizer(config_greedy, n_animals=2)
    trajectories_greedy = opt_greedy.optimize(tracklets)
    assert len(trajectories_greedy) >= 1, f"Expected greedy trajectories, got {len(trajectories_greedy)}"

    # 2. Test Hungarian
    config_hungarian = GlobalOptimizerConfig(method="hungarian", max_temporal_gap=50)
    opt_hungarian = GlobalTrackletOptimizer(config_hungarian, n_animals=2)
    trajectories_hungarian = opt_hungarian.optimize(tracklets)
    assert len(trajectories_hungarian) >= 1, f"Expected Hungarian trajectories, got {len(trajectories_hungarian)}"

    # 3. Test ILP DAG path cover and min-cost-flow alias
    config_ilp = GlobalOptimizerConfig(method="ilp", max_temporal_gap=50)
    trajectories_ilp = GlobalTrackletOptimizer(config_ilp, n_animals=2).optimize(tracklets)
    assert len(trajectories_ilp) == 2, f"Expected 2 ILP trajectories, got {len(trajectories_ilp)}"
    config_flow = GlobalOptimizerConfig(method="min_cost_flow", max_temporal_gap=50)
    trajectories_flow = GlobalTrackletOptimizer(config_flow, n_animals=2).optimize(tracklets)
    assert len(trajectories_flow) == 2, f"Expected 2 flow trajectories, got {len(trajectories_flow)}"

    print(
        "✅ Global optimizer: "
        f"greedy ({len(trajectories_greedy)}), "
        f"Hungarian ({len(trajectories_hungarian)}), "
        f"ILP ({len(trajectories_ilp)}), and flow alias ({len(trajectories_flow)}) OK"
    )


def test_hierarchical_discrete_geometry():
    """Test stable two-stage geometry tokens and robust endpoint velocity."""
    from densetrack.models import Detection, DetectionSource
    from densetrack.tracking.discrete_geometry import (
        HierarchicalGeometryEncoder,
        robust_detection_velocity,
    )

    encoder = HierarchicalGeometryEncoder(
        frame_size=(400, 200),
        coarse_bins=8,
        fine_bins=8,
        max_speed=40,
    )
    reference = encoder.encode((100, 60, 140, 90), (8, 0))
    small_jitter = encoder.encode((101, 60, 141, 90), (8.2, 0.1))
    identity_break = encoder.encode((250, 120, 290, 150), (-8, 0))
    assert encoder.distance(reference, small_jitter) < 0.05
    assert encoder.distance(reference, identity_break) > 0.25

    detections = [
        Detection(
            frame_idx=frame,
            bbox_xyxy=(50 + frame * 5, 70, 70 + frame * 5, 86),
            confidence=0.9,
            source=DetectionSource.BACKGROUND_SUB,
        )
        for frame in range(8)
    ]
    detections[-1].bbox_xyxy = (145, 70, 165, 86)  # one-frame endpoint jitter
    robust = robust_detection_velocity(detections, endpoint="end", window=7)
    assert np.allclose(robust, [5.0, 0.0])
    print("✅ Hierarchical geometry: jitter-stable tokens and robust velocity OK")


def test_long_term_stitching_reduces_id_switches():
    """Verify long-term geometry fixes a fragmented crossing that legacy motion swaps."""
    from densetrack.config import GlobalOptimizerConfig
    from densetrack.export.mot_format import MOTExporter
    from densetrack.models import Detection, DetectionSource, Tracklet
    from densetrack.tracking.global_optimizer import GlobalTrackletOptimizer
    from densetrack.utils.metrics import compute_mot_metrics

    def tracklet(
        tracklet_id: int,
        frames: list[int],
        xs: list[float],
        y: float,
    ) -> Tracklet:
        detections = [
            Detection(
                frame_idx=frame,
                bbox_xyxy=(x, y, x + 20, y + 16),
                confidence=0.9,
                source=DetectionSource.BACKGROUND_SUB,
            )
            for frame, x in zip(frames, xs)
        ]
        return Tracklet(
            tracklet_id=tracklet_id,
            start_frame=frames[0],
            end_frame=frames[-1],
            detections=detections,
            confidence=0.9,
        )

    before = list(range(20))
    after = list(range(30, 50))
    animal_a_before = [50 + 5 * frame for frame in before]
    animal_b_before = [300 - 5 * frame for frame in before]
    animal_a_before[-1] += 50  # corrupt only the legacy one-step endpoint velocity
    animal_b_before[-1] -= 60
    animal_a_after = [152 + 5 * (frame - 30) for frame in after]
    animal_b_after = [148 - 5 * (frame - 30) for frame in after]
    tracklets = [
        tracklet(0, before, animal_a_before, 60),
        tracklet(1, before, animal_b_before, 100),
        tracklet(2, after, animal_a_after, 60),
        tracklet(3, after, animal_b_after, 100),
    ]

    legacy = GlobalTrackletOptimizer(
        GlobalOptimizerConfig(
            method="ilp",
            max_temporal_gap=30,
            min_link_confidence=0.1,
        ),
        n_animals=2,
        frame_size=(400, 200),
    )
    legacy_trajectories = legacy.optimize(tracklets)
    long_term = GlobalTrackletOptimizer(
        GlobalOptimizerConfig(
            method="long_term",
            max_temporal_gap=30,
            min_link_confidence=0.1,
        ),
        n_animals=2,
        frame_size=(400, 200),
    )
    long_term_trajectories = long_term.optimize(tracklets)

    legacy_chains = {
        tuple(segment.tracklet.tracklet_id for segment in trajectory.segments)
        for trajectory in legacy_trajectories
    }
    long_term_chains = {
        tuple(segment.tracklet.tracklet_id for segment in trajectory.segments)
        for trajectory in long_term_trajectories
    }
    assert legacy_chains == {(0, 3), (1, 2)}
    assert long_term_chains == {(0, 2), (1, 3)}
    assert long_term.stats is not None
    assert long_term.stats.closed_set_achieved is True
    assert long_term.stats.discrete_geometry_enabled is True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gt_path = tmp_path / "ground_truth.mot"
        legacy_path = tmp_path / "legacy.mot"
        long_term_path = tmp_path / "long_term.mot"
        gt_rows = []
        for identity, parts in {
            0: (tracklets[0], tracklets[2]),
            1: (tracklets[1], tracklets[3]),
        }.items():
            for part in parts:
                for detection in part.detections:
                    x, y, width, height = detection.bbox_xywh
                    gt_rows.append(
                        f"{detection.frame_idx},{identity},{x},{y},{width},{height},1,-1,-1,-1"
                    )
        gt_path.write_text("\n".join(gt_rows) + "\n", encoding="utf-8")
        MOTExporter().export(legacy_trajectories, legacy_path)
        MOTExporter().export(long_term_trajectories, long_term_path)
        legacy_metrics = compute_mot_metrics(gt_path, legacy_path)
        long_term_metrics = compute_mot_metrics(gt_path, long_term_path)

    assert legacy_metrics["num_switches"] >= 2
    assert long_term_metrics["num_switches"] == 0
    assert long_term_metrics["idf1"] > legacy_metrics["idf1"]
    assert long_term_metrics["assa"] > legacy_metrics["assa"]
    print(
        "✅ Long-term identity: "
        f"ID switches {legacy_metrics['num_switches']:.0f} → "
        f"{long_term_metrics['num_switches']:.0f}, "
        f"AssA {legacy_metrics['assa']:.1%} → {long_term_metrics['assa']:.1%}"
    )


def test_dark_object_detector():
    """Test dark-object detection on synthetic bright bedding."""
    import cv2

    from densetrack.config import DetectionConfig
    from densetrack.detection.dark_object import DarkObjectDetector

    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    centers = [(120, 120), (320, 130), (150, 330), (460, 320)]
    for center in centers:
        cv2.ellipse(frame, center, (42, 24), 25, 0, 360, (18, 18, 18), -1)

    detector = DarkObjectDetector(DetectionConfig(min_mask_area=800, max_mask_area=6000))
    detections = detector.detect_frame(frame, frame_idx=0)

    assert len(detections) == 4, f"Expected 4 dark objects, got {len(detections)}"
    assert all(det.confidence > 0 for det in detections)

    print(f"✅ Dark-object detector: {len(detections)} synthetic animals detected")


def test_fixed_count_tracker():
    """Test fixed-count tracker keeps identities through a missing detection."""
    from densetrack.config import TrackletConfig
    from densetrack.models import Detection, DetectionSource
    from densetrack.tracking.fixed_count_tracker import FixedCountTrackletGenerator

    detections_per_frame = {}
    for frame_idx in range(30):
        dets = [
            Detection(
                frame_idx=frame_idx,
                bbox_xyxy=(50 + frame_idx * 2, 100, 90 + frame_idx * 2, 140),
                confidence=0.9,
                source=DetectionSource.DARK_OBJECT,
            )
        ]
        if not 10 <= frame_idx <= 14:
            dets.append(
                Detection(
                    frame_idx=frame_idx,
                    bbox_xyxy=(300, 100 + frame_idx * 2, 340, 140 + frame_idx * 2),
                    confidence=0.88,
                    source=DetectionSource.DARK_OBJECT,
                )
            )
        detections_per_frame[frame_idx] = dets

    tracker = FixedCountTrackletGenerator(
        TrackletConfig(backend="fixed_count", fixed_max_assignment_distance=100),
        n_animals=2,
    )
    tracklets = tracker.generate(detections_per_frame)

    assert len(tracklets) == 2, f"Expected 2 fixed-count tracklets, got {len(tracklets)}"
    assert all(len(tracklet.detections) == 30 for tracklet in tracklets)
    assert tracker.stats is not None
    assert tracker.stats.per_track_max_missing[1] >= 5

    print(
        "✅ Fixed-count tracker: "
        f"{len(tracklets)} tracks, observed {tracker.stats.avg_observed_tracks_per_frame:.1f}/2"
    )


def test_mac_adaptive_tracker():
    """Test zero-click adaptive bidirectional identity recovery through a crossing."""
    import cv2

    from densetrack.config import TrackletConfig
    from densetrack.models import Detection, DetectionSource
    from densetrack.tracking.mac_adaptive_tracker import MacAdaptiveTrackletGenerator

    def crop_bytes(intensity: int) -> bytes:
        crop = np.full((30, 42, 3), intensity, dtype=np.uint8)
        cv2.circle(crop, (12, 14), 6, (min(intensity + 40, 255),) * 3, -1)
        success, encoded = cv2.imencode(".jpg", crop)
        assert success
        return encoded.tobytes()

    detections_per_frame = {}
    for frame_idx in range(44):
        left_to_right_x = 55 + frame_idx * 7
        right_to_left_x = 355 - frame_idx * 7
        detections = [
            Detection(
                frame_idx=frame_idx,
                bbox_xyxy=(left_to_right_x, 100, left_to_right_x + 42, 130),
                confidence=0.94,
                source=DetectionSource.DARK_OBJECT,
                crop_bytes=crop_bytes(45),
            )
        ]
        if not 19 <= frame_idx <= 22:
            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    bbox_xyxy=(right_to_left_x, 100, right_to_left_x + 42, 130),
                    confidence=0.92,
                    source=DetectionSource.DARK_OBJECT,
                    crop_bytes=crop_bytes(155),
                )
            )
        else:
            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    bbox_xyxy=(405, 185, 447, 215),
                    confidence=0.99,
                    source=DetectionSource.DARK_OBJECT,
                    crop_bytes=crop_bytes(90),
                )
            )
        if frame_idx % 2:
            detections.reverse()
        detections_per_frame[frame_idx] = detections

    tracker = MacAdaptiveTrackletGenerator(
        TrackletConfig(
            backend="mac_adaptive",
            fixed_max_assignment_distance=95,
            adaptive_appearance_weight=0.60,
            adaptive_consistency_distance=80,
        ),
        n_animals=2,
        frame_size=(460, 240),
    )
    tracklets = tracker.generate(detections_per_frame, total_frames=44)

    assert len(tracklets) == 2
    assert all(len(tracklet.detections) == 44 for tracklet in tracklets)
    assert tracklets[0].detections[-1].centroid[0] > tracklets[1].detections[-1].centroid[0]
    assert tracker.stats is not None
    assert tracker.stats.algorithm.startswith("mac_adaptive_bidirectional")
    assert tracker.stats.identity_stability_score > 0.9
    assert tracker.stats.appearance_descriptor_ratio > 0.8
    assert tracker.stats.bidirectional_agreement_ratio > 0.8
    assert tracker.stats.quantized_geometry_enabled is True
    assert tracker.stats.geometry_token_agreement_ratio > 0.8
    assert tracker.stats.max_smoothing_states <= 128
    assert tracker.stats.per_track_max_missing[1] >= 4
    assert tracker.stats.large_jump_frames == 0
    assert tracker.stats.identity_switch_risk_events == 0
    with tempfile.TemporaryDirectory() as tmp:
        written = tracker.write_identity_risk(tmp)
        risk_payload = json.loads(Path(written["identity_risk_json"]).read_text())
        assert Path(written["identity_risk_csv"]).exists()
        assert risk_payload["summary"]["total_frames"] == 44
        assert "identity_stability_score" in risk_payload["summary"]
        assert risk_payload["summary"]["large_jump_frames"] == 0
        assert risk_payload["summary"]["identity_switch_risk_events"] == 0
        assert "motion_outlier_rejections" in risk_payload["summary"]
        assert "contact-aware ambiguity" in risk_payload["method"]["long_term"]
        assert risk_payload["summary"]["quantized_geometry_enabled"] is True
        assert risk_payload["method"]["profile"] == "hierarchical_discrete_geometry_long_term"
        assert len(risk_payload["frames"]) == 44

    assisted = MacAdaptiveTrackletGenerator(
        TrackletConfig(
            backend="mac_adaptive",
            fixed_max_assignment_distance=95,
            adaptive_appearance_weight=0.60,
            adaptive_consistency_distance=80,
        ),
        n_animals=2,
        frame_size=(460, 240),
        initialization_points=[
            {"id": 0, "x": 376, "y": 115},
            {"id": 1, "x": 76, "y": 115},
        ],
        initialization_frame=0,
    )
    assisted_tracklets = assisted.generate(detections_per_frame, total_frames=44)
    assert assisted.stats is not None
    assert assisted.stats.algorithm.startswith("mac_adaptive_bidirectional_click_assisted")
    assert assisted.stats.algorithm.endswith("identity_stable")
    assert assisted_tracklets[0].detections[-1].centroid[0] < assisted_tracklets[1].detections[-1].centroid[0]

    print(
        "✅ Mac-adaptive tracker: "
        f"agreement={tracker.stats.bidirectional_agreement_ratio:.1%}, "
        f"appearance={tracker.stats.appearance_descriptor_ratio:.1%}"
    )


def test_geometric_pose():
    """Test geometric pose estimation on synthetic frame."""
    from densetrack.config import PoseConfig
    from densetrack.models import Detection, DetectionSource
    from densetrack.pose.geometric import GeometricPoseEstimator

    config = PoseConfig(backend="geometry")
    estimator = GeometricPoseEstimator(config)

    # Create a synthetic frame with an elliptical white blob
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    import cv2
    cv2.ellipse(frame, (320, 240), (80, 40), 30, 0, 360, (255, 255, 255), -1)

    det = Detection(
        frame_idx=0,
        bbox_xyxy=(240, 190, 400, 290),
        confidence=0.9,
        source=DetectionSource.BACKGROUND_SUB,
    )

    pose = estimator.estimate(frame, det)
    
    assert pose.body_area is not None and pose.body_area > 0, "Body area should be positive"
    assert pose.orientation is not None, "Orientation should be computed"
    assert pose.centroid is not None, "Centroid should be computed"
    
    print(f"✅ Geometric pose: area={pose.body_area:.0f}, "
          f"orient={pose.orientation:.2f}rad, "
          f"length={pose.body_length}, "
          f"aspect={pose.aspect_ratio}")


def test_confidence_audit():
    """Test confidence auditor and report generation."""
    from densetrack.config import ConfidenceConfig
    from densetrack.confidence.auditor import ConfidenceAuditor
    from densetrack.confidence.report import ReportGenerator
    from densetrack.models import Detection, DetectionSource, Tracklet, Trajectory

    config = ConfidenceConfig()
    auditor = ConfidenceAuditor(config)
    
    # Build a trajectory with 2 segments
    dets1 = [Detection(frame_idx=f, bbox_xyxy=(100, 100, 200, 200),
                       confidence=0.9, source=DetectionSource.BACKGROUND_SUB)
             for f in range(20)]
    t1 = Tracklet(tracklet_id=0, start_frame=0, end_frame=19, detections=dets1, confidence=0.9)
    
    dets2 = [Detection(frame_idx=f, bbox_xyxy=(110, 105, 210, 205),
                       confidence=0.7, source=DetectionSource.BACKGROUND_SUB)
             for f in range(25, 45)]
    t2 = Tracklet(tracklet_id=1, start_frame=25, end_frame=44, detections=dets2, confidence=0.7)
    
    traj = Trajectory(global_id=0)
    traj.add_segment(t1, link_confidence=1.0, confidence_reason="first segment")
    traj.add_segment(t2, link_confidence=0.45, confidence_reason="遮挡恢复，运动模式匹配但外观不确定")
    
    report = auditor.generate_report([traj])
    
    assert report["summary"]["total_trajectories"] == 1
    assert report["summary"]["total_segments"] == 2
    assert len(report["risky_segments"]) >= 1  # The 0.45 link should be flagged
    
    # Format report
    gen = ReportGenerator()
    text = gen.to_text(report)
    assert "置信度审计报告" in text
    assert "🔴" in text or "🟠" in text  # Low confidence should appear
    
    # JSON export
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        gen.to_json(report, f.name)
        saved = json.loads(Path(f.name).read_text())
        assert "summary" in saved
    
    print(f"✅ Confidence audit: {report['summary']['total_segments']} segments, "
          f"{len(report['risky_segments'])} risky links")
    print(f"   High: {report['summary']['high_confidence_ratio']:.0%}, "
          f"Low: {report['summary']['low_confidence_ratio']:.0%}")


def test_export_mot():
    """Test MOT format export."""
    from densetrack.export.mot_format import MOTExporter
    from densetrack.models import Detection, DetectionSource, Tracklet, Trajectory

    dets = [Detection(frame_idx=f, bbox_xyxy=(100 + f, 200, 150 + f, 250),
                      confidence=0.9, source=DetectionSource.BACKGROUND_SUB)
            for f in range(10)]
    t = Tracklet(tracklet_id=0, start_frame=0, end_frame=9, detections=dets, confidence=0.9)
    traj = Trajectory(global_id=0)
    traj.add_segment(t)
    
    with tempfile.NamedTemporaryFile(suffix=".mot", delete=False, mode='w') as f:
        path = f.name
    
    exporter = MOTExporter()
    exporter.export([traj], path)
    
    content = Path(path).read_text()
    lines = [l for l in content.strip().split("\n") if not l.startswith("#")]
    assert len(lines) == 10, f"Expected 10 data lines, got {len(lines)}"
    
    print(f"✅ MOT export: {len(lines)} entries written")


def test_clean_overlay_preview_from_mot():
    """Test clean MOT overlay generation cuts trails across implausible jumps."""
    import cv2

    from densetrack.export.overlay_video import OverlayVideoGenerator

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "overlay_input.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (128, 96),
        )
        assert writer.isOpened()
        for _ in range(4):
            writer.write(np.zeros((96, 128, 3), dtype=np.uint8))
        writer.release()

        mot_path = tmp_path / "tracks.mot"
        mot_path.write_text(
            "# frame,id,x,y,w,h,conf\n"
            "0,0,10,55,12,12,0.95,-1,-1,-1\n"
            "1,0,15,55,12,12,0.95,-1,-1,-1\n"
            "2,0,90,55,12,12,0.95,-1,-1,-1\n"
            "3,0,95,55,12,12,0.95,-1,-1,-1\n",
            encoding="utf-8",
        )
        clean_path = tmp_path / "tracking_preview_clean.mp4"
        OverlayVideoGenerator(
            trail_length=4,
            max_trail_step_px=25,
            min_trail_confidence=0.5,
            show_confidence=False,
            bbox_thickness=1,
        ).generate_from_mot(video_path, mot_path, clean_path)

        assert clean_path.exists()
        capture = cv2.VideoCapture(str(clean_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ok, frame = capture.read()
        capture.release()
        assert ok
        middle_gap = frame[59:64, 48:58]
        right_box = frame[55:68, 90:103]
        assert float(middle_gap.mean()) < 12, "Unexpected long trail across identity jump"
        assert float(right_box.mean()) > 5, "Expected current-frame box to be visible"

    print("✅ Clean overlay: MOT preview generation cuts large trail jumps")


def test_export_sleap():
    """Test SLEAP-compatible JSON export."""
    import json

    from densetrack.export.sleap_format import SLEAPExporter
    from densetrack.models import Detection, DetectionSource, Tracklet, Trajectory

    dets = [
        Detection(
            frame_idx=f,
            bbox_xyxy=(100 + f, 200, 150 + f, 250),
            confidence=0.9,
            source=DetectionSource.DARK_OBJECT,
        )
        for f in range(3)
    ]
    tracklet = Tracklet(tracklet_id=0, start_frame=0, end_frame=2, detections=dets, confidence=0.9)
    traj = Trajectory(global_id=0)
    traj.add_segment(tracklet)

    with tempfile.NamedTemporaryFile(suffix=".sleap.json", delete=False) as f:
        path = f.name

    SLEAPExporter().export([traj], output_path=path, video_path="synthetic.mp4")
    payload = json.loads(Path(path).read_text())

    assert payload["format"] == "densetrack_sleap_json"
    assert len(payload["frames"]) == 3
    assert payload["frames"][0]["instances"][0]["track_id"] == 0

    print("✅ SLEAP export: JSON labels written")


def test_run_reexport():
    """Test re-exporting from a previous run directory."""
    import json

    from densetrack.export.run_reexport import reexport_run

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        mot_path = run_dir / "tracks.mot"
        mot_path.write_text(
            "# test\n"
            "0,0,10,20,30,40,0.90,-1,-1,-1\n"
            "1,0,11,20,30,40,0.40,-1,-1,-1\n"
        )
        (run_dir / "run_summary.json").write_text(json.dumps({
            "video_metadata": {"width": 640, "height": 480, "fps": 15}
        }))

        out_mot = reexport_run(run_dir, run_dir / "filtered.mot", "mot", min_confidence=0.5)
        out_sleap = reexport_run(run_dir, run_dir / "labels.sleap.json", "sleap", min_confidence=0.5)

        mot_lines = [line for line in Path(out_mot).read_text().splitlines() if not line.startswith("#")]
        sleap = json.loads(Path(out_sleap).read_text())

        assert len(mot_lines) == 1
        assert len(sleap["frames"]) == 1

    print("✅ Run re-export: MOT and SLEAP filters OK")


def test_behavior_analysis():
    """Test rule-based behavior analysis from MOT data."""
    import json

    from densetrack.behavior.rule_based import BehaviorConfig, RuleBasedBehaviorAnalyzer

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mot_path = tmp_path / "tracks.mot"
        mot_path.write_text(
            "# test\n"
            "0,0,0,0,20,20,0.9,-1,-1,-1\n"
            "0,1,40,0,20,20,0.9,-1,-1,-1\n"
            "1,0,10,0,20,20,0.9,-1,-1,-1\n"
            "1,1,42,0,20,20,0.9,-1,-1,-1\n"
            "2,0,20,0,20,20,0.9,-1,-1,-1\n"
            "2,1,44,0,20,20,0.9,-1,-1,-1\n"
        )
        report_path = tmp_path / "behavior.json"
        timeline_path = tmp_path / "timeline.csv"

        analyzer = RuleBasedBehaviorAnalyzer(
            BehaviorConfig(fps=10, contact_distance_px=60, activity_speed_px_s=20)
        )
        report = analyzer.analyze_mot(mot_path, report_path, timeline_path)
        saved = json.loads(report_path.read_text())

        assert report["summary"]["num_ids"] == 2
        assert saved["interactions"]["contact_frames"][0][1] >= 1
        assert timeline_path.exists()

    print("✅ Behavior analysis: activity/contact report OK")


def test_experiment_template_metrics():
    """Test Tracking-Master-style experiment template zone/time-bin metrics."""
    from densetrack.experiments import (
        ExperimentMetricsAnalyzer,
        ExperimentMetricsConfig,
        get_experiment_template,
        list_experiment_templates,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        mot_path = run_dir / "tracks.mot"
        lines = ["# synthetic open field"]
        for frame in range(20):
            lines.append(f"{frame},0,{40 + frame * 8},80,20,20,0.9,-1,-1,-1")
            lines.append(f"{frame},1,250,20,20,20,0.9,-1,-1,-1")
        mot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (run_dir / "run_summary.json").write_text(
            json.dumps({"video_metadata": {"width": 320, "height": 240, "fps": 10, "total_frames": 20}}),
            encoding="utf-8",
        )

        templates = list_experiment_templates()
        assert any(template.template_id == "open_field" for template in templates)
        assert get_experiment_template("open_field").name_zh == "旷场实验"

        report = ExperimentMetricsAnalyzer(
            ExperimentMetricsConfig(template_id="open_field", time_bin_s=1.0)
        ).analyze_run(run_dir)

        assert report["format"] == "densetrack_experiment_metrics"
        assert report["template"]["id"] == "open_field"
        assert report["summary"]["num_ids"] == 2
        assert report["summary"]["primary_zone_id"] == "center"
        assert report["summary"]["primary_zone_time_s"] > 0
        assert report["summary"]["mean_speed_px_s"] > 0
        assert report["transparent_baseline"]["total_distance_px"] > 0
        assert report["transparent_baseline"]["tracking_coverage_ratio"] == 1.0
        assert report["analysis_policy"]["default_tier"] == "transparent_parametric_baseline"
        assert (
            report["analysis_policy"]["evidence"]["doi"]
            == "10.64898/2026.07.20.739643"
        )
        assert (run_dir / "experiment_metrics.json").exists()
        assert (run_dir / "experiment_zone_metrics.csv").exists()
        assert (run_dir / "experiment_time_bins.csv").exists()
        zone_header = (run_dir / "experiment_zone_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "zone_id" in zone_header

        social = ExperimentMetricsAnalyzer(
            ExperimentMetricsConfig(template_id="three_chamber_social", time_bin_s=2.0)
        ).analyze_run(run_dir, output_json=run_dir / "social_metrics.json")
        assert social["template"]["id"] == "three_chamber_social"
        assert "social_chamber" in {zone["zone_id"] for zone in social["aggregate_zones"]}

    print("✅ Experiment templates: zone and time-bin metrics OK")


def test_config_yaml():
    """Test YAML config loading."""
    from densetrack.config import PipelineConfig

    config = PipelineConfig.from_yaml("configs/mice_topview.yaml")
    
    assert config.species.value == "mouse"
    assert config.detection.backend == "background_sub"
    assert config.pose.backend == "geometry"
    assert config.tracklet.backend == "bytetrack"
    assert config.reid.enabled is True
    assert config.experiment.enabled is True
    assert config.experiment.template_id == "open_field"
    assert "mot" in config.export.formats
    
    # Test resolve_device
    device = config.resolve_device()
    assert device in ("cpu", "cuda", "mps")
    
    print(f"✅ Config YAML: species={config.species.value}, device={device}")


def test_norfair_backend_contract():
    """Test mature OSS Norfair backend config and dependency boundary."""
    from densetrack.config import PipelineConfig, TrackletConfig
    from densetrack.tracking.norfair_tracker import NorfairTrackletGenerator

    config = PipelineConfig.from_yaml("configs/mice_topview_norfair.yaml")
    assert config.tracklet.backend == "norfair"
    assert config.reid.enabled is False

    generator = NorfairTrackletGenerator(TrackletConfig(backend="norfair"))
    try:
        result = generator.generate({})
    except ImportError as exc:
        message = str(exc)
        assert "norfair is not installed" in message
        assert "densetrack[norfair]" in message
    else:
        assert result == []

    print("✅ Norfair backend: config and optional dependency boundary OK")


def test_udmt_external_adapter():
    """Test UDMT external adapter validation and MOT import bridge."""
    from densetrack.external.udmt import UDMTAdapter, UDMTExternalConfig

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        udmt_home = tmp_path / "UDMT"
        (udmt_home / "udmt").mkdir(parents=True)
        (udmt_home / "README.md").write_text("# UDMT\n", encoding="utf-8")
        (udmt_home / "LICENSE").write_text("research use only\n", encoding="utf-8")

        blocked = UDMTAdapter(
            UDMTExternalConfig(
                udmt_home=udmt_home,
                python_executable=sys.executable,
                license_accepted=False,
            )
        )
        blocked_report = blocked.validate(check_import=False)
        assert not blocked_report["ok"]
        assert any("license" in err.lower() for err in blocked_report["errors"])

        adapter = UDMTAdapter(
            UDMTExternalConfig(
                udmt_home=udmt_home,
                python_executable=sys.executable,
                license_accepted=True,
            )
        )
        report = adapter.validate(check_import=False)
        assert report["ok"], report
        assert adapter.launch_command() == [sys.executable, "-m", "udmt.gui.launch_script"]

        source_mot = tmp_path / "udmt_tracks.mot"
        source_mot.write_text(
            "\n".join(
                [
                    "0,1,10,20,30,40,0.90,-1,-1,-1",
                    "1,1,12,22,30,40,0.88,-1,-1,-1",
                    "0,2,100,120,25,35,0.86,-1,-1,-1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "imported"
        summary = adapter.import_mot_tracks(
            source_mot,
            output_dir,
            video_metadata={"width": 320, "height": 240, "fps": 30},
        )

        assert summary["external_backend"] == "udmt"
        assert summary["tracking_summary"]["num_tracks"] == 2
        assert (output_dir / "tracks.mot").exists()
        assert (output_dir / "detections.coco.json").exists()
        assert (output_dir / "labels.sleap.json").exists()
        assert (output_dir / "run_summary.json").exists()

    print("✅ UDMT external adapter: validation and MOT import bridge OK")


def test_auto_initialization():
    """Test no-click automatic seed point generation on synthetic video."""
    import cv2

    from densetrack.config import DetectionConfig
    from densetrack.external.udmt import build_udmt_handoff_package
    from densetrack.initialization.auto_seed import (
        AutoInitConfig,
        create_manual_initialization_points,
        find_auto_initialization_points,
        write_auto_init_outputs,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "synthetic_init.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (640, 480),
        )
        centers = [(120, 120), (320, 130), (150, 330), (460, 320)]
        for _ in range(12):
            frame = np.full((480, 640, 3), 205, dtype=np.uint8)
            for center in centers:
                cv2.ellipse(frame, center, (42, 24), 25, 0, 360, (18, 18, 18), -1)
            writer.write(frame)
        writer.release()

        result = find_auto_initialization_points(
            video_path=video_path,
            n_animals=4,
            detection_config=DetectionConfig(
                backend="dark_object",
                min_mask_area=800,
                max_mask_area=6000,
            ),
            auto_config=AutoInitConfig(
                detector_backend="dark_object",
                max_frames=12,
                sample_every=1,
            ),
        )
        assert result.ok, result.payload
        assert len(result.payload["points"]) == 4
        files = write_auto_init_outputs(result, tmp_path / "auto_init")
        assert Path(files["json"]).exists()
        assert Path(files["image"]).exists()
        handoff = build_udmt_handoff_package(
            video_path=video_path,
            initialization_json=files["json"],
            output_dir=tmp_path / "udmt_handoff",
        )
        assert handoff["initialization"]["seed_points"] == 4
        assert handoff["automation_boundary"]["manual_training_annotation_required"] is False
        assert handoff["automation_boundary"]["manual_first_frame_click_required"] is False
        assert handoff["upstream_contract"]["direct_first_frame_drop_in_ready"] is True
        assert Path(handoff["files"]["udmt_start_pos_array"]).read_text(encoding="utf-8").count("\n") == 4
        assert Path(handoff["files"]["udmt_workspace_seed"]).exists()
        assert Path(handoff["files"]["manifest"]).exists()
        assert Path(handoff["files"]["instructions"]).exists()

        handoff_dir = tmp_path / "udmt_handoff"
        stale_seed = handoff_dir / "udmt_workspace_seed" / "tmp" / "stale" / "start_pos_array.txt"
        stale_seed.parent.mkdir(parents=True)
        stale_seed.write_text("1,2\n", encoding="utf-8")
        no_preview_dir = tmp_path / "no_preview"
        no_preview_dir.mkdir()
        no_preview_json = no_preview_dir / "initialization_points.json"
        no_preview_json.write_text(Path(files["json"]).read_text(encoding="utf-8"), encoding="utf-8")
        refreshed_handoff = build_udmt_handoff_package(
            video_path=video_path,
            initialization_json=no_preview_json,
            output_dir=handoff_dir,
        )
        assert refreshed_handoff["files"]["initialization_preview"] is None
        assert not stale_seed.exists()

        manual = create_manual_initialization_points(
            video_path=video_path,
            n_animals=4,
            points=[{"x": x, "y": y} for x, y in reversed(centers)],
        )
        assert manual.payload["initialization_mode"] == "manual_first_frame_click"
        assert manual.payload["provenance"]["manual_training_annotation_used"] is False
        assert manual.payload["provenance"]["manual_first_frame_click_used"] is True
        assert [point["id"] for point in manual.payload["points"]] == [0, 1, 2, 3]

    print("✅ Auto initialization: no-click seed generation and UDMT handoff OK")


def test_reid_self_supervised():
    """Test self-supervised ReID triplet mining, training, and similarity computation."""
    import cv2
    from densetrack.config import ReIDConfig
    from densetrack.models import Detection, DetectionSource, Tracklet
    from densetrack.reid.self_supervised import SelfSupervisedReID

    # Create dummy images for two distinct animals (e.g. one white, one black)
    black_crop = np.zeros((40, 40, 3), dtype=np.uint8)
    white_crop = np.full((40, 40, 3), 255, dtype=np.uint8)

    _, black_encoded = cv2.imencode('.jpg', black_crop)
    _, white_encoded = cv2.imencode('.jpg', white_crop)

    black_bytes = black_encoded.tobytes()
    white_bytes = white_encoded.tobytes()

    # Tracklet A: black mouse (frames 0-60)
    dets_a = []
    for f in range(60):
        dets_a.append(Detection(
            frame_idx=f,
            bbox_xyxy=(10, 10, 50, 50),
            confidence=0.95,
            source=DetectionSource.DARK_OBJECT,
            crop_bytes=black_bytes,
        ))
    t_a = Tracklet(tracklet_id=0, start_frame=0, end_frame=59, detections=dets_a, confidence=0.95)

    # Tracklet B: white mouse (frames 0-60)
    dets_b = []
    for f in range(60):
        dets_b.append(Detection(
            frame_idx=f,
            bbox_xyxy=(100, 10, 140, 50),
            confidence=0.95,
            source=DetectionSource.DARK_OBJECT,
            crop_bytes=white_bytes,
        ))
    t_b = Tracklet(tracklet_id=1, start_frame=0, end_frame=59, detections=dets_b, confidence=0.95)

    config = ReIDConfig(
        enabled=True,
        feat_dim=64,
        backbone="resnet18",
        confidence_threshold=0.8,
        min_triplets=5,
        epochs=3,
        learning_rate=1e-3,
        margin=0.3,
        max_iterations=1,
    )

    reid = SelfSupervisedReID(config, device="cpu")

    # Mine triplets
    triplets = reid.mine_training_data([t_a, t_b], None)
    assert len(triplets) >= 5, f"Expected at least 5 triplets, got {len(triplets)}"

    # Check triplet structure
    anchor, pos, neg = triplets[0]
    assert anchor.shape == (40, 40, 3)
    assert pos.shape == (40, 40, 3)
    assert neg.shape == (40, 40, 3)

    # Train ReID
    success = reid.train_on_video([t_a, t_b], None)
    assert success is True, "ReID training failed"

    # Compute similarities
    similarities = reid.compute_all_similarities([t_a, t_b], None)
    assert (0, 1) in similarities or (1, 0) in similarities, "Missing similarity between t_a and t_b"

    # Self similarity should be higher than cross similarity
    feat_extractor = reid._ensure_extractor()
    feat_a = feat_extractor.extract(black_crop)
    feat_b = feat_extractor.extract(white_crop)
    sim_cross = feat_extractor.compute_similarity(feat_a, feat_b)
    sim_self = feat_extractor.compute_similarity(feat_a, feat_a)

    assert sim_self > sim_cross, f"Self similarity {sim_self} should be greater than cross similarity {sim_cross}"

    print(f"✅ ReID self-supervised: mined {len(triplets)} triplets, trained successfully, self similarity={sim_self:.2f} > cross={sim_cross:.2f}")


def test_mot_metrics_standalone():
    """Test standalone MOT metrics parsing and IoU computation."""
    from densetrack.utils.metrics import compute_iou_matrix, compute_mot_metrics, load_mot_file

    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.mot"
        gt_path.write_text(
            "# ground truth\n"
            "0,0,10,20,30,40,1.0,-1,-1,-1\n"
            "0,1,100,120,25,35,1.0,-1,-1,-1\n"
            "1,0,12,22,30,40,1.0,-1,-1,-1\n"
            "1,1,102,122,25,35,1.0,-1,-1,-1\n"
        )

        pred_path = Path(tmp) / "pred.mot"
        pred_path.write_text(
            "# predictions\n"
            "0,0,11,21,30,40,0.95,-1,-1,-1\n"
            "0,1,101,121,25,35,0.90,-1,-1,-1\n"
            "1,0,13,23,30,40,0.92,-1,-1,-1\n"
            "1,1,103,123,25,35,0.88,-1,-1,-1\n"
        )

        gt_df = load_mot_file(gt_path)
        pred_df = load_mot_file(pred_path)

        assert len(gt_df) == 4, f"Expected 4 GT rows, got {len(gt_df)}"
        assert len(pred_df) == 4, f"Expected 4 pred rows, got {len(pred_df)}"
        assert "FrameId" in gt_df.columns
        assert "Confidence" in gt_df.columns

        # Test IoU computation
        gt_bboxes = gt_df[gt_df["FrameId"] == 0][["X", "Y", "Width", "Height"]].values
        pred_bboxes = pred_df[pred_df["FrameId"] == 0][["X", "Y", "Width", "Height"]].values

        iou = compute_iou_matrix(gt_bboxes, pred_bboxes)
        assert iou.shape == (2, 2)
        # Diagonal should have high IoU (nearly identical boxes, offset by 1px)
        assert iou[0, 0] > 0.8, f"GT[0] vs Pred[0] IoU too low: {iou[0, 0]}"
        assert iou[1, 1] > 0.8, f"GT[1] vs Pred[1] IoU too low: {iou[1, 1]}"
        # Off-diagonal should be ~0 (different locations)
        assert iou[0, 1] < 0.1, f"GT[0] vs Pred[1] IoU too high: {iou[0, 1]}"
        metrics = compute_mot_metrics(gt_path, pred_path)
        assert metrics["hota"] > 0.85
        assert metrics["assa"] > 0.85
        assert metrics["num_switches"] == 0

    print("✅ MOT metrics: HOTA/AssA/IDF1/ID switches and IoU computation OK")


def test_udmt_config_integration():
    """Test UDMT config section loads correctly from YAML."""
    from densetrack.config import PipelineConfig

    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "test_config.yaml"
        yaml_path.write_text(
            "species: mouse\n"
            "n_animals: 4\n"
            "tracklet:\n"
            "  backend: udmt\n"
            "udmt:\n"
            "  enabled: true\n"
            "  udmt_home: /tmp/UDMT\n"
            "  python_executable: /usr/bin/python3\n"
            "  license_accepted: true\n"
            "  auto_init: true\n"
        )

        cfg = PipelineConfig.from_yaml(yaml_path)

        assert cfg.tracklet.backend == "udmt"
        assert cfg.udmt.enabled is True
        assert cfg.udmt.udmt_home == "/tmp/UDMT"
        assert cfg.udmt.python_executable == "/usr/bin/python3"
        assert cfg.udmt.license_accepted is True
        assert cfg.udmt.auto_init is True

    print("✅ UDMT config integration: YAML loading and field mapping OK")


def test_unsupervised_behavior_discovery():
    """Test annotation-free clustering and visual outputs from MOT trajectories."""
    from densetrack.behavior.unsupervised import (
        BehaviorDiscoveryConfig,
        UnsupervisedBehaviorDiscovery,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mot_path = tmp_path / "tracks.mot"
        lines = ["# synthetic behavior motifs"]
        for frame in range(120):
            if frame < 40:
                x0, y0 = 40.0, 40.0
                x1, y1 = 300.0, 300.0
            elif frame < 80:
                x0, y0 = 40.0 + (frame - 40) * 6, 40.0
                x1, y1 = 300.0 - (frame - 40) * 3, 300.0
            else:
                x0, y0 = 240.0 + 8 * np.sin(frame / 3), 180.0 + 8 * np.cos(frame / 3)
                x1, y1 = x0 + 35.0, y0 + 20.0
            lines.append(f"{frame},0,{x0:.2f},{y0:.2f},30,20,0.9,-1,-1,-1")
            lines.append(f"{frame},1,{x1:.2f},{y1:.2f},30,20,0.9,-1,-1,-1")
        mot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        output_json = tmp_path / "clusters.json"
        output_csv = tmp_path / "timeline.csv"
        embedding = tmp_path / "embedding.jpg"
        timeline = tmp_path / "timeline.jpg"
        discovery = UnsupervisedBehaviorDiscovery(
            BehaviorDiscoveryConfig(
                fps=10.0,
                n_clusters=4,
                window_frames=7,
                sample_every=2,
            )
        )
        report = discovery.analyze_mot(
            mot_path,
            output_json=output_json,
            output_csv=output_csv,
            embedding_image=embedding,
            timeline_image=timeline,
        )

        assert report["annotation_required"] is False
        assert report["summary"]["num_ids"] == 2
        assert report["summary"]["num_clusters"] == 4
        assert report["summary"]["num_samples"] > 50
        assert len(report["clusters"]) == 4
        assert output_json.exists()
        assert output_csv.exists()
        assert embedding.exists()
        assert timeline.exists()

        pose_csv = tmp_path / "poses.csv"
        pose_lines = ["frame,id,x,y,confidence,orientation,body_length,body_area"]
        for frame in range(120):
            for track_id in (0, 1):
                pose_lines.append(
                    f"{frame},{track_id},0,0,0.9,{frame / 30:.3f},{50 + track_id},{900 + frame}"
                )
        pose_csv.write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
        pose_report = UnsupervisedBehaviorDiscovery(
            BehaviorDiscoveryConfig(
                fps=10.0,
                n_clusters=4,
                window_frames=7,
                sample_every=2,
                pose_csv=str(pose_csv),
                feature_mode="pose",
            )
        ).analyze_mot(mot_path)
        assert pose_report["summary"]["feature_mode"] == "pose_enriched"
        assert "pose_orientation_sin" in pose_report["features"]["names"]
        assert "pose_body_length_px" in pose_report["features"]["names"]

        try:
            import umap  # noqa: F401
        except ImportError:
            try:
                UnsupervisedBehaviorDiscovery(
                    BehaviorDiscoveryConfig(embedding_method="umap")
                ).analyze_mot(mot_path)
            except ImportError as exc:
                assert "pip install -e '.[behavior]'" in str(exc)
            else:
                raise AssertionError("Expected optional UMAP dependency guidance")

        try:
            import hdbscan  # noqa: F401
        except ImportError:
            pass
        else:
            hdbscan_report = UnsupervisedBehaviorDiscovery(
                BehaviorDiscoveryConfig(
                    fps=10.0,
                    window_frames=5,
                    sample_every=2,
                    clustering_method="hdbscan",
                    hdbscan_min_cluster_size=10,
                )
            ).analyze_mot(mot_path)
            effective_clusters = [
                cluster for cluster in hdbscan_report["clusters"] if cluster["cluster"] >= 0
            ]
            noise_clusters = [
                cluster for cluster in hdbscan_report["clusters"] if cluster["cluster"] == -1
            ]
            assert hdbscan_report["summary"]["num_clusters"] == len(effective_clusters)
            assert hdbscan_report["summary"]["num_noise_samples"] == sum(
                cluster["num_samples"] for cluster in noise_clusters
            )

    print("✅ Unsupervised behavior discovery: centroid, pose-enriched, and optional advanced paths OK")


def test_visual_ssl_representation():
    """Test BEAST3D-inspired annotation-free visual representation export."""
    from densetrack.representation.visual_ssl import (
        VisualSSLRepresentationAnalyzer,
        VisualSSLRepresentationConfig,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        mot_lines = ["# synthetic SSL run"]
        pose_lines = ["frame,id,x,y,confidence,orientation,body_length,body_area"]
        risk_rows = []
        for frame in range(60):
            x0 = 35.0 + frame * 2.5
            y0 = 60.0 + 8.0 * np.sin(frame / 6)
            x1 = 250.0 - frame * 1.8
            y1 = 150.0 + 10.0 * np.cos(frame / 7)
            mot_lines.append(f"{frame},0,{x0:.2f},{y0:.2f},28,18,0.92,-1,-1,-1")
            mot_lines.append(f"{frame},1,{x1:.2f},{y1:.2f},26,20,0.88,-1,-1,-1")
            pose_lines.append(f"{frame},0,0,0,0.9,{frame / 20:.3f},48,{700 + frame}")
            pose_lines.append(f"{frame},1,0,0,0.9,{frame / 22:.3f},45,{650 + frame}")
            risk_rows.append(
                {
                    "frame": frame,
                    "risk_score": 0.5 if frame % 13 == 0 else 0.05,
                }
            )
        (run_dir / "tracks.mot").write_text("\n".join(mot_lines) + "\n", encoding="utf-8")
        (run_dir / "poses.csv").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
        (run_dir / "identity_risk.json").write_text(
            json.dumps({"frames": risk_rows}),
            encoding="utf-8",
        )
        camera_manifest = run_dir / "cameras.yaml"
        camera_manifest.write_text(
            "views:\n"
            "  - id: cam0\n"
            "    intrinsics: {fx: 1000, fy: 1000, cx: 320, cy: 240}\n"
            "    extrinsics: {rotation: [1, 0, 0, 0], translation: [0, 0, 0]}\n"
            "  - id: cam1\n"
            "    intrinsics: {fx: 1000, fy: 1000, cx: 320, cy: 240}\n"
            "    extrinsics: {rotation: [1, 0, 0, 0], translation: [1, 0, 0]}\n",
            encoding="utf-8",
        )

        output_json = run_dir / "visual_ssl_report.json"
        output_csv = run_dir / "visual_ssl_embeddings.csv"
        report = VisualSSLRepresentationAnalyzer(
            VisualSSLRepresentationConfig(
                dim=6,
                window_frames=5,
                sample_every=3,
                camera_manifest=str(camera_manifest),
            )
        ).analyze_run(
            run_dir=run_dir,
            output_json=output_json,
            output_csv=output_csv,
        )

        assert report["annotation_required"] is False
        assert report["format"] == "densetrack_visual_ssl_report"
        assert report["summary"]["num_ids"] == 2
        assert report["summary"]["num_samples"] > 20
        assert report["summary"]["pose_features_used"] is True
        assert report["summary"]["temporal_positive_pairs"] > 0
        assert report["summary"]["same_frame_negative_pairs"] > 0
        assert report["summary"]["multiview_ready"] is True
        csv_header = output_csv.read_text(encoding="utf-8").splitlines()[0]
        assert "ssl_0" in csv_header
        assert "pose_orientation_sin" in csv_header

    print("✅ Visual SSL representation: trajectory, pose, risk, and camera readiness outputs OK")


def test_bioacoustic_ssl():
    """Test Dolph2Vec-inspired annotation-free bioacoustic SSL export."""
    from densetrack.audio.bioacoustic_ssl import BioacousticSSLAnalyzer, BioacousticSSLConfig

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sample_rate = 16000
        duration_s = 1.4
        time_axis = np.arange(int(sample_rate * duration_s)) / sample_rate
        signal = 0.08 * np.sin(2 * np.pi * 220 * time_axis)
        signal += 0.55 * np.sin(2 * np.pi * 2400 * time_axis)
        signal += 0.03 * np.random.default_rng(42).normal(size=len(time_axis))
        int16 = np.clip(signal, -1.0, 1.0)
        int16 = (int16 * 32767).astype("<i2")
        audio_path = tmp_path / "synthetic_whistle.wav"
        with wave.open(str(audio_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(int16.tobytes())

        output_json = tmp_path / "bioacoustic_ssl_report.json"
        output_csv = tmp_path / "bioacoustic_ssl_units.csv"
        report = BioacousticSSLAnalyzer(
            BioacousticSSLConfig(
                species="dolphin",
                window_ms=120,
                hop_ms=40,
                n_bands=12,
                codebook_size=5,
                whistle_min_frequency_hz=1000,
            )
        ).analyze_wav(
            audio_path=audio_path,
            output_json=output_json,
            output_csv=output_csv,
        )

        assert report["annotation_required"] is False
        assert report["format"] == "densetrack_bioacoustic_ssl_report"
        assert report["summary"]["num_windows"] > 0
        assert report["summary"]["codebook_units"] > 0
        assert report["summary"]["whistle_candidate_windows"] > 0
        csv_header = output_csv.read_text(encoding="utf-8").splitlines()[0]
        assert "codebook_unit" in csv_header
        assert "whistle_candidate" in csv_header

    print("✅ Bioacoustic SSL: WAV features, codebook units, and whistle candidates OK")


def test_static_dashboard():
    """Test standalone HTML dashboard generation and auto-init asset inclusion."""
    import cv2

    from densetrack.reporting.static_dashboard import build_static_dashboard

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_dir = tmp_path / "run"
        init_dir = tmp_path / "init"
        run_dir.mkdir()
        init_dir.mkdir()
        (run_dir / "tracks.mot").write_text("0,0,10,20,30,40,0.9,-1,-1,-1\n")
        (run_dir / "run_summary.json").write_text(json.dumps({
            "video_metadata": {"path": "synthetic.mp4", "fps": 10, "total_frames": 100},
            "fixed_count_stats": {"avg_observed_tracks_per_frame": 1.0, "mostly_observed_ratio": 1.0},
        }))
        (run_dir / "confidence_report.json").write_text(json.dumps({
            "summary": {"total_trajectories": 1},
            "trajectories": [{"global_id": 0, "effective_confidence": 0.9, "level": "high", "total_frames": 100}],
        }))
        (run_dir / "behavior_report.json").write_text(json.dumps({
            "per_id": {"0": {"frames_tracked": 100, "mean_speed_px_s": 12, "activity_ratio": 0.5, "distance_px": 120}},
        }))
        (run_dir / "behavior_clusters.json").write_text(json.dumps({
            "summary": {"num_ids": 1, "num_samples": 10, "num_clusters": 1},
            "clusters": [{
                "cluster": 0,
                "label": "low_activity_c0",
                "ratio": 1.0,
                "num_samples": 10,
                "feature_means": {},
            }],
        }))
        (run_dir / "identity_risk.json").write_text(json.dumps({
            "summary": {"high_risk_frames": 2, "mean_risk_score": 0.25},
            "frames": [{
                "frame": 12,
                "risk_score": 0.75,
                "disagreement_count": 1,
                "one_sided_count": 0,
                "prediction_count": 1,
                "reasons": "bidirectional_disagreement=1",
            }],
        }))
        (run_dir / "identity_continuity.json").write_text(json.dumps({
            "method": {
                "hierarchical_discrete_geometry": True,
            },
            "diagnostics": {
                "input_tracklets": 3,
                "output_trajectories": 1,
                "selected_stitches": 2,
                "closed_set_achieved": True,
                "tracker": {
                    "reacquisition_events": 2,
                    "deferred_ambiguous_assignments": 1,
                },
                "optimizer": {
                    "valid_candidate_links": 4,
                },
            },
            "scientific_validation": {
                "status_display": "需要人工审核 GT / reviewed ground truth required",
            },
            "references": [
                {"name": "HieDG", "url": "https://arxiv.org/abs/2607.00494"},
                {"name": "WaspMOT", "url": "https://arxiv.org/abs/2607.08729"},
            ],
        }, ensure_ascii=False))
        (run_dir / "visual_ssl_report.json").write_text(json.dumps({
            "summary": {
                "num_samples": 12,
                "num_ids": 1,
                "embedding_dim": 4,
                "temporal_positive_pairs": 11,
                "same_frame_negative_pairs": 0,
                "multiview_status": "single_view_only / 仅单视角",
                "multiview_ready": False,
            },
            "limitations": ["deterministic test visual SSL"],
        }))
        (run_dir / "visual_ssl_embeddings.csv").write_text("frame,track_id,ssl_0\n0,0,0.1\n")
        (run_dir / "bioacoustic_ssl_report.json").write_text(json.dumps({
            "config": {"species": "dolphin"},
            "summary": {
                "species": "dolphin",
                "num_windows": 8,
                "codebook_units": 3,
                "whistle_candidate_windows": 5,
                "whistle_candidate_ratio": 0.625,
            },
            "limitations": ["deterministic test audio SSL"],
        }))
        (run_dir / "bioacoustic_ssl_units.csv").write_text(
            "start_s,end_s,codebook_unit,whistle_candidate\n0,0.1,0,true\n"
        )
        (run_dir / "experiment_metrics.json").write_text(json.dumps({
            "template": {
                "id": "open_field",
                "name_zh": "旷场实验",
                "name_en": "Open field",
                "primary_zone_id": "center",
                "recommended_duration_s": 480,
                "description_zh": "中心区和周边区停留、进入、速度和路程分析。",
                "description_en": "Center/periphery occupancy, entries, speed, and distance analysis.",
            },
            "summary": {
                "num_ids": 1,
                "num_zones": 2,
                "primary_zone_id": "center",
                "primary_zone_time_s": 4.2,
                "primary_zone_entries": 2,
            },
            "aggregate_zones": [
                {
                    "zone_id": "center",
                    "zone_name_zh": "中心区",
                    "zone_name_en": "Center zone",
                    "time_s": 4.2,
                    "entries": 2,
                    "observations": 42,
                }
            ],
        }, ensure_ascii=False))
        (run_dir / "experiment_zone_metrics.csv").write_text(
            "template_id,track_id,zone_id,time_s\nopen_field,0,center,4.2\n",
            encoding="utf-8",
        )
        (run_dir / "experiment_time_bins.csv").write_text(
            "template_id,track_id,zone_id,bin_start_s,bin_end_s,time_s\nopen_field,0,center,0,1,1\n",
            encoding="utf-8",
        )
        image = np.full((120, 160, 3), 220, dtype=np.uint8)
        cv2.imwrite(str(init_dir / "initialization_points.jpg"), image)
        (init_dir / "initialization_points.json").write_text("{}")
        preview = cv2.VideoWriter(
            str(run_dir / "tracking_preview_clean.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (160, 120),
        )
        assert preview.isOpened()
        preview.write(image)
        preview.release()

        dashboard = build_static_dashboard(run_dir, auto_init_dir=init_dir)
        dashboard_html = dashboard.read_text()
        assert dashboard.exists()
        assert "Multi-animal tracking dashboard" in dashboard_html
        assert "产品状态 / Product state" in dashboard_html
        assert "干净轨迹回放 / Clean tracking playback" in dashboard_html
        assert (
            "tracking_preview_browser.mp4" in dashboard_html
            or "tracking_preview_clean.mp4" in dashboard_html
        )
        assert "visual tracking ready, metrics need review" in dashboard_html
        assert "low_activity_c0" in dashboard_html
        assert "Bidirectional consistency risk" in dashboard_html
        assert "Research identity continuity" in dashboard_html
        assert "HieDG" in dashboard_html
        assert "identity_continuity.json" in dashboard_html
        assert "bidirectional_disagreement=1" in dashboard_html
        assert "BEAST3D-inspired" in dashboard_html
        assert "Dolph2Vec-inspired" in dashboard_html
        assert "visual_ssl_embeddings.csv" in dashboard_html
        assert "bioacoustic_ssl_units.csv" in dashboard_html
        assert "实验范式 / Experiment template" in dashboard_html
        assert "Tracking Master style metrics" in dashboard_html
        assert "Transparent baseline" in dashboard_html
        assert "10.64898/2026.07.20.739643" in dashboard_html
        assert "experiment_zone_metrics.csv" in dashboard_html
        assert (run_dir / "auto_initialization_points.jpg").exists()
        assert (run_dir / "auto_initialization_points.json").exists()

    print("✅ Static dashboard: HTML generation and auto-init assets OK")


def test_complete_local_workflow():
    """Test the zero-click local workflow from synthetic video to dashboard."""
    import cv2

    from densetrack.config import PipelineConfig
    from densetrack.workflows.complete_local import run_complete_local

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "synthetic.mp4"
        output_dir = tmp_path / "complete"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            12.0,
            (320, 240),
        )
        assert writer.isOpened()
        for frame_idx in range(36):
            frame = np.full((240, 320, 3), 215, dtype=np.uint8)
            cv2.ellipse(frame, (65 + frame_idx * 2, 75), (20, 12), 10, 0, 360, (18, 18, 18), -1)
            cv2.ellipse(frame, (250 - frame_idx * 2, 165), (19, 13), -15, 0, 360, (24, 24, 24), -1)
            writer.write(frame)
        writer.release()

        cfg = PipelineConfig()
        cfg.n_animals = 2
        cfg.detection.backend = "dark_object"
        cfg.detection.min_mask_area = 250
        cfg.detection.max_mask_area = 2500
        cfg.sam2.enabled = False
        cfg.pose.backend = "geometry"
        cfg.tracklet.backend = "fixed_count"
        cfg.reid.enabled = False
        cfg.export.formats = ["mot", "csv"]
        cfg.export.overlay_video = False
        cfg.processing_resolution = 0

        result = run_complete_local(
            video_path=video_path,
            pipeline_config=cfg,
            output_dir=output_dir,
            motif_clusters=3,
            auto_init_max_frames=24,
            auto_init_sample_every=2,
        )

        assert result["manual_annotation_used"] is False
        assert result["automatic_initialization"]["status"] == "ok"
        assert result["automatic_initialization"]["points"] == 2
        assert (output_dir / "tracks.mot").exists()
        assert (output_dir / "behavior_report.json").exists()
        assert (output_dir / "behavior_clusters.json").exists()
        assert (output_dir / "identity_continuity.json").exists()
        assert (output_dir / "identity_stitches.csv").exists()
        assert (output_dir / "contact_events.json").exists()
        assert (output_dir / "contact_events.csv").exists()
        contact_payload = json.loads((output_dir / "contact_events.json").read_text(encoding="utf-8"))
        assert contact_payload["format"] == "densetrack_contact_events"
        assert "total_events" in contact_payload["summary"]
        assert (output_dir / "experiment_metrics.json").exists()
        assert (output_dir / "experiment_zone_metrics.csv").exists()
        assert result["experiment_metrics"]["template"]["id"] == "open_field"
        pose_header = (output_dir / "poses.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "orientation" in pose_header
        motif_report = json.loads((output_dir / "behavior_clusters.json").read_text(encoding="utf-8"))
        assert motif_report["summary"]["feature_mode"] == "pose_enriched"
        assert (output_dir / "complete_run_manifest.json").exists()
        assert (output_dir / "validation_review" / "prelabels.mot").exists()
        assert (output_dir / "validation_review" / "corrected_gt_template.mot").exists()
        assert (output_dir / "validation_review" / "review_grid.jpg").exists()
        dashboard_html = (output_dir / "dashboard.html").read_text(encoding="utf-8")
        assert "多动物追踪研究面板" in dashboard_html
        assert "manual annotation: false" in dashboard_html
        assert "实验范式 / Experiment template" in dashboard_html
        assert "稀疏 GT 审核包" in dashboard_html
        assert "Research identity continuity" in dashboard_html
        assert "接触事件 / Contact events" in dashboard_html

    print("✅ Complete workflow: zero-click video-to-dashboard package OK")


def test_webapp_surface_and_artifact_guard():
    """Test bilingual local workspace UI and safe result-file resolution."""
    import io
    from unittest.mock import patch

    import cv2

    from densetrack.webapp import JobRegistry, _parse_range, render_index_html

    html = render_index_html()
    assert "本地研究工作台" in html
    assert "Local Research Workspace" in html
    assert 'data-testid="video-input"' in html
    assert 'data-testid="submit-job"' in html
    assert 'data-testid="tracking-profile"' in html
    assert "科研长时身份（推荐） / Research long-term identity (recommended)" in html
    assert 'data-testid="experiment-template"' in html
    assert "三腔社交实验 / Three-chamber social test" in html
    assert "上传并计算指标 / Upload and evaluate" in html
    assert "首帧点击辅助 / First-frame click assist" in html
    assert "data-seed-job" in html
    assert "/validation" in html
    assert _parse_range("bytes=10-19", 100) == (10, 19)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        registry = JobRegistry(tmp_path / "workspace", tmp_path / "config.yaml")
        result_dir = registry.result_dir("abc123def456")
        result_dir.mkdir(parents=True)
        artifact = result_dir / "dashboard.html"
        artifact.write_text("<html></html>", encoding="utf-8")
        assert registry.safe_artifact_path("abc123def456", "dashboard.html") == artifact.resolve()
        try:
            registry.safe_artifact_path("abc123def456", "../../config.yaml")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Expected traversal attempt to be rejected")

        config_path = tmp_path / "config.yaml"
        config_path.write_text("n_animals: 2\nprocessing_resolution: 0\n", encoding="utf-8")
        video_path = tmp_path / "picker.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (240, 180),
        )
        assert writer.isOpened()
        frame = np.full((180, 240, 3), 220, dtype=np.uint8)
        cv2.circle(frame, (60, 90), 20, (25, 25, 25), -1)
        cv2.circle(frame, (180, 90), 20, (35, 35, 35), -1)
        writer.write(frame)
        writer.release()

        click_registry = JobRegistry(tmp_path / "click_workspace", config_path)
        with video_path.open("rb") as source:
            click_job = click_registry.submit(
                io.BytesIO(source.read()),
                "picker.avi",
                n_animals=2,
                clusters=3,
                initialization_mode="manual_first_frame_click",
                experiment_template="three_chamber_social",
            )
        assert click_job["status"] == "awaiting_seed"
        assert click_job["experiment_template"] == "three_chamber_social"
        assert click_job["tracking_profile"] == "research_long_term"
        picker = click_registry.get_initialization(click_job["id"])
        assert picker["training_annotation_required"] is False
        preview = click_registry.safe_job_artifact_path(click_job["id"], "initialization/frame.jpg")
        assert preview.exists()
        with patch("densetrack.webapp.threading.Thread") as thread:
            queued = click_registry.start_click_assisted(
                click_job["id"],
                [{"x": 60, "y": 90}, {"x": 180, "y": 90}],
            )
            thread.assert_called_once()
        assert queued["status"] == "queued"
        assert queued["initialization_points"] == 2

    print("✅ Web workspace: bilingual UI, click-assisted picker, byte range, and artifact guard OK")


def test_sparse_validation_review_guard():
    """Test pre-filled GT template generation and explicit review confirmation."""
    import cv2

    from densetrack.validation.review_package import (
        build_validation_review_package,
        evaluate_validation_review_package,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "validation.mp4"
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (240, 180),
        )
        assert writer.isOpened()
        mot_lines = ["# synthetic sparse validation"]
        for frame in range(20):
            image = np.full((180, 240, 3), 218, dtype=np.uint8)
            cv2.circle(image, (45 + frame * 3, 65), 14, (20, 20, 20), -1)
            cv2.circle(image, (190 - frame * 2, 120), 14, (25, 25, 25), -1)
            writer.write(image)
            mot_lines.append(f"{frame},0,{31 + frame * 3},51,28,28,0.95,-1,-1,-1")
            mot_lines.append(f"{frame},1,{176 - frame * 2},106,28,28,0.92,-1,-1,-1")
        writer.release()
        (run_dir / "tracks.mot").write_text("\n".join(mot_lines) + "\n", encoding="utf-8")
        (run_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "config": {"n_animals": 2},
                    "video_metadata": {"path": str(video_path), "fps": 10, "total_frames": 20},
                }
            ),
            encoding="utf-8",
        )

        stale_raw_dir = run_dir / "validation_review" / "raw_frames"
        stale_raw_dir.mkdir(parents=True)
        (stale_raw_dir / "frame_999999.jpg").write_bytes(b"stale")
        package = build_validation_review_package(run_dir, n_frames=8)
        package_dir = Path(package["manifest"]).parent
        assert package["selected_review_frames"] == 8
        assert package["annotation_boundary"]["review_completed"] is False
        assert (package_dir / "review_grid.jpg").exists()
        assert (package_dir / "frames").is_dir()
        assert (package_dir / "raw_frames").is_dir()
        assert not (package_dir / "raw_frames" / "frame_999999.jpg").exists()
        assert len(list((package_dir / "raw_frames").glob("frame_*.jpg"))) == 8
        review_data = json.loads((package_dir / "review_data.json").read_text(encoding="utf-8"))
        assert review_data["format"] == "densetrack_browser_gt_review"
        assert len(review_data["frames"]) == 8
        assert len(review_data["frames"][0]["annotations"]) == 2
        review_html = (package_dir / "review.html").read_text(encoding="utf-8")
        assert "DenseTrack-AI GT 审核器 / GT Reviewer" in review_html
        assert "reviewed_gt.mot" in review_html
        assert "pointerdown" in review_html

        template = package_dir / "corrected_gt_template.mot"
        incomplete = package_dir / "incomplete_gt.mot"
        incomplete.write_text(
            "\n".join(template.read_text(encoding="utf-8").splitlines()[:5]) + "\n",
            encoding="utf-8",
        )
        try:
            evaluate_validation_review_package(
                package_dir,
                ground_truth=incomplete,
                confirm_reviewed=True,
            )
        except ValueError as exc:
            assert "missing selected review frames" in str(exc)
        else:
            raise AssertionError("Expected incomplete reviewed GT to be rejected")

        try:
            evaluate_validation_review_package(package_dir, ground_truth=template)
        except ValueError as exc:
            assert "confirm-reviewed" in str(exc)
        else:
            raise AssertionError("Expected unconfirmed GT evaluation to be rejected")

        report = evaluate_validation_review_package(
            package_dir,
            ground_truth=template,
            confirm_reviewed=True,
        )
        assert report["review_confirmed"] is True
        assert report["scope"] == "sparse_reviewed_frames_only"
        assert report["metrics"]["mota"] == 1.0
        assert report["metrics"]["idf1"] == 1.0
        assert (package_dir / "evaluation_report.json").exists()
        assert (package_dir / "evaluation_report.md").exists()
        manifest = json.loads((package_dir / "validation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["annotation_boundary"]["review_completed"] is True

        import io

        from densetrack.webapp import JobRegistry

        workspace = tmp_path / "workspace"
        registry = JobRegistry(workspace, tmp_path / "unused.yaml")
        job_id = "abc123def456"
        job_dir = workspace / "jobs" / job_id
        job_dir.mkdir()
        (job_dir / "results").symlink_to(run_dir, target_is_directory=True)
        registry._save(
            {
                "id": job_id,
                "status": "completed",
                "status_display": "已完成 / completed",
                "filename": "validation.mp4",
                "n_animals": 2,
                "clusters": 3,
                "created_at": "2026-06-01T00:00:00",
                "updated_at": "2026-06-01T00:00:00",
                "result_url": f"/jobs/{job_id}/results/dashboard.html",
                "error": None,
            }
        )
        web_report = registry.evaluate_review(
            job_id=job_id,
            source=io.BytesIO(template.read_bytes()),
            filename="reviewed_gt.mot",
            confirm_reviewed=True,
        )
        assert web_report["metrics"]["mota"] == 1.0
        assert web_report["metrics"]["hota"] == 1.0
        assert web_report["metrics"]["assa"] == 1.0
        assert web_report["report_url"].endswith("evaluation_report.md")
        web_job = registry.get(job_id)
        assert web_job is not None
        assert web_job["validation_status"] == "reviewed"
        dashboard_html = (run_dir / "dashboard.html").read_text(encoding="utf-8")
        assert "ID Switches" in dashboard_html
        assert "HOTA" in dashboard_html
        assert "AssA" in dashboard_html

        refreshed_package = build_validation_review_package(run_dir, n_frames=6)
        assert refreshed_package["annotation_boundary"]["review_completed"] is False
        assert not (package_dir / "reviewed_gt.mot").exists()
        assert not (package_dir / "evaluation_report.json").exists()
        assert not (package_dir / "evaluation_report.md").exists()
        assert "100.0%" in dashboard_html

    print("✅ Sparse validation: prelabels, review guard, Web upload, and confirmed metrics OK")


def _contact_test_box(frame, track_id, cx, cy, w=40, h=30, observed=True):
    from densetrack.tracking.contact_events import FrameBox

    return FrameBox(
        frame=frame,
        track_id=track_id,
        x1=cx - w / 2,
        y1=cy - h / 2,
        x2=cx + w / 2,
        y2=cy + h / 2,
        observed=observed,
    )


def test_contact_event_layer():
    """Auditable contact events: merge/split records, top-K hypotheses, margins."""
    from densetrack.config import ContactEventConfig
    from densetrack.export.run_reexport import MOTRow
    from densetrack.reporting.static_dashboard import build_static_dashboard
    from densetrack.tracking.contact_events import (
        detect_contact_events,
        write_contact_events_report,
    )
    from densetrack.validation.review_package import _score_frame

    config = ContactEventConfig(contact_distance_px=75.0)

    # Scenario A: motion-continuous crossing with a merged-measurement segment.
    frames = {}
    for frame_idx in range(60):
        if frame_idx < 20:
            boxes = [
                _contact_test_box(frame_idx, 1, 100 + 5 * frame_idx, 100),
                _contact_test_box(frame_idx, 2, 300 - 5 * frame_idx, 100),
            ]
        elif frame_idx < 30:
            boxes = [
                _contact_test_box(frame_idx, 1, 200, 100, w=60, h=40),
                _contact_test_box(frame_idx, 2, 200, 100, observed=False),
            ]
        else:
            boxes = [
                _contact_test_box(frame_idx, 1, 200 + 5 * (frame_idx - 29), 100),
                _contact_test_box(frame_idx, 2, 200 - 5 * (frame_idx - 29), 100),
            ]
        frames[frame_idx] = boxes
    result = detect_contact_events(frames, config, n_animals=2)
    assert result["summary"]["total_events"] == 1
    event = result["events"][0]
    assert event.participant_ids == [1, 2]
    assert event.merged_frames >= 10
    assert event.resolved and event.uncertainty == "confident"
    assert event.score_margin is not None and event.score_margin > 0.9
    assert event.best_assignment == {1: 1, 2: 2}
    assert event.tracker_matches_best is True
    assert len(event.hypotheses) == 2
    assert abs(sum(h.probability for h in event.hypotheses) - 1.0) < 1e-6
    assert event.hypotheses[0].probability > event.hypotheses[1].probability

    # Scenario B: bounce-back — the shipped identity continuation contradicts
    # motion continuity, so the audit must disagree with the tracker output.
    bounce = {}
    for frame_idx in range(60):
        if frame_idx < 20:
            boxes = [
                _contact_test_box(frame_idx, 1, 100 + 5 * frame_idx, 300),
                _contact_test_box(frame_idx, 2, 300 - 5 * frame_idx, 300),
            ]
        elif frame_idx < 30:
            boxes = [
                _contact_test_box(frame_idx, 1, 200, 300, w=60, h=40),
                _contact_test_box(frame_idx, 2, 200, 300, observed=False),
            ]
        else:
            boxes = [
                _contact_test_box(frame_idx, 1, 200 - 5 * (frame_idx - 29), 300),
                _contact_test_box(frame_idx, 2, 200 + 5 * (frame_idx - 29), 300),
            ]
        bounce[frame_idx] = boxes
    bounce_event = detect_contact_events(bounce, config, n_animals=2)["events"][0]
    assert bounce_event.best_assignment == {1: 2, 2: 1}
    assert bounce_event.tracker_matches_best is False
    assert "tracker_assignment_differs_from_best_hypothesis" in bounce_event.notes

    # Scenario C: symmetric perpendicular exit — evidence cannot separate the
    # two assignments, so the margin must collapse and the event is uncertain.
    symmetric = {}
    for frame_idx in range(70):
        if frame_idx < 20:
            boxes = [
                _contact_test_box(frame_idx, 1, 155, 100),
                _contact_test_box(frame_idx, 2, 245, 100),
            ]
        elif frame_idx < 35:
            boxes = [
                _contact_test_box(frame_idx, 1, 200, 100, w=70, h=45),
                _contact_test_box(frame_idx, 2, 200, 100, observed=False),
            ]
        else:
            boxes = [
                _contact_test_box(frame_idx, 1, 200, 100 - 5 * (frame_idx - 34)),
                _contact_test_box(frame_idx, 2, 200, 100 + 5 * (frame_idx - 34)),
            ]
        symmetric[frame_idx] = boxes
    symmetric_event = detect_contact_events(symmetric, config, n_animals=2)["events"][0]
    assert symmetric_event.uncertainty == "uncertain"
    assert symmetric_event.score_margin is not None and symmetric_event.score_margin < 0.2
    assert symmetric_event.review_priority > 0.5

    # No-contact sanity: no events.
    quiet = {
        frame_idx: [
            _contact_test_box(frame_idx, 1, 100, 100),
            _contact_test_box(frame_idx, 2, 500, 500),
        ]
        for frame_idx in range(30)
    }
    assert detect_contact_events(quiet, config, n_animals=2)["summary"]["total_events"] == 0

    # Scenario D: repeated bounce-backs chain into inversion episodes, and
    # proximity-only frames are classified without merged measurements.
    chained = {}
    for frame_idx in range(500):
        in_bounce = False
        for base in (0, 100, 400):
            local = frame_idx - base
            if 0 <= local < 60:
                in_bounce = True
                if local < 20:
                    boxes = [
                        _contact_test_box(frame_idx, 1, 100 + 5 * local, 300),
                        _contact_test_box(frame_idx, 2, 300 - 5 * local, 300),
                    ]
                elif local < 30:
                    boxes = [
                        _contact_test_box(frame_idx, 1, 200, 300, w=60, h=40),
                        _contact_test_box(frame_idx, 2, 200, 300, observed=False),
                    ]
                else:
                    boxes = [
                        _contact_test_box(frame_idx, 1, 200 - 5 * (local - 29), 300),
                        _contact_test_box(frame_idx, 2, 200 + 5 * (local - 29), 300),
                    ]
                break
        if not in_bounce:
            boxes = [
                _contact_test_box(frame_idx, 1, 80, 300),
                _contact_test_box(frame_idx, 2, 320, 300),
            ]
        chained[frame_idx] = boxes
    chained_result = detect_contact_events(chained, config, n_animals=2)
    chained_events = chained_result["events"]
    assert len(chained_events) == 3
    assert all(event.tracker_matches_best is False for event in chained_events)
    assert all(event.event_kind == "measurement_ambiguous" for event in chained_events)
    episodes = chained_result["inversion_episodes"]
    assert len(episodes) == 2, f"expected 2 inversion episodes, got {len(episodes)}"
    assert episodes[0]["n_events"] == 2
    assert episodes[1]["n_events"] == 1
    summary = chained_result["summary"]
    assert summary["tracker_disagreement_events"] == 3
    assert summary["inversion_episode_count"] == 2
    assert summary["event_kind_counts"]["measurement_ambiguous"] == 3

    proximity = {}
    for frame_idx in range(50):
        # Parallel lanes 60 px apart: close enough for contact distance, but
        # the boxes never overlap and neither contains the other's center.
        boxes = [
            _contact_test_box(frame_idx, 1, 100 + 5 * frame_idx, 100),
            _contact_test_box(frame_idx, 2, 280 - 5 * frame_idx, 160),
        ]
        proximity[frame_idx] = boxes
    proximity_result = detect_contact_events(proximity, config, n_animals=2)
    proximity_events = proximity_result["events"]
    assert len(proximity_events) == 1
    assert proximity_events[0].event_kind == "proximity_only"
    assert proximity_events[0].merged_frames == 0
    assert proximity_result["summary"]["event_kind_counts"]["proximity_only"] == 1
    assert proximity_result["summary"]["inversion_episode_count"] == 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Artifact writing from MOT rows (CLI path uses the same entry point).
        mot_rows = [
            MOTRow(frame=f, track_id=box.track_id, x=box.x1, y=box.y1,
                   width=box.x2 - box.x1, height=box.y2 - box.y1, confidence=0.9)
            for f, boxes in frames.items()
            for box in boxes
        ]
        source_map = {
            (f, 2): "adaptive_prediction" if 20 <= f < 30 else "dark_object"
            for f in range(60)
        }
        paths, summary = write_contact_events_report(
            tmp_path,
            config=config,
            mot_rows=mot_rows,
            source_map=source_map,
            n_animals=2,
            source_description="functional_test",
        )
        assert summary["total_events"] == 1
        payload = json.loads(Path(paths["contact_events_json"]).read_text(encoding="utf-8"))
        assert payload["format"] == "densetrack_contact_events"
        assert payload["summary"]["uncertainty_counts"]["confident"] == 1
        assert payload["events"][0]["hypotheses"][0]["probability"] > 0.9
        assert payload["frame_review_priorities"]
        csv_lines = Path(paths["contact_events_csv"]).read_text(encoding="utf-8").splitlines()
        assert csv_lines[0].startswith("event_id,start_frame")
        assert len(csv_lines) == 2

        # Dashboard renders the bilingual contact-event section.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "tracks.mot").write_text(
            "\n".join(
                f"{row.frame},{row.track_id},{row.x:.1f},{row.y:.1f},"
                f"{row.width:.1f},{row.height:.1f},{row.confidence:.2f},-1,-1,-1"
                for row in mot_rows
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "contact_events.json").write_text(
            Path(paths["contact_events_json"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        dashboard_html = build_static_dashboard(run_dir).read_text(encoding="utf-8")
        assert "接触事件 / Contact events" in dashboard_html
        assert "合并/拆分身份假设审计" in dashboard_html
        assert "contact_events.json" in dashboard_html
        assert "标签反转段 / label-inversion episodes" in dashboard_html
        assert "仅接近事件 / Proximity-only" in dashboard_html

        # Review-package scoring boosts frames inside contact events.
        review_rows = [
            MOTRow(frame=25, track_id=1, x=180, y=85, width=60, height=40, confidence=0.9),
            MOTRow(frame=25, track_id=2, x=182, y=86, width=60, height=40, confidence=0.9),
        ]
        base = _score_frame(25, review_rows, {}, {}, 2, {})
        boosted = _score_frame(25, review_rows, {}, {}, 2, {25: 0.9})
        assert boosted.risk_score > base.risk_score
        assert boosted.contact_event_score == 0.9
        assert "contact_event_priority" in boosted.risk_reasons

    print("✅ Contact events: merge/split audit, top-K hypotheses, margins, dashboard, review boost OK")


def main():
    print("=" * 60)
    print("  DenseTrack-AI Functional Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_kalman_filter,
        test_iou_batch,
        test_linear_assignment,
        test_tracklet_generator,
        test_global_optimizer,
        test_hierarchical_discrete_geometry,
        test_long_term_stitching_reduces_id_switches,
        test_dark_object_detector,
        test_fixed_count_tracker,
        test_mac_adaptive_tracker,
        test_geometric_pose,
        test_confidence_audit,
        test_export_mot,
        test_clean_overlay_preview_from_mot,
        test_export_sleap,
        test_run_reexport,
        test_behavior_analysis,
        test_experiment_template_metrics,
        test_config_yaml,
        test_norfair_backend_contract,
        test_udmt_external_adapter,
        test_auto_initialization,
        test_reid_self_supervised,
        test_mot_metrics_standalone,
        test_udmt_config_integration,
        test_unsupervised_behavior_discovery,
        test_visual_ssl_representation,
        test_bioacoustic_ssl,
        test_static_dashboard,
        test_complete_local_workflow,
        test_webapp_surface_and_artifact_guard,
        test_sparse_validation_review_guard,
        test_contact_event_layer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
