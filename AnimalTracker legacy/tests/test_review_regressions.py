"""
回归测试：覆盖本轮 review 修复的关键路径。
"""

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from tracking.hungarian_matcher import MultiStageMatcher, Track
from tracking.enhanced_tracker import create_enhanced_tracker
from tracking.virtual_marker import IdentityAnchorManager, VirtualMarker, VirtualMarkerGenerator
from tracking.vmtracker import create_vm_tracker
from tracking.buffer_recover_tracker import KalmanBoxTracker, create_buffer_recover_tracker
from tracking.post_processor import fill_identity_gaps, split_impossible_jumps, stitch_frame_tracks
from tracking.animalmot import AnimalTrack, create_animalmot
from tracking.animal_bytetracker import AnimalTrack as ByteAnimalTrack, _extract_feat
from tracking.track_filter import create_track_filter
from tracking.tracker_factory import create_tracker, get_tracker_name
from tracking.automated_mot import build_auto_mot_plan
from detection.track_guided_detector import TrackGuidedThresholdDetector
from detection.detector import (
    YOLODetector,
    resolve_inference_device,
    supports_half_precision,
)
from evaluation.annotations import (
    AnimalAnnotation,
    annotations_to_frame_dict,
    load_annotations,
    save_annotations,
)
from evaluation.mot_metrics import evaluate_mot
from evaluation.hard_frame_mining import evaluate_frame_risk, select_hard_frames
from evaluation.auto_labeling import build_auto_annotations
from reporting.estimate_dashboard import create_scientific_estimate_dashboard
from tracking.offline_global_optimizer import GlobalTrackletOptimizer
from tracking.bidirectional_consistency import refine_bidirectional_centers
from pose.estimator import AnimalPose, Keypoint, PoseEstimator
from utils.analytics import (
    build_results_analytics,
    build_results_analytics_from_counts,
    filter_frame_tracks,
    filter_track_history,
    select_stable_track_ids,
)
from utils.behavior_analytics import build_behavior_analytics, normalize_pose_frames
from utils.group_comparison import build_group_comparison
from utils.interpretation import build_behavior_insights, build_parameter_interpretation
from utils.export import DataExporter
from visualization.visualizer import TrajectoryVisualizer


class MockTrack:
    def __init__(self, center):
        self.center = center
        self.bbox = (center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5)


class MockDetection:
    def __init__(self, center, bbox=None):
        self.center = center
        self.bbox = bbox or (center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5)
        self.confidence = 0.9
        self.class_id = 0


class MockInteraction:
    def __init__(self, id1=1, id2=2, start_frame=0, end_frame=5, duration=5, interaction_type="approach"):
        self.id1 = id1
        self.id2 = id2
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.duration = duration
        self.interaction_type = interaction_type


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _):
        return self.value


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBox:
    def __init__(self, confidence):
        self.conf = FakeScalar(confidence)
        self.cls = FakeScalar(0)
        self.xyxy = [FakeTensor([0, 0, 10, 10])]


class FakeResult:
    def __init__(self, confidences):
        self.boxes = [FakeBox(conf) for conf in confidences]


class FakeModel:
    def __init__(self, confidences):
        self.confidences = confidences
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.confidences)]


def test_distance_gating():
    matcher = MultiStageMatcher(max_distance=100)
    tracks = [MockTrack((0, 0)), MockTrack((1000, 1000))]
    detections = [MockDetection((400, 400)), MockDetection((1400, 1400))]
    matches = matcher.match(tracks, detections)
    assert matches == [], f"expected no matches, got {matches}"


def test_anchor_uniqueness():
    manager = IdentityAnchorManager(max_match_distance=80)
    markers = [
        VirtualMarker(1, (0, 0), 0.99, 0),
        VirtualMarker(2, (100, 0), 0.50, 0),
    ]
    used_marker_ids = set()
    anchor_1 = manager.update(10, markers, track_position=(95, 0), used_marker_ids=used_marker_ids)
    anchor_2 = manager.update(11, markers, track_position=(5, 0), used_marker_ids=used_marker_ids)

    assert anchor_1 != anchor_2
    assert manager.anchors[anchor_1].marker.marker_id == 2
    assert manager.anchors[anchor_2].marker.marker_id == 1


def test_virtual_marker_promotion():
    generator = VirtualMarkerGenerator(
        confidence_threshold=0.8,
        stability_frames=3,
        max_marker_distance=20.0
    )

    markers = []
    for frame_id in range(3):
        markers = generator.generate([((100 + frame_id, 100), 0.95)], frame_id)

    assert len(markers) == 1, f"expected one promoted marker, got {len(markers)}"


def test_export_and_animation():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        exporter = DataExporter(output_dir=str(temp_path))

        detection = MockDetection((5, 5))
        tracks = {0: {1: Track(1, detection)}}

        csv_path = Path(exporter.export_tracks_to_csv(tracks, filename='tracks.csv'))
        json_path = Path(exporter.export_to_json({'tracks': tracks}, filename='tracks.json'))
        records_csv_path = Path(exporter.export_records_to_csv(
            [{'frame': 0, 'detections': 1}],
            filename='frame_stats.csv'
        ))
        report_path = Path(exporter.export_summary_report(
            tracks,
            interactions=[],
            video_info={'fps': 5.0, 'frame_count': 3, 'width': 64, 'height': 64, 'duration': 0.6},
            filename='report.txt',
            tracker_stats={'match_rate': 0.5},
            frame_stats=[{'frame': 0, 'detections': 1, 'gap_frame': 0}],
            track_stats=[{'track_id': 1, 'duration_frames': 3, 'path_length_px': 10.0}]
        ))

        assert csv_path.exists() and csv_path.stat().st_size > 0
        assert json_path.exists() and json_path.stat().st_size > 0
        assert records_csv_path.exists() and records_csv_path.stat().st_size > 0
        assert report_path.exists() and report_path.stat().st_size > 0
        json_data = json.loads(json_path.read_text(encoding="utf-8"))
        assert json_data["tracks"]["0"]["1"]["center"] == [5, 5]

        visualizer = TrajectoryVisualizer(64, 64)
        animation_path = Path(visualizer.create_trajectory_animation(
            {
                0: {1: (10, 10)},
                1: {1: (20, 20)},
                2: {1: (30, 30)},
            },
            str(temp_path / 'trajectory.mp4'),
            fps=5
        ))
        assert animation_path.exists() and animation_path.stat().st_size > 0
        animation_cap = cv2.VideoCapture(str(animation_path))
        assert animation_cap.isOpened()
        animation_ok, _ = animation_cap.read()
        animation_cap.release()
        assert animation_ok

        input_video_path = temp_path / 'input.mp4'
        writer = cv2.VideoWriter(
            str(input_video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            5.0,
            (64, 64)
        )
        for idx in range(3):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.circle(frame, (10 + idx * 10, 10 + idx * 10), 5, (0, 255, 0), -1)
            writer.write(frame)
        writer.release()

        annotated_path = Path(visualizer.render_annotated_video(
            str(input_video_path),
            {
                0: {1: (10, 10)},
                1: {1: (20, 20)},
                2: {1: (30, 30)},
            },
            {
                0: [],
                1: [],
                2: [],
            },
            str(temp_path / 'annotated.mp4'),
            fps=5
        ))
        dashboard_path = Path(visualizer.render_dashboard_video(
            str(input_video_path),
            {
                0: {1: (10, 10)},
                1: {1: (20, 20)},
                2: {1: (30, 30)},
            },
            {
                0: [],
                1: [],
                2: [],
            },
            str(temp_path / 'dashboard.mp4'),
            fps=5
        ))
        assert annotated_path.exists() and annotated_path.stat().st_size > 0
        assert dashboard_path.exists() and dashboard_path.stat().st_size > 0
        annotated_cap = cv2.VideoCapture(str(annotated_path))
        assert annotated_cap.isOpened()
        annotated_ok, _ = annotated_cap.read()
        annotated_cap.release()
        assert annotated_ok
        dashboard_cap = cv2.VideoCapture(str(dashboard_path))
        assert dashboard_cap.isOpened()
        dashboard_ok, _ = dashboard_cap.read()
        dashboard_cap.release()
        assert dashboard_ok


def test_tracker_factory_overrides():
    enhanced = create_enhanced_tracker(
        preset='balanced',
        use_optical_flow=False,
        use_hungarian_matching=False,
        max_distance=55,
        max_disappeared=12,
        min_confirmed_hits=4,
        max_track_slots=5,
    )
    vm_tracker = create_vm_tracker(
        preset='balanced',
        use_optical_flow=False,
        use_hungarian_matching=False,
        max_distance=55,
        max_disappeared=12,
        min_confirmed_hits=4,
        max_track_slots=5,
    )

    assert enhanced.use_optical_flow is False
    assert enhanced.use_hungarian_matching is False
    assert enhanced.max_distance == 55
    assert enhanced.max_disappeared == 12
    assert enhanced.min_confirmed_hits == 4
    assert enhanced.max_track_slots == 5

    assert vm_tracker.use_optical_flow is False
    assert vm_tracker.use_hungarian_matching is False
    assert vm_tracker.max_distance == 55
    assert vm_tracker.max_disappeared == 12
    assert vm_tracker.min_confirmed_hits == 4
    assert vm_tracker.max_track_slots == 5


def test_animalmot_reuses_identity_slot_after_track_loss():
    tracker = create_animalmot(
        n_animals=1,
        max_distance=40,
        buffer_duration=5,
        buffer_entry_delay=0,
        use_identity_dictionary=True,
        identity_match_threshold=0.70,
    )
    tracker.reset()

    first = tracker.update([MockDetection((10, 10), bbox=(0, 0, 20, 20))])
    first_id = next(iter(first.keys()))
    lost_track = tracker.active_tracks.pop(first_id)
    tracker._refresh_identity_slot(lost_track)
    tracker._mark_identity_inactive(first_id)

    recovered = tracker.update([MockDetection((12, 10), bbox=(2, 0, 22, 20))])

    assert list(recovered.keys()) == [first_id]
    stats = tracker.get_stats()
    assert stats["identity_slot_recoveries"] == 1
    assert stats["identity_slots"] == 1


def test_animalmot_local_priority_preserves_nearest_ids():
    tracker = create_animalmot(
        n_animals=2,
        max_distance=80,
        local_match_radius=30,
        local_match_margin=10,
        use_identity_dictionary=True,
    )
    tracker.reset()

    tracker.update([
        MockDetection((10, 10), bbox=(0, 0, 20, 20)),
        MockDetection((100, 10), bbox=(90, 0, 110, 20)),
    ])
    tracks = tracker.update([
        MockDetection((98, 10), bbox=(88, 0, 108, 20)),
        MockDetection((12, 10), bbox=(2, 0, 22, 20)),
    ])

    assert set(tracks.keys()) == {1, 2}
    assert abs(tracks[1].center[0] - 12) < 10
    assert abs(tracks[2].center[0] - 98) < 10
    assert tracker.get_stats()["local_priority_matches"] >= 2


def test_animalmot_virtual_updates_do_not_expire_twice_as_fast():
    tracker = create_animalmot(
        n_animals=2,
        max_distance=80,
        high_conf_thresh=0.25,
        buffer_entry_delay=30,
        virtual_max_age=20,
        virtual_max_hits=20,
        use_identity_dictionary=True,
    )
    tracker.reset()
    tracker.update([
        MockDetection((10, 10), bbox=(0, 0, 20, 20)),
        MockDetection((80, 10), bbox=(70, 0, 90, 20)),
    ])

    tracks = tracker.update([MockDetection((10, 10), bbox=(0, 0, 20, 20))])

    missing_track = tracks[2]
    assert missing_track.virtual_hits == 1
    assert missing_track.time_since_update == 1
    assert 2 in tracks
    assert tracker.get_stats()["direct_virtual_matches"] == 1


def test_animalmot_accepts_frame_keyword_and_splits_merged_cluster_box():
    tracker = create_animalmot(
        n_animals=2,
        max_distance=80,
        high_conf_thresh=0.25,
        buffer_entry_delay=30,
        use_identity_dictionary=True,
    )
    tracker.reset()
    tracker.update([
        MockDetection((10, 10), bbox=(0, 0, 20, 20)),
        MockDetection((55, 10), bbox=(45, 0, 65, 20)),
    ])

    frame = np.zeros((80, 90, 3), dtype=np.uint8)
    tracks = tracker.update(
        [MockDetection((32, 10), bbox=(0, 0, 65, 20))],
        frame=frame,
    )

    assert set(tracks.keys()) == {1, 2}
    assert tracks[1].bbox[2] - tracks[1].bbox[0] < 35
    assert tracks[2].bbox[2] - tracks[2].bbox[0] < 35
    stats = tracker.get_stats()
    assert stats["cluster_detections_suppressed"] == 1
    assert stats["cluster_virtual_splits"] == 2
    assert stats["direct_virtual_matches"] == 2


def test_animalmot_enforces_expected_active_track_limit():
    tracker = create_animalmot(
        n_animals=2,
        max_distance=80,
        use_identity_dictionary=True,
    )
    tracker.reset()
    strong = AnimalTrack(np.array([0, 0, 20, 20]), 0.9, track_id=1)
    strong.hits = 10
    stable = AnimalTrack(np.array([50, 0, 70, 20]), 0.8, track_id=2)
    stable.hits = 8
    duplicate = AnimalTrack(np.array([52, 0, 72, 20]), 0.4, track_id=3)
    duplicate.hits = 1
    tracker.active_tracks = {
        1: strong,
        2: stable,
        3: duplicate,
    }

    tracker._enforce_active_limit()

    assert set(tracker.active_tracks.keys()) == {1, 2}
    assert 3 in tracker.buffered_tracks
    assert tracker.get_stats()["active_limit_suppressed"] == 1


def test_track_filter_suppresses_duplicate_detection_boxes():
    track_filter = create_track_filter(min_confidence=0.1)
    detections = [
        MockDetection((10, 10), bbox=(0, 0, 20, 20)),
        MockDetection((11, 10), bbox=(1, 0, 21, 20)),
        MockDetection((70, 10), bbox=(60, 0, 80, 20)),
    ]

    filtered = track_filter.filter_detections(detections, frame_size=(100, 80))

    assert len(filtered) == 2
    assert track_filter.get_stats()["duplicate_filtered"] == 1


def test_enhanced_tracker_caps_concurrent_track_slots():
    tracker = create_enhanced_tracker(
        preset='fast',
        use_optical_flow=False,
        use_hungarian_matching=False,
        max_distance=40,
        max_disappeared=3,
        min_confirmed_hits=1,
        max_track_slots=4,
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detections = [
        MockDetection((10, 10)),
        MockDetection((20, 20)),
        MockDetection((30, 30)),
        MockDetection((40, 40)),
        MockDetection((50, 50)),
    ]
    tracks = tracker.update(frame, detections)
    assert len(tracks) == 4
    stats = tracker.get_stats()
    assert stats['max_track_slots'] == 4
    assert stats['suppressed_new_tracks'] == 1


def test_enhanced_tracker_hides_unconfirmed_ids():
    tracker = create_enhanced_tracker(
        preset='balanced',
        use_optical_flow=False,
        use_hungarian_matching=False,
        max_distance=30,
        max_disappeared=5,
        min_confirmed_hits=3,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    visible_1 = tracker.update(frame, [MockDetection((10, 10))])
    visible_2 = tracker.update(frame, [MockDetection((11, 10))])
    visible_3 = tracker.update(frame, [MockDetection((12, 10))])

    assert visible_1 == {}
    assert visible_2 == {}
    assert list(visible_3.keys()) == [1]


def test_detection_threshold_counting():
    detector = YOLODetector("fake.pt", 0.1)
    detector.model = FakeModel([0.12, 0.18, 0.27])
    counts = detector.count_detections_by_threshold(frame=None, thresholds=[0.10, 0.15, 0.20, 0.25])
    assert counts == {
        0.10: 3,
        0.15: 2,
        0.20: 1,
        0.25: 1,
    }


def test_detection_prediction_summary():
    detector = YOLODetector("fake.pt", 0.1)
    detector.model = FakeModel([0.12, 0.18, 0.27])
    counts, confidences = detector.summarize_predictions(
        frame=None,
        thresholds=[0.10, 0.15, 0.20, 0.25]
    )
    assert counts == {
        0.10: 3,
        0.15: 2,
        0.20: 1,
        0.25: 1,
    }
    assert confidences == [0.12, 0.18, 0.27]


def test_track_guided_detector_splits_merged_blob_with_track_priors():
    frame = np.full((120, 160, 3), 210, dtype=np.uint8)
    cv2.ellipse(frame, (70, 60), (45, 18), 0, 0, 360, (15, 15, 15), -1)
    detector = TrackGuidedThresholdDetector(
        n_animals=2,
        threshold=80,
        min_area=80,
        border_margin=0,
    )
    priors = {
        1: MockTrack((45, 60)),
        2: MockTrack((95, 60)),
    }

    detections = detector.detect(frame, track_priors=priors)

    assert len(detections) == 2
    centers_x = sorted(det.center[0] for det in detections)
    assert centers_x[0] < 70 < centers_x[1]
    stats = detector.get_stats()
    assert stats["track_guided_splits"] >= 1


def test_track_guided_detector_adapts_initial_area_without_false_splits():
    frame = np.full((240, 260, 3), 190, dtype=np.uint8)
    expected_centers = [(55, 55), (190, 55), (55, 180), (190, 180)]
    for center in expected_centers:
        cv2.ellipse(frame, center, (32, 24), 20, 0, 360, (12, 12, 12), -1)

    detector = TrackGuidedThresholdDetector(
        n_animals=4,
        threshold=80,
        min_area=80,
        border_margin=0,
    )

    detections = detector.detect(frame)

    centers = [det.center for det in detections]
    assert len(detections) == 4
    for expected in expected_centers:
        assert min(np.hypot(center[0] - expected[0], center[1] - expected[1]) for center in centers) < 18
    assert detector.get_stats()["kmeans_splits"] == 0


def test_detector_runtime_config():
    detector = YOLODetector(
        "fake.pt",
        0.1,
        device="cpu",
        imgsz=512,
        vid_stride=3,
        half=True,
    )
    model = FakeModel([0.12, 0.18, 0.27])
    detector.model = model

    detector.predict_raw(frame=None)

    assert detector.get_runtime_config()["resolved_device"] == "cpu"
    assert detector.get_runtime_config()["imgsz"] == 512
    assert detector.get_runtime_config()["vid_stride"] == 3
    assert detector.get_runtime_config()["half"] is False
    assert detector.should_process_frame(0) is True
    assert detector.should_process_frame(1) is False
    assert model.calls[0]["device"] == "cpu"
    assert model.calls[0]["imgsz"] == 512
    assert model.calls[0]["half"] is False
    assert resolve_inference_device("cpu") == "cpu"
    assert supports_half_precision("cpu") is False
    assert supports_half_precision("cuda:0") is True


def test_results_analytics():
    frame_df, track_df = build_results_analytics(
        all_tracks={
            0: {1: (10, 10)},
            1: {1: (20, 20), 2: (5, 5)},
            2: {},
        },
        all_detections={
            0: [MockDetection((10, 10))],
            1: [MockDetection((20, 20)), MockDetection((5, 5))],
            2: [],
        },
        track_history={
            1: [(10, 10), (20, 20)],
            2: [(5, 5)],
        }
    )
    assert list(frame_df.columns) == ['frame', 'detections', 'tracks', 'cumulative_ids', 'gap_frame']
    assert int(frame_df['cumulative_ids'].iloc[-1]) == 2
    assert not track_df.empty
    assert int(track_df.iloc[0]['track_id']) == 1


def test_results_analytics_from_counts():
    frame_df, track_df = build_results_analytics_from_counts(
        all_tracks={
            0: {1: (10, 10)},
            1: {1: (20, 20), 2: (5, 5)},
            2: {},
        },
        detection_counts={
            0: 1,
            1: 2,
            2: 0,
        },
        track_history={
            1: [(10, 10), (20, 20)],
            2: [(5, 5)],
        }
    )
    assert list(frame_df['detections']) == [1, 2, 0]
    assert int(frame_df['cumulative_ids'].iloc[-1]) == 2
    assert int(track_df.iloc[0]['track_id']) == 1


def test_stable_track_filtering_for_visualization():
    track_history = {
        1: [(0, 0)] * 20,
        2: [(1, 1)] * 12,
        3: [(2, 2)] * 3,
    }
    all_tracks = {
        0: {1: (0, 0), 2: (1, 1), 3: (2, 2)},
        1: {1: (0, 0), 2: (1, 1)},
    }

    selected = select_stable_track_ids(track_history, min_duration_frames=10, max_tracks=2)
    filtered_history = filter_track_history(track_history, selected)
    filtered_frames = filter_frame_tracks(all_tracks, selected)

    assert selected == [1, 2]
    assert set(filtered_history.keys()) == {1, 2}
    assert set(filtered_frames[0].keys()) == {1, 2}


def test_behavior_analytics_following_and_standing():
    all_tracks = {
        0: {1: (10, 10), 2: (28, 10)},
        1: {1: (15, 10), 2: (33, 10)},
        2: {1: (20, 10), 2: (38, 10)},
        3: {1: (25, 10), 2: (43, 10)},
        4: {1: (30, 10), 2: (48, 10)},
    }
    all_detections = {
        0: [
            {"bbox": (0, 0, 20, 20), "center": (10, 10), "confidence": 0.9},
            {"bbox": (18, 0, 38, 20), "center": (28, 10), "confidence": 0.9},
        ],
        1: [
            {"bbox": (5, 0, 25, 20), "center": (15, 10), "confidence": 0.9},
            {"bbox": (23, 0, 43, 20), "center": (33, 10), "confidence": 0.9},
        ],
        2: [
            {"bbox": (15, 0, 25, 20), "center": (20, 10), "confidence": 0.9},
            {"bbox": (28, 0, 48, 20), "center": (38, 10), "confidence": 0.9},
        ],
        3: [
            {"bbox": (20, 0, 30, 20), "center": (25, 10), "confidence": 0.9},
            {"bbox": (33, 0, 53, 20), "center": (43, 10), "confidence": 0.9},
        ],
        4: [
            {"bbox": (25, 0, 35, 20), "center": (30, 10), "confidence": 0.9},
            {"bbox": (38, 0, 58, 20), "center": (48, 10), "confidence": 0.9},
        ],
    }
    track_history = {
        1: [(10, 10), (15, 10), (20, 10), (25, 10), (30, 10)],
        2: [(28, 10), (33, 10), (38, 10), (43, 10), (48, 10)],
    }

    analytics = build_behavior_analytics(
        all_tracks,
        all_detections,
        track_history,
        fps=10.0,
        interaction_distance=25.0,
        contact_distance=8.0,
        min_event_frames=2,
    )

    assert analytics["summary"]["interaction_events"] >= 1
    assert analytics["summary"]["following_events"] >= 1
    assert analytics["summary"]["standing_candidate_bouts"] >= 1
    assert "group_stats" in analytics
    assert analytics["summary"]["mean_nearest_neighbor_distance_px"] > 0
    assert analytics["summary"]["mean_arena_coverage_ratio"] > 0
    assert analytics["group_stats"]["mean_pair_distance_px"] > 0
    assert analytics["frame_stats"]
    assert any(row["event_type"] == "following" for row in analytics["event_rows"])
    assert any(row["event_type"] == "standing_candidate" for row in analytics["event_rows"])
    track1 = next(row for row in analytics["individual_stats"] if row["track_id"] == 1)
    assert track1["following_out_events"] >= 1
    assert "grooming_candidate_bouts" in track1


def test_pose_normalization_and_pose_assisted_behavior():
    pose_frames = {
        0: {
            1: {
                "track_id": 1,
                "bbox": (0, 0, 20, 20),
                "center": (10.0, 10.0),
                "orientation": 0.0,
                "head_position": (10.0, 8.0),
                "valid_keypoints": 7,
                "keypoints": {
                    "nose": {"x": 10.0, "y": 8.0, "confidence": 0.9},
                    "neck": {"x": 10.0, "y": 10.0, "confidence": 0.9},
                    "tail_base": {"x": 10.0, "y": 18.0, "confidence": 0.9},
                    "spine_base": {"x": 10.0, "y": 14.0, "confidence": 0.9},
                    "left_forepaw": {"x": 11.0, "y": 8.5, "confidence": 0.9},
                    "right_forepaw": {"x": 9.0, "y": 8.5, "confidence": 0.9},
                    "left_hindpaw": {"x": 8.0, "y": 16.0, "confidence": 0.9},
                    "right_hindpaw": {"x": 12.0, "y": 16.0, "confidence": 0.9},
                },
            }
        },
        1: {
            1: {
                "track_id": 1,
                "bbox": (0, 0, 20, 20),
                "center": (10.0, 10.0),
                "orientation": 0.0,
                "head_position": (10.0, 8.0),
                "valid_keypoints": 7,
                "keypoints": {
                    "nose": {"x": 10.0, "y": 8.0, "confidence": 0.9},
                    "neck": {"x": 10.0, "y": 10.0, "confidence": 0.9},
                    "tail_base": {"x": 10.0, "y": 18.0, "confidence": 0.9},
                    "spine_base": {"x": 10.0, "y": 14.0, "confidence": 0.9},
                    "left_forepaw": {"x": 10.8, "y": 8.2, "confidence": 0.9},
                    "right_forepaw": {"x": 9.2, "y": 8.3, "confidence": 0.9},
                    "left_hindpaw": {"x": 8.0, "y": 16.0, "confidence": 0.9},
                    "right_hindpaw": {"x": 12.0, "y": 16.0, "confidence": 0.9},
                },
            }
        },
    }
    normalized_pose_frames = normalize_pose_frames(pose_frames)
    analytics = build_behavior_analytics(
        {0: {1: (10, 10)}, 1: {1: (10, 10)}},
        {
            0: [{"bbox": (0, 0, 20, 20), "center": (10, 10), "confidence": 0.9}],
            1: [{"bbox": (0, 0, 20, 20), "center": (10, 10), "confidence": 0.9}],
        },
        {1: [(10, 10), (10, 10)]},
        pose_frames=normalized_pose_frames,
        fps=10.0,
        min_event_frames=1,
    )
    assert analytics["summary"]["pose_coverage_ratio"] > 0
    assert analytics["summary"]["standing_mode"] == "pose_assisted_candidate"
    assert analytics["summary"]["grooming_candidate_bouts"] >= 1
    assert analytics["summary"]["standing_candidate_bouts"] >= 1


def test_pose_estimator_detect_with_tracking_uses_unique_pose_assignment():
    estimator = PoseEstimator()
    pose_a = AnimalPose(
        [Keypoint(10, 10, 0.9, "nose") for _ in range(11)],
        (0, 0, 20, 20),
    )
    pose_b = AnimalPose(
        [Keypoint(50, 50, 0.9, "nose") for _ in range(11)],
        (40, 40, 60, 60),
    )
    estimator.detect = lambda frame: [pose_a, pose_b]
    track_map = estimator.detect_with_tracking(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        tracks={1: (10, 10), 2: (50, 50)},
    )
    assert set(track_map.keys()) == {1, 2}
    assert track_map[1] is not track_map[2]
    assert track_map[1].track_id == 1
    assert track_map[2].track_id == 2


def test_behavior_segments_and_pursuit_candidates():
    all_tracks = {
        0: {1: (10, 10), 2: (28, 10)},
        1: {1: (16, 10), 2: (30, 10)},
        2: {1: (23, 10), 2: (32, 10)},
        3: {1: (31, 10), 2: (34, 10)},
        4: {1: (39, 10), 2: (40, 10)},
        5: {1: (48, 10), 2: (52, 10)},
    }
    all_detections = {
        frame: [
            {"bbox": (pos1[0] - 5, 0, pos1[0] + 5, 20), "center": pos1, "confidence": 0.9},
            {"bbox": (pos2[0] - 5, 0, pos2[0] + 5, 20), "center": pos2, "confidence": 0.9},
        ]
        for frame, (pos1, pos2) in {
            0: ((10, 10), (28, 10)),
            1: ((16, 10), (30, 10)),
            2: ((23, 10), (32, 10)),
            3: ((31, 10), (34, 10)),
            4: ((39, 10), (40, 10)),
            5: ((48, 10), (52, 10)),
        }.items()
    }
    pose_frames = {
        frame: {
            1: {
                "track_id": 1,
                "bbox": (pos1[0] - 5, 0, pos1[0] + 5, 20),
                "center": pos1,
                "head_position": (pos1[0] + 3, pos1[1]),
                "valid_keypoints": 6,
                "keypoints": {},
            },
            2: {
                "track_id": 2,
                "bbox": (pos2[0] - 5, 0, pos2[0] + 5, 20),
                "center": pos2,
                "head_position": (pos2[0] + 2, pos2[1]),
                "valid_keypoints": 6,
                "keypoints": {},
            },
        }
        for frame, (pos1, pos2) in {
            0: ((10, 10), (28, 10)),
            1: ((16, 10), (30, 10)),
            2: ((23, 10), (32, 10)),
            3: ((31, 10), (34, 10)),
            4: ((39, 10), (40, 10)),
            5: ((48, 10), (52, 10)),
        }.items()
    }
    analytics = build_behavior_analytics(
        all_tracks,
        all_detections,
        {1: [row[1] for row in all_tracks.values()], 2: [row[2] for row in all_tracks.values()]},
        pose_frames=pose_frames,
        fps=5.0,
        interaction_distance=25.0,
        contact_distance=8.0,
        min_event_frames=1,
        segment_seconds=0.6,
    )

    assert analytics["summary"]["pursuit_events"] >= 1
    assert analytics["summary"]["aggressive_approach_events"] >= 1
    assert analytics["summary"]["escape_events"] >= 1
    assert any(row["event_type"] == "pursuit" for row in analytics["event_rows"])
    assert any(row["event_type"] == "aggressive_approach_candidate" for row in analytics["event_rows"])
    assert any(row["event_type"] == "escape_candidate" for row in analytics["event_rows"])
    assert analytics["segment_stats"]
    assert analytics["phase_comparison"]
    assert "dominant_behavior" in analytics["segment_stats"][0]


def test_manual_phase_configuration():
    all_tracks = {
        0: {1: (10, 10), 2: (20, 10)},
        1: {1: (12, 10), 2: (22, 10)},
        2: {1: (14, 10), 2: (24, 10)},
        3: {1: (16, 10), 2: (26, 10)},
        4: {1: (18, 10), 2: (28, 10)},
        5: {1: (20, 10), 2: (30, 10)},
    }
    all_detections = {
        frame: [
            {"bbox": (left[0] - 4, 0, left[0] + 4, 18), "center": left, "confidence": 0.9},
            {"bbox": (right[0] - 4, 0, right[0] + 4, 18), "center": right, "confidence": 0.9},
        ]
        for frame, (left, right) in {
            0: ((10, 10), (20, 10)),
            1: ((12, 10), (22, 10)),
            2: ((14, 10), (24, 10)),
            3: ((16, 10), (26, 10)),
            4: ((18, 10), (28, 10)),
            5: ((20, 10), (30, 10)),
        }.items()
    }
    analytics = build_behavior_analytics(
        all_tracks,
        all_detections,
        {1: [row[1] for row in all_tracks.values()], 2: [row[2] for row in all_tracks.values()]},
        fps=2.0,
        interaction_distance=15.0,
        contact_distance=6.0,
        min_event_frames=1,
        segment_seconds=1.0,
        phase_config={
            "mode": "manual",
            "labels": ["baseline", "stimulus", "recovery"],
            "boundaries_s": [1.5, 2.5],
        },
    )

    assert analytics["summary"]["phase_mode"] == "manual"
    assert analytics["summary"]["phase_labels"] == ["baseline", "stimulus", "recovery"]
    assert analytics["summary"]["phase_boundaries_s"] == [1.5, 2.5]
    assert [row["phase"] for row in analytics["phase_comparison"]] == ["baseline", "stimulus", "recovery"]
    assert all(row["phase_source"] == "manual" for row in analytics["phase_comparison"])


def test_interpretation_helpers():
    behavior_analysis = {
        "summary": {
            "avg_moving_ratio": 0.72,
            "avg_speed_px_per_s": 18.4,
            "mean_edge_ratio": 0.61,
            "mean_center_ratio": 0.12,
            "mean_arena_coverage_ratio": 0.18,
            "following_events": 2,
            "huddling_events": 1,
            "pursuit_events": 2,
            "aggressive_approach_events": 1,
            "contact_events": 3,
            "avoidance_events": 0,
            "grooming_candidate_bouts": 1,
            "standing_candidate_bouts": 2,
        },
        "pair_stats": [{"pair": "1-2", "following_events": 2, "contact_events": 3}],
        "group_stats": {"social_density_ratio": 0.5, "mean_nearest_neighbor_distance_px": 14.2},
        "segment_stats": [{"window_label": "0.0s-30.0s", "interaction_events": 4, "following_events": 2, "pursuit_events": 2, "aggressive_approach_events": 1, "dominant_behavior": "pursuit"}],
        "phase_comparison": [
            {"phase": "baseline", "interaction_events": 1, "escape_events": 0, "dominant_behavior": "contact"},
            {"phase": "recovery", "interaction_events": 4, "escape_events": 2, "dominant_behavior": "pursuit"},
        ],
    }
    tracker_stats = {"id_switches": 2, "match_rate": 0.55, "active_tracks": 6}
    parameter_rows = build_parameter_interpretation(
        "增强追踪器",
        tracking_config={
            "max_distance": 90,
            "max_disappeared": 40,
            "min_confirmed_hits": 3,
            "expected_animals": 2,
            "detection_buffer": 1,
            "interaction_distance": 50,
            "min_duration": 10,
            "behavior_window_seconds": 30,
            "phase_mode": "manual",
            "phase_labels": ["baseline", "stimulus", "recovery"],
            "phase_boundaries_s": [30, 60],
            "trail_length": 60,
        },
        inference_config={
            "resolved_device": "cuda:0",
            "imgsz": 416,
            "vid_stride": 1,
            "half": True,
            "confidence_threshold": 0.1,
        },
        threshold_analysis={"recommended_threshold": 0.15},
    )
    insights = build_behavior_insights(behavior_analysis, tracker_stats=tracker_stats, expected_animals=2)

    assert any(row["parameter"] == "device" for row in parameter_rows)
    assert any(row["parameter"] == "confidence_threshold" for row in parameter_rows)
    assert any(row["parameter"] == "phase_mode" for row in parameter_rows)
    assert any("贴边" in insight["headline"] or "贴边" in insight["detail"] for insight in insights)
    assert any("身份切换" in insight["headline"] for insight in insights)
    assert any("追逐" in insight["headline"] or "追逐" in insight["detail"] for insight in insights)
    assert any("后期" in insight["headline"] or "后期" in insight["detail"] for insight in insights)


def test_exporter_handles_empty_and_dict_inputs():
    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = DataExporter(output_dir=temp_dir)

        empty_tracks_path = Path(exporter.export_tracks_to_csv({}, filename="empty_tracks.csv"))
        empty_interactions_path = Path(
            exporter.export_interactions_to_csv([], filename="empty_interactions.csv")
        )
        dict_interactions_path = Path(
            exporter.export_interactions_to_csv(
                [
                    {
                        "id1": 1,
                        "id2": 2,
                        "start_frame": 0,
                        "end_frame": 3,
                        "duration": 3,
                        "type": "contact",
                    }
                ],
                filename="dict_interactions.csv",
            )
        )
        report_path = Path(
            exporter.export_summary_report(
                tracks_dict={},
                interactions=[MockInteraction(), {"id1": 3, "id2": 4, "duration": 2, "type": "follow"}],
                video_info={},
                filename="empty_report.txt",
                tracker_stats={"match_rate": np.float32(0.5)},
                frame_stats=[],
                track_stats=[],
            )
        )

        assert empty_tracks_path.exists()
        assert "frame,track_id,x,y,source_type" in empty_tracks_path.read_text(encoding="utf-8")
        assert empty_interactions_path.exists()
        assert "id1,id2,start_frame,end_frame,duration,type,source_type" in empty_interactions_path.read_text(
            encoding="utf-8"
        )
        assert dict_interactions_path.exists() and dict_interactions_path.stat().st_size > 0
        assert report_path.exists()
        report_text = report_path.read_text(encoding="utf-8")
        assert "Tracked Frames: 0" in report_text
        assert "Total Interactions: 2" in report_text


def test_export_report_includes_behavior_summary():
    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = DataExporter(output_dir=temp_dir)
        report_path = Path(
            exporter.export_summary_report(
                tracks_dict={0: {1: (10, 10)}},
                interactions=[],
                video_info={"fps": 10.0},
                filename="behavior_report.txt",
                behavior_analysis={
                    "summary": {
                        "interaction_events": 3,
                        "following_events": 1,
                        "standing_candidate_bouts": 2,
                    },
                    "behavior_counts": [{"behavior": "following", "count": 1}],
                    "individual_stats": [{"track_id": 1, "total_distance_px": 12.0, "following_out_events": 1, "standing_candidate_bouts": 2, "grooming_candidate_bouts": 1}],
                    "pair_stats": [{"pair": "1-2", "contact_events": 0, "close_proximity_events": 1, "following_events": 1}],
                    "group_stats": {"mean_pair_distance_px": 14.5},
                    "segment_stats": [{"window_label": "0.0s-30.0s", "interaction_events": 2, "following_events": 1, "pursuit_events": 1, "aggressive_approach_events": 0, "dominant_behavior": "following"}],
                    "phase_comparison": [{"phase": "early", "interaction_events": 1, "escape_events": 0, "dominant_behavior": "following"}],
                    "insights": [{"headline": "检测到跟随", "detail": "存在稳定跟随", "action": "继续复核"}],
                },
                parameter_interpretation=[{"group": "推理", "parameter": "imgsz", "current_value": "416", "interpretation": "较快", "suggestion": "需要更准时提高"}],
            )
        )
        report_text = report_path.read_text(encoding="utf-8")
        assert "Behavior Summary" in report_text
        assert "following_events: 1" in report_text
        assert "Most Standing Candidates" in report_text
        assert "Group Behavior Stats" in report_text
        assert "Segment Behavior Stats" in report_text
        assert "Phase Behavior Comparison" in report_text
        assert "Most Grooming Candidates" in report_text
        assert "Parameter Interpretation" in report_text
        assert "Behavior Insights" in report_text


def test_group_comparison_statistics():
    comparison = build_group_comparison([
        {
            "sample": "control_1",
            "group": "control",
            "interaction_events": 1.0,
            "following_events": 1.0,
            "pursuit_events": 0.0,
            "escape_events": 0.0,
            "avg_speed_px_per_s": 5.0,
            "mean_edge_ratio": 0.20,
            "tracked_animals": 2.0,
            "contact_events": 1.0,
            "aggressive_approach_events": 0.0,
            "standing_candidate_bouts": 1.0,
            "grooming_candidate_bouts": 1.0,
            "avg_moving_ratio": 0.25,
            "mean_arena_coverage_ratio": 0.20,
            "mean_center_ratio": 0.30,
            "mean_pair_distance_px": 22.0,
            "mean_nearest_neighbor_distance_px": 10.0,
            "id_switches": 0.0,
            "match_rate": 0.95,
        },
        {
            "sample": "control_2",
            "group": "control",
            "interaction_events": 2.0,
            "following_events": 1.0,
            "pursuit_events": 0.0,
            "escape_events": 0.0,
            "avg_speed_px_per_s": 6.0,
            "mean_edge_ratio": 0.25,
            "tracked_animals": 2.0,
            "contact_events": 1.0,
            "aggressive_approach_events": 0.0,
            "standing_candidate_bouts": 1.0,
            "grooming_candidate_bouts": 1.0,
            "avg_moving_ratio": 0.30,
            "mean_arena_coverage_ratio": 0.22,
            "mean_center_ratio": 0.28,
            "mean_pair_distance_px": 20.0,
            "mean_nearest_neighbor_distance_px": 9.0,
            "id_switches": 0.0,
            "match_rate": 0.96,
        },
        {
            "sample": "treated_1",
            "group": "treated",
            "interaction_events": 8.0,
            "following_events": 5.0,
            "pursuit_events": 3.0,
            "escape_events": 2.0,
            "avg_speed_px_per_s": 11.0,
            "mean_edge_ratio": 0.55,
            "tracked_animals": 2.0,
            "contact_events": 4.0,
            "aggressive_approach_events": 2.0,
            "standing_candidate_bouts": 2.0,
            "grooming_candidate_bouts": 1.0,
            "avg_moving_ratio": 0.60,
            "mean_arena_coverage_ratio": 0.42,
            "mean_center_ratio": 0.14,
            "mean_pair_distance_px": 14.0,
            "mean_nearest_neighbor_distance_px": 6.0,
            "id_switches": 1.0,
            "match_rate": 0.88,
        },
        {
            "sample": "treated_2",
            "group": "treated",
            "interaction_events": 9.0,
            "following_events": 6.0,
            "pursuit_events": 4.0,
            "escape_events": 1.0,
            "avg_speed_px_per_s": 12.0,
            "mean_edge_ratio": 0.60,
            "tracked_animals": 2.0,
            "contact_events": 5.0,
            "aggressive_approach_events": 2.0,
            "standing_candidate_bouts": 3.0,
            "grooming_candidate_bouts": 1.0,
            "avg_moving_ratio": 0.65,
            "mean_arena_coverage_ratio": 0.45,
            "mean_center_ratio": 0.10,
            "mean_pair_distance_px": 13.0,
            "mean_nearest_neighbor_distance_px": 5.0,
            "id_switches": 1.0,
            "match_rate": 0.87,
        },
    ], comparison_mode="independent")

    assert len(comparison["sample_rows"]) == 4
    assert any(row["group"] == "control" and row["metric"] == "interaction_events" for row in comparison["group_summary_rows"])
    assert any(row["metric"] == "interaction_events" and row["test_name"] in {"welch_t", "mannwhitney_u"} for row in comparison["omnibus_rows"])
    pairwise_interaction = next(row for row in comparison["pairwise_rows"] if row["metric"] == "interaction_events")
    assert pairwise_interaction["group_a"] == "control"
    assert pairwise_interaction["group_b"] == "treated"
    assert pairwise_interaction["delta_mean"] > 0
    assert comparison["insights"]


def test_group_comparison_multigroup_statistics():
    comparison = build_group_comparison([
        {"sample": "control_1", "group": "control", "interaction_events": 1.0, "avg_speed_px_per_s": 5.0},
        {"sample": "control_2", "group": "control", "interaction_events": 2.0, "avg_speed_px_per_s": 6.0},
        {"sample": "stress_1", "group": "stress", "interaction_events": 4.0, "avg_speed_px_per_s": 8.0},
        {"sample": "stress_2", "group": "stress", "interaction_events": 5.0, "avg_speed_px_per_s": 8.5},
        {"sample": "treated_1", "group": "treated", "interaction_events": 8.0, "avg_speed_px_per_s": 11.0},
        {"sample": "treated_2", "group": "treated", "interaction_events": 9.0, "avg_speed_px_per_s": 12.0},
    ], comparison_mode="independent")

    assert any(row["metric"] == "interaction_events" and row["test_name"] == "anova_oneway" for row in comparison["omnibus_rows"])
    assert any(row["metric"] == "interaction_events" and row["test_name"] == "kruskal_wallis" for row in comparison["omnibus_rows"])


def test_group_comparison_paired_statistics():
    comparison = build_group_comparison([
        {
            "sample": "control_mouse_a",
            "group": "control",
            "pair_id": "mouse_a",
            "interaction_events": 2.0,
            "avg_speed_px_per_s": 5.0,
        },
        {
            "sample": "treated_mouse_a",
            "group": "treated",
            "pair_id": "mouse_a",
            "interaction_events": 6.0,
            "avg_speed_px_per_s": 9.0,
        },
        {
            "sample": "control_mouse_b",
            "group": "control",
            "pair_id": "mouse_b",
            "interaction_events": 3.0,
            "avg_speed_px_per_s": 6.0,
        },
        {
            "sample": "treated_mouse_b",
            "group": "treated",
            "pair_id": "mouse_b",
            "interaction_events": 7.0,
            "avg_speed_px_per_s": 10.0,
        },
        {
            "sample": "control_mouse_c",
            "group": "control",
            "pair_id": "mouse_c",
            "interaction_events": 4.0,
            "avg_speed_px_per_s": 7.0,
        },
        {
            "sample": "treated_mouse_c",
            "group": "treated",
            "pair_id": "mouse_c",
            "interaction_events": 8.0,
            "avg_speed_px_per_s": 11.0,
        },
    ], comparison_mode="paired")

    assert any(row["metric"] == "interaction_events" and row["test_name"] == "paired_t" for row in comparison["omnibus_rows"])
    paired_row = next(row for row in comparison["pairwise_rows"] if row["metric"] == "interaction_events")
    assert paired_row["n_pairs"] == 3
    assert paired_row["effect_name"] == "cohens_dz"
    assert comparison["comparison_mode"] == "paired"


def test_export_bundle():
    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = DataExporter(output_dir=temp_dir)
        tracks = {0: {1: (10, 10)}}
        tracks_path = exporter.export_tracks_to_csv(tracks, filename="bundle_tracks.csv")
        json_path = exporter.export_to_json({"tracks": tracks}, filename="bundle_analysis.json")
        bundle_path = Path(
            exporter.export_bundle(
                {
                    "bundle_tracks.csv": tracks_path,
                    "bundle_analysis.json": json_path,
                },
                filename="bundle.zip",
            )
        )

        assert bundle_path.exists() and bundle_path.stat().st_size > 0
        with zipfile.ZipFile(bundle_path, "r") as archive:
            names = set(archive.namelist())
        assert {"bundle_tracks.csv", "bundle_analysis.json"} <= names


def test_interpretation_helpers():
    behavior_analysis = {
        "summary": {
            "avg_moving_ratio": 0.72,
            "avg_speed_px_per_s": 18.4,
            "mean_edge_ratio": 0.61,
            "mean_center_ratio": 0.12,
            "mean_arena_coverage_ratio": 0.18,
            "following_events": 2,
            "huddling_events": 1,
            "pursuit_events": 2,
            "aggressive_approach_events": 1,
            "contact_events": 3,
            "avoidance_events": 0,
            "grooming_candidate_bouts": 1,
            "standing_candidate_bouts": 2,
        },
        "pair_stats": [{"pair": "1-2", "following_events": 2, "contact_events": 3}],
        "group_stats": {"social_density_ratio": 0.5, "mean_nearest_neighbor_distance_px": 14.2},
        "segment_stats": [{"window_label": "0.0s-30.0s", "interaction_events": 4, "following_events": 2, "pursuit_events": 2, "aggressive_approach_events": 1, "dominant_behavior": "pursuit"}],
        "phase_comparison": [
            {"phase": "baseline", "interaction_events": 1, "escape_events": 0, "dominant_behavior": "contact"},
            {"phase": "recovery", "interaction_events": 4, "escape_events": 2, "dominant_behavior": "pursuit"},
        ],
    }
    tracker_stats = {"id_switches": 2, "match_rate": 0.55, "active_tracks": 6}
    parameter_rows = build_parameter_interpretation(
        "EnhancedTracker",
        tracking_config={
            "max_distance": 90,
            "max_disappeared": 40,
            "min_confirmed_hits": 3,
            "expected_animals": 2,
            "detection_buffer": 1,
            "interaction_distance": 50,
            "min_duration": 10,
            "behavior_window_seconds": 30,
            "phase_mode": "manual",
            "phase_labels": ["baseline", "stimulus", "recovery"],
            "phase_boundaries_s": [30, 60],
            "trail_length": 60,
        },
        inference_config={
            "resolved_device": "cuda:0",
            "imgsz": 416,
            "vid_stride": 1,
            "half": True,
            "confidence_threshold": 0.1,
        },
        threshold_analysis={"recommended_threshold": 0.15},
    )
    insights = build_behavior_insights(behavior_analysis, tracker_stats=tracker_stats, expected_animals=2)

    assert any(row["parameter"] == "device" for row in parameter_rows)
    assert any(row["parameter"] == "confidence_threshold" for row in parameter_rows)
    assert any(row["parameter"] == "phase_mode" for row in parameter_rows)
    assert any("贴边" in insight["headline"] or "贴边" in insight["detail"] for insight in insights)
    assert any("身份切换" in insight["headline"] for insight in insights)
    assert any("追逐" in insight["headline"] or "追逐" in insight["detail"] for insight in insights)
    assert any("recovery" in insight["headline"] or "recovery" in insight["detail"] for insight in insights)


def test_export_report_includes_behavior_summary():
    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = DataExporter(output_dir=temp_dir)
        report_path = Path(
            exporter.export_summary_report(
                tracks_dict={0: {1: (10, 10)}},
                interactions=[],
                video_info={"fps": 10.0},
                filename="behavior_report.txt",
                behavior_analysis={
                    "summary": {
                        "interaction_events": 3,
                        "following_events": 1,
                        "standing_candidate_bouts": 2,
                    },
                    "behavior_counts": [{"behavior": "following", "count": 1}],
                    "individual_stats": [{"track_id": 1, "total_distance_px": 12.0, "following_out_events": 1, "standing_candidate_bouts": 2, "grooming_candidate_bouts": 1}],
                    "pair_stats": [{"pair": "1-2", "contact_events": 0, "close_proximity_events": 1, "following_events": 1}],
                    "group_stats": {"mean_pair_distance_px": 14.5},
                    "segment_stats": [{"window_label": "0.0s-30.0s", "interaction_events": 2, "following_events": 1, "pursuit_events": 1, "aggressive_approach_events": 0, "dominant_behavior": "following"}],
                    "phase_comparison": [{"phase": "baseline", "phase_source": "manual", "interaction_events": 1, "escape_events": 0, "dominant_behavior": "following"}],
                    "phase_config": {"mode": "manual", "labels": ["baseline", "stimulus", "recovery"], "boundaries_s": [30.0, 60.0]},
                    "insights": [{"headline": "检测到跟随", "detail": "存在稳定跟随", "action": "继续复核"}],
                },
                parameter_interpretation=[{"group": "推理", "parameter": "imgsz", "current_value": "416", "interpretation": "较快", "suggestion": "需要更准时提高"}],
            )
        )
        report_text = report_path.read_text(encoding="utf-8")
        assert "Behavior Summary" in report_text
        assert "following_events: 1" in report_text
        assert "Most Standing Candidates" in report_text
        assert "Group Behavior Stats" in report_text
        assert "Behavior Phase Setup" in report_text
        assert "Segment Behavior Stats" in report_text
        assert "Phase Behavior Comparison" in report_text
        assert "Most Grooming Candidates" in report_text
        assert "Parameter Interpretation" in report_text
        assert "Behavior Insights" in report_text


def test_buffer_recover_kalman_keeps_aspect_ratio_constant():
    kf = KalmanBoxTracker(np.array([0, 0, 20, 10], dtype=float))
    kf.x[6, 0] = 10.0
    aspect_before = float(kf.x[3, 0])
    kf.predict()
    assert float(kf.x[3, 0]) == aspect_before


def test_sota_tracker_factory_and_offline_stitching():
    tracker = create_tracker(
        mode="sota",
        n_animals=2,
        max_distance=80,
        max_disappeared=30,
        high_conf_thresh=0.4,
    )
    assert get_tracker_name("sota") == "AnimalMOT (SOTA)"
    assert hasattr(tracker, "update")

    all_tracks = {
        0: {1: (0, 0), 3: (100, 100)},
        1: {1: (1, 0), 3: (101, 100)},
        5: {2: (5, 0)},
        6: {2: (6, 0)},
    }
    stitched, stats = stitch_frame_tracks(
        all_tracks,
        n_animals=2,
        max_distance=10,
        max_time_gap=10,
    )
    assert stats["merged_tracks"] == 1
    assert stats["output_tracks"] == 2
    assert 2 not in {tid for frame_tracks in stitched.values() for tid in frame_tracks}


def test_animal_bytetrack_factory_accepts_app_signature_and_fixed_slots():
    tracker = create_tracker(
        mode="animal_bytetrack",
        n_animals=2,
        max_disappeared=5,
        threshold=90,
        min_area=40,
        max_area=5000,
        lost_buffer=20,
    )
    assert get_tracker_name("animal_bytetrack") == "AnimalByteTracker (论文级稳定)"

    frame = np.full((180, 180, 3), 220, dtype=np.uint8)
    cv2.circle(frame, (55, 90), 12, (0, 0, 0), -1)
    cv2.circle(frame, (125, 90), 12, (0, 0, 0), -1)

    tracks = tracker.update(frame, [])
    stats = tracker.get_stats()

    assert 1 <= len(tracks) <= 2
    assert stats["total_created"] <= 2
    assert stats["max_concurrent"] <= 2
    assert len(tracker.last_tracking_detections) <= 2

    tracker.reset()
    reset_stats = tracker.get_stats()
    assert reset_stats["total_created"] == 0
    assert reset_stats["frames"] == 0


def test_automot_prefers_unsupervised_transfer_for_annotation_free_accuracy():
    plan = build_auto_mot_plan(
        n_animals=6,
        max_distance=80,
        max_disappeared=30,
        user_confidence_threshold=0.08,
        prefer_accuracy=True,
        video_info={"width": 1024, "height": 768, "fps": 30},
    )

    assert plan.tracker_mode == "unsupervised_transfer"


def test_animal_bytetrack_clamps_merged_blob_bbox_updates():
    track = ByteAnimalTrack(50, 50, 20, 20)
    track.update(51, 50, 21, 20)
    track.update(52, 50, 20, 21)

    clamped = track.update(53, 50, 120, 120)
    x1, y1, x2, y2 = track.bbox

    assert clamped & 1
    assert (x2 - x1) < 60
    assert (y2 - y1) < 60


def test_animal_bytetrack_low_conf_update_clamps_motion_and_velocity():
    track = ByteAnimalTrack(100, 100, 30, 30)
    track.update(102, 100, 30, 30)
    track.update(104, 100, 30, 30)

    clamped = track.update(190, 100, 30, 30, confidence=0.12)
    vx, vy = track.velocity

    assert clamped & 2
    assert clamped & 4
    assert abs(track.center[0] - 104) <= 45
    assert np.hypot(vx, vy) <= ByteAnimalTrack.MAX_VELOCITY + 1e-4


def test_animal_bytetrack_fixed_slots_do_not_expire_active_identities():
    from tracking.animal_bytetracker import AnimalByteTracker

    tracker = AnimalByteTracker(n_animals=2, max_disappeared=1, lost_buffer=2)
    t1 = ByteAnimalTrack(40, 40, 20, 20)
    t2 = ByteAnimalTrack(90, 40, 20, 20)
    tracker.tracks = {t1.id: t1, t2.id: t2}

    for track in tracker.tracks.values():
        track.time_since_update = 10
    tracker._lifecycle()

    assert set(tracker.tracks.keys()) == {t1.id, t2.id}
    assert tracker.lost == {}
    assert tracker.get_stats()["total_lost"] == 0


def test_animal_bytetrack_reactivates_stale_slot_with_unmatched_detection():
    from tracking.animal_bytetracker import AnimalByteTracker

    tracker = AnimalByteTracker(n_animals=2)
    stale = ByteAnimalTrack(20, 20, 20, 20, np.ones(40) / np.sqrt(40))
    fresh = ByteAnimalTrack(90, 20, 20, 20, np.ones(40) / np.sqrt(40))
    stale.time_since_update = 80
    tracker.tracks = {stale.id: stale, fresh.id: fresh}
    det = {
        "cx": 150.0,
        "cy": 30.0,
        "w": 24.0,
        "h": 22.0,
        "bbox": np.array([138, 19, 162, 41], dtype=float),
        "area": 528.0,
        "conf": 0.9,
        "feat": np.ones(40) / np.sqrt(40),
        "virtual": False,
    }

    remaining = tracker._reactivate_stale_tracks([det])

    assert remaining == []
    assert stale.time_since_update == 0
    assert stale.last_update_was_reactivation
    assert abs(stale.center[0] - 150) < 3
    assert stale.history == [stale.center]
    assert tracker.get_stats()["stale_reactivations"] == 1


def test_animal_bytetrack_does_not_reactivate_stale_slot_from_low_conf_noise():
    from tracking.animal_bytetracker import AnimalByteTracker

    tracker = AnimalByteTracker(n_animals=1)
    stale = ByteAnimalTrack(20, 20, 20, 20)
    stale.time_since_update = 80
    tracker.tracks = {stale.id: stale}
    det = {
        "cx": 150.0,
        "cy": 30.0,
        "w": 24.0,
        "h": 22.0,
        "bbox": np.array([138, 19, 162, 41], dtype=float),
        "area": 528.0,
        "conf": 0.1,
        "feat": None,
        "virtual": False,
    }

    remaining = tracker._reactivate_stale_tracks([det])

    assert remaining == [det]
    assert stale.time_since_update == 80
    assert tracker.get_stats().get("stale_reactivations", 0) == 0


def test_animal_bytetrack_feature_vector_is_normalized_for_matching():
    frame = np.full((80, 80, 3), 220, dtype=np.uint8)
    cnt = np.array([[[20, 20]], [[55, 22]], [[60, 45]], [[40, 65]], [[18, 48]]], dtype=np.int32)
    cv2.drawContours(frame, [cnt], -1, (0, 0, 0), -1)

    feat = _extract_feat(frame, (15, 15, 65, 70), cnt)

    assert feat.shape == (40,)
    assert np.isfinite(feat).all()
    assert np.linalg.norm(feat) <= 1.000001


def test_animal_bytetrack_replaces_merged_blob_with_virtual_splits():
    tracker = create_tracker(
        mode="animal_bytetrack",
        n_animals=2,
        max_disappeared=5,
        min_area=20,
    )
    t1 = ByteAnimalTrack(50, 50, 20, 20)
    t2 = ByteAnimalTrack(85, 55, 20, 20)
    tracker.tracks = {t1.id: t1, t2.id: t2}

    merged = {
        "cx": 67.5,
        "cy": 52.5,
        "w": 90.0,
        "h": 45.0,
        "bbox": np.array([25, 25, 115, 70], dtype=float),
        "area": 4050.0,
        "conf": 1.0,
        "feat": np.zeros(40),
    }

    prepared = tracker._prepare_detections_for_association([merged])

    assert len(prepared) == 2
    assert {d["target_id"] for d in prepared} == {t1.id, t2.id}
    assert all(d.get("virtual") for d in prepared)
    assert tracker.get_stats()["cluster_detections_suppressed"] == 1


def test_offline_post_processor_splits_impossible_jump():
    cleaned, stats = split_impossible_jumps(
        {
            1: {
                0: (10, 10),
                1: (12, 10),
                2: (300, 300),
                3: (302, 300),
            }
        },
        max_speed=50,
        max_frame_gap=10,
    )

    assert stats["splits"] == 1
    assert stats["output_tracks"] == 2
    assert sorted(cleaned.keys()) == [1, 2]
    assert set(cleaned[1].keys()) == {0, 1}
    assert set(cleaned[2].keys()) == {2, 3}


def test_fill_identity_gaps_keeps_expected_active_count():
    filled, stats = fill_identity_gaps(
        {
            7: {
                2: (10, 10),
                4: (20, 10),
            },
            9: {
                0: (100, 100),
                3: (130, 100),
            },
            11: {
                0: (300, 300),
            },
        },
        n_animals=2,
        total_frames=5,
        frame_size=(160, 120),
    )

    assert stats["output_tracks"] == 2
    assert sorted(filled.keys()) == [1, 2]
    assert all(set(history.keys()) == {0, 1, 2, 3, 4} for history in filled.values())
    per_frame_counts = {
        frame_idx: sum(1 for history in filled.values() if frame_idx in history)
        for frame_idx in range(5)
    }
    assert set(per_frame_counts.values()) == {2}
    assert filled[1][0] == (10.0, 10.0)
    assert filled[1][3] == (15.0, 10.0)
    assert filled[2][4] == (130.0, 100.0)


def test_research_annotations_round_trip_and_frame_dict():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "annotations.json"
        save_annotations(
            path,
            {
                0: [
                    AnimalAnnotation(
                        frame=0,
                        identity=1,
                        bbox=(10, 20, 30, 40),
                        mask=[[10, 20, 30, 20, 30, 40, 10, 40]],
                    )
                ],
                1: [
                    AnimalAnnotation(
                        frame=1,
                        identity=1,
                        bbox=(12, 20, 32, 40),
                        occluded=True,
                    )
                ],
            },
            videos=[{"id": "sample"}],
        )

        loaded = load_annotations(path)
        frame_dict = annotations_to_frame_dict(loaded)

        assert sorted(loaded.keys()) == [0, 1]
        assert loaded[1][0].occluded is True
        assert frame_dict[0][1] == (10.0, 20.0, 30.0, 40.0)


def test_mot_metrics_detects_identity_switch():
    gt = {
        0: {1: (0, 0, 20, 20), 2: (100, 0, 120, 20)},
        1: {1: (5, 0, 25, 20), 2: (105, 0, 125, 20)},
        2: {1: (10, 0, 30, 20), 2: (110, 0, 130, 20)},
    }
    pred = {
        0: {1: (0, 0, 20, 20), 2: (100, 0, 120, 20)},
        1: {1: (5, 0, 25, 20), 2: (105, 0, 125, 20)},
        2: {2: (10, 0, 30, 20), 1: (110, 0, 130, 20)},
    }

    metrics = evaluate_mot(gt, pred, iou_threshold=0.5)

    assert metrics.true_positives == 6
    assert metrics.id_switches == 2
    assert metrics.idf1 < 1.0


def test_global_tracklet_optimizer_links_short_fragments():
    optimizer = GlobalTrackletOptimizer(
        n_animals=2,
        max_distance=20,
        max_time_gap=10,
        max_speed=10,
    )
    optimized = optimizer.optimize({
        1: {0: (0, 0), 1: (5, 0)},
        2: {4: (15, 0), 5: (20, 0)},
        3: {0: (100, 100), 1: (105, 100), 5: (120, 100)},
    })

    assert len(optimized) == 2
    assert optimizer.last_stats["global_links"] == 1
    assert set(optimized[1].keys()) == {0, 1, 4, 5}


def test_hard_frame_mining_prioritizes_missing_and_close_contact():
    prev_tracks = {
        1: MockTrack((10, 10)),
        2: MockTrack((100, 10)),
    }
    tracks = {
        1: MockTrack((12, 10)),
        2: MockTrack((20, 10)),
    }
    detections = [MockDetection((12, 10))]

    risk = evaluate_frame_risk(
        frame_idx=5,
        target_animals=3,
        detections=detections,
        tracks=tracks,
        prev_tracks=prev_tracks,
        unique_ids_seen={1, 2, 3, 4},
        close_distance=15,
        max_jump=30,
    )
    selected = select_hard_frames([risk], top_k=1, min_score=1.0)

    assert risk.score > 0
    assert any(reason.startswith("detection_count_mismatch") for reason in risk.reasons)
    assert any(reason.startswith("active_count_mismatch") for reason in risk.reasons)
    assert any(reason.startswith("id_churn") for reason in risk.reasons)
    assert any(reason.startswith("large_jumps") for reason in risk.reasons)
    assert selected == [risk]


def test_auto_labeling_fills_missing_frames_and_marks_review():
    risk = evaluate_frame_risk(
        frame_idx=1,
        target_animals=2,
        detections=[MockDetection((10, 10))],
        tracks={1: MockTrack((10, 10))},
        unique_ids_seen={1, 2, 3},
    )
    annotations, stats = build_auto_annotations(
        raw_bboxes={
            1: {
                0: (0, 0, 20, 20),
                2: (10, 0, 30, 20),
            },
            2: {
                0: (100, 0, 120, 20),
                2: (110, 0, 130, 20),
            },
        },
        risks=[risk],
        n_animals=2,
        total_frames=3,
        frame_size=(160, 120),
        video_id="sample",
        review_score_threshold=1.0,
        max_speed=50,
        max_time_gap=5,
    )

    assert stats["output_annotations"] == 6
    assert len(annotations[1]) == 2
    assert all(ann.source == "interpolated" for ann in annotations[1])
    assert all(ann.review_required for ann in annotations[1])
    assert annotations[0][0].source == "auto"


def test_bidirectional_consistency_fuses_forward_reverse_tracks():
    refined, quality, stats = refine_bidirectional_centers(
        raw_bboxes={
            1: {
                0: (0, 0, 20, 20),
                1: (10, 0, 30, 20),
                2: (20, 0, 40, 20),
            },
            2: {
                0: (100, 0, 120, 20),
                1: (110, 0, 130, 20),
                2: (120, 0, 140, 20),
            },
        },
        n_animals=2,
        total_frames=3,
        frame_size=(160, 120),
        max_speed=50,
        max_time_gap=5,
    )

    assert set(refined) == {1, 2}
    assert all(set(history) == {0, 1, 2} for history in refined.values())
    assert stats["output_tracks"] == 2
    assert all(item.confidence > 0.5 for per_track in quality.values() for item in per_track.values())


def test_scientific_estimate_dashboard_is_generated():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        annotations_path = save_annotations(
            temp_path / "auto_annotations.json",
            {
                0: [
                    AnimalAnnotation(frame=0, identity=1, bbox=(0, 0, 20, 20), source="auto"),
                    AnimalAnnotation(frame=0, identity=2, bbox=(100, 0, 120, 20), source="auto"),
                ],
                1: [
                    AnimalAnnotation(frame=1, identity=1, bbox=(10, 0, 30, 20), source="interpolated", review_required=True, risk_score=2.0),
                    AnimalAnnotation(frame=1, identity=2, bbox=(110, 0, 130, 20), source="auto"),
                ],
            },
            videos=[{"id": "sample", "width": 160, "height": 120, "fps": 10.0}],
        )
        review_queue = temp_path / "review_queue.json"
        review_queue.write_text(
            json.dumps({
                "summary": {"risk_frames": 1, "mean_score": 1.0, "max_score": 2.0},
                "review_frames": [{"frame": 1, "score": 2.0, "reasons": ["active_count_mismatch:1/2"]}],
            }),
            encoding="utf-8",
        )

        dashboard_path = create_scientific_estimate_dashboard(
            annotations_path,
            review_queue,
            temp_path / "dashboard.png",
        )

        assert dashboard_path.exists() and dashboard_path.stat().st_size > 0
        image = cv2.imread(str(dashboard_path))
        assert image is not None
        assert image.shape[1] == 320


if __name__ == "__main__":
    test_distance_gating()
    test_anchor_uniqueness()
    test_virtual_marker_promotion()
    test_export_and_animation()
    test_tracker_factory_overrides()
    test_detection_threshold_counting()
    test_detection_prediction_summary()
    test_track_guided_detector_splits_merged_blob_with_track_priors()
    test_track_guided_detector_adapts_initial_area_without_false_splits()
    test_detector_runtime_config()
    test_results_analytics()
    test_stable_track_filtering_for_visualization()
    test_behavior_analytics_following_and_standing()
    test_pose_normalization_and_pose_assisted_behavior()
    test_pose_estimator_detect_with_tracking_uses_unique_pose_assignment()
    test_behavior_segments_and_pursuit_candidates()
    test_manual_phase_configuration()
    test_interpretation_helpers()
    test_exporter_handles_empty_and_dict_inputs()
    test_export_report_includes_behavior_summary()
    test_buffer_recover_kalman_keeps_aspect_ratio_constant()
    test_sota_tracker_factory_and_offline_stitching()
    test_animal_bytetrack_factory_accepts_app_signature_and_fixed_slots()
    test_automot_prefers_unsupervised_transfer_for_annotation_free_accuracy()
    test_animal_bytetrack_clamps_merged_blob_bbox_updates()
    test_animal_bytetrack_low_conf_update_clamps_motion_and_velocity()
    test_animal_bytetrack_fixed_slots_do_not_expire_active_identities()
    test_animal_bytetrack_reactivates_stale_slot_with_unmatched_detection()
    test_animal_bytetrack_does_not_reactivate_stale_slot_from_low_conf_noise()
    test_animal_bytetrack_feature_vector_is_normalized_for_matching()
    test_animal_bytetrack_replaces_merged_blob_with_virtual_splits()
    test_offline_post_processor_splits_impossible_jump()
    test_fill_identity_gaps_keeps_expected_active_count()
    test_animalmot_accepts_frame_keyword_and_splits_merged_cluster_box()
    test_track_filter_suppresses_duplicate_detection_boxes()
    test_research_annotations_round_trip_and_frame_dict()
    test_mot_metrics_detects_identity_switch()
    test_global_tracklet_optimizer_links_short_fragments()
    test_hard_frame_mining_prioritizes_missing_and_close_contact()
    test_auto_labeling_fills_missing_frames_and_marks_review()
    test_bidirectional_consistency_fuses_forward_reverse_tracks()
    test_scientific_estimate_dashboard_is_generated()
    test_animalmot_enforces_expected_active_track_limit()
    test_group_comparison_statistics()
    test_group_comparison_multigroup_statistics()
    test_group_comparison_paired_statistics()
    test_export_bundle()
    print("review regressions: OK")
