import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tracking.automated_mot import (
    AutomatedMOTSystem,
    build_auto_mot_plan,
)
from tracking.tracker_factory import create_tracker, get_tracker_name
from visualization.publication_figure import create_publication_trajectory_figure


class MockDetection:
    def __init__(self, center, confidence=0.9, bbox=None):
        self.center = center
        self.confidence = confidence
        self.class_id = 0
        self.bbox = bbox or (
            center[0] - 5,
            center[1] - 5,
            center[0] + 5,
            center[1] + 5,
        )


def test_auto_mot_plan_infers_target_animals_and_threshold():
    profile = {
        "recommended_threshold": 0.15,
        "summary": {
            0.10: {"avg_detections": 4.5, "max_detections": 6, "frame_hit_rate": 1.0},
            0.15: {"avg_detections": 4.0, "max_detections": 5, "frame_hit_rate": 0.95},
            0.25: {"avg_detections": 2.0, "max_detections": 3, "frame_hit_rate": 0.80},
        },
    }

    plan = build_auto_mot_plan(
        n_animals=0,
        max_distance=70,
        max_disappeared=15,
        user_confidence_threshold=0.08,
        video_info={"width": 640, "height": 480, "fps": 30},
        detection_profile=profile,
    )

    assert plan.target_animals == 4
    assert plan.tracker_mode == "unsupervised_transfer"
    assert plan.confidence_threshold == 0.15
    assert any(reason.startswith("inferred_animals") for reason in plan.reason_codes)
    assert "annotation_free_transfer" in plan.reason_codes


def test_automated_mot_generates_pseudo_labels_without_review_flags():
    system = AutomatedMOTSystem(
        n_animals=2,
        max_distance=50,
        max_disappeared=5,
        prefer_accuracy=False,
        video_info={"width": 160, "height": 120, "fps": 10},
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    for idx in range(3):
        system.update(
            frame,
            [
                MockDetection((20 + idx, 20), bbox=(15 + idx, 15, 25 + idx, 25)),
                MockDetection((90 + idx, 30), bbox=(85 + idx, 25, 95 + idx, 35)),
                MockDetection((140, 100), confidence=0.2, bbox=(135, 95, 145, 105)),
            ],
        )

    annotations, stats = system.finalize_auto_annotations(
        total_frames=3,
        frame_size=(160, 120),
        video_id="sample",
    )

    assert stats["fully_automatic"] is True
    assert stats["output_annotations"] == 6
    assert all(not ann.review_required for rows in annotations.values() for ann in rows)
    assert len(system.last_tracking_detections) <= 3


def test_publication_trajectory_figure_is_created():
    all_tracks = {
        0: {1: (10, 10), 2: (80, 20)},
        1: {1: (20, 15), 2: (90, 25)},
        2: {1: (30, 20), 2: (100, 30)},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "figure.png"
        path = create_publication_trajectory_figure(
            all_tracks,
            {"width": 160, "height": 120, "fps": 10},
            output,
            auto_label_stats={"output_annotations": 6, "interpolated_annotations": 0},
            quality_stats={"quality_score": 0.98, "id_churn": 0},
        )
        assert path.exists()
        assert path.stat().st_size > 0


def test_unsupervised_transfer_tracker_keeps_seeded_identities_and_repairs_tracks():
    tracker = create_tracker(
        mode="unsupervised_transfer",
        n_animals=2,
        max_distance=45,
        max_disappeared=8,
        freeze_identity_slots=True,
        enable_bidirectional_refine=True,
    )
    assert get_tracker_name("unsupervised_transfer") == "Annotation-free Transfer Tracker / 无标注迁移追踪"

    for frame_idx in range(6):
        frame = np.full((120, 180, 3), 210, dtype=np.uint8)
        cv2.circle(frame, (30 + frame_idx * 3, 40), 8, (20, 20, 20), -1)
        cv2.circle(frame, (130 - frame_idx * 2, 75), 9, (55, 55, 55), -1)
        detections = [
            MockDetection((30 + frame_idx * 3, 40), bbox=(22 + frame_idx * 3, 32, 38 + frame_idx * 3, 48)),
            MockDetection((130 - frame_idx * 2, 75), bbox=(121 - frame_idx * 2, 66, 139 - frame_idx * 2, 84)),
            MockDetection((165, 15), confidence=0.35, bbox=(160, 10, 170, 20)),
        ]
        tracks = tracker.update(frame, detections)

    assert len(tracks) == 2
    refined, stats = tracker.finalize_tracks(total_frames=6, frame_size=(180, 120))
    assert stats["enabled"] is True
    assert stats["output_tracks"] == 2
    assert set(refined[0]) == {1, 2}

    tracker_stats = tracker.get_stats()["unsupervised_transfer"]
    assert tracker_stats["seeded_identities"] == 2
    assert tracker_stats["self_supervised_updates"] >= 8
    assert tracker_stats["suppressed_new_tracks"] >= 1
