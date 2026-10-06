#!/usr/bin/env python3
"""
Phase 0 Benchmark Script — DenseTrack-AI

Tests each pipeline component independently on real video data
and generates a Phase 0 evaluation report.

Usage:
    python scripts/benchmark.py --video cage.mp4 --n-animals 5

Outputs:
    results/phase0/
    ├── detection_results.json      # Detection metrics per backend
    ├── sam2_propagation_test.json   # SAM2 mask stability
    ├── pose_test.json               # Pose estimation results
    ├── tracking_baseline.json       # ByteTrack baseline metrics
    ├── phase0_report.md             # Human-readable summary
    └── sample_frames/               # Annotated sample frames
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from densetrack.config import DetectionConfig, PipelineConfig, PoseConfig, TrackletConfig
from densetrack.utils.video import VideoReader, read_video_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def benchmark_detection(video_path: str, output_dir: Path, n_frames: int = 100) -> dict:
    """
    Benchmark detection backends on sampled frames.

    Tests:
    1. Background subtraction (always available)
    2. Grounding DINO (if installed)
    """
    logger.info("=" * 50)
    logger.info("Benchmarking: Detection")

    meta = read_video_metadata(video_path)
    # Sample frames uniformly
    sample_indices = np.linspace(
        max(50, 0),  # Skip first 50 frames (warmup)
        meta.total_frames - 1,
        min(n_frames, meta.total_frames),
    ).astype(int).tolist()

    results: dict[str, dict] = {}

    # --- Background Subtraction ---
    logger.info("  Testing: Background Subtraction (MOG2)")
    from densetrack.detection.background_sub import BackgroundSubDetector

    config = DetectionConfig(backend="background_sub")
    detector = BackgroundSubDetector(config)

    frames: dict[int, np.ndarray] = {}
    t0 = time.time()
    with VideoReader(video_path) as reader:
        for frame_idx, frame in reader:
            # Need to process ALL frames for background model
            dets = detector.detect_frame(frame, frame_idx)
            if frame_idx in sample_indices:
                frames[frame_idx] = frame

    bg_sub_time = time.time() - t0

    # Re-detect on sample frames to get counts
    detector.reset()
    det_counts = []
    with VideoReader(video_path) as reader:
        for frame_idx, frame in reader:
            dets = detector.detect_frame(frame, frame_idx)
            if frame_idx in sample_indices:
                det_counts.append(len(dets))

    results["background_sub"] = {
        "detections_per_frame_mean": float(np.mean(det_counts)) if det_counts else 0,
        "detections_per_frame_std": float(np.std(det_counts)) if det_counts else 0,
        "total_time_s": bg_sub_time,
        "fps": meta.total_frames / max(bg_sub_time, 1e-6),
        "available": True,
    }
    logger.info(f"    → {results['background_sub']['detections_per_frame_mean']:.1f} det/frame, "
                f"{results['background_sub']['fps']:.1f} FPS")

    # --- Grounding DINO ---
    try:
        from densetrack.detection.grounding_dino import GroundingDINODetector

        logger.info("  Testing: Grounding DINO")
        config = DetectionConfig(
            backend="grounding_dino",
            detection_prompt="top-view laboratory mouse . white mouse . black mouse",
        )
        gd_detector = GroundingDINODetector(config)

        det_counts = []
        t0 = time.time()
        for frame_idx in sample_indices[:20]:  # Test on fewer frames (slow)
            if frame_idx in frames:
                dets = gd_detector.detect_frame(frames[frame_idx], frame_idx)
                det_counts.append(len(dets))

        gd_time = time.time() - t0

        results["grounding_dino"] = {
            "detections_per_frame_mean": float(np.mean(det_counts)) if det_counts else 0,
            "detections_per_frame_std": float(np.std(det_counts)) if det_counts else 0,
            "frames_tested": len(det_counts),
            "total_time_s": gd_time,
            "fps": len(det_counts) / max(gd_time, 1e-6),
            "available": True,
        }
        logger.info(
            f"    → {results['grounding_dino']['detections_per_frame_mean']:.1f} det/frame, "
            f"{results['grounding_dino']['fps']:.1f} FPS"
        )
    except (ImportError, RuntimeError) as e:
        results["grounding_dino"] = {"available": False, "error": str(e)}
        logger.info(f"    → Not available: {e}")

    # Save results
    with open(output_dir / "detection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def benchmark_pose(video_path: str, output_dir: Path) -> dict:
    """Benchmark pose estimation backends."""
    logger.info("=" * 50)
    logger.info("Benchmarking: Pose Estimation")

    results = {}

    # Geometric (always available)
    logger.info("  Testing: Geometric features")
    from densetrack.pose.geometric import GeometricPoseEstimator
    from densetrack.models import Detection, DetectionSource

    config = PoseConfig(backend="geometry")
    estimator = GeometricPoseEstimator(config)

    # Test on a synthetic detection
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_det = Detection(
        frame_idx=0,
        bbox_xyxy=(100, 100, 250, 200),
        confidence=0.9,
        source=DetectionSource.BACKGROUND_SUB,
    )

    t0 = time.time()
    for _ in range(100):
        pose = estimator.estimate(test_frame, test_det)
    geo_time = time.time() - t0

    results["geometric"] = {
        "available": True,
        "time_per_detection_ms": geo_time / 100 * 1000,
        "sample_output": {
            "orientation": pose.orientation,
            "body_length": pose.body_length,
            "body_area": pose.body_area,
            "aspect_ratio": pose.aspect_ratio,
        },
    }
    logger.info(f"    → {results['geometric']['time_per_detection_ms']:.1f} ms/detection")

    # DLC SuperAnimal: check whether the optional dependency is actually installed.
    try:
        from densetrack.pose import dlc_superanimal as dlc_module

        if dlc_module._dlc_available:
            results["dlc_superanimal"] = {
                "available": True,
                "note": "DeepLabCut import succeeded; model loading/inference still requires a dedicated test",
            }
            logger.info("    → DLC SuperAnimal: dependency installed (model not loaded)")
        else:
            results["dlc_superanimal"] = {
                "available": False,
                "error": "DeepLabCut is not installed",
            }
            logger.info("    → DLC SuperAnimal: not installed")
    except Exception as e:
        results["dlc_superanimal"] = {"available": False, "error": str(e)}
        logger.info("    → DLC SuperAnimal: not available")

    with open(output_dir / "pose_test.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def benchmark_tracking(video_path: str, output_dir: Path, n_animals: int) -> dict:
    """Benchmark the tracking pipeline (detection → tracklets)."""
    logger.info("=" * 50)
    logger.info("Benchmarking: Tracking (background_sub → ByteTrack)")

    from densetrack.detection.background_sub import BackgroundSubDetector
    from densetrack.tracking.tracklet_generator import TrackletGenerator

    det_config = DetectionConfig(backend="background_sub")
    detector = BackgroundSubDetector(det_config)

    track_config = TrackletConfig()
    generator = TrackletGenerator(track_config)

    # Detect
    detections_per_frame: dict[int, list] = {}
    t0 = time.time()
    with VideoReader(video_path) as reader:
        for frame_idx, frame in reader:
            dets = detector.detect_frame(frame, frame_idx)
            detections_per_frame[frame_idx] = dets
    det_time = time.time() - t0

    # Track
    t0 = time.time()
    tracklets = generator.generate(detections_per_frame)
    track_time = time.time() - t0

    meta = read_video_metadata(video_path)
    total_time = det_time + track_time
    fps = meta.total_frames / max(total_time, 1e-6)

    results = {
        "total_frames": meta.total_frames,
        "total_detections": sum(len(d) for d in detections_per_frame.values()),
        "total_tracklets": len(tracklets),
        "avg_tracklet_length": float(np.mean([t.duration for t in tracklets])) if tracklets else 0,
        "detection_time_s": det_time,
        "tracking_time_s": track_time,
        "total_time_s": total_time,
        "fps": fps,
        "tracklet_confidences": {
            "mean": float(np.mean([t.confidence for t in tracklets])) if tracklets else 0,
            "min": float(np.min([t.confidence for t in tracklets])) if tracklets else 0,
            "max": float(np.max([t.confidence for t in tracklets])) if tracklets else 0,
        },
    }

    logger.info(f"    → {results['total_tracklets']} tracklets, {fps:.1f} FPS")

    with open(output_dir / "tracking_baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def generate_report(
    det_results: dict,
    pose_results: dict,
    tracking_results: dict,
    output_dir: Path,
    n_animals: int,
) -> None:
    """Generate the Phase 0 markdown report."""
    lines = [
        "# DenseTrack-AI Phase 0 Evaluation Report",
        "",
        "## Detection",
        "",
        "| Backend | Avg det/frame | FPS | Available |",
        "|---------|--------------|-----|-----------|",
    ]

    for name, data in det_results.items():
        if data.get("available"):
            lines.append(
                f"| {name} | {data.get('detections_per_frame_mean', 0):.1f} | "
                f"{data.get('fps', 0):.1f} | ✅ |"
            )
        else:
            lines.append(f"| {name} | - | - | ❌ {data.get('error', '')} |")

    lines.extend([
        "",
        "## Pose Estimation",
        "",
    ])
    for name, data in pose_results.items():
        status = "✅" if data.get("available") else "❌"
        lines.append(f"- **{name}**: {status}")
        if data.get("time_per_detection_ms"):
            lines.append(f"  - Speed: {data['time_per_detection_ms']:.1f} ms/detection")

    lines.extend([
        "",
        "## Tracking Baseline",
        "",
        f"- Total frames: {tracking_results.get('total_frames', 0)}",
        f"- Total detections: {tracking_results.get('total_detections', 0)}",
        f"- Tracklets generated: {tracking_results.get('total_tracklets', 0)}",
        f"- Avg tracklet length: {tracking_results.get('avg_tracklet_length', 0):.1f} frames",
        f"- Pipeline FPS: {tracking_results.get('fps', 0):.1f}",
        f"- Expected animals: {n_animals}",
        "",
        "## Phase 0 Go / No-Go Assessment",
        "",
        "| Gate | Status | Notes |",
        "|------|--------|-------|",
    ])

    # Auto-assess go/no-go. Without ground truth, detection count is only a proxy.
    bg_det = det_results.get("background_sub", {})
    expected_dets = bg_det.get("detections_per_frame_mean", 0)

    det_proxy_ok = expected_dets >= n_animals * 0.8
    lines.append(
        f"| Detection | {'⚠️ Proxy OK' if det_proxy_ok else '⚠️ Review'} | "
        f"{expected_dets:.1f} det/frame vs {n_animals} expected; needs GT recall check |"
    )

    fps = tracking_results.get("fps", 0)
    lines.append(
        f"| Speed | {'✅ Go' if fps > 1 else '⚠️ Slow'} | {fps:.1f} FPS (offline OK) |"
    )

    tracklets = tracking_results.get("total_tracklets", 0)
    avg_len = tracking_results.get("avg_tracklet_length", 0)
    mean_conf = tracking_results.get("tracklet_confidences", {}).get("mean", 0)
    tracklet_ok = tracklets > 0 and avg_len >= 15 and mean_conf >= 0.2
    lines.append(
        f"| Tracklets | {'✅ Go' if tracklet_ok else '⚠️ Review'} | "
        f"{tracklets} tracklets, avg length {avg_len:.1f}, mean confidence {mean_conf:.3f} |"
    )

    if not tracklet_ok:
        lines.append(
            "| ID Tracking | ⚠️ No-Go for identity claims | "
            "Tracklets are too fragmented/low confidence for reliable global IDs |"
        )

    lines.extend([
        "",
        "## Next Steps",
        "",
        "- [ ] Collect 100-frame ground-truth annotations for MOTA/IDF1 evaluation",
        "- [ ] Test Grounding DINO with species-specific prompts",
        "- [ ] Test SAM 2 mask propagation stability",
        "- [ ] Test DLC SuperAnimal zero-shot pose",
        "- [ ] Run full pipeline and generate overlay video",
    ])

    report_path = output_dir / "phase0_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Phase 0 report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="DenseTrack-AI Phase 0 Benchmark")
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--n-animals", type=int, default=4, help="Expected number of animals")
    parser.add_argument("--output", default="results/phase0", help="Output directory")
    parser.add_argument("--n-frames", type=int, default=100, help="Number of frames to sample")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DenseTrack-AI Phase 0 Benchmark")
    logger.info(f"Video: {args.video}")
    logger.info(f"Expected animals: {args.n_animals}")
    logger.info(f"Output: {output_dir}")

    det_results = benchmark_detection(args.video, output_dir, args.n_frames)
    pose_results = benchmark_pose(args.video, output_dir)
    tracking_results = benchmark_tracking(args.video, output_dir, args.n_animals)

    generate_report(det_results, pose_results, tracking_results, output_dir, args.n_animals)

    logger.info("=" * 50)
    logger.info("Phase 0 Benchmark Complete!")
    logger.info(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
