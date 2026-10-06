"""Automatic first-frame seed generation for multi-animal tracking.

This module is designed to replace manual first-frame clicks with detector
derived seed points whenever the video quality allows it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from densetrack.config import DetectionConfig
from densetrack.detection.background_sub import BackgroundSubDetector
from densetrack.detection.dark_object import DarkObjectDetector
from densetrack.models import Detection
from densetrack.utils.video import VideoReader

__all__ = [
    "AutoInitConfig",
    "AutoInitResult",
    "create_manual_initialization_points",
    "find_auto_initialization_points",
    "write_auto_init_outputs",
]


@dataclass(frozen=True)
class AutoInitConfig:
    """Controls automatic seed-point discovery."""

    detector_backend: str = "dark_object"
    max_frames: int = 300
    sample_every: int = 5
    start_frame: int = 0
    resize_height: int = 0
    min_score: float = 0.55


@dataclass
class AutoInitResult:
    """Serializable automatic initialization output."""

    payload: dict[str, Any]
    frame_bgr: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.payload.get("status") == "ok"


def create_manual_initialization_points(
    video_path: str | Path,
    n_animals: int,
    points: list[dict[str, Any]],
    resize_height: int = 0,
    selected_frame: int = 0,
) -> AutoInitResult:
    """Create a first-frame click payload without treating clicks as training labels."""
    video = Path(video_path).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Input video not found / 输入视频不存在: {video}")
    if len(points) != n_animals:
        raise ValueError(
            "Click exactly one point per animal / 请为每只动物点击一个点: "
            f"expected {n_animals}, got {len(points)}"
        )

    frame_bgr: np.ndarray | None = None
    with VideoReader(
        video,
        resize_height=resize_height,
        start_frame=selected_frame,
        end_frame=selected_frame + 1,
    ) as reader:
        for _, frame in reader:
            frame_bgr = frame.copy()
            break
    if frame_bgr is None:
        raise ValueError("Could not read initialization frame / 无法读取初始化帧")

    height, width = frame_bgr.shape[:2]
    normalized = []
    for seed_id, point in enumerate(points):
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each click must include numeric x and y / 每个点击必须包含数值 x 和 y") from exc
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(
                "Click is outside the initialization frame / 点击超出初始化帧范围: "
                f"id={seed_id}, x={x:.2f}, y={y:.2f}"
            )
        normalized.append(
            {
                "id": seed_id,
                "x": x,
                "y": y,
                "confidence": 1.0,
                "source": "manual_first_frame_click",
            }
        )

    payload = {
        "format": "densetrack_auto_initialization_points",
        "format_version": "1.1",
        "created_at": datetime.now().isoformat(),
        "status": "ok",
        "initialization_mode": "manual_first_frame_click",
        "initialization_mode_display": "首帧点击辅助 / first-frame click assist",
        "video": {
            "path": str(video),
            "width": int(width),
            "height": int(height),
        },
        "n_animals_requested": int(n_animals),
        "selected_frame": int(selected_frame),
        "score": 1.0,
        "points": normalized,
        "candidate_frames": [],
        "provenance": {
            "generator": "densetrack.initialization.auto_seed.create_manual_initialization_points",
            "purpose": "Use lightweight first-frame instance selection without training labels.",
            "manual_annotation_used": False,
            "manual_training_annotation_used": False,
            "manual_first_frame_click_used": True,
        },
    }
    return AutoInitResult(payload=payload, frame_bgr=frame_bgr)


def find_auto_initialization_points(
    video_path: str | Path,
    n_animals: int,
    detection_config: DetectionConfig,
    auto_config: AutoInitConfig | None = None,
) -> AutoInitResult:
    """Find detector-derived seed points for the first tracking initialization."""
    auto_cfg = auto_config or AutoInitConfig(detector_backend=detection_config.backend)
    detector = _build_detector(auto_cfg.detector_backend, detection_config)
    video_path = Path(video_path)

    best: dict[str, Any] | None = None
    best_frame: np.ndarray | None = None
    candidates: list[dict[str, Any]] = []
    end_frame = auto_cfg.start_frame + max(auto_cfg.max_frames, 1)

    with VideoReader(
        video_path,
        resize_height=auto_cfg.resize_height,
        start_frame=auto_cfg.start_frame,
        end_frame=end_frame,
    ) as reader:
        metadata = reader.metadata
        for frame_idx, frame in reader:
            detections = detector.detect_frame(frame, frame_idx)
            if (frame_idx - auto_cfg.start_frame) % max(auto_cfg.sample_every, 1) != 0:
                continue

            candidate = _candidate_from_detections(frame_idx, frame, detections, n_animals)
            candidates.append(_candidate_summary(candidate))
            if best is None or candidate["score"] > best["score"]:
                best = candidate
                best_frame = frame.copy()

    if best is None:
        payload = _empty_payload(video_path, n_animals, auto_cfg)
        return AutoInitResult(payload=payload, frame_bgr=None)

    status = "ok" if len(best["points"]) == n_animals and best["score"] >= auto_cfg.min_score else "partial"
    payload = {
        "format": "densetrack_auto_initialization_points",
        "format_version": "1.1",
        "created_at": datetime.now().isoformat(),
        "status": status,
        "initialization_mode": "automatic",
        "initialization_mode_display": "零点击自动初始化 / zero-click automatic initialization",
        "video": {
            "path": str(video_path),
            "width": int(best["frame_width"]),
            "height": int(best["frame_height"]),
        },
        "n_animals_requested": int(n_animals),
        "detector_backend": auto_cfg.detector_backend,
        "selected_frame": int(best["frame_idx"]),
        "score": float(best["score"]),
        "points": best["points"],
        "candidate_frames": sorted(candidates, key=lambda item: item["score"], reverse=True)[:20],
        "provenance": {
            "generator": "densetrack.initialization.auto_seed",
            "purpose": "Replace first-frame manual clicks with detector-derived seed points.",
            "manual_annotation_used": False,
            "manual_training_annotation_used": False,
            "manual_first_frame_click_used": False,
        },
    }
    if not payload["points"]:
        payload["status"] = "failed"
    return AutoInitResult(payload=payload, frame_bgr=best_frame)


def write_auto_init_outputs(
    result: AutoInitResult,
    output_dir: str | Path,
    json_name: str = "initialization_points.json",
    image_name: str = "initialization_points.jpg",
) -> dict[str, str]:
    """Write initialization JSON and, when available, a visual QA image."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / json_name
    json_path.write_text(json.dumps(result.payload, indent=2), encoding="utf-8")
    files = {"json": str(json_path)}

    if result.frame_bgr is not None:
        image = draw_initialization_points(result.frame_bgr, result.payload)
        image_path = output / image_name
        cv2.imwrite(str(image_path), image)
        files["image"] = str(image_path)

    return files


def draw_initialization_points(frame_bgr: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    """Draw seed points and boxes on the selected frame."""
    image = frame_bgr.copy()
    colors = [
        (30, 144, 255),
        (50, 205, 50),
        (255, 99, 71),
        (238, 130, 238),
        (255, 215, 0),
        (0, 206, 209),
    ]
    for point in payload.get("points", []):
        idx = int(point["id"])
        color = colors[idx % len(colors)]
        x = int(round(point["x"]))
        y = int(round(point["y"]))
        bbox = point.get("bbox_xywh", [x - 10, y - 10, 20, 20])
        bx, by, bw, bh = [int(round(v)) for v in bbox]
        cv2.rectangle(image, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.circle(image, (x, y), 7, color, -1)
        cv2.putText(
            image,
            f"id {idx}",
            (x + 9, max(18, y - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    label = f"auto-init frame {payload.get('selected_frame')} score {payload.get('score', 0):.3f}"
    cv2.rectangle(image, (8, 8), (420, 40), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (16, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return image


def _build_detector(backend: str, config: DetectionConfig):
    backend = backend.lower()
    if backend == "dark_object":
        return DarkObjectDetector(config)
    if backend == "background_sub":
        return BackgroundSubDetector(config)
    raise ValueError(
        "auto-init currently supports detector_backend='dark_object' or 'background_sub'. "
        f"Got: {backend}"
    )


def _candidate_from_detections(
    frame_idx: int,
    frame: np.ndarray,
    detections: list[Detection],
    n_animals: int,
) -> dict[str, Any]:
    detections = sorted(
        detections,
        key=lambda det: (det.confidence, det.area or 0),
        reverse=True,
    )
    selected = detections[:n_animals]
    count = len(detections)
    count_score = _count_score(count, n_animals)
    mean_conf = float(np.mean([det.confidence for det in selected])) if selected else 0.0
    separation = _separation_score(selected, frame.shape)
    area_consistency = _area_consistency_score(selected)
    score = 0.45 * count_score + 0.25 * mean_conf + 0.15 * separation + 0.15 * area_consistency

    points = []
    for seed_id, det in enumerate(_stable_seed_order(selected)):
        cx, cy = det.centroid or ((det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2, (det.bbox_xyxy[1] + det.bbox_xyxy[3]) / 2)
        x, y, w, h = det.bbox_xywh
        points.append(
            {
                "id": seed_id,
                "x": float(cx),
                "y": float(cy),
                "bbox_xywh": [float(x), float(y), float(w), float(h)],
                "confidence": float(det.confidence),
                "area": float(det.area or 0.0),
                "source": det.source.value,
            }
        )

    return {
        "frame_idx": frame_idx,
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
        "num_detections": count,
        "num_selected": len(points),
        "score": float(score),
        "count_score": float(count_score),
        "mean_confidence": mean_conf,
        "separation_score": float(separation),
        "area_consistency": float(area_consistency),
        "points": points,
    }


def _stable_seed_order(detections: list[Detection]) -> list[Detection]:
    return sorted(
        detections,
        key=lambda det: (
            round((det.centroid or (0.0, 0.0))[1] / 80.0),
            (det.centroid or (0.0, 0.0))[0],
        ),
    )


def _count_score(count: int, n_animals: int) -> float:
    if n_animals <= 0:
        return 0.0
    if count == n_animals:
        return 1.0
    if count > n_animals:
        return max(0.70, 1.0 - 0.08 * (count - n_animals))
    return 0.75 * count / n_animals


def _separation_score(detections: list[Detection], frame_shape: tuple[int, ...]) -> float:
    if len(detections) < 2:
        return 0.0
    centers = np.array([det.centroid for det in detections if det.centroid is not None], dtype=float)
    if len(centers) < 2:
        return 0.0
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    dists[dists == 0] = np.inf
    min_dist = float(np.min(dists))
    h, w = frame_shape[:2]
    target = max(25.0, 0.08 * float(np.hypot(w, h)))
    return float(np.clip(min_dist / target, 0.0, 1.0))


def _area_consistency_score(detections: list[Detection]) -> float:
    areas = np.array([det.area for det in detections if det.area and det.area > 0], dtype=float)
    if len(areas) < 2:
        return 0.5 if len(areas) == 1 else 0.0
    cv = float(np.std(areas) / max(np.mean(areas), 1.0))
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_idx": int(candidate["frame_idx"]),
        "num_detections": int(candidate["num_detections"]),
        "num_selected": int(candidate["num_selected"]),
        "score": float(candidate["score"]),
        "count_score": float(candidate["count_score"]),
        "mean_confidence": float(candidate["mean_confidence"]),
        "separation_score": float(candidate["separation_score"]),
        "area_consistency": float(candidate["area_consistency"]),
    }


def _empty_payload(video_path: Path, n_animals: int, auto_cfg: AutoInitConfig) -> dict[str, Any]:
    return {
        "format": "densetrack_auto_initialization_points",
        "format_version": "1.1",
        "created_at": datetime.now().isoformat(),
        "status": "failed",
        "initialization_mode": "automatic",
        "initialization_mode_display": "零点击自动初始化 / zero-click automatic initialization",
        "video": {"path": str(video_path)},
        "n_animals_requested": int(n_animals),
        "detector_backend": auto_cfg.detector_backend,
        "selected_frame": None,
        "score": 0.0,
        "points": [],
        "candidate_frames": [],
        "provenance": {
            "generator": "densetrack.initialization.auto_seed",
            "purpose": "Replace first-frame manual clicks with detector-derived seed points.",
            "manual_annotation_used": False,
            "manual_training_annotation_used": False,
            "manual_first_frame_click_used": False,
        },
    }
