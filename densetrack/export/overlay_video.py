"""
Overlay visualization video generator.

Draws tracking results on top of the original video:
- Colored bounding boxes per ID
- ID labels with confidence emoji
- Trajectory trails (last N positions)
- Semi-transparent mask overlays (if available)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from densetrack.models import ConfidenceLevel, Trajectory

__all__ = ["OverlayVideoGenerator"]

logger = logging.getLogger(__name__)

# Distinguishable color palette (BGR format)
DEFAULT_COLORS = [
    (255, 64, 64),    # Blue-ish red
    (64, 255, 64),    # Green
    (64, 64, 255),    # Red-ish blue
    (255, 255, 64),   # Cyan
    (255, 64, 255),   # Magenta
    (64, 255, 255),   # Yellow
    (255, 128, 0),    # Orange-blue
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring green
    (128, 255, 0),    # Chartreuse
    (0, 128, 255),    # Azure
    (255, 0, 128),    # Rose
]


@dataclass(frozen=True)
class _OverlayEntry:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    centroid: tuple[float, float] | None
    confidence: float


class OverlayVideoGenerator:
    """Generate visualization video with tracking overlays."""

    def __init__(
        self,
        color_palette: list[tuple[int, int, int]] | None = None,
        trail_length: int = 8,
        show_confidence: bool = True,
        show_trails: bool = True,
        bbox_thickness: int = 2,
        max_trail_step_px: float = 75.0,
        min_trail_confidence: float = 0.45,
    ):
        self.colors = color_palette or DEFAULT_COLORS
        self.trail_length = trail_length
        self.show_confidence = show_confidence
        self.show_trails = show_trails
        self.bbox_thickness = bbox_thickness
        self.max_trail_step_px = max_trail_step_px
        self.min_trail_confidence = min_trail_confidence

    def generate(
        self,
        video_path: str,
        trajectories: list[Trajectory],
        output_path: str | Path,
        fps: float | None = None,
    ) -> None:
        """
        Generate overlay visualization video.

        Args:
            video_path: Path to input video.
            trajectories: Tracking results.
            output_path: Output video path.
            fps: Output FPS (None = match input).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        frame_index: dict[int, list[_OverlayEntry]] = {}

        for traj in trajectories:
            for seg in traj.segments:
                for det in seg.tracklet.detections:
                    if det.frame_idx not in frame_index:
                        frame_index[det.frame_idx] = []
                    frame_index[det.frame_idx].append(
                        _OverlayEntry(
                            track_id=traj.global_id,
                            bbox_xyxy=det.bbox_xyxy,
                            centroid=det.centroid,
                            confidence=det.confidence,
                        )
                    )

        self._generate_from_frame_index(
            video_path=video_path,
            frame_index=frame_index,
            output_path=path,
            fps=fps,
        )
        logger.info(f"Overlay video saved to {path}")

    def generate_from_mot(
        self,
        video_path: str | Path,
        mot_path: str | Path,
        output_path: str | Path,
        fps: float | None = None,
    ) -> None:
        """Generate a clean overlay directly from an existing MOT file."""
        from densetrack.export.run_reexport import load_mot_rows

        frame_index: dict[int, list[_OverlayEntry]] = {}
        for row in load_mot_rows(mot_path):
            frame_index.setdefault(row.frame, []).append(
                _OverlayEntry(
                    track_id=row.track_id,
                    bbox_xyxy=(row.x, row.y, row.x + row.width, row.y + row.height),
                    centroid=(row.cx, row.cy),
                    confidence=row.confidence,
                )
            )
        self._generate_from_frame_index(
            video_path=video_path,
            frame_index=frame_index,
            output_path=output_path,
            fps=fps,
        )
        logger.info(f"Overlay video saved to {output_path}")

    def _generate_from_frame_index(
        self,
        video_path: str | Path,
        frame_index: dict[int, list[_OverlayEntry]],
        output_path: str | Path,
        fps: float | None,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        output_fps = fps or input_fps
        trail_history: dict[int, list[tuple[float, float]]] = {}

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, output_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise OSError(f"Cannot write overlay video: {path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw overlays for this frame
            for entry in frame_index.get(frame_idx, []):
                color = self.colors[entry.track_id % len(self.colors)]

                # Bounding box
                x1, y1, x2, y2 = [int(v) for v in entry.bbox_xyxy]
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)

                # ID label with confidence
                label = f"ID:{entry.track_id}"
                if self.show_confidence:
                    level = ConfidenceLevel.from_score(entry.confidence)
                    label += f" {level.value[0].upper()} {entry.confidence:.2f}"

                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y1 = max(0, y1 - th - 8)
                label_y2 = max(th + 8, y1)
                label_x2 = min(width - 1, x1 + tw + 4)
                cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
                cv2.putText(
                    frame, label, (x1 + 2, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

                # Trajectory trail
                if self.show_trails and entry.centroid:
                    history = trail_history.setdefault(entry.track_id, [])
                    if history and (
                        entry.confidence < self.min_trail_confidence
                        or np.linalg.norm(
                            np.asarray(entry.centroid, dtype=float)
                            - np.asarray(history[-1], dtype=float)
                        )
                        > self.max_trail_step_px
                    ):
                        history.clear()
                    history.append(entry.centroid)
                    # Keep only last N positions
                    if len(history) > self.trail_length:
                        del history[:-self.trail_length]

                    pts = history
                    for i in range(1, len(pts)):
                        alpha = i / len(pts)  # Fade older points
                        thickness = max(1, int(alpha * 3))
                        pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                        pt2 = (int(pts[i][0]), int(pts[i][1]))
                        cv2.line(frame, pt1, pt2, color, thickness)

            # Frame counter
            cv2.putText(
                frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        logger.info(f"Overlay video saved to {path} ({frame_idx} frames)")
