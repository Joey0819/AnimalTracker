"""Video I/O utilities for DenseTrack-AI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["VideoReader", "VideoMetadata", "read_video_metadata", "extract_frames"]

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str

    def __str__(self) -> str:
        return (
            f"Video: {Path(self.path).name} | "
            f"{self.width}x{self.height} @ {self.fps:.1f}fps | "
            f"{self.total_frames} frames ({self.duration_seconds:.1f}s)"
        )


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Read metadata from a video file without loading frames."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        duration = total_frames / fps if fps > 0 else 0.0

        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec,
        )
    finally:
        cap.release()


class VideoReader:
    """
    Lazy frame-by-frame video reader with optional resizing.

    Usage:
        reader = VideoReader("video.mp4", resize_height=720)
        for frame_idx, frame in reader:
            process(frame)

        # Or random access:
        frame = reader[100]
    """

    def __init__(
        self,
        video_path: str | Path,
        resize_height: int = 0,
        start_frame: int = 0,
        end_frame: int | None = None,
    ):
        self.path = str(video_path)
        self.resize_height = resize_height
        self.start_frame = start_frame
        self.metadata = read_video_metadata(self.path)
        self.end_frame = end_frame or self.metadata.total_frames

        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        if self.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        return self

    def __exit__(self, *args) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self):
        """Iterate over (frame_idx, frame_bgr) tuples."""
        if self._cap is None:
            raise RuntimeError("Use VideoReader as context manager: `with VideoReader(...) as r:`")

        frame_idx = self.start_frame
        while frame_idx < self.end_frame:
            ret, frame = self._cap.read()
            if not ret:
                break
            frame = self._resize(frame)
            yield frame_idx, frame
            frame_idx += 1

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """Random access to a specific frame."""
        if self._cap is None:
            raise RuntimeError("Use VideoReader as context manager")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            raise IndexError(f"Cannot read frame {frame_idx}")
        return self._resize(frame)

    def __len__(self) -> int:
        return self.end_frame - self.start_frame

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_height > 0 and frame.shape[0] != self.resize_height:
            h, w = frame.shape[:2]
            scale = self.resize_height / h
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, self.resize_height))
        return frame


def extract_frames(
    video_path: str | Path,
    frame_indices: list[int] | None = None,
    sample_every: int = 1,
    resize_height: int = 0,
    max_frames: int = 0,
) -> dict[int, np.ndarray]:
    """
    Extract specific frames or sample frames from a video.

    Args:
        video_path: Path to video file.
        frame_indices: Specific frame indices to extract. If None, sample uniformly.
        sample_every: Sample every N-th frame (only used if frame_indices is None).
        resize_height: Resize frames to this height (0 = no resize).
        max_frames: Maximum number of frames to extract (0 = no limit).

    Returns:
        Dict mapping frame_idx -> BGR numpy array.
    """
    meta = read_video_metadata(video_path)
    frames: dict[int, np.ndarray] = {}

    if frame_indices is None:
        frame_indices = list(range(0, meta.total_frames, sample_every))

    if max_frames > 0:
        frame_indices = frame_indices[:max_frames]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        for idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Cannot read frame {idx}, skipping")
                continue
            if resize_height > 0:
                h, w = frame.shape[:2]
                scale = resize_height / h
                frame = cv2.resize(frame, (int(w * scale), resize_height))
            frames[idx] = frame
    finally:
        cap.release()

    logger.info(f"Extracted {len(frames)} frames from {Path(video_path).name}")
    return frames
