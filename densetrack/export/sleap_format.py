"""SLEAP-compatible JSON exporter.

This is a lightweight interchange format, not SLEAP's native binary .slp file.
It preserves frames, tracks, centroid points, bounding boxes, confidence, and
DenseTrack provenance so the output can be converted downstream if needed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from densetrack.models import Trajectory

__all__ = ["SLEAPExporter"]

logger = logging.getLogger(__name__)


class SLEAPExporter:
    """Export trajectories as SLEAP-style labels JSON."""

    def export(
        self,
        trajectories: list[Trajectory],
        output_path: str | Path = "labels.sleap.json",
        video_path: str | None = None,
        video_metadata: dict | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        frames: dict[int, dict] = {}
        tracks = [{"name": f"animal_{traj.global_id}", "id": traj.global_id} for traj in trajectories]

        for traj in trajectories:
            for seg in traj.segments:
                effective_conf = min(seg.link_confidence, seg.confidence)
                if effective_conf < min_confidence:
                    continue

                for det in seg.tracklet.detections:
                    frame = frames.setdefault(
                        det.frame_idx,
                        {"frame_idx": det.frame_idx, "instances": []},
                    )
                    x, y, w, h = det.bbox_xywh
                    cx, cy = det.centroid if det.centroid else (x + w / 2, y + h / 2)
                    frame["instances"].append(
                        {
                            "track": f"animal_{traj.global_id}",
                            "track_id": traj.global_id,
                            "points": {
                                "centroid": {
                                    "x": float(cx),
                                    "y": float(cy),
                                    "score": float(det.confidence),
                                }
                            },
                            "bbox_xywh": [float(x), float(y), float(w), float(h)],
                            "score": float(det.confidence),
                            "densetrack_effective_confidence": float(effective_conf),
                            "densetrack_source": det.source.value,
                        }
                    )

        labels = {
            "format": "densetrack_sleap_json",
            "format_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "video": {
                "path": video_path,
                **(video_metadata or {}),
            },
            "skeleton": {
                "nodes": [{"name": "centroid"}],
                "edges": [],
            },
            "tracks": tracks,
            "frames": [frames[idx] for idx in sorted(frames)],
            "provenance": {
                "generator": "DenseTrack-AI",
                "note": "SLEAP-compatible JSON; convert to native .slp with downstream tooling if required.",
                "min_confidence": min_confidence,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)

        total_instances = sum(len(frame["instances"]) for frame in labels["frames"])
        logger.info(
            "Exported %s SLEAP-style instances across %s frames to %s",
            total_instances,
            len(labels["frames"]),
            path,
        )
