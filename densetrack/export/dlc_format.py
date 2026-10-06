"""
DeepLabCut H5/CSV format exporter.

Exports keypoint data in DLC-compatible CSV format for downstream analysis.
Native DLC H5 export remains delegated to DeepLabCut's own conversion utilities.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from densetrack.models import PoseResult, Trajectory

__all__ = ["DLCExporter"]

logger = logging.getLogger(__name__)


class DLCExporter:
    """Export pose data in DeepLabCut-compatible format."""

    def export(
        self,
        trajectories: list[Trajectory],
        poses: dict[str, PoseResult] | None = None,
        output_path: str | Path = "poses.csv",
        min_confidence: float = 0.0,
    ) -> None:
        """
        Export keypoint data as CSV (DLC-compatible).

        For full H5 format, use DeepLabCut's own export utilities
        after importing this CSV.

        Args:
            trajectories: Tracking results.
            poses: Optional dict mapping detection_id -> PoseResult.
            output_path: Output file path.
            min_confidence: Minimum confidence filter.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for traj in trajectories:
            for seg in traj.segments:
                for det in seg.tracklet.detections:
                    row = {
                        "frame": det.frame_idx,
                        "id": traj.global_id,
                        "x": det.centroid[0] if det.centroid else 0,
                        "y": det.centroid[1] if det.centroid else 0,
                        "confidence": det.confidence,
                    }

                    # Add pose data if available
                    if poses and det.detection_id in poses:
                        pose = poses[det.detection_id]
                        if pose.has_keypoints:
                            for i, (kx, ky) in enumerate(pose.keypoints_xy):
                                conf = (
                                    pose.keypoint_conf[i]
                                    if pose.keypoint_conf is not None
                                    else 0.0
                                )
                                row[f"kp{i}_x"] = float(kx)
                                row[f"kp{i}_y"] = float(ky)
                                row[f"kp{i}_conf"] = float(conf)
                        if pose.orientation is not None:
                            row["orientation"] = pose.orientation
                        if pose.body_length is not None:
                            row["body_length"] = pose.body_length
                        if pose.body_area is not None:
                            row["body_area"] = pose.body_area

                    rows.append(row)

        if not rows:
            logger.warning("No data to export")
            return

        # Write CSV
        fieldnames = sorted(set().union(*[r.keys() for r in rows]))
        # Ensure standard columns come first
        priority = ["frame", "id", "x", "y", "confidence", "orientation", "body_length", "body_area"]
        ordered = [f for f in priority if f in fieldnames]
        ordered += [f for f in fieldnames if f not in priority]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
            for row in sorted(rows, key=lambda r: (r["frame"], r["id"])):
                writer.writerow(row)

        logger.info(f"Exported {len(rows)} pose entries to {path}")
