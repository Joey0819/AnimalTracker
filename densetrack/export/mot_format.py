"""
MOT Challenge format exporter.

Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from densetrack.models import Trajectory

__all__ = ["MOTExporter"]

logger = logging.getLogger(__name__)


class MOTExporter:
    """Export tracking results in MOT Challenge format."""

    def export(
        self,
        trajectories: list[Trajectory],
        output_path: str | Path,
        min_confidence: float = 0.0,
    ) -> None:
        """
        Export trajectories to MOT format file.

        Args:
            trajectories: List of Trajectory objects.
            output_path: Output file path (.mot or .txt).
            min_confidence: Minimum segment confidence to include.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Provenance header
        lines.append(f"# DenseTrack-AI MOT Export")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append(f"# Trajectories: {len(trajectories)}")
        lines.append(f"# Min confidence filter: {min_confidence}")
        lines.append(f"# Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1")

        total_entries = 0

        for traj in trajectories:
            for seg in traj.segments:
                if seg.link_confidence < min_confidence and seg.link_confidence > 0:
                    continue

                for det in seg.tracklet.detections:
                    x, y, w, h = det.bbox_xywh
                    line = (
                        f"{det.frame_idx},{traj.global_id},"
                        f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                        f"{det.confidence:.4f},-1,-1,-1"
                    )
                    lines.append(line)
                    total_entries += 1

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Exported {total_entries} entries to {path}")
