"""Research-facing identity-continuity diagnostics and provenance."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from densetrack.config import PipelineConfig
from densetrack.models import Tracklet, Trajectory

__all__ = ["write_identity_continuity_report"]


def write_identity_continuity_report(
    output_dir: str | Path,
    *,
    config: PipelineConfig,
    tracklets: list[Tracklet],
    trajectories: list[Trajectory],
    tracker_stats: Any = None,
    optimizer_stats: Any = None,
) -> dict[str, str]:
    """Write method provenance and long-term association diagnostics."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "identity_continuity.json"
    csv_path = output / "identity_stitches.csv"

    stitches = []
    for trajectory in trajectories:
        for segment_index, segment in enumerate(trajectory.segments):
            if segment_index == 0:
                continue
            previous = trajectory.segments[segment_index - 1]
            stitches.append(
                {
                    "global_id": trajectory.global_id,
                    "from_tracklet": previous.tracklet.tracklet_id,
                    "to_tracklet": segment.tracklet.tracklet_id,
                    "gap_frames": max(segment.start - previous.end - 1, 0),
                    "link_confidence": float(segment.link_confidence),
                    "reason": segment.confidence_reason,
                }
            )

    tracker_payload = (
        asdict(tracker_stats)
        if tracker_stats is not None and is_dataclass(tracker_stats)
        else {}
    )
    optimizer_payload = (
        asdict(optimizer_stats)
        if optimizer_stats is not None and is_dataclass(optimizer_stats)
        else {}
    )
    discrete_geometry = bool(
        config.tracklet.adaptive_discrete_geometry
        and config.tracklet.backend.lower() == "mac_adaptive"
    ) or config.global_optimizer.method.lower() == "long_term"
    payload = {
        "created_at": datetime.now().isoformat(),
        "format": "densetrack_identity_continuity",
        "format_version": "1.0",
        "profile": "research_long_term_identity",
        "profile_display": "科研长时身份连续性 / Research long-term identity continuity",
        "method": {
            "closed_set_known_count": config.n_animals,
            "bidirectional_full_video_smoothing": bool(
                config.tracklet.backend.lower() == "mac_adaptive"
                and config.tracklet.adaptive_identity_smoothing
            ),
            "hierarchical_discrete_geometry": discrete_geometry,
            "geometry_components": [
                "normalized_position",
                "normalized_scale",
                "polar_velocity",
            ],
            "geometry_quantization": {
                "stages": 2,
                "coarse_bins": config.tracklet.adaptive_geometry_coarse_bins,
                "fine_bins": config.tracklet.adaptive_geometry_fine_bins,
                "learned_codebook": False,
                "boundary": (
                    "Training-free deterministic adaptation of HieDG's representation "
                    "principle; not the learned HieDG model."
                ),
            },
            "tracklet_stitching": config.global_optimizer.method.lower() == "long_term",
            "ambiguity_deferral": bool(
                config.tracklet.backend.lower() == "mac_adaptive"
                and config.tracklet.adaptive_reacquisition_min_margin > 0
            ),
        },
        "diagnostics": {
            "input_tracklets": len(tracklets),
            "output_trajectories": len(trajectories),
            "selected_stitches": len(stitches),
            "closed_set_target": config.n_animals,
            "closed_set_achieved": len(trajectories) == config.n_animals,
            "tracker": tracker_payload,
            "optimizer": optimizer_payload,
        },
        "stitches": stitches,
        "scientific_validation": {
            "status": "reviewed_ground_truth_required",
            "status_display": "需要人工审核 GT / reviewed ground truth required",
            "primary_metrics": ["HOTA", "DetA", "AssA", "IDF1", "ID switches"],
            "warning": (
                "Annotation-free stability and risk diagnostics do not prove identity "
                "accuracy. Report scientific claims only from reviewed ground truth."
            ),
        },
        "references": [
            {
                "name": "HieDG",
                "url": "https://arxiv.org/abs/2607.00494",
                "used_for": "hierarchical discrete position, scale, and polar-velocity cues",
            },
            {
                "name": "WaspMOT",
                "url": "https://arxiv.org/abs/2607.08729",
                "used_for": "closed-set long-sequence evaluation and spatial tracklet stitching",
            },
        ],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [
        "global_id",
        "from_tracklet",
        "to_tracklet",
        "gap_frames",
        "link_confidence",
        "reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stitches)
    return {
        "identity_continuity_json": str(json_path),
        "identity_stitches_csv": str(csv_path),
    }
