"""
Confidence auditor — scores and classifies tracking results.

Every output gets a confidence level:
🟢 HIGH (≥0.8)     — auto-accept
🟡 MEDIUM (0.5-0.8) — spot-check recommended
🟠 LOW (0.2-0.5)    — manual review needed
🔴 UNCERTAIN (<0.2) — identity may be wrong
"""

from __future__ import annotations

import logging

from densetrack.config import ConfidenceConfig
from densetrack.models import ConfidenceLevel, Trajectory

__all__ = ["ConfidenceAuditor"]

logger = logging.getLogger(__name__)


class ConfidenceAuditor:
    """
    Confidence scoring and audit report generation.

    Reports which trajectories and segments are reliable,
    and which need human review. Essential for scientific use:
    researchers need to know WHERE the tracking might be wrong.
    """

    def __init__(self, config: ConfidenceConfig):
        self.config = config

    def generate_report(self, trajectories: list[Trajectory]) -> dict:
        """
        Generate a comprehensive confidence audit report.

        Returns:
            Dict with:
            - summary: aggregate statistics
            - per_trajectory: detailed per-ID confidence info
            - risky_segments: sorted list of lowest-confidence segments
        """
        report: dict = {
            "scope": {
                "name": "trajectory_coverage_and_link_confidence",
                "display": "轨迹覆盖与连接置信度 / Trajectory coverage and link confidence",
                "identity_accuracy": False,
            },
            "scientific_validation": {
                "status": "reviewed_ground_truth_required",
                "warning": (
                    "High trajectory confidence does not prove identity accuracy. "
                    "Use reviewed ground truth with HOTA, AssA, IDF1, and ID switches."
                ),
            },
            "summary": {
                "total_trajectories": len(trajectories),
                "total_frames": 0,
                "total_segments": 0,
                "high_confidence_ratio": 0.0,
                "medium_confidence_ratio": 0.0,
                "low_confidence_ratio": 0.0,
                "uncertain_ratio": 0.0,
                "needs_review_frames": 0,
                "total_occlusion_events": 0,
            },
            "per_trajectory": [],
            "risky_segments": [],
        }

        if not trajectories:
            return report

        all_segments = []
        total_frames = 0

        for traj in trajectories:
            trajectory_confidence = min(traj.mean_confidence, traj.min_link_confidence)
            trajectory_level = ConfidenceLevel.from_score(trajectory_confidence)
            traj_report = {
                "global_id": traj.global_id,
                "frame_range": [traj.start_frame, traj.end_frame],
                "duration_frames": traj.duration,
                "overall_confidence": traj.mean_confidence,
                "effective_confidence": trajectory_confidence,
                "min_link_confidence": traj.min_link_confidence,
                "confidence_level": trajectory_level.value,
                "confidence_emoji": trajectory_level.emoji,
                "num_segments": len(traj.segments),
                "num_occlusion_events": traj.num_occlusions,
                "segments": [],
            }

            for i, seg in enumerate(traj.segments):
                effective_confidence = min(seg.link_confidence, seg.confidence)
                level = ConfidenceLevel.from_score(effective_confidence)
                seg_info = {
                    "segment_index": i,
                    "frames": [seg.start, seg.end],
                    "duration": seg.duration,
                    "tracklet_confidence": seg.confidence,
                    "link_confidence": seg.link_confidence,
                    "effective_confidence": effective_confidence,
                    "level": level.value,
                    "emoji": level.emoji,
                    "description": level.description,
                    "reason": seg.confidence_reason,
                }
                traj_report["segments"].append(seg_info)
                all_segments.append(seg_info)
                total_frames += seg.duration

                # Track risky segments
                if effective_confidence < 0.5:
                    report["risky_segments"].append({
                        "trajectory_id": traj.global_id,
                        "segment_index": i,
                        "frame": seg.start,
                        "link_confidence": seg.link_confidence,
                        "effective_confidence": effective_confidence,
                        "reason": (
                            f"轨迹片段有效置信度仅 {effective_confidence:.2f}"
                            if not seg.confidence_reason
                            else seg.confidence_reason
                        ),
                        "suggestion": "建议人工确认此处 ID 是否正确",
                    })

            report["per_trajectory"].append(traj_report)

        # Sort risky segments by confidence (lowest first)
        report["risky_segments"].sort(key=lambda x: x["link_confidence"])

        # Summary statistics
        report["summary"]["total_frames"] = total_frames
        report["summary"]["total_segments"] = len(all_segments)
        report["summary"]["total_occlusion_events"] = sum(
            t.num_occlusions for t in trajectories
        )

        # Confidence distribution
        if all_segments:
            confs = [s["effective_confidence"] for s in all_segments]
            total = len(confs)
            report["summary"]["high_confidence_ratio"] = sum(
                1 for c in confs if c >= 0.8
            ) / total
            report["summary"]["medium_confidence_ratio"] = sum(
                1 for c in confs if 0.5 <= c < 0.8
            ) / total
            report["summary"]["low_confidence_ratio"] = sum(
                1 for c in confs if 0.2 <= c < 0.5
            ) / total
            report["summary"]["uncertain_ratio"] = sum(
                1 for c in confs if c < 0.2
            ) / total
            report["summary"]["needs_review_frames"] = sum(
                s["duration"] for s in all_segments if s["link_confidence"] < 0.5
            )

        return report
