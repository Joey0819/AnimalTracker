"""
Report generation and formatting for confidence audit results.
Supports JSON, human-readable text, and Rich console output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

__all__ = ["ReportGenerator"]

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Format and output confidence audit reports."""

    def to_json(self, report: dict, output_path: str | Path) -> None:
        """Save report as JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Confidence report saved to {path}")

    def to_text(self, report: dict) -> str:
        """Generate human-readable text report."""
        lines = []
        s = report.get("summary", {})

        lines.append("═" * 60)
        lines.append("  DenseTrack-AI 轨迹覆盖置信度审计报告（非身份准确率）")
        lines.append("═" * 60)
        lines.append("  身份准确率需使用人工审核 GT 的 HOTA / AssA / IDF1 / ID switch。")
        lines.append("")
        lines.append(f"  轨迹数量:     {s.get('total_trajectories', 0)}")
        lines.append(f"  总帧数:       {s.get('total_frames', 0)}")
        lines.append(f"  总段数:       {s.get('total_segments', 0)}")
        lines.append(f"  遮挡事件:     {s.get('total_occlusion_events', 0)}")
        lines.append("")
        lines.append("  置信度分布:")
        lines.append(f"    🟢 高置信:   {s.get('high_confidence_ratio', 0):.1%}")
        lines.append(f"    🟡 中等:     {s.get('medium_confidence_ratio', 0):.1%}")
        lines.append(f"    🟠 低置信:   {s.get('low_confidence_ratio', 0):.1%}")
        lines.append(f"    🔴 不确定:   {s.get('uncertain_ratio', 0):.1%}")
        lines.append(f"    需审查帧数:  {s.get('needs_review_frames', 0)}")
        lines.append("")

        # Per-trajectory summary
        lines.append("─" * 60)
        lines.append("  每条轨迹概览")
        lines.append("─" * 60)

        for traj in report.get("per_trajectory", []):
            emoji = traj.get("confidence_emoji", "")
            lines.append(
                f"  {emoji} ID {traj['global_id']:3d} | "
                f"帧 {traj['frame_range'][0]:5d}-{traj['frame_range'][1]:5d} | "
                f"段数 {traj['num_segments']:2d} | "
                f"遮挡 {traj['num_occlusion_events']:2d} | "
                f"有效置信 {traj.get('effective_confidence', traj['min_link_confidence']):.2f}"
            )

        # Risky segments
        risky = report.get("risky_segments", [])
        if risky:
            lines.append("")
            lines.append("─" * 60)
            lines.append(f"  ⚠️  需审查片段 ({len(risky)} 处)")
            lines.append("─" * 60)

            for seg in risky[:20]:  # Show top 20
                lines.append(
                    f"  🔴 轨迹 ID {seg['trajectory_id']} | "
                    f"帧 {seg['frame']:5d} | "
                    f"置信度 {seg.get('effective_confidence', seg['link_confidence']):.2f} | "
                    f"{seg['reason']}"
                )

            if len(risky) > 20:
                lines.append(f"  ... 还有 {len(risky) - 20} 处需审查")

        lines.append("")
        lines.append("═" * 60)
        return "\n".join(lines)

    def to_rich_console(self, report: dict) -> None:
        """Print formatted report using Rich library."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            # Summary table
            summary_table = Table(title="DenseTrack-AI 轨迹覆盖置信度审计报告（非身份准确率）")
            summary_table.add_column("指标", style="cyan")
            summary_table.add_column("值", style="bold")

            s = report.get("summary", {})
            summary_table.add_row("轨迹数量", str(s.get("total_trajectories", 0)))
            summary_table.add_row("总帧数", str(s.get("total_frames", 0)))
            summary_table.add_row("🟢 高置信", f"{s.get('high_confidence_ratio', 0):.1%}")
            summary_table.add_row("🟡 中等", f"{s.get('medium_confidence_ratio', 0):.1%}")
            summary_table.add_row("🟠 低置信", f"{s.get('low_confidence_ratio', 0):.1%}")
            summary_table.add_row("🔴 不确定", f"{s.get('uncertain_ratio', 0):.1%}")
            summary_table.add_row("需审查帧数", str(s.get("needs_review_frames", 0)))
            console.print(summary_table)

            # Risky segments
            risky = report.get("risky_segments", [])
            if risky:
                risk_table = Table(title=f"⚠️ 需审查片段 ({len(risky)} 处)")
                risk_table.add_column("轨迹 ID", style="red")
                risk_table.add_column("帧", style="yellow")
                risk_table.add_column("置信度", style="bold red")
                risk_table.add_column("原因")

                for seg in risky[:20]:
                    risk_table.add_row(
                        str(seg["trajectory_id"]),
                        str(seg["frame"]),
                        f"{seg.get('effective_confidence', seg['link_confidence']):.2f}",
                        seg["reason"],
                    )
                console.print(risk_table)

        except ImportError:
            # Fall back to plain text
            print(self.to_text(report))
