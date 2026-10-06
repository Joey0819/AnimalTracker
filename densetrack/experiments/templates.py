"""Behavior experiment templates and zone/time-bin metric exports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from densetrack.utils.metrics import load_mot_file

__all__ = [
    "ExperimentMetricsAnalyzer",
    "ExperimentMetricsConfig",
    "ExperimentTemplate",
    "ZoneTemplate",
    "get_experiment_template",
    "list_experiment_templates",
]


@dataclass(frozen=True)
class ZoneTemplate:
    """Normalized zone definition used before a researcher draws custom ROIs."""

    zone_id: str
    name_zh: str
    name_en: str
    shape: str
    params: dict[str, float]
    role: str = "analysis"


@dataclass(frozen=True)
class ExperimentTemplate:
    """Bilingual behavior experiment template."""

    template_id: str
    name_zh: str
    name_en: str
    category_zh: str
    category_en: str
    description_zh: str
    description_en: str
    primary_zone_id: str
    recommended_duration_s: float
    zones: tuple[ZoneTemplate, ...]
    primary_metrics: tuple[str, ...]

    @property
    def display_name(self) -> str:
        return f"{self.name_zh} / {self.name_en}"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.template_id,
            "name_zh": self.name_zh,
            "name_en": self.name_en,
            "display_name": self.display_name,
            "category_zh": self.category_zh,
            "category_en": self.category_en,
            "description_zh": self.description_zh,
            "description_en": self.description_en,
            "primary_zone_id": self.primary_zone_id,
            "recommended_duration_s": self.recommended_duration_s,
            "primary_metrics": list(self.primary_metrics),
            "zones": [
                {
                    "id": zone.zone_id,
                    "name_zh": zone.name_zh,
                    "name_en": zone.name_en,
                    "shape": zone.shape,
                    "params": dict(zone.params),
                    "role": zone.role,
                }
                for zone in self.zones
            ],
        }


@dataclass(frozen=True)
class ExperimentMetricsConfig:
    """Configuration for template-driven zone analysis."""

    template_id: str = "open_field"
    fps: float = 0.0
    time_bin_s: float = 60.0


_TEMPLATES: dict[str, ExperimentTemplate] = {
    "open_field": ExperimentTemplate(
        template_id="open_field",
        name_zh="旷场实验",
        name_en="Open field",
        category_zh="焦虑与探索",
        category_en="Anxiety and exploration",
        description_zh="中心区和周边区停留、进入、速度和路程分析。",
        description_en="Center/periphery occupancy, entries, speed, and distance analysis.",
        primary_zone_id="center",
        recommended_duration_s=480.0,
        primary_metrics=("center_time_s", "center_entries", "center_latency_s", "distance_px"),
        zones=(
            ZoneTemplate("arena", "全场", "Arena", "rect", {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}),
            ZoneTemplate("center", "中心区", "Center zone", "rect", {"x0": 0.25, "y0": 0.25, "x1": 0.75, "y1": 0.75}, "primary"),
        ),
    ),
    "morris_water_maze": ExperimentTemplate(
        template_id="morris_water_maze",
        name_zh="Morris 水迷宫",
        name_en="Morris water maze",
        category_zh="空间学习记忆",
        category_en="Spatial learning and memory",
        description_zh="目标象限、平台区潜伏期、停留和路径长度分析。",
        description_en="Target quadrant, platform latency, occupancy, and path-length analysis.",
        primary_zone_id="platform",
        recommended_duration_s=90.0,
        primary_metrics=("platform_latency_s", "platform_entries", "target_quadrant_time_s", "distance_px"),
        zones=(
            ZoneTemplate("arena", "水池", "Pool", "circle", {"cx": 0.5, "cy": 0.5, "r": 0.48}),
            ZoneTemplate("target_quadrant", "目标象限", "Target quadrant", "rect", {"x0": 0.5, "y0": 0.0, "x1": 1.0, "y1": 0.5}, "primary"),
            ZoneTemplate("platform", "平台区", "Platform zone", "circle", {"cx": 0.72, "cy": 0.28, "r": 0.07}, "primary"),
        ),
    ),
    "three_chamber_social": ExperimentTemplate(
        template_id="three_chamber_social",
        name_zh="三腔社交实验",
        name_en="Three-chamber social test",
        category_zh="社会行为",
        category_en="Social behavior",
        description_zh="左、中、右三腔停留和进入，用于社交偏好或社交新颖性分析。",
        description_en="Left, center, and right chamber occupancy and entries for social preference analysis.",
        primary_zone_id="social_chamber",
        recommended_duration_s=300.0,
        primary_metrics=("social_chamber_time_s", "empty_chamber_time_s", "center_chamber_time_s"),
        zones=(
            ZoneTemplate("empty_chamber", "空笼侧", "Empty/object chamber", "rect", {"x0": 0.0, "y0": 0.0, "x1": 0.33, "y1": 1.0}),
            ZoneTemplate("center_chamber", "中间腔", "Center chamber", "rect", {"x0": 0.33, "y0": 0.0, "x1": 0.67, "y1": 1.0}),
            ZoneTemplate("social_chamber", "社交目标侧", "Social target chamber", "rect", {"x0": 0.67, "y0": 0.0, "x1": 1.0, "y1": 1.0}, "primary"),
        ),
    ),
    "light_dark_box": ExperimentTemplate(
        template_id="light_dark_box",
        name_zh="明暗箱实验",
        name_en="Light-dark box",
        category_zh="焦虑样行为",
        category_en="Anxiety-like behavior",
        description_zh="明箱/暗箱停留、进入和首次进入潜伏期。",
        description_en="Light/dark chamber occupancy, entries, and first-entry latency.",
        primary_zone_id="light_chamber",
        recommended_duration_s=600.0,
        primary_metrics=("light_chamber_time_s", "light_chamber_entries", "light_chamber_latency_s"),
        zones=(
            ZoneTemplate("dark_chamber", "暗箱", "Dark chamber", "rect", {"x0": 0.0, "y0": 0.0, "x1": 0.34, "y1": 1.0}),
            ZoneTemplate("light_chamber", "明箱", "Light chamber", "rect", {"x0": 0.34, "y0": 0.0, "x1": 1.0, "y1": 1.0}, "primary"),
        ),
    ),
    "elevated_plus_maze": ExperimentTemplate(
        template_id="elevated_plus_maze",
        name_zh="高架十字迷宫",
        name_en="Elevated plus maze",
        category_zh="焦虑样行为",
        category_en="Anxiety-like behavior",
        description_zh="开放臂、闭合臂和中心区停留与进入分析。",
        description_en="Open-arm, closed-arm, and center occupancy and entries.",
        primary_zone_id="open_arms",
        recommended_duration_s=300.0,
        primary_metrics=("open_arms_time_s", "open_arms_entries", "closed_arms_time_s"),
        zones=(
            ZoneTemplate("center", "中心区", "Center", "rect", {"x0": 0.40, "y0": 0.40, "x1": 0.60, "y1": 0.60}),
            ZoneTemplate("open_arms", "开放臂", "Open arms", "cross", {"x0": 0.0, "y0": 0.42, "x1": 1.0, "y1": 0.58}, "primary"),
            ZoneTemplate("closed_arms", "闭合臂", "Closed arms", "cross", {"x0": 0.42, "y0": 0.0, "x1": 0.58, "y1": 1.0}),
        ),
    ),
    "cpp": ExperimentTemplate(
        template_id="cpp",
        name_zh="条件性位置偏爱",
        name_en="Conditioned place preference",
        category_zh="奖赏学习",
        category_en="Reward learning",
        description_zh="左/中/右隔间停留时间和 CPP 分数所需基础表。",
        description_en="Left/center/right chamber occupancy for downstream CPP score calculation.",
        primary_zone_id="paired_chamber",
        recommended_duration_s=900.0,
        primary_metrics=("paired_chamber_time_s", "unpaired_chamber_time_s", "center_chamber_time_s"),
        zones=(
            ZoneTemplate("unpaired_chamber", "非配对隔间", "Unpaired chamber", "rect", {"x0": 0.0, "y0": 0.0, "x1": 0.40, "y1": 1.0}),
            ZoneTemplate("center_chamber", "中间隔间", "Center chamber", "rect", {"x0": 0.40, "y0": 0.0, "x1": 0.60, "y1": 1.0}),
            ZoneTemplate("paired_chamber", "药物配对隔间", "Drug-paired chamber", "rect", {"x0": 0.60, "y0": 0.0, "x1": 1.0, "y1": 1.0}, "primary"),
        ),
    ),
}


def list_experiment_templates() -> list[ExperimentTemplate]:
    """Return all bundled behavior experiment templates."""
    return list(_TEMPLATES.values())


def get_experiment_template(template_id: str) -> ExperimentTemplate:
    """Return a bundled template by id."""
    key = template_id.strip().lower()
    try:
        return _TEMPLATES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_TEMPLATES))
        raise ValueError(f"Unknown experiment template / 未知实验模板: {template_id}. Choices: {choices}") from exc


class ExperimentMetricsAnalyzer:
    """Compute template-driven zone and time-bin metrics from MOT outputs."""

    def __init__(self, config: ExperimentMetricsConfig | None = None):
        self.config = config or ExperimentMetricsConfig()
        self.template = get_experiment_template(self.config.template_id)

    def analyze_run(
        self,
        run_dir: str | Path,
        output_json: str | Path | None = None,
        output_zone_csv: str | Path | None = None,
        output_timebin_csv: str | Path | None = None,
    ) -> dict[str, Any]:
        run_path = Path(run_dir).expanduser().resolve()
        return self.analyze_mot(
            mot_path=run_path / "tracks.mot",
            run_summary_path=run_path / "run_summary.json",
            output_json=output_json or run_path / "experiment_metrics.json",
            output_zone_csv=output_zone_csv or run_path / "experiment_zone_metrics.csv",
            output_timebin_csv=output_timebin_csv or run_path / "experiment_time_bins.csv",
        )

    def analyze_mot(
        self,
        mot_path: str | Path,
        run_summary_path: str | Path | None = None,
        output_json: str | Path | None = None,
        output_zone_csv: str | Path | None = None,
        output_timebin_csv: str | Path | None = None,
    ) -> dict[str, Any]:
        mot = Path(mot_path).expanduser().resolve()
        df = load_mot_file(mot)
        summary = _read_json(run_summary_path) if run_summary_path else {}
        video_meta = summary.get("video_metadata", {})
        fps = self.config.fps or float(video_meta.get("fps", 0) or 15.0)
        width, height = _resolve_extent(df, video_meta)
        time_bin_s = max(float(self.config.time_bin_s), 1.0)

        if df.empty:
            report = self._empty_report(mot, fps, width, height, time_bin_s)
            self._write_outputs(report, [], [], output_json, output_zone_csv, output_timebin_csv)
            return report

        df = df.copy()
        df["Cx"] = df["X"] + df["Width"] / 2
        df["Cy"] = df["Y"] + df["Height"] / 2
        df = df.sort_values(["Id", "FrameId"])

        zone_rows: list[dict[str, Any]] = []
        timebin_rows: list[dict[str, Any]] = []
        per_id_summary: dict[str, dict[str, Any]] = {}
        aggregate: dict[str, dict[str, Any]] = {
            zone.zone_id: {
                "zone_id": zone.zone_id,
                "zone_name_zh": zone.name_zh,
                "zone_name_en": zone.name_en,
                "time_s": 0.0,
                "entries": 0,
                "observations": 0,
            }
            for zone in self.template.zones
        }

        min_frame = int(df["FrameId"].min())
        max_frame = int(df["FrameId"].max())
        bin_frames = max(int(round(time_bin_s * fps)), 1)

        for track_id, group in df.groupby("Id"):
            group = group.sort_values("FrameId").copy()
            track_key = str(int(track_id))
            distance_px = _distance_px(group)
            track_duration_s = max(
                float(
                    (
                        int(group["FrameId"].max())
                        - int(group["FrameId"].min())
                        + 1
                    )
                    / fps
                ),
                1.0 / fps,
            )
            per_id_summary[track_key] = {
                "distance_px": distance_px,
                "mean_speed_px_s": float(distance_px / track_duration_s),
                "frames_tracked": int(len(group)),
                "first_frame": int(group["FrameId"].min()),
                "last_frame": int(group["FrameId"].max()),
            }
            for zone in self.template.zones:
                inside = _inside_zone(group, zone, width, height)
                entries = _count_entries(inside)
                frames_inside = int(np.count_nonzero(inside))
                first_inside = group.loc[inside, "FrameId"]
                latency_s = (
                    float((int(first_inside.iloc[0]) - int(group["FrameId"].min())) / fps)
                    if len(first_inside)
                    else None
                )
                time_s = float(frames_inside / fps)
                row = {
                    "template_id": self.template.template_id,
                    "track_id": int(track_id),
                    "zone_id": zone.zone_id,
                    "zone_name_zh": zone.name_zh,
                    "zone_name_en": zone.name_en,
                    "time_s": time_s,
                    "entries": int(entries),
                    "latency_s": latency_s,
                    "observations": frames_inside,
                    "distance_px": distance_px,
                }
                zone_rows.append(row)
                aggregate[zone.zone_id]["time_s"] += time_s
                aggregate[zone.zone_id]["entries"] += int(entries)
                aggregate[zone.zone_id]["observations"] += frames_inside
                per_id_summary[track_key][f"{zone.zone_id}_time_s"] = time_s
                per_id_summary[track_key][f"{zone.zone_id}_entries"] = int(entries)
                per_id_summary[track_key][f"{zone.zone_id}_latency_s"] = latency_s

                for start in range(min_frame, max_frame + 1, bin_frames):
                    end = min(start + bin_frames - 1, max_frame)
                    mask = (group["FrameId"] >= start) & (group["FrameId"] <= end)
                    if not bool(mask.any()):
                        continue
                    bin_inside = inside[mask.to_numpy()]
                    timebin_rows.append(
                        {
                            "template_id": self.template.template_id,
                            "track_id": int(track_id),
                            "zone_id": zone.zone_id,
                            "bin_start_s": float((start - min_frame) / fps),
                            "bin_end_s": float((end - min_frame + 1) / fps),
                            "time_s": float(np.count_nonzero(bin_inside) / fps),
                            "observations": int(np.count_nonzero(bin_inside)),
                        }
                    )

        primary = aggregate.get(self.template.primary_zone_id, {})
        total_frames = int(video_meta.get("total_frames", 0) or (max_frame - min_frame + 1))
        num_ids = int(df["Id"].nunique())
        total_distance_px = float(
            sum(float(row.get("distance_px", 0.0)) for row in per_id_summary.values())
        )
        mean_speed_px_s = float(
            np.mean(
                [float(row.get("mean_speed_px_s", 0.0)) for row in per_id_summary.values()]
            )
        ) if per_id_summary else 0.0
        tracking_coverage_ratio = float(
            len(df) / max(num_ids * total_frames, 1)
        )
        report = {
            "format": "densetrack_experiment_metrics",
            "created_at": datetime.now().isoformat(),
            "source": str(mot),
            "template": self.template.to_public_dict(),
            "config": {
                "template_id": self.template.template_id,
                "fps": fps,
                "time_bin_s": time_bin_s,
            },
            "scale": {
                "width_px": width,
                "height_px": height,
                "calibrated_real_world_units": False,
            },
            "summary": {
                "num_ids": num_ids,
                "num_zones": len(self.template.zones),
                "frame_range": [min_frame, max_frame],
                "total_frames": total_frames,
                "duration_s": float(total_frames / fps),
                "total_distance_px": total_distance_px,
                "mean_distance_px_per_id": total_distance_px / max(num_ids, 1),
                "mean_speed_px_s": mean_speed_px_s,
                "tracking_coverage_ratio": tracking_coverage_ratio,
                "primary_zone_id": self.template.primary_zone_id,
                "primary_zone_time_s": float(primary.get("time_s", 0.0)),
                "primary_zone_entries": int(primary.get("entries", 0)),
            },
            "analysis_policy": _transparent_analysis_policy(),
            "transparent_baseline": {
                "method": "parametric_aggregation",
                "method_display": "透明参数聚合 / Transparent parametric aggregation",
                "total_distance_px": total_distance_px,
                "mean_distance_px_per_id": total_distance_px / max(num_ids, 1),
                "mean_speed_px_s": mean_speed_px_s,
                "tracking_coverage_ratio": tracking_coverage_ratio,
                "primary_zone_id": self.template.primary_zone_id,
                "primary_zone_time_s": float(primary.get("time_s", 0.0)),
                "primary_zone_entries": int(primary.get("entries", 0)),
                "time_bin_s": time_bin_s,
            },
            "aggregate_zones": list(aggregate.values()),
            "per_id": per_id_summary,
            "zone_metrics": zone_rows,
            "time_bins": timebin_rows,
            "boundary": (
                "Template zones are normalized defaults; replace with researcher-drawn arena.json "
                "for calibrated study-specific ROIs. / 模板区域是归一化默认区域；正式研究应替换为研究者绘制的 arena.json。"
            ),
        }
        self._write_outputs(report, zone_rows, timebin_rows, output_json, output_zone_csv, output_timebin_csv)
        return report

    def _empty_report(
        self,
        mot_path: Path,
        fps: float,
        width: float,
        height: float,
        time_bin_s: float,
    ) -> dict[str, Any]:
        return {
            "format": "densetrack_experiment_metrics",
            "created_at": datetime.now().isoformat(),
            "source": str(mot_path),
            "template": self.template.to_public_dict(),
            "config": {"template_id": self.template.template_id, "fps": fps, "time_bin_s": time_bin_s},
            "scale": {"width_px": width, "height_px": height, "calibrated_real_world_units": False},
            "summary": {
                "num_ids": 0,
                "num_zones": len(self.template.zones),
                "frame_range": [0, 0],
                "total_frames": 0,
                "duration_s": 0.0,
                "primary_zone_id": self.template.primary_zone_id,
                "primary_zone_time_s": 0.0,
                "primary_zone_entries": 0,
            },
            "analysis_policy": _transparent_analysis_policy(),
            "transparent_baseline": {
                "method": "parametric_aggregation",
                "method_display": "透明参数聚合 / Transparent parametric aggregation",
                "total_distance_px": 0.0,
                "mean_distance_px_per_id": 0.0,
                "mean_speed_px_s": 0.0,
                "tracking_coverage_ratio": 0.0,
                "primary_zone_id": self.template.primary_zone_id,
                "primary_zone_time_s": 0.0,
                "primary_zone_entries": 0,
                "time_bin_s": time_bin_s,
            },
            "aggregate_zones": [],
            "per_id": {},
            "zone_metrics": [],
            "time_bins": [],
            "boundary": "No MOT rows found / 未找到 MOT 轨迹行。",
        }

    def _write_outputs(
        self,
        report: dict[str, Any],
        zone_rows: list[dict[str, Any]],
        timebin_rows: list[dict[str, Any]],
        output_json: str | Path | None,
        output_zone_csv: str | Path | None,
        output_timebin_csv: str | Path | None,
    ) -> None:
        if output_json:
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if output_zone_csv:
            _write_csv(
                output_zone_csv,
                zone_rows,
                [
                    "template_id",
                    "track_id",
                    "zone_id",
                    "zone_name_zh",
                    "zone_name_en",
                    "time_s",
                    "entries",
                    "latency_s",
                    "observations",
                    "distance_px",
                ],
            )
        if output_timebin_csv:
            _write_csv(
                output_timebin_csv,
                timebin_rows,
                [
                    "template_id",
                    "track_id",
                    "zone_id",
                    "bin_start_s",
                    "bin_end_s",
                    "time_s",
                    "observations",
                ],
            )


def _transparent_analysis_policy() -> dict[str, Any]:
    return {
        "default_tier": "transparent_parametric_baseline",
        "default_tier_display": "透明基础聚合优先 / Transparent baseline first",
        "recommended_for": [
            "controlled_treatment_effect_detection",
            "dose_discrimination",
            "small_to_medium_preclinical_studies",
        ],
        "advanced_representation_for": [
            "open_ended_behavior_discovery",
            "social_interaction_structure",
            "long_term_identity_questions",
            "large_scale_behavior_atlas",
        ],
        "interpretation": (
            "Complex segmentation is an optional discovery layer, not a default "
            "replacement for distance, speed, occupancy, entries, and time-bin metrics."
        ),
        "interpretation_zh": (
            "复杂行为分割是可选发现层，不默认替代路程、速度、区域停留、进入次数和时间箱指标。"
        ),
        "evidence": {
            "title": (
                "Comparison of multiple video tracking-based behavioral summary "
                "approaches for compound discrimination"
            ),
            "source_type": "bioRxiv preprint",
            "doi": "10.64898/2026.07.20.739643",
            "design": "314 recordings, eight treatment-dose pairs, five summary approaches",
            "finding_boundary": (
                "Under the tested controlled treatment conditions, simple aggregate "
                "tracking measures and machine-learning behavior summaries performed "
                "similarly for effect detection and treatment-dose discrimination."
            ),
        },
    }


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def _resolve_extent(df, video_meta: dict[str, Any]) -> tuple[float, float]:
    width = float(video_meta.get("width", 0) or 0)
    height = float(video_meta.get("height", 0) or 0)
    if width <= 0 and not df.empty:
        width = float((df["X"] + df["Width"]).max())
    if height <= 0 and not df.empty:
        height = float((df["Y"] + df["Height"]).max())
    return max(width, 1.0), max(height, 1.0)


def _inside_zone(group, zone: ZoneTemplate, width: float, height: float) -> np.ndarray:
    x = group["Cx"].to_numpy(dtype=float)
    y = group["Cy"].to_numpy(dtype=float)
    params = zone.params
    if zone.shape == "rect":
        x0 = float(params["x0"]) * width
        y0 = float(params["y0"]) * height
        x1 = float(params["x1"]) * width
        y1 = float(params["y1"]) * height
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if zone.shape == "circle":
        cx = float(params["cx"]) * width
        cy = float(params["cy"]) * height
        radius = float(params["r"]) * min(width, height)
        return ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2
    if zone.shape == "cross":
        x0 = float(params["x0"]) * width
        y0 = float(params["y0"]) * height
        x1 = float(params["x1"]) * width
        y1 = float(params["y1"]) * height
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    raise ValueError(f"Unsupported zone shape / 不支持的区域形状: {zone.shape}")


def _count_entries(inside: np.ndarray) -> int:
    if len(inside) == 0:
        return 0
    shifted = np.concatenate(([False], inside[:-1]))
    return int(np.count_nonzero(inside & ~shifted))


def _distance_px(group) -> float:
    coords = group[["Cx", "Cy"]].to_numpy(dtype=float)
    if len(coords) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
