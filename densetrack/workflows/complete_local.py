"""One-command annotation-free local tracking workflow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from densetrack.behavior.rule_based import BehaviorConfig, RuleBasedBehaviorAnalyzer
from densetrack.behavior.unsupervised import (
    BehaviorDiscoveryConfig,
    UnsupervisedBehaviorDiscovery,
)
from densetrack.config import PipelineConfig
from densetrack.experiments import ExperimentMetricsAnalyzer, ExperimentMetricsConfig
from densetrack.initialization.auto_seed import (
    AutoInitConfig,
    create_manual_initialization_points,
    find_auto_initialization_points,
    write_auto_init_outputs,
)
from densetrack.pipeline import Pipeline
from densetrack.reporting.static_dashboard import build_static_dashboard
from densetrack.validation.review_package import build_validation_review_package

__all__ = ["run_complete_local"]


def run_complete_local(
    video_path: str | Path,
    pipeline_config: PipelineConfig,
    output_dir: str | Path | None = None,
    motif_clusters: int = 6,
    validation_frames: int = 100,
    auto_init_max_frames: int = 450,
    auto_init_sample_every: int = 3,
    initialization_override: dict[str, Any] | None = None,
    experiment_template: str | None = None,
    experiment_time_bin_s: float | None = None,
) -> dict[str, Any]:
    """Run the local annotation-free workflow and write a bilingual result package."""
    video = Path(video_path).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Input video not found / 输入视频不存在: {video}")

    output = Path(output_dir or pipeline_config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig.model_validate(pipeline_config.model_dump())
    config.video_path = str(video)
    config.output_dir = str(output)

    auto_init_dir = output / "auto_init"
    if initialization_override and initialization_override.get("initialization_mode") == "manual_first_frame_click":
        auto_init_result = create_manual_initialization_points(
            video_path=video,
            n_animals=config.n_animals,
            points=initialization_override.get("points", []),
            resize_height=config.processing_resolution,
            selected_frame=int(initialization_override.get("selected_frame", 0) or 0),
        )
    else:
        auto_init_result = find_auto_initialization_points(
            video_path=video,
            n_animals=config.n_animals,
            detection_config=config.detection,
            auto_config=AutoInitConfig(
                detector_backend=config.detection.backend,
                max_frames=auto_init_max_frames,
                sample_every=auto_init_sample_every,
                resize_height=config.processing_resolution,
            ),
        )
    auto_init_files = write_auto_init_outputs(auto_init_result, auto_init_dir)

    tracking_result = Pipeline(config, initialization=auto_init_result.payload).run()
    fps = float(tracking_result.get("video_metadata", {}).get("fps", 0) or 15.0)
    mot_path = output / "tracks.mot"

    rule_report_path = output / "behavior_report.json"
    rule_timeline_path = output / "behavior_timeline.csv"
    rule_report = RuleBasedBehaviorAnalyzer(BehaviorConfig(fps=fps)).analyze_mot(
        mot_path=mot_path,
        output_json=rule_report_path,
        output_timeline_csv=rule_timeline_path,
    )

    motif_report_path = output / "behavior_clusters.json"
    motif_timeline_path = output / "behavior_clusters_timeline.csv"
    motif_embedding_path = output / "behavior_clusters_embedding.jpg"
    motif_timeline_image_path = output / "behavior_clusters_timeline.jpg"
    motif_report = UnsupervisedBehaviorDiscovery(
        BehaviorDiscoveryConfig(
            fps=fps,
            n_clusters=motif_clusters,
            pose_csv=str(output / "poses.csv") if (output / "poses.csv").exists() else None,
        )
    ).analyze_mot(
        mot_path=mot_path,
        output_json=motif_report_path,
        output_csv=motif_timeline_path,
        embedding_image=motif_embedding_path,
        timeline_image=motif_timeline_image_path,
    )

    experiment_report: dict[str, Any] = {}
    experiment_files: dict[str, str | None] = {
        "experiment_metrics": None,
        "experiment_zone_metrics": None,
        "experiment_time_bins": None,
    }
    resolved_experiment_template = experiment_template
    if resolved_experiment_template is None and config.experiment.enabled:
        resolved_experiment_template = config.experiment.template_id
    if resolved_experiment_template and resolved_experiment_template.lower() not in {"none", "off", "disabled"}:
        experiment_report_path = output / "experiment_metrics.json"
        experiment_zone_path = output / "experiment_zone_metrics.csv"
        experiment_timebin_path = output / "experiment_time_bins.csv"
        experiment_report = ExperimentMetricsAnalyzer(
            ExperimentMetricsConfig(
                template_id=resolved_experiment_template,
                fps=fps,
                time_bin_s=experiment_time_bin_s or config.experiment.time_bin_s,
            )
        ).analyze_run(
            run_dir=output,
            output_json=experiment_report_path,
            output_zone_csv=experiment_zone_path,
            output_timebin_csv=experiment_timebin_path,
        )
        experiment_files = {
            "experiment_metrics": str(experiment_report_path),
            "experiment_zone_metrics": str(experiment_zone_path),
            "experiment_time_bins": str(experiment_timebin_path),
        }

    validation_review = build_validation_review_package(
        run_dir=output,
        n_frames=validation_frames,
        video_path=video,
    )
    dashboard = build_static_dashboard(output, auto_init_dir=auto_init_dir)
    manifest_path = output / "complete_run_manifest.json"
    tracking_stats = tracking_result.get("fixed_count_stats")
    tracking_stats_payload = (
        tracking_stats.__dict__
        if tracking_stats is not None and hasattr(tracking_stats, "__dict__")
        else None
    )
    visual_ssl_report = _read_json(output / "visual_ssl_report.json")
    bioacoustic_ssl_report = _read_json(output / "bioacoustic_ssl_report.json")
    adaptive_tracking = config.tracklet.backend.lower() == "mac_adaptive"
    initialization_mode = auto_init_result.payload.get("initialization_mode", "automatic")
    click_assisted = initialization_mode == "manual_first_frame_click"
    manifest = {
        "created_at": datetime.now().isoformat(),
        "workflow": "densetrack_annotation_free_local_complete",
        "workflow_display": (
            "DenseTrack 首帧点击辅助本地完整流程 / DenseTrack first-frame click-assisted local workflow"
            if click_assisted
            else "DenseTrack 零点击本地完整流程 / DenseTrack zero-click local complete workflow"
        ),
        "video": str(video),
        "output_dir": str(output),
        "manual_annotation_required": False,
        "manual_annotation_used": False,
        "manual_training_annotation_required": False,
        "manual_training_annotation_used": False,
        "manual_first_frame_click_used": click_assisted,
        "initialization_mode": initialization_mode,
        "initialization_mode_display": auto_init_result.payload.get("initialization_mode_display"),
        "automatic_initialization": {
            "status": auto_init_result.payload.get("status"),
            "selected_frame": auto_init_result.payload.get("selected_frame"),
            "score": auto_init_result.payload.get("score"),
            "points": len(auto_init_result.payload.get("points", [])),
            "requested_animals": config.n_animals,
            "files": auto_init_files,
        },
        "initialization": {
            "mode": initialization_mode,
            "mode_display": auto_init_result.payload.get("initialization_mode_display"),
            "selected_frame": auto_init_result.payload.get("selected_frame"),
            "points": len(auto_init_result.payload.get("points", [])),
            "manual_training_annotation_used": False,
            "manual_first_frame_click_used": click_assisted,
            "files": auto_init_files,
        },
        "tracking": {
            "backend": config.tracklet.backend,
            "backend_display": (
                "Mac 原生自适应双向追踪 / Mac-native adaptive bidirectional tracking"
                if adaptive_tracking
                else config.tracklet.backend
            ),
            "num_trajectories": len(tracking_result.get("trajectories", [])),
            "stats": tracking_stats_payload,
            "files": tracking_result.get("output_files", {}),
        },
        "rule_based_behavior": rule_report.get("summary", {}),
        "unsupervised_behavior": motif_report.get("summary", {}),
        "experiment_metrics": {
            "template": experiment_report.get("template", {}),
            "summary": experiment_report.get("summary", {}),
            "files": experiment_files,
        },
        "frontier_representations": {
            "manual_training_annotation_required": False,
            "visual_ssl": visual_ssl_report.get("summary", {}),
            "bioacoustic_ssl": bioacoustic_ssl_report.get("summary", {}),
            "files": {
                "visual_ssl_report": str(output / "visual_ssl_report.json")
                if (output / "visual_ssl_report.json").exists()
                else None,
                "visual_ssl_embeddings": str(output / "visual_ssl_embeddings.csv")
                if (output / "visual_ssl_embeddings.csv").exists()
                else None,
                "bioacoustic_ssl_report": str(output / "bioacoustic_ssl_report.json")
                if (output / "bioacoustic_ssl_report.json").exists()
                else None,
                "bioacoustic_ssl_units": str(output / "bioacoustic_ssl_units.csv")
                if (output / "bioacoustic_ssl_units.csv").exists()
                else None,
            },
        },
        "validation_review": {
            "selected_review_frames": validation_review["selected_review_frames"],
            "manual_review_required_for_scientific_metrics": True,
            "review_completed": False,
            "manifest": validation_review["manifest"],
        },
        "dashboard": str(dashboard),
        "limitations": [
            (
                "Mac 原生算法使用检测、轻量外观模板和双向一致性融合；可选首帧点击仅用于实例选择，不是训练标注；"
                "复杂遮挡下仍需查看置信度审计。 / The Mac-native algorithm uses detections, lightweight appearance "
                "templates, and bidirectional consistency fusion; optional first-frame clicks select instances and "
                "are not training annotations; inspect confidence audit under complex occlusion."
                if adaptive_tracking
                else
                "当前本地基线使用检测框和质心自动追踪；复杂遮挡下仍需查看置信度审计。 "
                "/ The local baseline uses detector boxes and centroids; inspect confidence audit under complex occlusion."
            ),
            "无监督模体是重复运动模式，不是经过验证的语义行为标签。 "
            "/ Unsupervised motifs are recurring motion patterns, not validated semantic behavior labels.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest["manifest"] = str(manifest_path)
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
