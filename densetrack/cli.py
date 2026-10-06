"""
DenseTrack-AI CLI.

Commands:
    densetrack run       — Run the full AutoLabel MOT pipeline on a video
    densetrack evaluate  — Evaluate predictions against ground-truth
    densetrack export    — Re-export results in a different format
    densetrack complete  — Run the zero-click local workflow and build a dashboard
    densetrack serve     — Start the local upload/process/download workspace
    densetrack build-report — Build a standalone research dashboard
    densetrack prepare-validation — Build sparse pre-filled GT review package
    densetrack evaluate-validation — Evaluate confirmed sparse GT review package
    densetrack discover-behaviors — Discover recurring motifs without labels
    densetrack list-experiment-templates — List bundled behavior experiment templates
    densetrack analyze-experiment — Export template-driven zone/time-bin metrics
    densetrack extract-representations — Export BEAST3D-inspired visual SSL features
    densetrack analyze-audio-ssl — Export Dolph2Vec-inspired bioacoustic SSL units
    densetrack auto-init — Generate no-click initialization seed points
    densetrack udmt-handoff — Build an automatic external UDMT seed package
    densetrack udmt-check — Validate an external UDMT research backend
    densetrack info      — Show video and config info
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="densetrack",
    help="DenseTrack-AI: Foundation-Model AutoLabel MOT for laboratory animal tracking",
    add_completion=False,
)

console = Console()


def _json_csv_outputs(
    output: str | None,
    default_dir: Path,
    json_name: str,
    csv_name: str,
) -> tuple[Path, Path]:
    if not output:
        return default_dir / json_name, default_dir / csv_name
    target = Path(output).expanduser()
    if target.suffix.lower() == ".json":
        return target, target.with_name(csv_name)
    return target / json_name, target / csv_name


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def run(
    video: str = typer.Argument(..., help="Path to input video file"),
    species: str = typer.Option("mouse", help="Species: mouse | insect | zebrafish | dolphin"),
    n_animals: int = typer.Option(4, "--n-animals", "-n", help="Number of animals in video"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    output: str = typer.Option("results/", "--output", "-o", help="Output directory"),
    detection_backend: str | None = typer.Option(
        None, "--detector", help="Override detection backend"
    ),
    optimizer: str | None = typer.Option(
        None,
        "--optimizer",
        help="Override global optimizer: greedy | hungarian | ilp | min_cost_flow | long_term",
    ),
    device: str = typer.Option("auto", help="Device: auto | cpu | cuda | mps"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full AutoLabel MOT pipeline on a video."""
    setup_logging(verbose)

    from densetrack.config import PipelineConfig, Species
    from densetrack.pipeline import Pipeline

    # Load config
    if config:
        cfg = PipelineConfig.from_yaml(config)
    else:
        cfg = PipelineConfig()

    # Override with CLI arguments
    cfg.video_path = video
    cfg.n_animals = n_animals
    cfg.output_dir = output
    cfg.device = device

    if species:
        cfg.species = Species(species)

    if detection_backend:
        cfg.detection.backend = detection_backend
    if optimizer:
        cfg.global_optimizer.method = optimizer

    # Run
    console.print(f"\n[bold cyan]DenseTrack-AI[/bold cyan] — Processing [bold]{video}[/bold]")
    console.print(f"  Species: {species} | Animals: {n_animals} | Device: {cfg.resolve_device()}")
    console.print()

    pipeline = Pipeline(cfg)
    results = pipeline.run()

    # Summary
    console.print("\n[bold green]✅ Processing complete![/bold green]")
    console.print(f"  Output directory: [cyan]{output}[/cyan]")
    for name, path in results.get("output_files", {}).items():
        console.print(f"  {name}: [dim]{path}[/dim]")


@app.command()
def evaluate(
    prediction: str = typer.Argument(..., help="Path to predicted MOT file"),
    ground_truth: str = typer.Argument(..., help="Path to ground-truth MOT file"),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for matching"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate tracking predictions against ground-truth."""
    setup_logging(verbose)

    from densetrack.utils.metrics import compute_mot_metrics, summarize_metrics

    console.print(f"\n[bold cyan]DenseTrack-AI[/bold cyan] — Evaluation")
    console.print(f"  Prediction:   {prediction}")
    console.print(f"  Ground-truth: {ground_truth}")
    console.print(f"  IoU threshold: {iou_threshold}")
    console.print()

    metrics = compute_mot_metrics(ground_truth, prediction, iou_threshold)
    console.print(summarize_metrics(metrics))


@app.command("export")
def export_cmd(
    run_dir: str = typer.Argument(..., help="Path to a previous run's output directory"),
    format: str = typer.Option("mot", "--format", "-f", help="Export format: mot | coco | csv | dlc | sleap"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Minimum confidence filter"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-export results from a previous run in a different format."""
    setup_logging(verbose)
    from densetrack.export.run_reexport import reexport_run

    console.print(f"\n[bold cyan]DenseTrack-AI[/bold cyan] — Export")
    console.print(f"  Run: {run_dir}")
    console.print(f"  Format: {format}")
    console.print(f"  Min confidence: {min_confidence}")

    suffix = {
        "mot": ".mot",
        "csv": ".csv",
        "dlc": ".csv",
        "coco": ".coco.json",
        "sleap": ".sleap.json",
    }.get(format.lower(), f".{format}")
    output_path = output or str(Path(run_dir) / f"reexport_{format.lower()}{suffix}")
    written = reexport_run(
        run_dir=run_dir,
        output_path=output_path,
        fmt=format,
        min_confidence=min_confidence,
    )
    console.print(f"[green]Exported:[/green] [cyan]{written}[/cyan]")


@app.command()
def analyze(
    run_dir: str = typer.Argument(..., help="Path to a previous run's output directory"),
    fps: float = typer.Option(0.0, "--fps", help="Video FPS; 0 = read from run_summary.json if present"),
    activity_speed: float = typer.Option(35.0, "--activity-speed", help="Activity threshold in px/s"),
    contact_distance: float = typer.Option(95.0, "--contact-distance", help="Contact threshold in pixels"),
    output: str | None = typer.Option(None, "--output", "-o", help="Behavior report JSON path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate rule-based behavior statistics from a completed run."""
    setup_logging(verbose)

    import json

    from densetrack.behavior.rule_based import BehaviorConfig, RuleBasedBehaviorAnalyzer

    run_path = Path(run_dir)
    mot_path = run_path / "tracks.mot"
    if not mot_path.exists():
        raise typer.BadParameter(f"tracks.mot not found in {run_dir}")

    resolved_fps = fps
    summary_path = run_path / "run_summary.json"
    if resolved_fps <= 0 and summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        resolved_fps = float(summary.get("video_metadata", {}).get("fps", 0) or 0)
    if resolved_fps <= 0:
        resolved_fps = 15.0

    output_json = Path(output) if output else run_path / "behavior_report.json"
    output_timeline = output_json.with_name("behavior_timeline.csv")
    analyzer = RuleBasedBehaviorAnalyzer(
        BehaviorConfig(
            fps=resolved_fps,
            activity_speed_px_s=activity_speed,
            contact_distance_px=contact_distance,
        )
    )
    report = analyzer.analyze_mot(
        mot_path=mot_path,
        output_json=output_json,
        output_timeline_csv=output_timeline,
    )

    console.print(f"\n[bold cyan]DenseTrack-AI[/bold cyan] — Behavior Analysis")
    console.print(f"  IDs: {report['summary']['num_ids']} | Frames: {report['summary']['num_frames']}")
    console.print(f"  Report: [cyan]{output_json}[/cyan]")
    console.print(f"  Timeline: [cyan]{output_timeline}[/cyan]")


@app.command("discover-behaviors")
def discover_behaviors(
    run_dir: str = typer.Argument(..., help="Path to a previous run's output directory"),
    fps: float = typer.Option(0.0, "--fps", help="Video FPS; 0 = read from run_summary.json"),
    clusters: int = typer.Option(5, "--clusters", "-k", help="Number of unsupervised motifs"),
    window_frames: int = typer.Option(15, "--window-frames", help="Rolling feature window"),
    sample_every: int = typer.Option(3, "--sample-every", help="Cluster every Nth frame"),
    contact_distance: float = typer.Option(
        95.0,
        "--contact-distance",
        help="Distance threshold for close-neighbor feature",
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Behavior clusters JSON path"),
    pose_csv: str | None = typer.Option(
        None,
        "--pose-csv",
        help="Optional poses.csv for geometric/keypoint enrichment / 可选姿态 CSV",
    ),
    feature_mode: str = typer.Option(
        "auto",
        "--feature-mode",
        help="Feature mode: auto | centroid | pose / 特征模式",
    ),
    embedding: str = typer.Option(
        "pca",
        "--embedding",
        help="Embedding: pca | umap / 嵌入方法",
    ),
    clustering: str = typer.Option(
        "kmeans",
        "--clustering",
        help="Clustering: kmeans | hdbscan / 聚类方法",
    ),
    hdbscan_min_cluster_size: int = typer.Option(
        20,
        "--hdbscan-min-cluster-size",
        help="HDBSCAN minimum cluster size / HDBSCAN 最小聚类样本数",
    ),
    hdbscan_min_samples: int | None = typer.Option(
        None,
        "--hdbscan-min-samples",
        help="Optional HDBSCAN density conservatism / 可选 HDBSCAN 密度保守度",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Discover recurring behavior motifs without behavior annotations."""
    setup_logging(verbose)

    import json

    from densetrack.behavior.unsupervised import (
        BehaviorDiscoveryConfig,
        UnsupervisedBehaviorDiscovery,
    )

    run_path = Path(run_dir)
    mot_path = run_path / "tracks.mot"
    if not mot_path.exists():
        raise typer.BadParameter(f"tracks.mot not found in {run_dir}")

    resolved_fps = fps
    summary_path = run_path / "run_summary.json"
    if resolved_fps <= 0 and summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        resolved_fps = float(summary.get("video_metadata", {}).get("fps", 0) or 0)
    if resolved_fps <= 0:
        resolved_fps = 15.0

    output_json = Path(output) if output else run_path / "behavior_clusters.json"
    resolved_pose_csv = Path(pose_csv) if pose_csv else run_path / "poses.csv"
    output_csv = output_json.with_name(f"{output_json.stem}_timeline.csv")
    embedding_image = output_json.with_name(f"{output_json.stem}_embedding.jpg")
    timeline_image = output_json.with_name(f"{output_json.stem}_timeline.jpg")

    discovery = UnsupervisedBehaviorDiscovery(
        BehaviorDiscoveryConfig(
            fps=resolved_fps,
            n_clusters=clusters,
            window_frames=window_frames,
            sample_every=sample_every,
            contact_distance_px=contact_distance,
            pose_csv=str(resolved_pose_csv) if resolved_pose_csv.exists() or pose_csv else None,
            feature_mode=feature_mode,
            embedding_method=embedding,
            clustering_method=clustering,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
        )
    )
    report = discovery.analyze_mot(
        mot_path=mot_path,
        output_json=output_json,
        output_csv=output_csv,
        embedding_image=embedding_image,
        timeline_image=timeline_image,
    )

    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — Unsupervised Behavior Discovery")
    console.print(
        f"  IDs: {report['summary']['num_ids']} | "
        f"Samples: {report['summary']['num_samples']} | "
        f"Clusters: {report['summary']['num_clusters']}"
    )
    if report["summary"].get("num_noise_samples"):
        console.print(
            "  噪声样本 / Noise samples: "
            f"{report['summary']['num_noise_samples']}"
        )
    for cluster in report["clusters"]:
        console.print(
            f"  c{cluster['cluster']}: {cluster['label']} "
            f"({cluster['ratio']:.1%}, {cluster['num_samples']} samples)"
        )
    console.print(f"  Report: [cyan]{output_json}[/cyan]")
    console.print(f"  Timeline CSV: [cyan]{output_csv}[/cyan]")
    console.print(f"  Embedding: [cyan]{embedding_image}[/cyan]")
    console.print(f"  Timeline image: [cyan]{timeline_image}[/cyan]")


@app.command("extract-representations")
def extract_representations(
    run_dir: str = typer.Argument(..., help="Path to a completed run directory / 已完成运行目录"),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory or JSON path / 输出目录或 JSON 路径",
    ),
    dim: int = typer.Option(8, "--dim", help="Embedding dimension / 嵌入维度"),
    window_frames: int = typer.Option(
        15,
        "--window-frames",
        help="Temporal smoothing window / 时间平滑窗口",
    ),
    sample_every: int = typer.Option(
        3,
        "--sample-every",
        help="Sample every Nth tracked frame / 每 N 帧采样",
    ),
    camera_manifest: str | None = typer.Option(
        None,
        "--camera-manifest",
        help="Optional synchronized multi-view camera manifest / 可选多视角相机清单",
    ),
    refresh_dashboard: bool = typer.Option(
        True,
        "--refresh-dashboard/--no-refresh-dashboard",
        help="Refresh dashboard when output stays in the run directory / 刷新运行目录中的面板",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Export annotation-free visual SSL representations from MOT tracks."""
    setup_logging(verbose)

    from densetrack.representation.visual_ssl import (
        VisualSSLRepresentationAnalyzer,
        VisualSSLRepresentationConfig,
    )

    run_path = Path(run_dir).expanduser().resolve()
    if not (run_path / "tracks.mot").exists():
        raise typer.BadParameter(f"tracks.mot not found / 缺少 tracks.mot: {run_path}")
    output_json, output_csv = _json_csv_outputs(
        output,
        run_path,
        "visual_ssl_report.json",
        "visual_ssl_embeddings.csv",
    )
    report = VisualSSLRepresentationAnalyzer(
        VisualSSLRepresentationConfig(
            dim=dim,
            window_frames=window_frames,
            sample_every=sample_every,
            camera_manifest=camera_manifest,
        )
    ).analyze_run(
        run_dir=run_path,
        output_json=output_json,
        output_csv=output_csv,
    )

    if refresh_dashboard and output_json.parent.resolve() == run_path and (run_path / "dashboard.html").exists():
        from densetrack.reporting.static_dashboard import build_static_dashboard

        auto_init_dir = run_path / "auto_init"
        build_static_dashboard(
            run_dir=run_path,
            auto_init_dir=auto_init_dir if auto_init_dir.exists() else None,
        )

    summary = report.get("summary", {})
    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — 视觉 SSL 表征 / Visual SSL Representations")
    console.print("  训练标注 / Training annotation: [green]false[/green]")
    console.print(f"  样本 / Samples: {int(summary.get('num_samples', 0)):,}")
    console.print(f"  个体 / IDs: {int(summary.get('num_ids', 0)):,}")
    console.print(f"  嵌入维度 / Embedding: {int(summary.get('embedding_dim', 0))}-D")
    console.print(f"  多视角状态 / Multi-view: {summary.get('multiview_status', 'single_view_only')}")
    console.print(f"  报告 / Report: [cyan]{output_json}[/cyan]")
    console.print(f"  嵌入 CSV / Embeddings CSV: [cyan]{output_csv}[/cyan]")


@app.command("list-experiment-templates")
def list_experiment_templates_cmd() -> None:
    """List bundled bilingual behavior experiment templates."""
    from densetrack.experiments import list_experiment_templates

    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — 实验范式模板 / Experiment Templates")
    for template in list_experiment_templates():
        console.print(
            f"  [bold]{template.template_id}[/bold] — "
            f"{template.name_zh} / {template.name_en}"
        )
        console.print(
            f"    {template.category_zh} / {template.category_en}; "
            f"{len(template.zones)} zones; primary={template.primary_zone_id}"
        )


@app.command("analyze-experiment")
def analyze_experiment(
    run_dir: str = typer.Argument(..., help="Completed run directory / 已完成运行目录"),
    template: str = typer.Option(
        "open_field",
        "--template",
        "-t",
        help="Experiment template id; run list-experiment-templates / 实验范式模板 ID",
    ),
    time_bin_s: float = typer.Option(60.0, "--time-bin-s", help="Time-bin size / 时间箱秒数"),
    fps: float = typer.Option(0.0, "--fps", help="Override FPS; 0 = use run summary / 覆盖 FPS"),
    refresh_dashboard: bool = typer.Option(
        True,
        "--refresh-dashboard/--no-refresh-dashboard",
        help="Refresh dashboard when output stays in the run directory / 刷新运行目录中的面板",
    ),
) -> None:
    """Export Tracking-Master-style zone and time-bin experiment metrics."""
    from densetrack.experiments import ExperimentMetricsAnalyzer, ExperimentMetricsConfig

    run_path = Path(run_dir).expanduser().resolve()
    if not (run_path / "tracks.mot").exists():
        raise typer.BadParameter(f"tracks.mot not found / 缺少 tracks.mot: {run_path}")
    try:
        report = ExperimentMetricsAnalyzer(
            ExperimentMetricsConfig(template_id=template, fps=fps, time_bin_s=time_bin_s)
        ).analyze_run(run_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if refresh_dashboard and (run_path / "dashboard.html").exists():
        from densetrack.reporting.static_dashboard import build_static_dashboard

        auto_init_dir = run_path / "auto_init"
        build_static_dashboard(
            run_dir=run_path,
            auto_init_dir=auto_init_dir if auto_init_dir.exists() else None,
        )

    template_payload = report["template"]
    summary = report["summary"]
    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — 实验指标 / Experiment Metrics")
    console.print(
        f"  模板 / Template: {template_payload['name_zh']} / {template_payload['name_en']} "
        f"({template_payload['id']})"
    )
    console.print(f"  个体 / IDs: {summary['num_ids']} | 区域 / Zones: {summary['num_zones']}")
    console.print(
        "  主区域 / Primary zone: "
        f"{summary['primary_zone_id']} | {summary['primary_zone_time_s']:.1f}s | "
        f"{summary['primary_zone_entries']} entries"
    )
    console.print(f"  报告 / Report: [cyan]{run_path / 'experiment_metrics.json'}[/cyan]")
    console.print(f"  区域 CSV / Zone CSV: [cyan]{run_path / 'experiment_zone_metrics.csv'}[/cyan]")
    console.print(f"  时间箱 CSV / Time-bin CSV: [cyan]{run_path / 'experiment_time_bins.csv'}[/cyan]")


@app.command("contact-events")
def contact_events_cmd(
    run_dir: str = typer.Argument(..., help="Completed run directory / 已完成运行目录"),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory; default writes into the run directory / 输出目录，默认写回运行目录",
    ),
    contact_distance: float = typer.Option(
        75.0, "--contact-distance", help="Contact centroid distance in px / 接触质心距离阈值"
    ),
    top_k: int = typer.Option(
        3, "--top-k", help="Identity hypotheses kept per event / 每事件保留的身份假设数"
    ),
    recovery_window: int = typer.Option(
        30, "--recovery-window", help="Frames to search for post-event split / 拆分搜索帧窗口"
    ),
    refresh_dashboard: bool = typer.Option(
        True,
        "--refresh-dashboard/--no-refresh-dashboard",
        help="Refresh dashboard when output stays in the run directory / 刷新运行目录中的面板",
    ),
) -> None:
    """Recompute the auditable contact-event layer for an existing run.

    Reads tracks.mot plus labels.sleap.json provenance as immutable inputs and
    writes contact_events.json/csv with merge/split records, top-K
    incoming-to-outgoing identity hypotheses, and best-vs-second-best margins.
    """
    from densetrack.config import ContactEventConfig
    from densetrack.export.run_reexport import load_mot_rows, load_run_metadata
    from densetrack.tracking.contact_events import (
        load_sleap_source_map,
        write_contact_events_report,
    )

    run_path = Path(run_dir).expanduser().resolve()
    if not (run_path / "tracks.mot").exists():
        raise typer.BadParameter(f"tracks.mot not found / 缺少 tracks.mot: {run_path}")

    metadata = load_run_metadata(run_path)
    n_animals = metadata.get("config", {}).get("n_animals")
    mot_rows = load_mot_rows(run_path / "tracks.mot")
    if not mot_rows:
        raise typer.BadParameter(f"No MOT rows found / MOT 结果为空: {run_path}")
    source_map = load_sleap_source_map(run_path / "labels.sleap.json")

    output_path = Path(output).expanduser().resolve() if output else run_path
    config = ContactEventConfig(
        contact_distance_px=contact_distance,
        top_k=top_k,
        recovery_window=recovery_window,
    )
    paths, summary = write_contact_events_report(
        output_path,
        config=config,
        mot_rows=mot_rows,
        source_map=source_map,
        n_animals=n_animals,
        source_description=f"cli_recompute:{run_path.name}",
    )

    if refresh_dashboard and output_path == run_path and (run_path / "dashboard.html").exists():
        from densetrack.reporting.static_dashboard import build_static_dashboard

        auto_init_dir = run_path / "auto_init"
        build_static_dashboard(
            run_dir=run_path,
            auto_init_dir=auto_init_dir if auto_init_dir.exists() else None,
        )

    uncertainty = summary.get("uncertainty_counts", {})
    kind_counts = summary.get("event_kind_counts", {})
    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — 接触事件审计 / Contact-event audit")
    console.print(f"  事件 / Events: {summary.get('total_events', 0)}")
    console.print(
        "  测量歧义/仅接近 / measurement-ambiguous/proximity-only: "
        f"{kind_counts.get('measurement_ambiguous', 0)}/{kind_counts.get('proximity_only', 0)}"
    )
    console.print(
        "  确定/歧义/不确定/未解析 / confident/ambiguous/uncertain/unresolved: "
        f"{uncertainty.get('confident', 0)}/{uncertainty.get('ambiguous', 0)}/"
        f"{uncertainty.get('uncertain', 0)}/{summary.get('unresolved_events', 0)}"
    )
    console.print(
        "  标签反转段(审核单位)/ Inversion episodes (review units): "
        f"{summary.get('inversion_episode_count', 0)}"
    )
    console.print(
        "  原始不一致事件(会级联)/ Raw disagreements (cascading): "
        f"{summary.get('tracker_disagreement_events', 0)}"
    )
    console.print(f"  报告 / Report: [cyan]{paths['contact_events_json']}[/cyan]")
    console.print(f"  表格 / CSV: [cyan]{paths['contact_events_csv']}[/cyan]")


@app.command("analyze-audio-ssl")
def analyze_audio_ssl(
    audio: str = typer.Argument(..., help="Path to WAV audio / WAV 音频路径"),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory or JSON path / 输出目录或 JSON 路径",
    ),
    species: str = typer.Option("dolphin", "--species", help="Species/domain label / 物种标签"),
    window_ms: float = typer.Option(120.0, "--window-ms", help="Window size in ms / 窗口毫秒"),
    hop_ms: float = typer.Option(40.0, "--hop-ms", help="Hop size in ms / 步长毫秒"),
    bands: int = typer.Option(16, "--bands", help="Number of spectral bands / 频带数量"),
    codebook_size: int = typer.Option(12, "--codebook-size", help="Acoustic unit count / 声学单元数量"),
    whistle_min_frequency: float = typer.Option(
        1000.0,
        "--whistle-min-frequency",
        help="Minimum peak frequency for whistle candidates / 哨声候选最低峰值频率",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Export species-specific bioacoustic SSL units from unlabeled WAV audio."""
    setup_logging(verbose)

    from densetrack.audio.bioacoustic_ssl import BioacousticSSLAnalyzer, BioacousticSSLConfig

    audio_path = Path(audio).expanduser().resolve()
    if not audio_path.exists():
        raise typer.BadParameter(f"WAV not found / 缺少 WAV: {audio_path}")
    output_json, output_csv = _json_csv_outputs(
        output,
        audio_path.parent,
        "bioacoustic_ssl_report.json",
        "bioacoustic_ssl_units.csv",
    )
    report = BioacousticSSLAnalyzer(
        BioacousticSSLConfig(
            species=species,
            window_ms=window_ms,
            hop_ms=hop_ms,
            n_bands=bands,
            codebook_size=codebook_size,
            whistle_min_frequency_hz=whistle_min_frequency,
        )
    ).analyze_wav(
        audio_path=audio_path,
        output_json=output_json,
        output_csv=output_csv,
    )

    summary = report.get("summary", {})
    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — 音频 SSL / Bioacoustic SSL")
    console.print("  训练标注 / Training annotation: [green]false[/green]")
    console.print(f"  物种 / Species: {summary.get('species', species)}")
    console.print(f"  窗口 / Windows: {int(summary.get('num_windows', 0)):,}")
    console.print(f"  声学单元 / Acoustic units: {int(summary.get('codebook_units', 0)):,}")
    console.print(
        "  哨声候选 / Whistle candidates: "
        f"{int(summary.get('whistle_candidate_windows', 0)):,}"
    )
    console.print(f"  报告 / Report: [cyan]{output_json}[/cyan]")
    console.print(f"  单元 CSV / Units CSV: [cyan]{output_csv}[/cyan]")


@app.command("build-report")
def build_report(
    run_dir: str = typer.Argument(..., help="Path to a completed run's output directory"),
    auto_init_dir: str | None = typer.Option(
        None,
        "--auto-init-dir",
        help="Optional auto-init result directory to include in the dashboard",
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Dashboard HTML path"),
) -> None:
    """Build a standalone researcher-facing HTML result dashboard."""
    from densetrack.reporting.static_dashboard import build_static_dashboard

    written = build_static_dashboard(
        run_dir=run_dir,
        output_path=output,
        auto_init_dir=auto_init_dir,
    )
    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — Research Dashboard")
    console.print(f"  Dashboard: [cyan]{written}[/cyan]")


@app.command("prepare-validation")
def prepare_validation(
    run_dir: str = typer.Argument(..., help="Path to a completed run / 已完成运行目录"),
    frames: int = typer.Option(100, "--frames", "-n", help="Review frame count / 审核帧数量"),
    video: str | None = typer.Option(None, "--video", help="Override source video path / 覆盖源视频路径"),
    output: str | None = typer.Option(None, "--output", "-o", help="Review package directory / 审核包目录"),
) -> None:
    """Create risk-prioritized pre-filled frames for sparse scientific GT review."""
    from densetrack.validation.review_package import build_validation_review_package

    result = build_validation_review_package(
        run_dir=run_dir,
        output_dir=output,
        n_frames=frames,
        video_path=video,
    )
    dashboard = None
    run_path = Path(run_dir)
    if output is None and (run_path / "dashboard.html").exists():
        from densetrack.reporting.static_dashboard import build_static_dashboard

        auto_init_dir = run_path / "auto_init"
        dashboard = build_static_dashboard(
            run_dir=run_path,
            auto_init_dir=auto_init_dir if auto_init_dir.exists() else None,
        )
    files = result["files"]
    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — 稀疏 GT 审核包 / Sparse GT Review Package")
    console.print(f"  审核帧 / Review frames: {result['selected_review_frames']}")
    console.print("  追踪人工标注 / Tracking manual annotation: [green]false[/green]")
    console.print("  科学指标人工审核 / Manual review for scientific metrics: [yellow]required[/yellow]")
    console.print(f"  联系表 / Contact sheet: [cyan]{files['review_grid']}[/cyan]")
    console.print(f"  修正模板 / Correction template: [cyan]{files['corrected_gt_template']}[/cyan]")
    console.print(f"  说明 / Instructions: [cyan]{files['instructions']}[/cyan]")
    if dashboard:
        console.print(f"  已刷新面板 / Refreshed dashboard: [cyan]{dashboard}[/cyan]")


@app.command("evaluate-validation")
def evaluate_validation(
    package_dir: str = typer.Argument(..., help="Sparse validation package / 稀疏验证审核包"),
    ground_truth: str | None = typer.Option(
        None,
        "--ground-truth",
        "-g",
        help="Reviewed GT MOT; default package/reviewed_gt.mot / 已审核 GT",
    ),
    confirm_reviewed: bool = typer.Option(
        False,
        "--confirm-reviewed",
        help="Confirm GT was manually reviewed / 确认 GT 已人工审核",
    ),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold / IoU 阈值"),
) -> None:
    """Evaluate sparse GT only after explicit manual-review confirmation."""
    from densetrack.validation.review_package import evaluate_validation_review_package

    try:
        result = evaluate_validation_review_package(
            package_dir=package_dir,
            ground_truth=ground_truth,
            confirm_reviewed=confirm_reviewed,
            iou_threshold=iou_threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    from densetrack.utils.metrics import summarize_metrics

    run_dir = Path(package_dir).expanduser().resolve().parent
    if (run_dir / "dashboard.html").exists():
        from densetrack.reporting.static_dashboard import build_static_dashboard

        auto_init_dir = run_dir / "auto_init"
        build_static_dashboard(
            run_dir=run_dir,
            auto_init_dir=auto_init_dir if auto_init_dir.exists() else None,
        )
    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — 稀疏审核评估 / Sparse Reviewed Evaluation")
    console.print(summarize_metrics(result["metrics"]))
    console.print("[yellow]  指标仅限已审核稀疏帧 / Metrics apply only to reviewed sparse frames.[/yellow]")
    for name, path in result["files"].items():
        console.print(f"  {name}: [cyan]{path}[/cyan]")


@app.command("auto-init")
def auto_init(
    video: str = typer.Argument(..., help="Path to input video file"),
    n_animals: int = typer.Option(
        0,
        "--n-animals",
        "-n",
        help="Number of animals; 0 = use config value",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    detector: str | None = typer.Option(
        None,
        "--detector",
        help="Detector backend for initialization: dark_object | background_sub",
    ),
    output: str = typer.Option(
        "results/auto_init",
        "--output",
        "-o",
        help="Output directory for initialization JSON and image",
    ),
    max_frames: int = typer.Option(
        300,
        "--max-frames",
        help="Number of initial frames to scan",
    ),
    sample_every: int = typer.Option(
        5,
        "--sample-every",
        help="Evaluate every Nth frame while scanning",
    ),
    start_frame: int = typer.Option(0, "--start-frame", help="Frame to start scanning"),
    min_score: float = typer.Option(
        0.55,
        "--min-score",
        help="Minimum score required for status=ok",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate detector-derived seed points to replace manual first-frame clicks."""
    setup_logging(verbose)

    from densetrack.config import PipelineConfig
    from densetrack.initialization.auto_seed import (
        AutoInitConfig,
        find_auto_initialization_points,
        write_auto_init_outputs,
    )

    cfg = PipelineConfig.from_yaml(config) if config else PipelineConfig()
    resolved_n = n_animals if n_animals > 0 else cfg.n_animals
    detector_backend = detector or cfg.detection.backend

    result = find_auto_initialization_points(
        video_path=video,
        n_animals=resolved_n,
        detection_config=cfg.detection,
        auto_config=AutoInitConfig(
            detector_backend=detector_backend,
            max_frames=max_frames,
            sample_every=sample_every,
            start_frame=start_frame,
            resize_height=cfg.processing_resolution,
            min_score=min_score,
        ),
    )
    files = write_auto_init_outputs(result, output)

    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — Auto Initialization")
    console.print(f"  Video: [cyan]{video}[/cyan]")
    console.print(f"  Detector: {detector_backend} | Animals: {resolved_n}")
    console.print(
        f"  Status: [bold]{result.payload['status']}[/bold] | "
        f"Frame: {result.payload.get('selected_frame')} | "
        f"Score: {result.payload.get('score', 0):.3f}"
    )
    console.print(f"  Points: {len(result.payload.get('points', []))}/{resolved_n}")
    for name, path in files.items():
        console.print(f"  {name}: [dim]{path}[/dim]")


@app.command("udmt-handoff")
def udmt_handoff(
    video: str = typer.Argument(..., help="Path to input video file"),
    n_animals: int = typer.Option(
        0,
        "--n-animals",
        "-n",
        help="Number of animals; 0 = use config value",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    detector: str | None = typer.Option(
        None,
        "--detector",
        help="Detector backend for initialization: dark_object | background_sub",
    ),
    initialization_json: str | None = typer.Option(
        None,
        "--initialization-json",
        help="Reuse an existing initialization_points.json instead of rescanning video",
    ),
    output: str = typer.Option(
        "results/udmt_handoff",
        "--output",
        "-o",
        help="Output directory for the external UDMT handoff package",
    ),
    max_frames: int = typer.Option(
        1,
        "--max-frames",
        help="Initial frames to scan; keep 1 for direct UDMT first-frame drop-in",
    ),
    sample_every: int = typer.Option(5, "--sample-every", help="Evaluate every Nth frame"),
    start_frame: int = typer.Option(0, "--start-frame", help="Frame to start scanning"),
    min_score: float = typer.Option(0.55, "--min-score", help="Minimum score for no-click readiness"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build a file handoff package that replaces UDMT first-frame clicks when possible."""
    setup_logging(verbose)

    from densetrack.config import PipelineConfig
    from densetrack.external.udmt import build_udmt_handoff_package
    from densetrack.initialization.auto_seed import (
        AutoInitConfig,
        find_auto_initialization_points,
        write_auto_init_outputs,
    )

    cfg = PipelineConfig.from_yaml(config) if config else PipelineConfig()
    resolved_n = n_animals if n_animals > 0 else cfg.n_animals
    output_path = Path(output)
    resolved_initialization = initialization_json
    if not resolved_initialization:
        detector_backend = detector or cfg.detection.backend
        result = find_auto_initialization_points(
            video_path=video,
            n_animals=resolved_n,
            detection_config=cfg.detection,
            auto_config=AutoInitConfig(
                detector_backend=detector_backend,
                max_frames=max_frames,
                sample_every=sample_every,
                start_frame=start_frame,
                resize_height=cfg.processing_resolution,
                min_score=min_score,
            ),
        )
        files = write_auto_init_outputs(result, output_path)
        resolved_initialization = files["json"]

    manifest = build_udmt_handoff_package(
        video_path=video,
        initialization_json=resolved_initialization,
        output_dir=output_path,
    )
    initialization = manifest["initialization"]
    boundary = manifest["automation_boundary"]
    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — UDMT 自动衔接包 / External Handoff")
    console.print(f"  视频 / Video: [cyan]{video}[/cyan]")
    console.print(f"  状态 / Status: [bold]{initialization['status_display']}[/bold]")
    console.print(
        "  种子点 / Seed points: "
        f"{initialization['seed_points']}/{initialization['requested_animals']}"
    )
    console.print(
        "  第一帧人工点击 / Manual first-frame clicks: "
        f"[bold]{str(boundary['manual_first_frame_click_required']).lower()}[/bold]"
    )
    console.print(
        "  UDMT 首帧直写 / Direct first-frame drop-in: "
        f"[bold]{str(manifest['upstream_contract']['direct_first_frame_drop_in_ready']).lower()}[/bold]"
    )
    for name, path in manifest["files"].items():
        if path:
            console.print(f"  {name}: [dim]{path}[/dim]")


@app.command("udmt-check")
def udmt_check(
    udmt_home: str | None = typer.Option(
        None,
        "--udmt-home",
        envvar="UDMT_HOME",
        help="External UDMT checkout path; can also be set with UDMT_HOME",
    ),
    udmt_python: str = typer.Option(
        "python",
        "--udmt-python",
        envvar="UDMT_PYTHON",
        help="Python executable for the UDMT environment",
    ),
    accept_license: bool = typer.Option(
        False,
        "--accept-license",
        help="Confirm UDMT's academic/non-profit non-commercial research license boundary",
    ),
    check_import: bool = typer.Option(
        True,
        "--check-import/--skip-import-check",
        help="Run a lightweight 'import udmt' check using the UDMT Python executable",
    ),
    cuda_visible_devices: str | None = typer.Option(
        None,
        "--cuda-visible-devices",
        help="Optional CUDA_VISIBLE_DEVICES value for UDMT GUI launch",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Validate an external UDMT installation without bundling it."""
    setup_logging(verbose)

    from densetrack.external.udmt import UDMTAdapter, UDMTExternalConfig

    if not udmt_home:
        raise typer.BadParameter("Set --udmt-home or UDMT_HOME to the external UDMT checkout.")

    adapter = UDMTAdapter(
        UDMTExternalConfig(
            udmt_home=udmt_home,
            python_executable=udmt_python,
            license_accepted=accept_license,
            cuda_visible_devices=cuda_visible_devices,
        )
    )
    report = adapter.validate(check_import=check_import)

    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — UDMT External Backend Check")
    console.print(f"  UDMT_HOME: [cyan]{report['udmt_home']}[/cyan]")
    console.print(f"  Python:    [cyan]{report['python_executable']}[/cyan]")
    console.print(f"  Launch:    [dim]{report['launch_command']}[/dim]")

    for warning in report["warnings"]:
        console.print(f"  [yellow]Warning:[/yellow] {warning}")
    for error in report["errors"]:
        console.print(f"  [red]Error:[/red] {error}")

    if not report["ok"]:
        raise typer.Exit(1)
    console.print("  [green]Status:[/green] UDMT external environment looks usable.")


@app.command("udmt-import")
def udmt_import(
    tracks: str = typer.Argument(..., help="UDMT-derived MOT-like track file"),
    output: str = typer.Option("results/udmt_import", "--output", "-o", help="DenseTrack output dir"),
    video_width: int = typer.Option(0, "--video-width", help="Optional source video width"),
    video_height: int = typer.Option(0, "--video-height", help="Optional source video height"),
    fps: float = typer.Option(0.0, "--fps", help="Optional source video FPS"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Import UDMT-derived MOT tracks and re-export DenseTrack labels."""
    setup_logging(verbose)

    from densetrack.external.udmt import UDMTAdapter, UDMTExternalConfig

    video_metadata = {}
    if video_width > 0:
        video_metadata["width"] = video_width
    if video_height > 0:
        video_metadata["height"] = video_height
    if fps > 0:
        video_metadata["fps"] = fps

    adapter = UDMTAdapter(UDMTExternalConfig(udmt_home="."))
    summary = adapter.import_mot_tracks(
        source_path=tracks,
        output_dir=output,
        video_metadata=video_metadata,
    )

    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — UDMT Import")
    console.print(f"  Tracks: [cyan]{tracks}[/cyan]")
    console.print(f"  Output: [cyan]{output}[/cyan]")
    console.print(
        "  Imported: "
        f"{summary['tracking_summary']['num_tracks']} tracks, "
        f"{summary['tracking_summary']['num_rows']} rows"
    )
    for name, path in summary["output_files"].items():
        console.print(f"  {name}: [dim]{path}[/dim]")


@app.command()
def info(
    video: str | None = typer.Option(None, help="Video file to inspect"),
    config: str | None = typer.Option(None, help="Config file to inspect"),
) -> None:
    """Show video metadata and/or config info."""
    setup_logging(False)

    if video:
        from densetrack.utils.video import read_video_metadata

        meta = read_video_metadata(video)
        console.print(f"\n[bold]Video Info:[/bold]")
        console.print(f"  {meta}")

    if config:
        from densetrack.config import PipelineConfig

        cfg = PipelineConfig.from_yaml(config)
        console.print(f"\n[bold]Config:[/bold]")
        console.print(f"  Species:    {cfg.species}")
        console.print(f"  Detection:  {cfg.detection.backend}")
        console.print(f"  Pose:       {cfg.pose.backend}")
        console.print(f"  Tracking:   {cfg.tracklet.backend}")
        console.print(f"  ReID:       {'enabled' if cfg.reid.enabled else 'disabled'}")
        console.print(f"  Optimizer:  {cfg.global_optimizer.method}")
        console.print(f"  Exports:    {', '.join(cfg.export.formats)}")

    if not video and not config:
        from densetrack import __version__

        console.print(f"\n[bold cyan]DenseTrack-AI[/bold cyan] v{__version__}")
        console.print("  Foundation-Model AutoLabel MOT for laboratory animal tracking")
        console.print("  Use --help for usage information")


@app.command()
def complete(
    video: str = typer.Argument(..., help="Path to input video file / 输入视频路径"),
    config: str = typer.Option(
        "configs/mice_black_topview_mac_adaptive.yaml",
        "--config",
        "-c",
        help="Pipeline YAML config / 流程配置文件",
    ),
    output: str = typer.Option(
        "results/complete_run",
        "--output",
        "-o",
        help="Output directory / 输出目录",
    ),
    n_animals: int = typer.Option(
        0,
        "--n-animals",
        "-n",
        help="Override number of animals; 0 = use config / 动物数量，0 表示使用配置",
    ),
    clusters: int = typer.Option(6, "--clusters", "-k", help="Unsupervised motif clusters / 无监督模体数量"),
    validation_frames: int = typer.Option(
        100,
        "--validation-frames",
        help="Pre-filled sparse GT review frame count / 预填 GT 审核帧数量",
    ),
    experiment_template: str | None = typer.Option(
        None,
        "--experiment-template",
        help=(
            "Experiment template id; default uses config, use none to disable / "
            "实验范式模板 ID，默认使用配置，none 表示关闭"
        ),
    ),
    experiment_time_bin_s: float | None = typer.Option(
        None,
        "--experiment-time-bin-s",
        help="Experiment time-bin size in seconds / 实验时间箱秒数",
    ),
    optimizer: str | None = typer.Option(
        None,
        "--optimizer",
        help="Global optimizer: greedy | hungarian | ilp | min_cost_flow | long_term / 全局优化器",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run zero-click tracking, analysis, motif discovery, and dashboard generation."""
    setup_logging(verbose)

    from densetrack.config import PipelineConfig
    from densetrack.workflows.complete_local import run_complete_local

    cfg = PipelineConfig.from_yaml(config)
    if n_animals > 0:
        cfg.n_animals = n_animals
    if optimizer:
        cfg.global_optimizer.method = optimizer

    console.print("\n[bold cyan]DenseTrack-AI[/bold cyan] — 零点击完整流程 / Zero-click Complete Workflow")
    console.print(f"  视频 / Video: [cyan]{video}[/cyan]")
    console.print(f"  动物数量 / Animals: {cfg.n_animals}")
    result = run_complete_local(
        video_path=video,
        pipeline_config=cfg,
        output_dir=output,
        motif_clusters=clusters,
        validation_frames=validation_frames,
        experiment_template=experiment_template,
        experiment_time_bin_s=experiment_time_bin_s,
    )
    init = result["automatic_initialization"]
    tracking = result["tracking"]
    motifs = result["unsupervised_behavior"]
    console.print("\n[bold green]处理完成 / Processing complete[/bold green]")
    console.print(
        f"  自动初始化 / Auto-init: {init['status']} | "
        f"{init['points']}/{init['requested_animals']} points | score={float(init['score'] or 0):.3f}"
    )
    console.print(f"  轨迹 / Trajectories: {tracking['num_trajectories']}")
    console.print(f"  行为模体 / Behavior motifs: {motifs.get('num_clusters', 0)}")
    if result.get("experiment_metrics", {}).get("summary"):
        experiment = result["experiment_metrics"]
        template = experiment.get("template", {})
        summary = experiment.get("summary", {})
        console.print(
            "  实验范式 / Experiment: "
            f"{template.get('name_zh', '')} / {template.get('name_en', '')} | "
            f"{summary.get('primary_zone_time_s', 0):.1f}s primary zone"
        )
    console.print(f"  GT 审核包 / GT review package: {result['validation_review']['selected_review_frames']} frames")
    console.print(f"  面板 / Dashboard: [cyan]{result['dashboard']}[/cyan]")
    console.print(f"  清单 / Manifest: [cyan]{result['manifest']}[/cyan]")


@app.command()
def serve(
    workspace: str = typer.Option(
        "results/web_workspace",
        "--workspace",
        "-w",
        help="Web job workspace / Web 任务目录",
    ),
    config: str = typer.Option(
        "configs/mice_black_topview_mac_adaptive.yaml",
        "--config",
        "-c",
        help="Pipeline YAML config / 流程配置文件",
    ),
    host: str = typer.Option("127.0.0.1", help="Bind host / 监听地址"),
    port: int = typer.Option(8765, help="Bind port / 监听端口"),
    max_upload_mb: int = typer.Option(2048, "--max-upload-mb", help="Upload size limit / 上传大小限制"),
) -> None:
    """Start the local bilingual upload, process, and download workspace."""
    from densetrack.webapp import serve_webapp

    serve_webapp(
        workspace=workspace,
        config_path=config,
        host=host,
        port=port,
        max_upload_mb=max_upload_mb,
    )


if __name__ == "__main__":
    app()
