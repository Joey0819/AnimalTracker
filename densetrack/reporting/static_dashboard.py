"""Generate a standalone researcher-facing HTML dashboard for a DenseTrack run."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

import cv2

__all__ = ["build_static_dashboard"]


def build_static_dashboard(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    auto_init_dir: str | Path | None = None,
) -> Path:
    """Build a standalone HTML result dashboard inside a completed run directory."""
    run_path = Path(run_dir).expanduser().resolve()
    if not (run_path / "tracks.mot").exists():
        raise FileNotFoundError(f"tracks.mot not found in run directory: {run_path}")

    summary = _read_json(run_path / "run_summary.json")
    confidence = _read_json(run_path / "confidence_report.json")
    behavior = _read_json(run_path / "behavior_report.json")
    motifs = _read_json(run_path / "behavior_clusters.json")
    advanced_motifs = _read_json(run_path / "behavior_clusters_umap_hdbscan.json")
    identity_risk = _read_json(run_path / "identity_risk.json")
    identity_continuity = _read_json(run_path / "identity_continuity.json")
    contact_events = _read_json(run_path / "contact_events.json")
    visual_ssl = _read_json(run_path / "visual_ssl_report.json")
    bioacoustic_ssl = _read_json(run_path / "bioacoustic_ssl_report.json")
    experiment_metrics = _read_json(run_path / "experiment_metrics.json")
    validation = _read_json(run_path / "validation_review" / "validation_manifest.json")
    validation_evaluation = _read_json(run_path / "validation_review" / "evaluation_report.json")
    auto_init = _copy_auto_init_assets(auto_init_dir, run_path)
    browser_preview = _ensure_browser_preview(run_path)

    output = Path(output_path).expanduser().resolve() if output_path else run_path / "dashboard.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _dashboard_html(
            run_path=run_path,
            summary=summary,
            confidence=confidence,
            behavior=behavior,
            motifs=motifs,
            advanced_motifs=advanced_motifs,
            identity_risk=identity_risk,
            identity_continuity=identity_continuity,
            contact_events=contact_events,
            visual_ssl=visual_ssl,
            bioacoustic_ssl=bioacoustic_ssl,
            experiment_metrics=experiment_metrics,
            validation=validation,
            validation_evaluation=validation_evaluation,
            auto_init=auto_init,
            browser_preview=browser_preview,
        ),
        encoding="utf-8",
    )
    return output


def _dashboard_html(
    run_path: Path,
    summary: dict[str, Any],
    confidence: dict[str, Any],
    behavior: dict[str, Any],
    motifs: dict[str, Any],
    advanced_motifs: dict[str, Any],
    identity_risk: dict[str, Any],
    identity_continuity: dict[str, Any],
    contact_events: dict[str, Any],
    visual_ssl: dict[str, Any],
    bioacoustic_ssl: dict[str, Any],
    experiment_metrics: dict[str, Any],
    validation: dict[str, Any],
    validation_evaluation: dict[str, Any],
    auto_init: dict[str, Any],
    browser_preview: str | None,
) -> str:
    video_meta = summary.get("video_metadata", {})
    fixed_stats = summary.get("fixed_count_stats") or {}
    provenance = summary.get("provenance") or {}
    tracking_backend = str(provenance.get("tracklet_backend", "unknown"))
    confidence_summary = confidence.get("summary", {})
    motif_summary = motifs.get("summary", {})
    per_id = behavior.get("per_id", {})

    total_frames = int(video_meta.get("total_frames", 0) or 0)
    fps = float(video_meta.get("fps", 0) or 0)
    duration = total_frames / fps if fps > 0 else 0
    num_ids = int(
        motif_summary.get("num_ids")
        or confidence_summary.get("total_trajectories")
        or len(per_id)
        or 0
    )
    observed = float(fixed_stats.get("avg_observed_tracks_per_frame", 0) or 0)
    mostly_observed = float(fixed_stats.get("mostly_observed_ratio", 0) or 0)
    adaptive_algorithm = str(fixed_stats.get("algorithm", ""))
    adaptive_agreement = float(fixed_stats.get("bidirectional_agreement_ratio", 0) or 0)
    appearance_ratio = float(fixed_stats.get("appearance_descriptor_ratio", 0) or 0)
    identity_stability = float(fixed_stats.get("identity_stability_score", 0) or 0)
    motif_count = int(motif_summary.get("num_clusters", 0) or 0)
    motif_samples = int(motif_summary.get("num_samples", 0) or 0)
    motif_feature_mode = str(motif_summary.get("feature_mode", "centroid_bbox"))
    motif_feature_display = (
        "姿态增强 / pose enriched"
        if motif_feature_mode == "pose_enriched"
        else "质心框 / centroid + bbox"
    )
    advanced_motif_summary = advanced_motifs.get("summary", {})
    advanced_motif_status = ""
    if advanced_motif_summary:
        advanced_motif_status = (
            '<span class="status">'
            "UMAP + HDBSCAN: "
            f"{int(advanced_motif_summary.get('num_clusters', 0) or 0)} 有效模体 / motifs, "
            f"{int(advanced_motif_summary.get('num_noise_samples', 0) or 0):,} 噪声 / noise"
            "</span>"
        )
    validation_frames = int(validation.get("selected_review_frames", 0) or 0)
    identity_risk_section = _identity_risk_section(identity_risk)
    identity_continuity_section = _identity_continuity_section(identity_continuity)
    contact_events_section = _contact_events_section(contact_events)
    adaptive_status = ""
    if adaptive_algorithm.startswith("mac_adaptive"):
        adaptive_status_class = "good" if identity_stability >= 0.90 else "warn"
        adaptive_status = (
            f'<span class="status {adaptive_status_class}">'
            "Mac 自适应双向追踪 / Mac adaptive bidirectional: "
            f"{adaptive_agreement * 100:.1f}% 一致 / agreement, "
            f"{identity_stability * 100:.1f}% 连续性代理 / continuity proxy, "
            f"{appearance_ratio * 100:.1f}% 外观模板 / appearance"
            "</span>"
        )
    tracking_status_label = (
        "Mac 自适应双向追踪 / Mac adaptive bidirectional"
        if adaptive_algorithm.startswith("mac_adaptive")
        else "无标注基线 / annotation-free baseline"
    )

    downloads = _download_links(run_path)
    per_id_rows = _per_id_rows(per_id)
    motif_rows = _motif_rows(motifs.get("clusters", []))
    confidence_rows = _confidence_rows(confidence)
    representation_section = _representation_section(run_path, visual_ssl, bioacoustic_ssl)
    experiment_section = _experiment_section(run_path, experiment_metrics)
    readiness_section = _readiness_section(
        auto_init=auto_init,
        browser_preview=browser_preview,
        validation=validation,
        validation_evaluation=validation_evaluation,
        identity_risk=identity_risk,
        adaptive_agreement=adaptive_agreement,
        identity_stability=identity_stability,
        observed=observed,
        num_ids=num_ids,
    )

    auto_init_section = ""
    if auto_init:
        click_assisted = auto_init.get("initialization_mode") == "manual_first_frame_click"
        initialization_title = (
            "首帧点击辅助种子 / First-frame click-assisted seeds"
            if click_assisted
            else "零点击种子点选择 / No-click seed selection"
        )
        initialization_status = (
            "训练标注 / training annotation: false · 首帧点击 / first-frame clicks: true"
            if click_assisted
            else "人工标注 / manual annotation: false"
        )
        auto_init_section = f"""
        <section class="media-panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">初始化 / Initialization</p>
              <h2>{initialization_title}</h2>
            </div>
            <span class="status good">{initialization_status}</span>
          </div>
          <img class="media-image" src="{html.escape(auto_init['image'])}" alt="初始化点 / Initialization points">
        </section>
        """

    overlay_section = ""
    if browser_preview:
        overlay_section = f"""
        <section class="media-panel video-panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">追踪 / Tracking</p>
              <h2>干净轨迹回放 / Clean tracking playback</h2>
            </div>
            <span class="status">短尾迹 / short trails · {html.escape(tracking_backend)}</span>
          </div>
          <video class="tracking-video" controls preload="metadata" src="{html.escape(browser_preview)}"></video>
        </section>
        """

    motif_media = ""
    if (run_path / "behavior_clusters_embedding.jpg").exists():
        motif_media += """
        <figure>
          <figcaption>行为模体嵌入 / Motif embedding</figcaption>
          <img src="behavior_clusters_embedding.jpg" alt="无监督行为模体嵌入 / Unsupervised behavior motif embedding">
        </figure>
        """
    if (run_path / "behavior_clusters_timeline.jpg").exists():
        motif_media += """
        <figure>
          <figcaption>逐 ID 模体时间轴 / Per-ID motif timeline</figcaption>
          <img src="behavior_clusters_timeline.jpg" alt="行为模体时间轴 / Behavior motif timeline">
        </figure>
        """
    if (run_path / "behavior_clusters_umap_hdbscan_embedding.jpg").exists():
        motif_media += """
        <figure>
          <figcaption>高级 UMAP + HDBSCAN 嵌入 / Advanced embedding</figcaption>
          <img src="behavior_clusters_umap_hdbscan_embedding.jpg" alt="UMAP HDBSCAN 行为模体嵌入 / behavior motif embedding">
        </figure>
        """
    if (run_path / "behavior_clusters_umap_hdbscan_timeline.jpg").exists():
        motif_media += """
        <figure>
          <figcaption>高级 HDBSCAN 时间轴 / Advanced motif timeline</figcaption>
          <img src="behavior_clusters_umap_hdbscan_timeline.jpg" alt="HDBSCAN 行为模体时间轴 / behavior motif timeline">
        </figure>
        """

    validation_section = ""
    if validation:
        validation_grid = ""
        evaluation_summary = ""
        if (run_path / "validation_review" / "review_grid.jpg").exists():
            validation_grid = """
            <img class="media-image validation-grid" src="validation_review/review_grid.jpg"
              alt="稀疏 GT 审核联系表 / Sparse GT review contact sheet">
            """
        if validation_evaluation:
            metrics = validation_evaluation.get("metrics", {})
            evaluation_summary = f"""
            <div class="metric-strip">
              {_metric("HOTA", f"{float(metrics.get('hota', 0)):.1%}")}
              {_metric("AssA", f"{float(metrics.get('assa', 0)):.1%}")}
              {_metric("MOTA", f"{float(metrics.get('mota', 0)):.1%}")}
              {_metric("IDF1", f"{float(metrics.get('idf1', 0)):.1%}")}
              {_metric("ID Switches", f"{float(metrics.get('num_switches', 0)):.0f}")}
              {_metric("Precision", f"{float(metrics.get('precision', 0)):.1%}")}
              {_metric("Recall", f"{float(metrics.get('recall', 0)):.1%}")}
            </div>
            <p class="panel-copy">
              指标仅适用于已人工确认的稀疏审核帧，不能替代多视频完整验证。
              / Metrics apply only to manually confirmed sparse review frames and do not replace
              complete multi-video validation.
            </p>
            """
        validation_section = f"""
        <section class="data-panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">科学验证准备 / Validation preparation</p>
              <h2>稀疏 GT 审核包 / Sparse GT review package</h2>
            </div>
            <span class="status">{validation_frames} 帧预填审核 / pre-filled frames</span>
          </div>
          <p class="panel-copy">
            追踪本身未使用人工标注。为计算 HOTA / AssA / MOTA / IDF1，系统已按风险优先策略预填审核帧；
            这些预填结果不是 GT，需要研究者修正后再确认评估。
            / Tracking used no manual annotation. To compute HOTA / AssA / MOTA / IDF1,
            risk-prioritized
            review frames are pre-filled; these predictions are not GT until corrected and confirmed.
          </p>
          {evaluation_summary}
          {validation_grid}
          <div class="downloads">
            <a href="validation_review/review.html">打开浏览器审核器 / Open browser reviewer</a>
            <a href="validation_review/README.md">审核说明 / Review instructions</a>
            <a href="validation_review/review_frames.csv" download>审核帧 CSV / Review frames CSV</a>
            <a href="validation_review/prelabels.mot" download>预填 MOT / Prelabels MOT</a>
            <a href="validation_review/corrected_gt_template.mot" download>GT 修正模板 / GT correction template</a>
          </div>
        </section>
        """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DenseTrack-AI 多动物追踪研究面板 / Multi-animal Tracking Research Dashboard</title>
  <style>
    :root {{
      --ink: #182126;
      --muted: #62727a;
      --line: #d9e1e3;
      --panel: #ffffff;
      --canvas: #f4f7f6;
      --teal: #167c80;
      --green: #2f855a;
      --amber: #b76a12;
      --coral: #bc4e43;
      --blue: #316bb5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--canvas);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    html, body {{ max-width: 100%; overflow-x: hidden; }}
    header {{
      background: #18343a;
      color: white;
      padding: 22px 28px 20px;
      border-bottom: 4px solid #e4a33d;
    }}
    header .topline, .section-heading, .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{ margin: 2px 0 4px; font-size: 25px; font-weight: 720; overflow-wrap: anywhere; }}
    h2 {{ margin: 2px 0 0; font-size: 17px; }}
    .subtitle, .eyebrow {{ color: #b8d1d2; margin: 0; font-size: 12px; }}
    .eyebrow {{ color: var(--teal); font-weight: 700; text-transform: uppercase; }}
    main {{ width: 100%; max-width: 1480px; margin: 0 auto; padding: 18px 22px 34px; }}
    header .topline > *, .toolbar > *, .section-heading > * {{ min-width: 0; }}
    .toolbar {{ margin-bottom: 16px; }}
    .toolbar p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .download-link {{
      color: white;
      background: var(--teal);
      border-radius: 4px;
      padding: 8px 11px;
      text-decoration: none;
      font-size: 12px;
      font-weight: 700;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .kpi, .media-panel, .data-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      min-width: 0;
    }}
    .kpi {{ padding: 12px 13px; min-height: 78px; }}
    .kpi span {{ display: block; color: var(--muted); font-size: 11px; }}
    .kpi strong {{ display: block; margin-top: 6px; font-size: 21px; }}
    .media-grid {{ display: grid; grid-template-columns: 1.25fr 0.9fr; gap: 14px; }}
    .media-panel, .data-panel {{ padding: 14px; margin-bottom: 14px; }}
    .status {{
      border: 1px solid #bed2d4;
      border-radius: 999px;
      color: #31666a;
      padding: 4px 8px;
      font-size: 11px;
      white-space: nowrap;
    }}
    .status.good {{ color: var(--green); border-color: #b8d9c7; }}
    .status.warn {{ color: var(--amber); border-color: #e4c69e; }}
    .status-group {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .tracking-video, .media-image, figure img {{
      display: block;
      width: 100%;
      margin-top: 12px;
      border: 1px solid var(--line);
      background: #edf1f1;
    }}
    .tracking-video {{ aspect-ratio: 1.025; max-height: 680px; }}
    .media-image {{ aspect-ratio: 1.025; object-fit: contain; }}
    .motif-media {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    figure {{ margin: 12px 0 0; }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 12px; }}
    .table-scroll {{ width: 100%; min-width: 0; overflow-x: auto; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; }}
    th {{ color: var(--muted); font-size: 11px; }}
    .lower-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .downloads {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }}
    .downloads a {{
      color: var(--blue);
      border: 1px solid #c7d7ea;
      border-radius: 4px;
      padding: 7px 9px;
      text-decoration: none;
      font-size: 12px;
    }}
    .panel-copy {{ color: var(--muted); font-size: 12px; line-height: 1.7; margin: 10px 0 0; }}
    .validation-grid {{ max-height: 760px; object-fit: contain; }}
    .metric-strip {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }}
    .metric {{ border-left: 3px solid var(--green); background: #f3f8f5; padding: 9px 10px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 10px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 17px; }}
    .readiness-strip {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .readiness-strip .metric strong {{ font-size: 13px; line-height: 1.35; overflow-wrap: anywhere; }}
    .ssl-block {{ border-top: 1px solid var(--line); margin-top: 13px; padding-top: 13px; }}
    .ssl-block:first-of-type {{ border-top: 0; margin-top: 0; padding-top: 0; }}
    .ssl-block h3 {{ margin: 0; font-size: 14px; }}
    .ssl-block .metric-strip {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    @media (max-width: 940px) {{
      .kpis {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .media-grid, .lower-grid, .motif-media {{ grid-template-columns: 1fr; }}
      .metric-strip {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      header .topline {{ align-items: flex-start; flex-direction: column; }}
    }}
    @media (max-width: 560px) {{
      header {{ padding: 18px 20px 16px; }}
      main {{ padding: 15px 14px 26px; }}
      h1 {{ font-size: 22px; }}
      .toolbar {{ align-items: flex-start; flex-wrap: wrap; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .metric-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .section-heading {{ align-items: flex-start; flex-direction: column; gap: 7px; }}
      .toolbar .status {{
        border-radius: 4px;
        max-width: 100%;
        overflow-wrap: anywhere;
        white-space: normal;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topline">
      <div>
        <p class="subtitle">DenseTrack-AI / 研究运行 / Research run</p>
        <h1>多动物追踪研究面板 / Multi-animal tracking dashboard</h1>
        <p class="subtitle">{html.escape(str(video_meta.get('path', 'video')))}</p>
      </div>
      <span class="status good">{tracking_status_label}</span>
    </div>
  </header>
  <main>
    <div class="toolbar">
      <p>基于审核运行产物生成 / Generated from audited run artifacts</p>
      {adaptive_status}
      <a class="download-link" href="tracks.mot" download>下载 MOT / Download MOT</a>
    </div>
    <section class="kpis">
      {_kpi("追踪 ID / Tracked IDs", str(num_ids))}
      {_kpi("帧数 / Frames", f"{total_frames:,}")}
      {_kpi("时长 / Duration", f"{duration / 60:.1f} min")}
      {_kpi("逐帧观测 / Observed per frame", f"{observed:.2f}")}
      {_kpi("主要可见 / Mostly observed", f"{mostly_observed:.1%}")}
      {_kpi("模体聚类 / Motif clusters", str(motif_count))}
    </section>
    {readiness_section}
    {experiment_section}
    <section class="media-grid">
      {overlay_section}
      {auto_init_section}
    </section>
    <section class="data-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">无监督发现 / Unsupervised discovery</p>
          <h2>重复行为模体 / Recurring behavior motifs</h2>
        </div>
        <div class="status-group">
          <span class="status">{motif_feature_display}</span>
          <span class="status">{motif_samples:,} 个样本 / samples</span>
          {advanced_motif_status}
        </div>
      </div>
      <div class="motif-media">{motif_media}</div>
      <div class="table-scroll"><table>
        <thead><tr><th>聚类 / Cluster</th><th>模体 / Motif</th><th>比例 / Ratio</th><th>样本 / Samples</th>
        <th>平均速度 / Mean speed</th><th>最近邻 / Nearest neighbor</th></tr></thead>
        <tbody>{motif_rows}</tbody>
      </table></div>
    </section>
    {representation_section}
    <section class="lower-grid">
      <section class="data-panel">
        <p class="eyebrow">活动 / Activity</p>
        <h2>逐 ID 运动摘要 / Per-ID movement summary</h2>
        <div class="table-scroll"><table>
          <thead><tr><th>ID</th><th>帧数 / Frames</th><th>平均速度 / Mean speed</th><th>活动 / Activity</th><th>距离 / Distance</th></tr></thead>
          <tbody>{per_id_rows}</tbody>
        </table></div>
      </section>
      <section class="data-panel">
        <p class="eyebrow">审计 / Audit</p>
        <h2>覆盖与连接置信度 / Coverage and link confidence</h2>
        <p class="panel-copy">
          此分数衡量检测覆盖和轨迹连接质量，不是身份准确率。
          / This score measures detection coverage and trajectory links, not identity accuracy.
        </p>
        <div class="table-scroll"><table>
          <thead><tr><th>ID</th><th>有效置信度 / Effective confidence</th><th>等级 / Level</th><th>帧数 / Frames</th></tr></thead>
          <tbody>{confidence_rows}</tbody>
        </table></div>
      </section>
    </section>
    {identity_continuity_section}
    {identity_risk_section}
    {contact_events_section}
    {validation_section}
    <section class="data-panel">
      <p class="eyebrow">产物 / Artifacts</p>
      <h2>下载 / Downloads</h2>
      <div class="downloads">{downloads}</div>
    </section>
  </main>
</body>
</html>
"""


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'


def _metric(label: str, value: str) -> str:
    return f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'


def _readiness_section(
    auto_init: dict[str, Any],
    browser_preview: str | None,
    validation: dict[str, Any],
    validation_evaluation: dict[str, Any],
    identity_risk: dict[str, Any],
    adaptive_agreement: float,
    identity_stability: float,
    observed: float,
    num_ids: int,
) -> str:
    """Render a bilingual product-state boundary for researcher review."""
    click_assisted = auto_init.get("initialization_mode") == "manual_first_frame_click"
    initialization_label = (
        "首帧点击辅助 / first-frame click assist"
        if click_assisted
        else "零点击自动种子 / zero-click auto seeds"
    )
    validation_confirmed = bool(validation_evaluation)
    validation_label = (
        "稀疏指标已确认 / sparse metrics confirmed"
        if validation_confirmed
        else "待稀疏 GT 审核 / sparse GT review pending"
    )
    risk_summary = identity_risk.get("summary", {})
    high_risk_frames = int(risk_summary.get("high_risk_frames", 0) or 0)
    risk_label = (
        f"{high_risk_frames:,} 高风险帧 / high-risk frames"
        if high_risk_frames
        else "未发现高风险帧 / no high-risk frames"
    )
    clean_preview_label = (
        "干净回放已生成 / clean playback ready"
        if browser_preview
        else "未生成回放 / playback missing"
    )
    status_class = "good" if validation_confirmed and high_risk_frames == 0 else "warn"
    status_text = (
        "可交付科学审核 / ready for scientific review"
        if validation_confirmed
        else "可查看追踪效果，需审核后给出科学指标 / visual tracking ready, metrics need review"
    )
    agreement_text = "N/A" if adaptive_agreement <= 0 else f"{adaptive_agreement:.1%}"
    stability_text = "N/A" if identity_stability <= 0 else f"{identity_stability:.1%}"
    validation_frames = int(validation.get("selected_review_frames", 0) or 0)
    return f"""
    <section class="data-panel readiness-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">产品状态 / Product state</p>
          <h2>无训练标注追踪交付边界 / No-training-label tracking boundary</h2>
        </div>
        <span class="status {status_class}">{status_text}</span>
      </div>
      <div class="metric-strip readiness-strip">
        {_metric("训练标注 / Training labels", "false")}
        {_metric("初始化 / Initialization", initialization_label)}
        {_metric("干净回放 / Clean playback", clean_preview_label)}
        {_metric("审核指标 / Review metrics", validation_label)}
      </div>
      <p class="panel-copy">
        当前页面展示的是自动追踪和自动伪标注结果；短尾迹回放会在低置信或大跳变时断开，避免把身份跳变画成很长的混乱轨迹。
        科学 HOTA / AssA / MOTA / IDF1 仍必须基于研究者修正并确认的稀疏 GT。
        / This page shows automatic tracking and prelabel results. The short-trail playback cuts low-confidence
        or large-jump trail segments so identity jumps are not drawn as long confusing paths.
        Scientific HOTA / AssA / MOTA / IDF1
        still requires researcher-corrected and confirmed sparse GT.
      </p>
      <p class="panel-copy">
        当前摘要 / Current summary: {num_ids} IDs, {observed:.2f} observed tracks per frame,
        bidirectional agreement {agreement_text}, continuity proxy {stability_text},
        {risk_label}, {validation_frames} review frames.
      </p>
    </section>
    """


def _per_id_rows(per_id: dict[str, Any]) -> str:
    rows = []
    for track_id in sorted(per_id, key=lambda value: int(value)):
        item = per_id[track_id]
        rows.append(
            "<tr>"
            f"<td>{html.escape(track_id)}</td>"
            f"<td>{int(item.get('frames_tracked', 0)):,}</td>"
            f"<td>{float(item.get('mean_speed_px_s', 0)):.1f} px/s</td>"
            f"<td>{float(item.get('activity_ratio', 0)):.1%}</td>"
            f"<td>{float(item.get('distance_px', 0)):,.0f} px</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="5">暂无行为报告 / No behavior report</td></tr>'


def _motif_rows(clusters: list[dict[str, Any]]) -> str:
    rows = []
    for cluster in clusters:
        features = cluster.get("feature_means", {})
        rows.append(
            "<tr>"
            f"<td>c{int(cluster.get('cluster', 0))}</td>"
            f"<td>{html.escape(_bilingual_motif_label(str(cluster.get('label', 'motif'))))}</td>"
            f"<td>{float(cluster.get('ratio', 0)):.1%}</td>"
            f"<td>{int(cluster.get('num_samples', 0)):,}</td>"
            f"<td>{float(features.get('speed_px_s', 0)):.1f} px/s</td>"
            f"<td>{float(features.get('nearest_neighbor_px', 0)):.1f} px</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="6">请先运行行为发现 / Run behavior discovery to populate motifs</td></tr>'


def _confidence_rows(confidence: dict[str, Any]) -> str:
    rows = []
    for item in confidence.get("per_trajectory", confidence.get("trajectories", [])):
        rows.append(
            "<tr>"
            f"<td>{int(item.get('global_id', item.get('trajectory_id', 0)))}</td>"
            f"<td>{float(item.get('effective_confidence', item.get('overall_confidence', 0))):.3f}</td>"
            f"<td>{html.escape(_bilingual_confidence_level(str(item.get('confidence_level', item.get('level', 'unknown')))))}</td>"
            f"<td>{int(item.get('duration_frames', item.get('total_frames', 0))):,}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="4">暂无置信度详情 / No confidence details</td></tr>'


def _identity_continuity_section(identity_continuity: dict[str, Any]) -> str:
    if not identity_continuity:
        return ""
    method = identity_continuity.get("method", {})
    diagnostics = identity_continuity.get("diagnostics", {})
    tracker = diagnostics.get("tracker", {})
    optimizer = diagnostics.get("optimizer", {})
    validation = identity_continuity.get("scientific_validation", {})
    closed_set = bool(diagnostics.get("closed_set_achieved", False))
    geometry_enabled = bool(method.get("hierarchical_discrete_geometry", False))
    references = identity_continuity.get("references", [])
    reference_links = " · ".join(
        f'<a href="{html.escape(str(reference.get("url", "")))}" target="_blank" '
        f'rel="noreferrer">{html.escape(str(reference.get("name", "paper")))}</a>'
        for reference in references
        if reference.get("url")
    )
    return f"""
    <section class="data-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">长时身份 / Long-term identity</p>
          <h2>科研身份连续性 / Research identity continuity</h2>
        </div>
        <span class="status {'good' if closed_set else 'warn'}">
          {'闭集轨迹完整 / closed-set complete' if closed_set else '轨迹仍有碎片 / trajectories remain fragmented'}
        </span>
      </div>
      <div class="metric-strip">
        {_metric("离散几何 / Discrete geometry", "启用 / on" if geometry_enabled else "关闭 / off")}
        {_metric("输入片段 / Tracklets", f"{int(diagnostics.get('input_tracklets', 0)):,}")}
        {_metric("全局轨迹 / Trajectories", f"{int(diagnostics.get('output_trajectories', 0)):,}")}
        {_metric("长时拼接 / Stitches", f"{int(diagnostics.get('selected_stitches', 0)):,}")}
        {_metric("遮挡重获 / Reacquisitions", f"{int(tracker.get('reacquisition_events', 0)):,}")}
        {_metric("歧义延迟 / Deferred", f"{int(tracker.get('deferred_ambiguous_assignments', 0)):,}")}
        {_metric("接触冻结 / Contact holds", f"{int(tracker.get('contact_deferred_assignments', 0)):,}")}
        {_metric("合并框拒绝 / Merged rejects", f"{int(tracker.get('merged_detection_rejections', 0)):,}")}
        {_metric("运动离群拒绝 / Motion rejects", f"{int(tracker.get('motion_outlier_rejections', 0)):,}")}
        {_metric("候选连接 / Candidate links", f"{int(optimizer.get('valid_candidate_links', 0)):,}")}
      </div>
      <p class="panel-copy">
        位置、尺度与极坐标速度使用固定两阶段残差离散编码，并与双向全视频平滑及闭集轨迹拼接结合。
        这是 HieDG 表征思想与 WaspMOT 长序列协议的可复现本地适配，不是两篇论文训练模型的复现。
        / Position, scale, and polar velocity use deterministic two-stage residual tokens,
        combined with bidirectional full-video smoothing and closed-set tracklet stitching.
      </p>
      <p class="panel-copy">
        {html.escape(str(validation.get('status_display', '需要人工审核 GT / reviewed ground truth required')))}：
        HOTA、AssA、IDF1 与 ID switch 只在人工审核 GT 上构成科学准确率。{reference_links}
      </p>
    </section>
    """


def _identity_risk_section(identity_risk: dict[str, Any]) -> str:
    if not identity_risk:
        return ""
    summary = identity_risk.get("summary", {})
    frames = sorted(
        identity_risk.get("frames", []),
        key=lambda frame: (-float(frame.get("risk_score", 0)), int(frame.get("frame", 0))),
    )[:12]
    rows = "".join(
        "<tr>"
        f"<td>{int(frame.get('frame', 0)):,}</td>"
        f"<td>{float(frame.get('risk_score', 0)):.3f}</td>"
        f"<td>{int(frame.get('disagreement_count', 0))}</td>"
        f"<td>{int(frame.get('one_sided_count', 0))}</td>"
        f"<td>{int(frame.get('prediction_count', 0))}</td>"
        f"<td>{html.escape(str(frame.get('reasons', '')))}</td>"
        "</tr>"
        for frame in frames
    )
    return f"""
    <section class="data-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">身份审计 / Identity audit</p>
          <h2>正反向一致性风险 / Bidirectional consistency risk</h2>
        </div>
        <div class="status-group">
          <span class="status warn">{int(summary.get('high_risk_frames', 0)):,} 高风险帧 / high-risk frames</span>
          <span class="status warn">{int(summary.get('identity_switch_risk_events', 0)):,} 换 ID 风险段 / switch-risk episodes</span>
          <span class="status">{int(summary.get('large_jump_frames', 0)):,} 大跳变帧 / large-jump frames</span>
          <span class="status">{float(summary.get('mean_risk_score', 0)):.3f} 平均风险 / mean risk</span>
        </div>
      </div>
      <p class="panel-copy">
        这是无标注内部审计信号，不是科学准确率。高风险帧会优先进入稀疏 GT 审核包。
        / This is an annotation-free internal audit signal, not a scientific accuracy metric.
        High-risk frames are prioritized in the sparse GT review package.
      </p>
      <div class="table-scroll"><table>
        <thead><tr><th>帧 / Frame</th><th>风险 / Risk</th><th>分歧 / Disagreement</th>
        <th>单边 / One-sided</th><th>预测补帧 / Predictions</th><th>原因 / Reasons</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>
    """


def _contact_events_section(contact_events: dict[str, Any]) -> str:
    if not contact_events:
        return ""
    summary = contact_events.get("summary", {})
    uncertainty_counts = summary.get("uncertainty_counts", {})
    kind_counts = summary.get("event_kind_counts", {})
    episodes = contact_events.get("inversion_episodes", [])
    events = sorted(
        contact_events.get("events", []),
        key=lambda event: (
            -float(event.get("review_priority", 0)),
            int(event.get("start_frame", 0)),
        ),
    )[:12]

    def _assignment_display(assignment: Any) -> str:
        if not assignment:
            return "—"
        return ", ".join(
            f"{incoming}→{outgoing}"
            for incoming, outgoing in sorted(
                ((int(key), int(value)) for key, value in dict(assignment).items())
            )
        )

    row_parts: list[str] = []
    for event in events:
        margin = event.get("score_margin")
        tracker_matches = event.get("tracker_matches_best")
        if tracker_matches is False:
            status = "⚠ 与追踪器不一致 / disagrees"
        else:
            status = str(event.get("uncertainty", ""))
        margin_cell = f"<td>{float(margin):.3f}</td>" if margin is not None else "<td>—</td>"
        kind = event.get("event_kind", "")
        kind_display = (
            "测量歧义 / ambiguous" if kind == "measurement_ambiguous" else "仅接近 / proximity"
        )
        row = (
            "<tr>"
            f"<td>{html.escape(str(event.get('event_id', '')))}</td>"
            f"<td>{int(event.get('start_frame', 0)):,}–{int(event.get('end_frame', 0)):,}</td>"
            f"<td>{html.escape('|'.join(str(pid) for pid in event.get('participant_ids', [])))}</td>"
            f"<td>{html.escape(kind_display)}</td>"
            f"<td>{int(event.get('duration_frames', 0))}</td>"
            f"<td>{int(event.get('merged_frames', 0))}</td>"
            f"<td>{html.escape(_assignment_display(event.get('best_assignment')))}</td>"
            + margin_cell
            + f"<td>{float(event.get('review_priority', 0)):.3f}</td>"
            f"<td>{html.escape(status)}</td>"
            "</tr>"
        )
        row_parts.append(row)
    rows = "".join(row_parts)
    return f"""
    <section class="data-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">接触事件 / Contact events</p>
          <h2>合并/拆分身份假设审计 / Merge–split identity hypothesis audit</h2>
        </div>
        <div class="status-group">
          <span class="status warn">{len(episodes):,} 标签反转段 / label-inversion episodes</span>
          <span class="status warn">{int(uncertainty_counts.get('uncertain', 0)) + int(summary.get('unresolved_events', 0)):,} 待审核事件 / events need review</span>
          <span class="status">{int(summary.get('total_events', 0)):,} 接触事件 / contact events</span>
        </div>
      </div>
      <div class="metric-strip">
        {_metric("测量歧义事件 / Ambiguous-measurement", f"{int(kind_counts.get('measurement_ambiguous', 0)):,}")}
        {_metric("仅接近事件 / Proximity-only", f"{int(kind_counts.get('proximity_only', 0)):,}")}
        {_metric("确定 / Confident", f"{int(uncertainty_counts.get('confident', 0)):,}")}
        {_metric("歧义 / Ambiguous", f"{int(uncertainty_counts.get('ambiguous', 0)):,}")}
        {_metric("不确定 / Uncertain", f"{int(uncertainty_counts.get('uncertain', 0)):,}")}
        {_metric("未解析 / Unresolved", f"{int(summary.get('unresolved_events', 0)):,}")}
        {_metric("原始不一致 / Raw disagreements", f"{int(summary.get('tracker_disagreement_events', 0)):,}")}
        {_metric("平均 margin / Mean margin", f"{float(summary['mean_score_margin']):.3f}" if summary.get('mean_score_margin') is not None else "—")}
      </div>
      <p class="panel-copy">
        每个接触事件记录参与者、合并测量帧和拆分边界的前 K 个入→出身份假设。
        margin 是固定打分函数给出的最优与次优假设差值，是无标注相对分数，未经 GT 校准。
        原始不一致事件会沿持续标签反转级联，因此审核单位是“标签反转段”而不是原始事件数；
        不确定和反转段涉及的事件会优先进入稀疏 GT 审核包。
        / Each contact event records participants, merged-measurement frames, and the top-K
        incoming→outgoing identity hypotheses at the split boundary. The margin is the
        best-vs-second-best difference from a fixed scoring function — an annotation-free
        relative score, not GT-calibrated. Raw disagreement events cascade along a
        persistent label inversion, so the review unit is the label-inversion episode,
        not the raw event count. Uncertain events and episodes are prioritized in the
        sparse GT review package.
      </p>
      <div class="table-scroll"><table>
        <thead><tr><th>事件 / Event</th><th>帧范围 / Frames</th><th>参与者 / IDs</th>
        <th>类型 / Kind</th><th>持续 / Duration</th><th>合并帧 / Merged</th>
        <th>最优假设 / Best hypothesis</th>
        <th>Margin</th><th>审核优先级 / Review priority</th><th>状态 / Status</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>
    """


def _experiment_section(run_path: Path, experiment_metrics: dict[str, Any]) -> str:
    if not experiment_metrics:
        return ""
    template = experiment_metrics.get("template", {})
    summary = experiment_metrics.get("summary", {})
    baseline = experiment_metrics.get("transparent_baseline", {})
    analysis_policy = experiment_metrics.get("analysis_policy", {})
    aggregate = experiment_metrics.get("aggregate_zones", [])
    primary_zone_id = str(summary.get("primary_zone_id", template.get("primary_zone_id", "")))
    primary_zone = next(
        (zone for zone in aggregate if str(zone.get("zone_id")) == primary_zone_id),
        {},
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(zone.get('zone_name_zh', zone.get('zone_id', ''))))} / "
        f"{html.escape(str(zone.get('zone_name_en', zone.get('zone_id', ''))))}</td>"
        f"<td>{float(zone.get('time_s', 0)):.1f} s</td>"
        f"<td>{int(zone.get('entries', 0))}</td>"
        f"<td>{int(zone.get('observations', 0)):,}</td>"
        "</tr>"
        for zone in aggregate
    )
    downloads = _inline_downloads(
        run_path,
        [
            ("experiment_metrics.json", "实验指标 JSON / Experiment metrics JSON"),
            ("experiment_zone_metrics.csv", "区域指标 CSV / Zone metrics CSV"),
            ("experiment_time_bins.csv", "时间箱 CSV / Time-bin CSV"),
        ],
    )
    return f"""
    <section class="data-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">实验范式 / Experiment template</p>
          <h2>{html.escape(str(template.get('name_zh', '行为实验')))} / {html.escape(str(template.get('name_en', 'Behavior experiment')))}</h2>
        </div>
        <span class="status good">透明基础聚合 / Transparent baseline</span>
      </div>
      <div class="metric-strip">
        {_metric("区域 / Zones", f"{int(summary.get('num_zones', 0))}")}
        {_metric("个体 / IDs", f"{int(summary.get('num_ids', 0))}")}
        {_metric("总路程 / Distance", f"{float(baseline.get('total_distance_px', summary.get('total_distance_px', 0))):,.0f} px")}
        {_metric("平均速度 / Mean speed", f"{float(baseline.get('mean_speed_px_s', summary.get('mean_speed_px_s', 0))):.1f} px/s")}
        {_metric("主区域时间 / Primary zone time", f"{float(summary.get('primary_zone_time_s', 0)):.1f} s")}
        {_metric("主区域进入 / Primary entries", f"{int(summary.get('primary_zone_entries', 0))}")}
        {_metric("追踪覆盖 / Coverage", f"{float(baseline.get('tracking_coverage_ratio', summary.get('tracking_coverage_ratio', 0))):.1%}")}
      </div>
      <p class="panel-copy">
        {html.escape(str(template.get('description_zh', '')))}
        / {html.escape(str(template.get('description_en', '')))}
        当前是模板默认归一化区域；正式研究可替换为研究者绘制并校准的 arena.json。
        / These are normalized default template zones; formal studies can replace them with researcher-drawn
        and calibrated arena.json regions.
      </p>
      <p class="panel-copy">
        {html.escape(str(analysis_policy.get('interpretation_zh', '受控效应和剂量比较优先使用透明基础指标。')))}
        / {html.escape(str(analysis_policy.get('interpretation', 'Use transparent baseline metrics first for controlled effect and dose comparisons.')))}
        Tracking Master style metrics / Tracking Master 风格指标继续提供区域和时间箱明细。
        开放式行为发现、社会交互结构和长期身份问题可再启用无监督模体等复杂表征。
        / Add unsupervised motifs for open-ended discovery, social structure, or long-term identity questions.
        Evidence: <a href="https://www.biorxiv.org/content/10.64898/2026.07.20.739643v1"
        target="_blank" rel="noreferrer">bioRxiv 10.64898/2026.07.20.739643</a>.
      </p>
      <div class="table-scroll"><table>
        <thead><tr><th>区域 / Zone</th><th>停留 / Time</th><th>进入 / Entries</th><th>观测 / Observations</th></tr></thead>
        <tbody>{rows or '<tr><td colspan="4">暂无实验指标 / No experiment metrics</td></tr>'}</tbody>
      </table></div>
      <p class="panel-copy">
        主区域 / Primary zone: {html.escape(str(primary_zone.get('zone_name_zh', primary_zone_id)))} /
        {html.escape(str(primary_zone.get('zone_name_en', primary_zone_id)))}.
      </p>
      <div class="downloads">{downloads}</div>
    </section>
    """


def _representation_section(
    run_path: Path,
    visual_ssl: dict[str, Any],
    bioacoustic_ssl: dict[str, Any],
) -> str:
    if not visual_ssl and not bioacoustic_ssl:
        return ""

    blocks = []
    if visual_ssl:
        summary = visual_ssl.get("summary", {})
        camera = visual_ssl.get("camera_manifest", {})
        downloads = _inline_downloads(
            run_path,
            [
                ("visual_ssl_report.json", "视觉 SSL 报告 / Visual SSL report"),
                ("visual_ssl_embeddings.csv", "视觉 SSL 嵌入 / Visual SSL embeddings"),
            ],
        )
        limitations = " ".join(str(item) for item in visual_ssl.get("limitations", [])[:2])
        blocks.append(
            f"""
            <div class="ssl-block">
              <div class="section-heading">
                <div>
                  <p class="eyebrow">自监督视觉表征 / Self-supervised visual representation</p>
                  <h3>BEAST3D 启发的无标注轨迹-姿态嵌入 / BEAST3D-inspired unlabeled tracking-pose embeddings</h3>
                </div>
                <span class="status">
                  {html.escape(str(summary.get('multiview_status', camera.get('status', 'single_view_only'))))}
                </span>
              </div>
              <div class="metric-strip">
                {_metric("Samples / 样本", f"{int(summary.get('num_samples', 0)):,}")}
                {_metric("IDs / 个体", f"{int(summary.get('num_ids', 0)):,}")}
                {_metric("Embedding / 嵌入维度", f"{int(summary.get('embedding_dim', 0))}-D")}
                {_metric("Positive pairs / 正样本对", f"{int(summary.get('temporal_positive_pairs', 0)):,}")}
              </div>
              <p class="panel-copy">
                该模块不需要训练标注，使用已追踪轨迹、几何姿态、身份风险和可选相机清单生成可迁移表征。
                / This module needs no training annotation; it exports transferable features from trajectories,
                geometric pose, identity-risk signals, and optional camera metadata.
                {html.escape(limitations)}
              </p>
              <div class="downloads">{downloads}</div>
            </div>
            """
        )

    if bioacoustic_ssl:
        summary = bioacoustic_ssl.get("summary", {})
        species = str(summary.get("species", bioacoustic_ssl.get("config", {}).get("species", "animal")))
        downloads = _inline_downloads(
            run_path,
            [
                ("bioacoustic_ssl_report.json", "音频 SSL 报告 / Bioacoustic SSL report"),
                ("bioacoustic_ssl_units.csv", "音频单元 CSV / Acoustic units CSV"),
            ],
        )
        limitations = " ".join(str(item) for item in bioacoustic_ssl.get("limitations", [])[:2])
        blocks.append(
            f"""
            <div class="ssl-block">
              <div class="section-heading">
                <div>
                  <p class="eyebrow">动物通信 / Animal communication</p>
                  <h3>Dolph2Vec 启发的物种特异音频 SSL / Dolph2Vec-inspired species-specific audio SSL</h3>
                </div>
                <span class="status">{html.escape(species)}</span>
              </div>
              <div class="metric-strip">
                {_metric("Windows / 窗口", f"{int(summary.get('num_windows', 0)):,}")}
                {_metric("Units / 声学单元", f"{int(summary.get('codebook_units', 0)):,}")}
                {_metric("Whistle candidates / 哨声候选", f"{int(summary.get('whistle_candidate_windows', 0)):,}")}
                {_metric("Candidate ratio / 候选比例", f"{float(summary.get('whistle_candidate_ratio', 0)):.1%}")}
              </div>
              <p class="panel-copy">
                该模块从未标注 WAV 中提取声学单元和哨声候选，用于把通信行为与视频追踪结果并列审计。
                / This module extracts acoustic units and whistle candidates from unlabeled WAV audio so
                communication behavior can be audited alongside video tracking.
                {html.escape(limitations)}
              </p>
              <div class="downloads">{downloads}</div>
            </div>
            """
        )

    return f"""
    <section class="data-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">前沿表征 / Frontier representations</p>
          <h2>无训练标注 SSL 扩展 / No-training-annotation SSL extensions</h2>
        </div>
        <span class="status good">manual training annotation: false</span>
      </div>
      {"".join(blocks)}
    </section>
    """


def _bilingual_motif_label(label: str) -> str:
    translations = {
        "close_interaction": "近距离互动",
        "low_activity": "低活动",
        "high_activity": "高活动",
        "turning_exploration": "转向探索",
        "locomotion": "移动",
        "motif": "模体",
    }
    for english, chinese in translations.items():
        if label.startswith(english):
            return f"{chinese} / {label}"
    return label


def _bilingual_confidence_level(level: str) -> str:
    translations = {
        "high": "高 / high",
        "medium": "中 / medium",
        "low": "低 / low",
        "unknown": "未知 / unknown",
    }
    return translations.get(level.lower(), level)


def _download_links(run_path: Path) -> str:
    names = [
        "tracks.mot",
        "detections.coco.json",
        "poses.csv",
        "labels.sleap.json",
        "confidence_report.json",
        "behavior_report.json",
        "behavior_timeline.csv",
        "behavior_clusters.json",
        "behavior_clusters_timeline.csv",
        "behavior_clusters_umap_hdbscan.json",
        "behavior_clusters_umap_hdbscan_timeline.csv",
        "behavior_clusters_umap_hdbscan_embedding.jpg",
        "behavior_clusters_umap_hdbscan_timeline.jpg",
        "identity_risk.json",
        "identity_risk.csv",
        "identity_continuity.json",
        "identity_stitches.csv",
        "contact_events.json",
        "contact_events.csv",
        "visual_ssl_report.json",
        "visual_ssl_embeddings.csv",
        "bioacoustic_ssl_report.json",
        "bioacoustic_ssl_units.csv",
        "experiment_metrics.json",
        "experiment_zone_metrics.csv",
        "experiment_time_bins.csv",
        "run_summary.json",
        "tracking_preview_clean.mp4",
        "tracking_preview_browser.mp4",
        "overlay.mp4",
        "overlay_before_after_frame75.jpg",
        "identity_stable_before_after_frame129.jpg",
        "auto_initialization_points.json",
        "validation_review/README.md",
        "validation_review/review_frames.csv",
        "validation_review/prelabels.mot",
        "validation_review/corrected_gt_template.mot",
        "validation_review/validation_manifest.json",
        "validation_review/review.html",
        "validation_review/review_data.json",
        "validation_review/reviewed_gt.mot",
        "validation_review/evaluation_report.json",
        "validation_review/evaluation_report.md",
    ]
    links = []
    for name in names:
        if (run_path / name).exists():
            links.append(f'<a href="{html.escape(name)}" download>{html.escape(name)}</a>')
    return "".join(links)


def _inline_downloads(run_path: Path, items: list[tuple[str, str]]) -> str:
    links = []
    for name, label in items:
        if (run_path / name).exists():
            links.append(f'<a href="{html.escape(name)}" download>{html.escape(label)}</a>')
    return "".join(links)


def _copy_auto_init_assets(auto_init_dir: str | Path | None, run_path: Path) -> dict[str, Any]:
    if not auto_init_dir:
        return {}
    source = Path(auto_init_dir).expanduser().resolve()
    image_source = source / "initialization_points.jpg"
    json_source = source / "initialization_points.json"
    if not image_source.exists():
        return {}

    image_target = run_path / "auto_initialization_points.jpg"
    shutil.copy2(image_source, image_target)
    result = {"image": image_target.name}
    if json_source.exists():
        json_target = run_path / "auto_initialization_points.json"
        shutil.copy2(json_source, json_target)
        result["json"] = json_target.name
        try:
            initialization = json.loads(json_source.read_text(encoding="utf-8"))
            result["initialization_mode"] = initialization.get("initialization_mode", "automatic")
        except json.JSONDecodeError:
            result["initialization_mode"] = "automatic"
    return result


def _ensure_browser_preview(run_path: Path) -> str | None:
    candidates = [
        run_path / "tracking_preview_clean.mp4",
        run_path / "tracking_preview_77s_92s.mp4",
        run_path / "overlay.mp4",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return None

    target = run_path / "tracking_preview_browser.mp4"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target.name

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return source.name
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return source.name

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    return target.name


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
