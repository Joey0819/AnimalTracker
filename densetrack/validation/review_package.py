"""Build a low-effort sparse GT review package from DenseTrack auto-labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from densetrack.export.run_reexport import MOTRow, load_mot_rows, load_run_metadata
from densetrack.utils.metrics import compute_mot_metrics, summarize_metrics

__all__ = [
    "build_validation_review_package",
    "evaluate_validation_review_package",
]


@dataclass
class ReviewFrame:
    """Frame-level auto-label review priority."""

    frame: int
    risk_score: float
    selection_reason: str
    risk_reasons: str
    track_count: int
    prediction_count: int
    min_confidence: float
    mean_confidence: float
    nearest_neighbor_px: float | None
    identity_risk_score: float
    identity_disagreement_count: int
    contact_event_score: float = 0.0


def build_validation_review_package(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    n_frames: int = 100,
    video_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create pre-filled sparse labels and visual QA images for GT correction."""
    run_path = Path(run_dir).expanduser().resolve()
    mot_path = run_path / "tracks.mot"
    if not mot_path.exists():
        raise FileNotFoundError(f"tracks.mot not found / 未找到 tracks.mot: {run_path}")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive / n_frames 必须为正数")

    output = Path(output_dir).expanduser().resolve() if output_dir else run_path / "validation_review"
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    raw_frames_dir = output / "raw_frames"
    raw_frames_dir.mkdir(exist_ok=True)
    for stale_review_artifact in (
        output / "reviewed_gt.mot",
        output / "evaluation_report.json",
        output / "evaluation_report.md",
    ):
        stale_review_artifact.unlink(missing_ok=True)
    for image_dir in (frames_dir, raw_frames_dir):
        for stale_image in image_dir.glob("frame_*.jpg"):
            stale_image.unlink()

    rows = load_mot_rows(mot_path)
    if not rows:
        raise ValueError(f"No MOT rows found / MOT 结果为空: {mot_path}")
    metadata = load_run_metadata(run_path)
    resolved_video = _resolve_video_path(video_path, metadata, run_path)
    source_map = _load_detection_sources(run_path / "labels.sleap.json")
    identity_risk = _load_identity_risk(run_path / "identity_risk.json")
    contact_priorities = _load_contact_event_priorities(run_path / "contact_events.json")
    by_frame: dict[int, list[MOTRow]] = defaultdict(list)
    for row in rows:
        by_frame[row.frame].append(row)
    available_frames = sorted(by_frame)
    expected_animals = int(
        metadata.get("config", {}).get("n_animals")
        or max(len(frame_rows) for frame_rows in by_frame.values())
    )

    candidates = [
        _score_frame(
            frame, by_frame[frame], source_map, identity_risk, expected_animals,
            contact_priorities,
        )
        for frame in available_frames
    ]
    selected = _select_review_frames(candidates, min(n_frames, len(candidates)))
    selected_rows = [
        row
        for frame in sorted(item.frame for item in selected)
        for row in sorted(by_frame[frame], key=lambda value: value.track_id)
    ]

    prelabels_path = output / "prelabels.mot"
    template_path = output / "corrected_gt_template.mot"
    _write_mot_rows(
        selected_rows,
        prelabels_path,
        [
            "DenseTrack-AI sparse validation prelabels",
            "AUTO-GENERATED PREDICTIONS. These are not reviewed GT.",
        ],
    )
    _write_mot_rows(
        selected_rows,
        template_path,
        [
            "DenseTrack-AI corrected GT template",
            "REVIEW REQUIRED: correct IDs/boxes, remove false positives, add misses, then save as reviewed_gt.mot.",
        ],
    )

    frame_images = _write_review_images(
        resolved_video,
        frames_dir,
        raw_frames_dir,
        selected,
        by_frame,
    )
    grid_path = output / "review_grid.jpg"
    _write_contact_sheet(frame_images, grid_path)
    csv_path = output / "review_frames.csv"
    _write_review_csv(selected, csv_path)
    readme_path = output / "README.md"
    _write_review_readme(readme_path, len(selected), resolved_video)
    review_data_path = output / "review_data.json"
    _write_review_data(review_data_path, selected, by_frame)
    review_app_path = output / "review.html"
    _write_review_html(review_app_path)

    reason_counts = Counter(item.selection_reason for item in selected)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "format": "densetrack_sparse_validation_review_package",
        "format_version": "1.0",
        "run_dir": str(run_path),
        "video": str(resolved_video),
        "requested_review_frames": int(n_frames),
        "selected_review_frames": len(selected),
        "selected_frame_ids": [item.frame for item in selected],
        "expected_animals": expected_animals,
        "selection_strategy": {
            "description": (
                "Risk-prioritized sparse sampling with temporal coverage. "
                "风险优先稀疏采样，并保留全视频时间覆盖。"
            ),
            "selection_reason_counts": dict(reason_counts),
            "risk_signals": [
                "tracker prediction carry-forward / 追踪器预测补帧",
                "low detection confidence / 低检测置信度",
                "close-neighbor crowding / 近距离拥挤接触",
                "unexpected per-frame track count / 每帧轨迹数异常",
                "bidirectional identity disagreement / 正反向身份分歧",
                "contact-event hypothesis uncertainty / 接触事件身份假设不确定度",
            ],
        },
        "annotation_boundary": {
            "manual_annotation_used_for_tracking": False,
            "manual_review_required_for_scientific_metrics": True,
            "review_completed": False,
            "warning": (
                "prelabels.mot and corrected_gt_template.mot are auto-generated predictions, "
                "not ground truth. / 两个文件均为自动预填预测，不是 GT。"
            ),
        },
        "files": {
            "prelabels": str(prelabels_path),
            "corrected_gt_template": str(template_path),
            "review_frames_csv": str(csv_path),
            "review_grid": str(grid_path),
            "frames_dir": str(frames_dir),
            "raw_frames_dir": str(raw_frames_dir),
            "instructions": str(readme_path),
            "review_data": str(review_data_path),
            "review_app": str(review_app_path),
        },
    }
    manifest_path = output / "validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest["manifest"] = str(manifest_path)
    _update_run_manifests(run_path, manifest)
    return manifest


def evaluate_validation_review_package(
    package_dir: str | Path,
    ground_truth: str | Path | None = None,
    confirm_reviewed: bool = False,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate sparse reviewed GT while refusing unreviewed pseudo-label claims."""
    package = Path(package_dir).expanduser().resolve()
    manifest_path = package / "validation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"validation_manifest.json not found / 未找到验证清单: {package}")
    if not confirm_reviewed:
        raise ValueError(
            "Refusing to compute scientific metrics without --confirm-reviewed. "
            "请先人工审核并确认 GT，再计算科学指标。"
        )

    gt_path = Path(ground_truth).expanduser().resolve() if ground_truth else package / "reviewed_gt.mot"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Reviewed GT not found / 未找到已审核 GT: {gt_path}. "
            "Correct corrected_gt_template.mot and save it as reviewed_gt.mot."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gt_rows = load_mot_rows(gt_path)
    _validate_reviewed_gt_rows(gt_rows, manifest)
    prediction_path = package / "prelabels.mot"
    metrics = compute_mot_metrics(gt_path, prediction_path, iou_threshold=iou_threshold)
    report = {
        "created_at": datetime.now().isoformat(),
        "format": "densetrack_sparse_validation_evaluation",
        "scope": "sparse_reviewed_frames_only",
        "scope_display": "仅限稀疏人工审核帧 / sparse reviewed frames only",
        "review_confirmed": True,
        "ground_truth": str(gt_path),
        "prediction": str(prediction_path),
        "iou_threshold": iou_threshold,
        "metrics": metrics,
        "warning": (
            "Metrics apply only to the reviewed sparse frame package. "
            "指标仅适用于已审核的稀疏帧包，不能替代多视频完整验证。"
        ),
    }
    report_json = package / "evaluation_report.json"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_md = package / "evaluation_report.md"
    report_md.write_text(_evaluation_markdown(report), encoding="utf-8")
    report["files"] = {"json": str(report_json), "markdown": str(report_md)}
    boundary = manifest.setdefault("annotation_boundary", {})
    boundary["review_completed"] = True
    boundary["review_confirmed_at"] = report["created_at"]
    boundary["reviewed_gt"] = str(gt_path)
    manifest.setdefault("files", {})["reviewed_gt"] = str(gt_path)
    manifest["files"]["evaluation_report_json"] = str(report_json)
    manifest["files"]["evaluation_report_markdown"] = str(report_md)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    run_dir = Path(manifest.get("run_dir", package.parent)).expanduser().resolve()
    _update_run_manifests(run_dir, {**manifest, "manifest": str(manifest_path)})
    return report


def _validate_reviewed_gt_rows(rows: list[MOTRow], manifest: dict[str, Any]) -> None:
    """Reject incomplete or malformed reviewed GT before publishing metrics."""
    if not rows:
        raise ValueError("Reviewed GT is empty / 已审核 GT 为空")
    selected = {int(frame) for frame in manifest.get("selected_frame_ids", [])}
    reviewed = {row.frame for row in rows}
    extra = sorted(reviewed - selected)
    missing = sorted(selected - reviewed)
    if extra:
        raise ValueError(
            "Reviewed GT contains frames outside the sparse package / 已审核 GT 包含审核包外帧: "
            + ", ".join(map(str, extra[:8]))
        )
    if missing:
        raise ValueError(
            "Reviewed GT is missing selected review frames / 已审核 GT 缺少选定审核帧: "
            + ", ".join(map(str, missing[:8]))
        )
    for row in rows:
        values = (row.x, row.y, row.width, row.height, row.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Reviewed GT contains non-finite values / 已审核 GT 包含非法数值")
        if row.width <= 0 or row.height <= 0:
            raise ValueError("Reviewed GT contains non-positive boxes / 已审核 GT 包含无效框")


def _score_frame(
    frame: int,
    rows: list[MOTRow],
    source_map: dict[tuple[int, int], str],
    identity_risk: dict[int, dict[str, Any]],
    expected_animals: int,
    contact_priorities: dict[int, float] | None = None,
) -> ReviewFrame:
    confidences = [row.confidence for row in rows]
    centers = np.array([[row.cx, row.cy] for row in rows], dtype=float)
    nearest = _nearest_neighbor_distance(centers)
    prediction_count = sum(
        source_map.get((frame, row.track_id), "") in {"fixed_count_prediction", "adaptive_prediction"}
        for row in rows
    )
    prediction_ratio = prediction_count / max(expected_animals, 1)
    count_gap = abs(len(rows) - expected_animals) / max(expected_animals, 1)
    crowding = math.exp(-nearest / 90.0) if nearest is not None else 0.0
    low_confidence = 1.0 - min(confidences)
    identity_frame = identity_risk.get(frame, {})
    identity_risk_score = float(identity_frame.get("risk_score", 0.0) or 0.0)
    identity_disagreement_count = int(identity_frame.get("disagreement_count", 0) or 0)
    contact_event_score = float((contact_priorities or {}).get(frame, 0.0) or 0.0)
    risk_score = min(
        1.0,
        0.30 * prediction_ratio
        + 0.15 * crowding
        + 0.10 * low_confidence
        + 0.10 * count_gap
        + 0.20 * identity_risk_score
        + 0.15 * contact_event_score,
    )
    reasons = []
    if prediction_count:
        reasons.append(f"prediction_carry_forward={prediction_count}")
    if nearest is not None and nearest < 100:
        reasons.append(f"crowding_nn={nearest:.1f}px")
    if min(confidences) < 0.6:
        reasons.append(f"low_confidence={min(confidences):.3f}")
    if len(rows) != expected_animals:
        reasons.append(f"track_count={len(rows)}/{expected_animals}")
    if identity_disagreement_count:
        reasons.append(f"bidirectional_disagreement={identity_disagreement_count}")
    if contact_event_score > 0:
        reasons.append(f"contact_event_priority={contact_event_score:.3f}")
    if not reasons:
        reasons.append("coverage_check")
    return ReviewFrame(
        frame=frame,
        risk_score=float(risk_score),
        selection_reason="risk_priority",
        risk_reasons="; ".join(reasons),
        track_count=len(rows),
        prediction_count=prediction_count,
        min_confidence=float(min(confidences)),
        mean_confidence=float(np.mean(confidences)),
        nearest_neighbor_px=nearest,
        identity_risk_score=identity_risk_score,
        identity_disagreement_count=identity_disagreement_count,
        contact_event_score=contact_event_score,
    )


def _select_review_frames(candidates: list[ReviewFrame], n_frames: int) -> list[ReviewFrame]:
    if n_frames >= len(candidates):
        return sorted(candidates, key=lambda item: item.frame)
    by_frame = {item.frame: item for item in candidates}
    available = sorted(by_frame)
    selected: dict[int, ReviewFrame] = {}
    min_spacing = max(1, len(available) // max(n_frames * 4, 1))

    def add(item: ReviewFrame, reason: str, enforce_spacing: bool = True) -> bool:
        if item.frame in selected:
            return False
        if enforce_spacing and any(abs(item.frame - frame) < min_spacing for frame in selected):
            return False
        selected[item.frame] = ReviewFrame(**{**asdict(item), "selection_reason": reason})
        return True

    add(by_frame[available[0]], "boundary", enforce_spacing=False)
    add(by_frame[available[-1]], "boundary", enforce_spacing=False)

    risk_target = max(1, int(round(n_frames * 0.65)))
    for item in sorted(candidates, key=lambda value: (-value.risk_score, value.frame)):
        if sum(entry.selection_reason == "risk_priority" for entry in selected.values()) >= risk_target:
            break
        add(item, "risk_priority")

    for frame in np.linspace(available[0], available[-1], n_frames, dtype=int):
        nearest_frame = min(available, key=lambda available_frame: abs(available_frame - int(frame)))
        add(by_frame[nearest_frame], "temporal_coverage", enforce_spacing=False)
        if len(selected) >= n_frames:
            break

    for item in sorted(candidates, key=lambda value: (-value.risk_score, value.frame)):
        if len(selected) >= n_frames:
            break
        add(item, "risk_fill", enforce_spacing=False)
    return sorted(selected.values(), key=lambda item: item.frame)


def _write_review_images(
    video_path: Path,
    frames_dir: Path,
    raw_frames_dir: Path,
    selected: list[ReviewFrame],
    by_frame: dict[int, list[MOTRow]],
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video / 无法打开视频: {video_path}")
    selected_by_frame = {item.frame: item for item in selected}
    images = []
    try:
        for frame_idx in sorted(selected_by_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok:
                continue
            raw_path = raw_frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(raw_path), image)
            annotated = _draw_review_frame(image, selected_by_frame[frame_idx], by_frame[frame_idx])
            image_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(image_path), annotated)
            images.append(image_path)
    finally:
        cap.release()
    return images


def _draw_review_frame(image: np.ndarray, review: ReviewFrame, rows: list[MOTRow]) -> np.ndarray:
    annotated = image.copy()
    colors = [(255, 64, 64), (64, 255, 64), (64, 64, 255), (255, 255, 64), (255, 64, 255)]
    for row in rows:
        color = colors[row.track_id % len(colors)]
        x1, y1 = int(round(row.x)), int(round(row.y))
        x2, y2 = int(round(row.x + row.width)), int(round(row.y + row.height))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"ID {row.track_id} {row.confidence:.2f}",
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    label = (
        f"review frame {review.frame} | risk {review.risk_score:.3f} | "
        f"pred {review.prediction_count} | nn "
        f"{review.nearest_neighbor_px:.1f}px"
        if review.nearest_neighbor_px is not None
        else f"review frame {review.frame} | risk {review.risk_score:.3f}"
    )
    cv2.rectangle(annotated, (7, 7), (min(annotated.shape[1] - 7, 650), 42), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        label,
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return annotated


def _write_contact_sheet(images: list[Path], output_path: Path) -> None:
    loaded = [cv2.imread(str(path)) for path in images[:20]]
    loaded = [image for image in loaded if image is not None]
    if not loaded:
        return
    thumb_w, thumb_h = 320, 240
    cols = 4
    rows = math.ceil(len(loaded) / cols)
    canvas = np.full((rows * thumb_h, cols * thumb_w, 3), 238, dtype=np.uint8)
    for idx, image in enumerate(loaded):
        resized = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(idx, cols)
        canvas[row * thumb_h : (row + 1) * thumb_h, col * thumb_w : (col + 1) * thumb_w] = resized
    cv2.imwrite(str(output_path), canvas)


def _write_review_csv(selected: list[ReviewFrame], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(selected[0]).keys()))
        writer.writeheader()
        for item in sorted(selected, key=lambda value: (-value.risk_score, value.frame)):
            writer.writerow(asdict(item))


def _write_mot_rows(rows: list[MOTRow], output_path: Path, comments: list[str]) -> None:
    lines = [f"# {comment}" for comment in comments]
    lines.append("# Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1")
    for row in sorted(rows, key=lambda value: (value.frame, value.track_id)):
        lines.append(
            f"{row.frame},{row.track_id},{row.x:.2f},{row.y:.2f},"
            f"{row.width:.2f},{row.height:.2f},{row.confidence:.4f},-1,-1,-1"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_readme(output_path: Path, count: int, video_path: Path) -> None:
    output_path.write_text(
        f"""# DenseTrack-AI 稀疏 GT 审核包 / Sparse GT Review Package

视频 / Video: `{video_path}`

本包自动挑选了 `{count}` 帧，并预填追踪结果。追踪阶段没有使用人工标注。
This package selected `{count}` frames automatically and pre-filled tracking predictions. No manual annotation was used for tracking.

## 审核步骤 / Review Steps

1. 打开 `review.html`，逐帧检查预填框。
2. 修正 ID 和框，删除误检，补充漏检，并标记已审核。
3. 下载审核完成的 `reviewed_gt.mot`。
4. 运行：

```bash
densetrack evaluate-validation {output_path.parent} --confirm-reviewed
```

## 重要边界 / Important Boundary

`prelabels.mot` 和 `corrected_gt_template.mot` 是自动预测，不是 GT。只有经过人工确认并保存为 `reviewed_gt.mot` 后，才能用于科学指标。

`prelabels.mot` and `corrected_gt_template.mot` are auto-generated predictions, not GT. Use scientific metrics only after manual confirmation and saving the reviewed file as `reviewed_gt.mot`.
""",
        encoding="utf-8",
    )


def _write_review_data(
    output_path: Path,
    selected: list[ReviewFrame],
    by_frame: dict[int, list[MOTRow]],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "format": "densetrack_browser_gt_review",
        "format_version": "1.0",
        "storage_key": (
            "densetrack-gt-review:"
            + hashlib.sha1(str(output_path.parent.resolve()).encode("utf-8")).hexdigest()[:12]
        ),
        "warning": (
            "Pre-filled predictions are not GT until reviewed. "
            "/ 预填预测在人工审核前不是 GT。"
        ),
        "frames": [
            {
                **asdict(review),
                "image": f"raw_frames/frame_{review.frame:06d}.jpg",
                "annotations": [
                    {
                        "id": row.track_id,
                        "x": row.x,
                        "y": row.y,
                        "width": row.width,
                        "height": row.height,
                        "confidence": row.confidence,
                    }
                    for row in sorted(by_frame[review.frame], key=lambda value: value.track_id)
                ],
            }
            for review in selected
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_review_html(output_path: Path) -> None:
    output_path.write_text(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DenseTrack-AI GT 审核器 / GT Reviewer</title>
  <style>
    :root {
      --ink:#182126;--muted:#62727a;--line:#d9e1e3;--panel:#fff;--canvas:#f4f7f6;
      --teal:#167c80;--teal-dark:#0f666a;--green:#2f855a;--amber:#b76a12;--coral:#bc4e43;
    }
    * { box-sizing:border-box; }
    html,body { margin:0;max-width:100%;overflow-x:hidden; }
    body { background:var(--canvas);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0; }
    header { background:#18343a;color:#fff;border-bottom:4px solid #e4a33d;padding:17px 22px;display:flex;align-items:center;justify-content:space-between;gap:14px; }
    h1 { margin:0;font-size:21px;overflow-wrap:anywhere; }
    h2 { margin:0;font-size:16px; }
    p { margin:4px 0 0;color:var(--muted);font-size:12px;line-height:1.55; }
    header p { color:#b8d1d2; }
    .warning { color:#f5d08d;font-size:12px;font-weight:700;text-align:right; }
    main { width:100%;max-width:1480px;margin:0 auto;padding:14px; }
    .toolbar,.frame-panel,.annotation-panel { background:var(--panel);border:1px solid var(--line);border-radius:6px; }
    .toolbar { display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px 13px;margin-bottom:12px; }
    .toolbar-group,.actions { display:flex;align-items:center;gap:8px;flex-wrap:wrap; }
    button,a.button { border:0;border-radius:4px;background:var(--teal);color:#fff;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;font:inherit;font-size:12px;font-weight:750;padding:8px 10px;text-decoration:none; }
    button:hover,a.button:hover { background:var(--teal-dark); }
    button.secondary,a.secondary { background:#fff;border:1px solid #bdd0d2;color:var(--teal-dark); }
    button.danger { background:#fff;border:1px solid #e1bab5;color:var(--coral); }
    button:disabled { cursor:not-allowed;opacity:.45; }
    .status { color:var(--teal-dark);font-size:12px;font-weight:700; }
    .review-layout { display:grid;grid-template-columns:minmax(0,1fr) 390px;gap:12px;align-items:start; }
    .frame-panel,.annotation-panel { min-width:0;padding:12px; }
    canvas { display:block;width:100%;height:auto;background:#edf1f1;border:1px solid var(--line);margin-top:10px; }
    .panel-heading { display:flex;align-items:start;justify-content:space-between;gap:10px; }
    .risk { color:var(--amber);font-size:12px;font-weight:750; }
    .canvas-help { color:var(--muted);font-size:11px; }
    .annotation-list { display:grid;gap:8px;margin-top:10px; }
    .annotation { border-top:1px solid var(--line);cursor:pointer;padding-top:8px; }
    .annotation:first-child { border-top:0;padding-top:0; }
    .annotation.selected { background:#edf7f6;border-left:3px solid var(--teal);padding:8px; }
    .annotation-heading { display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:7px; }
    .fields { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px; }
    label { display:grid;gap:3px;color:var(--muted);font-size:10px;font-weight:700; }
    input { min-width:0;width:100%;border:1px solid #bfd0d3;border-radius:3px;color:var(--ink);font:inherit;font-size:11px;padding:6px; }
    .empty { color:var(--muted);font-size:12px; }
    .verified { display:flex;align-items:center;gap:6px;color:var(--green);font-size:12px;font-weight:750; }
    .verified input { width:auto; }
    @media (max-width:900px) {
      header { align-items:start;flex-direction:column; }
      .warning { text-align:left; }
      .review-layout { grid-template-columns:1fr; }
      .toolbar { align-items:start;flex-direction:column; }
      .annotation-panel { order:2; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>DenseTrack-AI GT 审核器 / GT Reviewer</h1>
      <p>风险优先稀疏审核 / Risk-prioritized sparse review</p>
    </div>
    <div class="warning">预填预测不是 GT / Pre-filled predictions are not GT</div>
  </header>
  <main>
    <div class="toolbar">
      <div class="toolbar-group">
        <button class="secondary" id="previous" type="button">上一帧 / Previous</button>
        <button class="secondary" id="next" type="button">下一帧 / Next</button>
        <span class="status" id="progress"></span>
      </div>
      <div class="actions">
        <label class="verified"><input id="verified" type="checkbox">已审核 / Reviewed</label>
        <button id="download" type="button">下载 reviewed_gt.mot / Download GT</button>
      </div>
    </div>
    <div class="review-layout">
      <section class="frame-panel">
        <div class="panel-heading">
          <div><h2 id="frame-title"></h2><p id="reasons"></p><p class="canvas-help">拖动框移动；拖动右下角调整大小 / Drag a box to move it; drag its bottom-right handle to resize it</p></div>
          <span class="risk" id="risk"></span>
        </div>
        <canvas id="canvas"></canvas>
      </section>
      <aside class="annotation-panel">
        <div class="panel-heading">
          <div><h2>预填框 / Pre-filled boxes</h2><p>仅修正错误项 / Correct errors only</p></div>
          <button class="secondary" id="add" type="button">新增 / Add</button>
        </div>
        <div class="annotation-list" id="annotation-list"></div>
      </aside>
    </div>
  </main>
  <script>
    const palette=["#316bb5","#2f855a","#bc4e43","#167c80","#a14c91","#b76a12"];
    const state={frames:[],index:0,images:new Map(),selected:-1,drag:null,storageKey:"densetrack-gt-review:"+location.pathname,createdAt:""};
    const els={
      canvas:document.querySelector("#canvas"),list:document.querySelector("#annotation-list"),
      title:document.querySelector("#frame-title"),reasons:document.querySelector("#reasons"),
      risk:document.querySelector("#risk"),progress:document.querySelector("#progress"),
      previous:document.querySelector("#previous"),next:document.querySelector("#next"),
      verified:document.querySelector("#verified")
    };
    const ctx=els.canvas.getContext("2d");
    const escapeHtml=value=>String(value).replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
    const save=()=>localStorage.setItem(state.storageKey,JSON.stringify({created_at:state.createdAt,frames:state.frames,index:state.index}));
    const current=()=>state.frames[state.index];
    function restore(frames) {
      const saved=localStorage.getItem(state.storageKey);
      if (!saved) return frames.map(frame=>({...frame,reviewed:false}));
      try {
        const parsed=JSON.parse(saved);
        if(parsed.created_at!==state.createdAt)return frames.map(frame=>({...frame,reviewed:false}));
        state.index=Math.min(parsed.index||0,frames.length-1);
        return parsed.frames.length===frames.length?parsed.frames:frames.map(frame=>({...frame,reviewed:false}));
      } catch { return frames.map(frame=>({...frame,reviewed:false})); }
    }
    function loadImage(path) {
      if (state.images.has(path)) return state.images.get(path);
      const image=new Image();
      const promise=new Promise(resolve=>{image.onload=()=>resolve(image);});
      image.src=path;state.images.set(path,promise);return promise;
    }
    async function draw() {
      const frame=current();const image=await loadImage(frame.image);
      els.canvas.width=image.naturalWidth;els.canvas.height=image.naturalHeight;
      ctx.drawImage(image,0,0);
      frame.annotations.forEach((box,index)=>{
        const color=palette[Math.abs(Number(box.id))%palette.length];
        ctx.strokeStyle=color;ctx.lineWidth=index===state.selected?5:3;ctx.strokeRect(box.x,box.y,box.width,box.height);
        ctx.fillStyle=color;ctx.font="bold 17px sans-serif";ctx.fillText(`ID ${box.id}`,box.x+3,Math.max(19,box.y-5));
        if(index===state.selected){ctx.fillRect(box.x+box.width-7,box.y+box.height-7,14,14);}
      });
    }
    function renderFields() {
      const frame=current();
      els.list.innerHTML=frame.annotations.length?frame.annotations.map((box,index)=>`
        <div class="annotation ${index===state.selected?"selected":""}" data-select="${index}">
          <div class="annotation-heading"><strong>ID ${escapeHtml(box.id)}</strong><button class="danger" data-remove="${index}" type="button">删除 / Remove</button></div>
          <div class="fields">
            ${["id","x","y","width","height"].map(key=>`<label>${key}<input data-index="${index}" data-key="${key}" type="number" step="0.1" value="${escapeHtml(box[key])}"></label>`).join("")}
          </div>
        </div>`).join(""):`<p class="empty">暂无框 / No boxes</p>`;
    }
    async function render() {
      const frame=current();const reviewed=state.frames.filter(item=>item.reviewed).length;
      els.title.textContent=`帧 / Frame ${frame.frame}`;
      els.reasons.textContent=frame.risk_reasons;
      els.risk.textContent=`风险 / Risk ${Number(frame.risk_score).toFixed(3)}`;
      els.progress.textContent=`${state.index+1} / ${state.frames.length} · 已审核 / Reviewed ${reviewed}`;
      els.verified.checked=Boolean(frame.reviewed);
      els.previous.disabled=state.index===0;els.next.disabled=state.index===state.frames.length-1;
      renderFields();await draw();save();
    }
    els.list.addEventListener("input",event=>{
      const input=event.target.closest("input[data-index]");if(!input)return;
      current().annotations[Number(input.dataset.index)][input.dataset.key]=Number(input.value);render();
    });
    els.list.addEventListener("click",event=>{
      const button=event.target.closest("button[data-remove]");
      if(button){current().annotations.splice(Number(button.dataset.remove),1);state.selected=-1;render();return;}
      const item=event.target.closest("[data-select]");if(!item)return;
      state.selected=Number(item.dataset.select);render();
    });
    document.querySelector("#add").addEventListener("click",()=>{
      current().annotations.push({id:0,x:20,y:20,width:60,height:40,confidence:1});state.selected=current().annotations.length-1;render();
    });
    function canvasPoint(event){
      const rect=els.canvas.getBoundingClientRect();
      return{x:(event.clientX-rect.left)*els.canvas.width/rect.width,y:(event.clientY-rect.top)*els.canvas.height/rect.height};
    }
    function hitTest(point){
      const boxes=current().annotations;
      for(let index=boxes.length-1;index>=0;index--){
        const box=boxes[index],handle=Math.hypot(point.x-(box.x+box.width),point.y-(box.y+box.height))<=18;
        if(handle)return{index,mode:"resize"};
        if(point.x>=box.x&&point.x<=box.x+box.width&&point.y>=box.y&&point.y<=box.y+box.height)return{index,mode:"move"};
      }
      return null;
    }
    els.canvas.addEventListener("pointerdown",event=>{
      const point=canvasPoint(event),hit=hitTest(point);if(!hit){state.selected=-1;render();return;}
      state.selected=hit.index;const box=current().annotations[hit.index];
      state.drag={...hit,start:point,box:{...box}};els.canvas.setPointerCapture(event.pointerId);render();
    });
    els.canvas.addEventListener("pointermove",event=>{
      if(!state.drag)return;const point=canvasPoint(event),drag=state.drag,box=current().annotations[drag.index];
      const dx=point.x-drag.start.x,dy=point.y-drag.start.y;
      if(drag.mode==="move"){
        box.x=Math.min(els.canvas.width-drag.box.width,Math.max(0,drag.box.x+dx));
        box.y=Math.min(els.canvas.height-drag.box.height,Math.max(0,drag.box.y+dy));
      }else{
        box.width=Math.min(els.canvas.width-drag.box.x,Math.max(4,drag.box.width+dx));
        box.height=Math.min(els.canvas.height-drag.box.y,Math.max(4,drag.box.height+dy));
      }
      renderFields();draw();
    });
    els.canvas.addEventListener("pointerup",()=>{if(state.drag){state.drag=null;save();render();}});
    els.previous.addEventListener("click",()=>{if(state.index>0){state.index--;state.selected=-1;render();}});
    els.next.addEventListener("click",()=>{if(state.index<state.frames.length-1){state.index++;state.selected=-1;render();}});
    els.verified.addEventListener("change",()=>{current().reviewed=els.verified.checked;render();});
    document.querySelector("#download").addEventListener("click",()=>{
      const reviewed=state.frames.filter(frame=>frame.reviewed).length;
      if(reviewed<state.frames.length&&!confirm(`仅审核 ${reviewed}/${state.frames.length} 帧。仍要下载吗？ / Only ${reviewed}/${state.frames.length} reviewed. Download anyway?`))return;
      const lines=["# DenseTrack-AI reviewed sparse GT","# REVIEWED_GT_CONFIRMATION_REQUIRED_BEFORE_EVALUATION","# Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1"];
      state.frames.forEach(frame=>frame.annotations.forEach(box=>lines.push(`${frame.frame},${box.id},${Number(box.x).toFixed(2)},${Number(box.y).toFixed(2)},${Number(box.width).toFixed(2)},${Number(box.height).toFixed(2)},${Number(box.confidence||1).toFixed(4)},-1,-1,-1`)));
      const anchor=document.createElement("a");anchor.href=URL.createObjectURL(new Blob([lines.join("\\n")+"\\n"],{type:"text/plain"}));anchor.download="reviewed_gt.mot";anchor.click();URL.revokeObjectURL(anchor.href);
    });
    fetch("review_data.json").then(response=>response.json()).then(payload=>{state.storageKey=payload.storage_key||state.storageKey;state.createdAt=payload.created_at;state.frames=restore(payload.frames);render();});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _load_detection_sources(path: Path) -> dict[tuple[int, int], str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sources = {}
    for frame in payload.get("frames", []):
        frame_idx = int(frame.get("frame_idx", 0))
        for instance in frame.get("instances", []):
            track_id = int(instance.get("track_id", 0))
            sources[(frame_idx, track_id)] = str(instance.get("densetrack_source", ""))
    return sources


def _load_identity_risk(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        int(frame.get("frame", 0)): frame
        for frame in payload.get("frames", [])
    }


def _load_contact_event_priorities(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    priorities = payload.get("frame_review_priorities", {})
    if not isinstance(priorities, dict):
        return {}
    result: dict[int, float] = {}
    for frame, priority in priorities.items():
        try:
            result[int(frame)] = float(priority)
        except (TypeError, ValueError):
            continue
    return result


def _resolve_video_path(
    video_path: str | Path | None,
    metadata: dict[str, Any],
    run_path: Path,
) -> Path:
    raw = video_path or metadata.get("video_metadata", {}).get("path")
    if not raw:
        raise FileNotFoundError("Video path is required / 缺少源视频路径")
    candidate = Path(raw).expanduser()
    search = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, run_path / candidate]
    for path in search:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Video not found / 未找到源视频: {raw}")


def _nearest_neighbor_distance(centers: np.ndarray) -> float | None:
    if len(centers) < 2:
        return None
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _evaluation_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    return f"""# DenseTrack-AI 稀疏审核评估 / Sparse Reviewed Evaluation

范围 / Scope: {report["scope_display"]}

> 指标仅适用于已审核稀疏帧，不能替代多视频完整科学验证。
> Metrics apply only to reviewed sparse frames and do not replace multi-video scientific validation.

```text
{summarize_metrics(metrics)}
```
"""


def _update_run_manifests(run_path: Path, validation_manifest: dict[str, Any]) -> None:
    boundary = validation_manifest.get("annotation_boundary", {})
    summary = {
        "selected_review_frames": validation_manifest["selected_review_frames"],
        "manual_review_required_for_scientific_metrics": True,
        "review_completed": bool(boundary.get("review_completed", False)),
        "manifest": validation_manifest["manifest"],
    }
    if boundary.get("review_confirmed_at"):
        summary["review_confirmed_at"] = boundary["review_confirmed_at"]
    complete_path = run_path / "complete_run_manifest.json"
    if complete_path.exists():
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            complete = {}
        complete["validation_review"] = summary
        complete_path.write_text(
            json.dumps(complete, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    run_summary_path = run_path / "run_summary.json"
    if run_summary_path.exists():
        try:
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            run_summary = {}
        output_files = run_summary.setdefault("output_files", {})
        output_files["validation_review_manifest"] = validation_manifest["manifest"]
        run_summary_path.write_text(
            json.dumps(run_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
