"""
AnimalTracker - 小动物行为追踪系统
Animal Behavior Tracking System

使用方法:
    streamlit run app.py
"""

import json
import sys
# 设置 UTF-8 输出（仅在有控制台时）
if hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.video import VideoProcessor
from utils.analytics import (
    build_results_analytics_from_counts,
    filter_frame_tracks,
    filter_track_history,
    select_stable_track_ids,
)
from utils.behavior_analytics import (
    build_behavior_analytics,
    normalize_detections,
    normalize_pose_frames,
)
from utils.group_comparison import (
    build_group_comparison,
    build_sample_row,
    discover_analysis_exports,
    infer_group_label,
    infer_pairing_label,
    load_analysis_payload,
)
from utils.interpretation import build_behavior_insights, build_parameter_interpretation
from detection.detector import (
    YOLODetector,
    get_available_inference_devices,
    resolve_inference_device,
    supports_half_precision,
)
from detection.enhanced_detector import EnhancedYOLODetector, create_enhanced_detector
from tracking.tracker import MultiObjectTracker
from tracking.improved_tracker import ImprovedTracker
from tracking.enhanced_tracker import EnhancedTracker, create_enhanced_tracker
from visualization.visualizer import TrajectoryVisualizer
from visualization.publication_figure import create_publication_trajectory_figure
from behavior.detector import SocialInteractionDetector
from evaluation.annotations import save_annotations
from utils.export import DataExporter

PROJECT_ROOT = Path(__file__).parent
RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="animaltracker-render")

try:
    from tracking.track_filter import create_track_filter
    TRACK_FILTER_AVAILABLE = True
except ImportError:
    TRACK_FILTER_AVAILABLE = False

try:
    from tracking.post_processor import stitch_frame_tracks
    OFFLINE_STITCH_AVAILABLE = True
except ImportError:
    OFFLINE_STITCH_AVAILABLE = False

# 姿态估计模块
try:
    from pose.estimator import PoseEstimator, AnimalPose
    from pose.visualizer import PoseVisualizer, PoseHistory
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False
    print("[WARNING] 姿态估计模块不可用")

# 统一追踪器工厂 — 内部路由到最优引擎
try:
    from tracking.tracker_factory import create_tracker, get_available_modes, get_tracker_name, get_mode_info
    TRACKER_FACTORY_AVAILABLE = True
except ImportError:
    TRACKER_FACTORY_AVAILABLE = False
    print("[WARNING] 追踪器工厂不可用")

try:
    from tracking.automated_mot import build_auto_mot_plan
    AUTOMATED_MOT_AVAILABLE = True
except ImportError:
    AUTOMATED_MOT_AVAILABLE = False
    print("[WARNING] AutoMOT 自动系统不可用")

# 尝试导入深度分割模块
try:
    from segmentation.deep_segmenter import DeepSegmenter
    DEEP_SEGMENTER_AVAILABLE = True
except ImportError:
    DEEP_SEGMENTER_AVAILABLE = False
    print("[WARNING] 深度分割模块不可用")

# 尝试导入自适应检测器
try:
    from detection.adaptive_detector import AdaptiveDetector
    ADAPTIVE_DETECTOR_AVAILABLE = True
except ImportError:
    ADAPTIVE_DETECTOR_AVAILABLE = False
    print("[WARNING] 自适应检测器不可用")

# ========================================
# 页面配置
# ========================================

st.set_page_config(
    page_title="AnimalTracker - 小动物行为追踪",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# 初始化 Session State
# ========================================

if 'tracking_results' not in st.session_state:
    st.session_state.tracking_results = {
        'all_tracks': {},
        'all_detections': {},
        'all_poses': {},
        'track_history': {},
        'video_info': {},
        'visualizer': None,
        'pose_visualizer': None,
        'pose_history': {},
        'interaction_detector': None,
        'tracker_stats': {},
        'tracker_name': None,
        'model_path': None,
        'confidence_threshold': None,
        'inference_config': {},
        'pose_config': {},
        'trajectory_video_path': None,
        'annotated_video_path': None,
        'dashboard_video_path': None,
        'cluster_safe_video_path': None,
        'render_jobs': {},
        'export_paths': {},
        'export_bundle_path': None,
        'completed': False
    }

if 'threshold_analysis' not in st.session_state:
    st.session_state.threshold_analysis = {}


def discover_local_models(root_dir):
    """Discover local YOLO weight files for quick switching in the UI."""
    search_roots = [
        Path(root_dir),
        Path(root_dir) / "data" / "models",
    ]
    discovered = []
    seen = set()

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for model_file in search_root.rglob("*.pt"):
            try:
                display_path = str(model_file.relative_to(root_dir)).replace("\\", "/")
            except ValueError:
                display_path = str(model_file)

            if display_path not in seen:
                seen.add(display_path)
                discovered.append(display_path)

    preferred_order = {"yolov8n.pt": 0, "yolov8s.pt": 1, "yolov8m.pt": 2, "yolov8l.pt": 3}
    discovered.sort(key=lambda item: (preferred_order.get(Path(item).name, 99), item))
    return discovered


def extract_track_position(track_data):
    """兼容不同 tracker 的轨迹位置输出。"""
    if hasattr(track_data, 'center'):
        position = track_data.center
    elif hasattr(track_data, 'get_last_position'):
        position = track_data.get_last_position()
    elif hasattr(track_data, 'get_position'):
        position = track_data.get_position()
    else:
        position = track_data

    return tuple(map(int, position))


def ensure_track_history(results, visualizer):
    """优先使用持久化的轨迹历史，必要时按帧结果重建。"""
    track_history = results.get('track_history') or {}
    if track_history:
        visualizer.track_history = {
            int(track_id): [tuple(map(int, point)) for point in points]
            for track_id, points in track_history.items()
        }
        return visualizer.track_history

    rebuilt_history = {}
    visualizer.track_history = rebuilt_history
    for frame_tracks in results.get('all_tracks', {}).values():
        visualizer.update_track_history(frame_tracks)

    results['track_history'] = visualizer.track_history
    return visualizer.track_history


def snapshot_tracks(tracks):
    """冻结当前帧轨迹，避免后续 tracker 原地更新污染历史帧。"""
    return {
        int(track_id): extract_track_position(track_data)
        for track_id, track_data in tracks.items()
    }


def limit_detections_for_tracking(detections, expected_animals=0, extra_slots=0):
    """Optionally cap detections while preserving spatially diverse animals."""
    if expected_animals <= 0:
        return detections

    max_detections = max(int(expected_animals) + max(int(extra_slots), 0), 1)
    if len(detections) <= max_detections:
        return detections

    def det_bbox(det):
        bbox = getattr(det, 'bbox', None)
        return np.array(bbox, dtype=float) if bbox is not None and len(bbox) == 4 else None

    def bbox_center(bbox):
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])

    def bbox_iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-6)

    candidates = sorted(
        detections,
        key=lambda det: float(getattr(det, 'confidence', 0.0)),
        reverse=True
    )
    selected = []
    deferred = []

    for det in candidates:
        bbox = det_bbox(det)
        if bbox is None:
            deferred.append(det)
            continue
        is_duplicate = False
        for kept in selected:
            kept_bbox = det_bbox(kept)
            if kept_bbox is None:
                continue
            iou = bbox_iou(bbox, kept_bbox)
            center_dist = float(np.linalg.norm(bbox_center(bbox) - bbox_center(kept_bbox)))
            diag = min(
                np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                np.hypot(kept_bbox[2] - kept_bbox[0], kept_bbox[3] - kept_bbox[1]),
            )
            if iou >= 0.88 or center_dist < diag * 0.18:
                is_duplicate = True
                break
        if is_duplicate:
            deferred.append(det)
            continue
        selected.append(det)
        if len(selected) >= max_detections:
            break

    if len(selected) < max_detections:
        for det in deferred:
            if all(det is not existing for existing in selected):
                selected.append(det)
            if len(selected) >= max_detections:
                break

    return selected


def select_display_tracks(frame_tracks,
                          track_history,
                          expected_animals=0,
                          extra_slots=0,
                          preferred_track_ids=None):
    """Pick a bounded set of representative tracks for per-frame visualization."""
    if not frame_tracks:
        return {}

    preferred = {int(track_id) for track_id in (preferred_track_ids or [])}
    ranked = sorted(
        frame_tracks.items(),
        key=lambda item: (
            int(item[0]) in preferred,
            len(track_history.get(int(item[0]), [])),
            int(item[0]),
        ),
        reverse=True,
    )

    if int(expected_animals) > 0:
        limit = max(int(expected_animals) + max(int(extra_slots), 0), 1)
        ranked = ranked[:limit]

    return {
        int(track_id): track_data
        for track_id, track_data in ranked
    }


def build_display_tracks(all_tracks,
                         track_history,
                         expected_animals=0,
                         extra_slots=0,
                         preferred_track_ids=None):
    """Apply representative per-frame track filtering across the full result set."""
    return {
        int(frame_idx): select_display_tracks(
            frame_tracks,
            track_history,
            expected_animals=expected_animals,
            extra_slots=extra_slots,
            preferred_track_ids=preferred_track_ids,
        )
        for frame_idx, frame_tracks in all_tracks.items()
    }


@st.cache_data(show_spinner=False)
def load_video_frame(video_path, frame_idx):
    """Load a single frame by direct seek for preview rendering."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    finally:
        cap.release()


@st.cache_data(show_spinner=False)
def get_cached_results_analytics(all_tracks, detection_counts, track_history):
    """Avoid rebuilding analytics tables on every Streamlit rerun."""
    return build_results_analytics_from_counts(all_tracks, detection_counts, track_history)


@st.cache_data(show_spinner=False)
def get_cached_behavior_analytics(all_tracks,
                                  normalized_detections,
                                  normalized_pose_frames,
                                  track_history,
                                  fps,
                                  interaction_distance,
                                  min_duration,
                                  segment_seconds,
                                  phase_config_json):
    """Avoid rebuilding behavior analytics on every rerun."""
    phase_config = None
    if phase_config_json:
        try:
            phase_config = json.loads(phase_config_json)
        except json.JSONDecodeError:
            phase_config = None
    return build_behavior_analytics(
        all_tracks,
        normalized_detections,
        track_history,
        pose_frames=normalized_pose_frames,
        fps=fps,
        interaction_distance=interaction_distance,
        min_event_frames=min_duration,
        segment_seconds=segment_seconds,
        phase_config=phase_config,
    )


@st.cache_data(show_spinner=False)
def get_cached_analysis_payload(path):
    """Load an exported analysis JSON with Streamlit caching."""
    return load_analysis_payload(path)


@st.cache_data(show_spinner=False)
def get_cached_video_probe(video_path, file_size, modified_time_ns):
    """Inspect a rendered video file and return lightweight playback metadata."""
    probe = {
        'path': video_path,
        'exists': False,
        'playable': False,
        'size_bytes': int(file_size or 0),
        'size_mb': round((file_size or 0) / (1024 * 1024), 2),
        'frame_count': 0,
        'fps': 0.0,
        'width': 0,
        'height': 0,
        'error': None,
    }
    path = Path(video_path)
    if not path.exists() or file_size <= 0:
        probe['error'] = "渲染输出文件不存在或为空"
        return probe

    probe['exists'] = True
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        probe['error'] = "视频文件已生成，但播放器无法打开容器"
        return probe

    try:
        probe['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        probe['fps'] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        probe['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        probe['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, _ = cap.read()
        probe['playable'] = bool(ok)
        if not ok:
            probe['error'] = "视频文件已生成，但无法读取首帧"
    finally:
        cap.release()

    return probe


def get_video_probe(video_path):
    """Return probe info for a rendered video path."""
    if not video_path:
        return {
            'path': None,
            'exists': False,
            'playable': False,
            'size_bytes': 0,
            'size_mb': 0.0,
            'frame_count': 0,
            'fps': 0.0,
            'width': 0,
            'height': 0,
            'error': "尚未生成视频文件",
        }

    path = Path(video_path)
    if not path.exists():
        return {
            'path': str(path),
            'exists': False,
            'playable': False,
            'size_bytes': 0,
            'size_mb': 0.0,
            'frame_count': 0,
            'fps': 0.0,
            'width': 0,
            'height': 0,
            'error': "渲染输出文件不存在",
        }

    stat = path.stat()
    return get_cached_video_probe(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@st.cache_data(show_spinner=False)
def get_cached_video_bytes(video_path, file_size, modified_time_ns):
    """Load a rendered video as bytes so Streamlit serves it with a stable MIME type."""
    return Path(video_path).read_bytes()


def render_video_probe_caption(probe):
    """Format probe info for user-facing status text."""
    parts = []
    if probe.get('frame_count'):
        parts.append(f"{probe['frame_count']} 帧")
    if probe.get('fps'):
        parts.append(f"{probe['fps']:.2f} FPS")
    if probe.get('width') and probe.get('height'):
        parts.append(f"{probe['width']}x{probe['height']}")
    if probe.get('size_mb'):
        parts.append(f"{probe['size_mb']:.2f} MB")
    return " | ".join(parts)


def create_result_visualizer(video_info):
    """Create an isolated visualizer for render jobs."""
    return TrajectoryVisualizer(
        video_info['width'],
        video_info['height'],
        max_history_length=video_info.get('frame_count')
    )


def submit_render_job(results, job_key, output_field, render_callable, *args, **kwargs):
    """Submit a render task to the background executor."""
    render_jobs = results.setdefault('render_jobs', {})
    existing = render_jobs.get(job_key)
    if existing and existing.get('status') == 'running':
        return False

    future = RENDER_EXECUTOR.submit(render_callable, *args, **kwargs)
    render_jobs[job_key] = {
        'future': future,
        'status': 'running',
        'output_field': output_field,
        'output_path': None,
        'output_probe': None,
        'error': None,
        'submitted_at': datetime.now().isoformat(timespec='seconds'),
        'completed_at': None,
    }
    return True


def refresh_render_jobs(results):
    """Harvest finished render jobs without blocking the UI thread."""
    render_jobs = results.setdefault('render_jobs', {})
    for job in render_jobs.values():
        future = job.get('future')
        if future is None or job.get('status') != 'running':
            continue

        if not future.done():
            continue

        try:
            output_path = future.result()
            results[job['output_field']] = output_path
            job['output_path'] = output_path
            job['output_probe'] = get_video_probe(output_path)
            job['completed_at'] = datetime.now().isoformat(timespec='seconds')
            if not job['output_probe'].get('playable'):
                job['status'] = 'failed'
                job['error'] = job['output_probe'].get('error') or '渲染完成，但视频文件不可播放'
            else:
                job['status'] = 'completed'
                job['error'] = None
        except Exception as exc:
            job['status'] = 'failed'
            job['error'] = str(exc)
            job['completed_at'] = datetime.now().isoformat(timespec='seconds')
        finally:
            job['future'] = None


def render_cluster_safe_dashboard(video_path, output_path, threshold=70):
    """Render the detector-locked, cluster-aware dashboard video."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.render_detection_locked_trajectory import render_video

    stats = render_video(
        Path(video_path),
        Path(output_path),
        threshold=int(threshold),
        max_frames=0,
    )
    summary_path = Path(output_path).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(Path(output_path))


@st.fragment(run_every=2)
def render_job_status_panel(results):
    """Show background render job progress with periodic refresh."""
    refresh_render_jobs(results)
    render_jobs = results.get('render_jobs', {})
    if not render_jobs:
        return

    active_jobs = [job for job in render_jobs.values() if job.get('status') == 'running']
    if active_jobs:
        st.caption("渲染任务在后台执行，状态每 2 秒自动刷新。")

    for job_key, job in render_jobs.items():
        label = {
            'trajectory_video': "动态轨迹视频",
            'annotated_video': "原视频叠加追踪",
            'dashboard_video': "轨迹看板视频",
            'cluster_safe_video': "虚拟标记轨迹看板",
        }.get(job_key, job_key)

        if job.get('status') == 'running':
            st.info(f"{label}：后台渲染中")
        elif job.get('status') == 'failed':
            st.error(f"{label}：渲染失败 - {job.get('error', 'unknown error')}")
        elif job.get('status') == 'completed':
            output_probe = job.get('output_probe') or get_video_probe(results.get(job.get('output_field')))
            summary = render_video_probe_caption(output_probe)
            success_text = f"{label}：已生成，可直接播放或下载"
            if summary:
                success_text = f"{success_text} | {summary}"
            st.success(success_text)


@st.fragment(run_every=2)
def render_video_output_panel(results, job_key, output_field, label, download_key):
    """Auto-refresh a single rendered video block with clear ready/failure hints."""
    refresh_render_jobs(results)
    render_jobs = results.setdefault('render_jobs', {})
    job = render_jobs.get(job_key, {})
    status = job.get('status')

    if status == 'running':
        st.info(f"{label} 正在后台渲染中，完成后会在这里提示。")
    elif status == 'failed':
        st.error(f"{label} 渲染失败：{job.get('error', 'unknown error')}")

    video_path = results.get(output_field)
    if not video_path:
        return

    probe = job.get('output_probe') or get_video_probe(video_path)
    if not probe.get('exists'):
        st.warning(f"{label} 输出文件尚未写出。")
        return

    if not probe.get('playable'):
        st.error(f"{label} 文件已生成，但当前播放器无法读取。")
        if probe.get('error'):
            st.caption(probe['error'])
    else:
        message = f"[OK] {label} 已渲染完成，可直接播放或下载。"
        if job.get('completed_at'):
            message += f" 完成时间：{job['completed_at']}"
        st.success(message)
        summary = render_video_probe_caption(probe)
        if summary:
            st.caption(summary)
        video_bytes = get_cached_video_bytes(
            probe['path'],
            probe['size_bytes'],
            Path(probe['path']).stat().st_mtime_ns,
        )
        st.video(video_bytes, format="video/mp4")

    download_label = f"[DOWNLOAD] 下载{label}"
    video_bytes = get_cached_video_bytes(
        probe['path'],
        probe['size_bytes'],
        Path(probe['path']).stat().st_mtime_ns,
    )
    st.download_button(
        download_label,
        data=video_bytes,
        file_name=Path(probe['path']).name,
        mime="video/mp4",
        key=f"{download_key}_download",
    )


def analyze_detection_thresholds(video_path, model_path, device="auto", imgsz=640,
                                 vid_stride=1, half=False, thresholds=None, sample_frames=20):
    """抽样分析不同置信度阈值下的检测数量。"""
    thresholds = thresholds or [0.10, 0.15, 0.20, 0.25]
    detector = YOLODetector(
        model_path,
        min(thresholds),
        device=device,
        imgsz=imgsz,
        vid_stride=vid_stride,
        half=half,
    )
    processor = VideoProcessor(video_path)
    if not processor.open():
        raise RuntimeError("无法打开视频进行阈值分析")

    stats = {
        threshold: {
            'counts': [],
            'frames_with_detection': 0,
        }
        for threshold in thresholds
    }
    confidence_values = []
    frame_confidence_rows = []

    try:
        processed_frames = 0
        for frame_idx, frame in processor.frames_generator():
            if not detector.should_process_frame(frame_idx):
                continue

            if processed_frames >= sample_frames:
                break

            counts, confidences = detector.summarize_predictions(frame, thresholds)
            confidence_values.extend(confidences)
            frame_confidence_rows.append({
                'frame': frame_idx,
                'prediction_count': len(confidences),
                'mean_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'max_confidence': float(max(confidences)) if confidences else 0.0,
            })
            for threshold in thresholds:
                count = counts.get(threshold, 0)
                stats[threshold]['counts'].append(count)
                if count > 0:
                    stats[threshold]['frames_with_detection'] += 1
            processed_frames += 1
    finally:
        processor.release()

    summary = {}
    for threshold in thresholds:
        counts = stats[threshold]['counts']
        total_frames = len(counts)
        avg_count = sum(counts) / total_frames if total_frames else 0
        frame_hit_rate = (
            stats[threshold]['frames_with_detection'] / total_frames
            if total_frames else 0
        )
        summary[threshold] = {
            'avg_detections': avg_count,
            'max_detections': max(counts) if counts else 0,
            'frame_hit_rate': frame_hit_rate,
            'sample_frames': total_frames
        }

    min_threshold = min(thresholds)
    baseline_avg = summary[min_threshold]['avg_detections']
    baseline_hit_rate = summary[min_threshold]['frame_hit_rate']
    recommended_threshold = min_threshold
    for threshold in sorted(thresholds):
        if (
            summary[threshold]['avg_detections'] >= baseline_avg * 0.8 and
            summary[threshold]['frame_hit_rate'] >= baseline_hit_rate * 0.8
        ):
            recommended_threshold = threshold

    confidence_summary = {
        'total_predictions': len(confidence_values),
        'mean_confidence': float(np.mean(confidence_values)) if confidence_values else 0.0,
        'median_confidence': float(np.median(confidence_values)) if confidence_values else 0.0,
        'p90_confidence': float(np.percentile(confidence_values, 90)) if confidence_values else 0.0,
        'below_recommended_ratio': (
            float(sum(conf < recommended_threshold for conf in confidence_values)) / len(confidence_values)
            if confidence_values else 0.0
        )
    }

    histogram_rows = []
    if confidence_values:
        histogram, bin_edges = np.histogram(confidence_values, bins=np.linspace(0.0, 1.0, 11))
        for start, end, count in zip(bin_edges[:-1], bin_edges[1:], histogram):
            histogram_rows.append({
                'confidence_bin': f"{start:.1f}-{end:.1f}",
                'count': int(count),
            })

    return {
        'summary': summary,
        'recommended_threshold': recommended_threshold,
        'confidence_summary': confidence_summary,
        'confidence_histogram': histogram_rows,
        'frame_confidence': frame_confidence_rows,
    }


def build_analysis_payload(video_info, all_tracks, track_history, tracker_stats,
                           frame_df, track_df, interactions, tracker_name,
                           video_path, model_path, confidence,
                           threshold_analysis=None, tracking_config=None,
                           inference_config=None, behavior_analysis=None,
                           parameter_interpretation=None):
    """Build a stable export payload for JSON and bundle exports."""
    return {
        'video_info': video_info,
        'config': {
            'tracker_name': tracker_name,
            'video_path': str(video_path),
            'model_path': model_path,
            'confidence_threshold': confidence,
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'tracking_config': tracking_config or {},
            'inference_config': inference_config or {},
        },
        'tracks': all_tracks,
        'track_history': track_history,
        'tracker_stats': tracker_stats,
        'frame_stats': frame_df.to_dict('records') if not frame_df.empty else [],
        'track_stats': track_df.to_dict('records') if not track_df.empty else [],
        'interactions': interactions,
        'threshold_analysis': threshold_analysis or {},
        'behavior_analysis': behavior_analysis or {},
        'parameter_interpretation': parameter_interpretation or [],
    }


# ========================================
# 自定义CSS样式
# ========================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #4CAF50;
    }
    .sub-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2E86AB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# 侧边栏配置
# ========================================

with st.sidebar:
    st.title("AnimalTracker")
    st.markdown("---")
    
    # 检测器设置
    st.subheader("检测器设置")
    available_models = discover_local_models(PROJECT_ROOT)
    default_model_path = "yolov8n.pt"
    if available_models:
        default_index = available_models.index(default_model_path) if default_model_path in available_models else 0
        selected_model_path = st.selectbox("本地 YOLO 模型", available_models, index=default_index, help="优先选择本地可用的 YOLO 权重。模型越大通常越准，但推理越慢。")
        model_path = st.text_input("YOLO 模型路径", value=selected_model_path, help="可直接填写 .pt 权重路径。")
        st.caption(f"已找到 {len(available_models)} 个本地模型权重文件")
    else:
        model_path = st.text_input("YOLO 模型路径", value=default_model_path, help="填写 YOLO 权重路径，例如 yolov8n.pt。")
    available_devices = get_available_inference_devices()
    device_options = [option[0] for option in available_devices]
    device_labels = {option[0]: option[1] for option in available_devices}
    device = st.selectbox(
        "Inference Device",
        device_options,
        index=0,
        format_func=lambda option: device_labels[option],
        help="Auto 会按 CUDA > MPS > CPU 的顺序自动选择。"
    )
    resolved_device = resolve_inference_device(device)
    imgsz = st.select_slider("imgsz", options=[320, 416, 512, 640, 768, 960, 1280], value=640, help="检测输入尺寸。越大越利于小目标，但速度越慢。")
    vid_stride = st.slider("vid_stride", 1, 8, 1, 1, help="视频采样步长。1 为逐帧分析；大于 1 会跳帧提速。")
    half_available = supports_half_precision(resolved_device)
    half = st.checkbox(
        "half (FP16)",
        value=half_available,
        disabled=not half_available,
        help="仅在支持的 GPU 上启用。通常可以提速，但极少数环境兼容性较弱。"
    )
    st.caption(f"Resolved device: {resolved_device}")
    if vid_stride > 1:
        st.caption("vid_stride > 1 会跳帧提速，但会降低时间分辨率和追踪稳定性")
    confidence = st.slider("置信度阈值", 0.03, 1.0, 0.08, 0.01, help="ByteTrack/AnimalMOT 需要保留低置信遮挡目标；误检过多时再升到 0.10-0.15。")
    st.caption("4+ 动物建议先用 0.10-0.15；误报过多再逐步调高")

    # v1.0.1 新增: 增强检测选项
    use_enhanced_detector = st.checkbox(
        "启用增强检测",
        value=True,
        help="启用图像预处理和时序平滑，提高低对比度场景的检测稳定性"
    )
    
    # 追踪器设置
    st.subheader("追踪器设置")
    max_disappeared = st.slider("最大消失帧数", 10, 100, 30, 5, help="轨迹失配后最多保留多少帧。越大越利于遮挡恢复，但 stale ID 会保留更久。")
    max_distance = st.slider("最大匹配距离", 50, 200, 100, 10, help="相邻帧允许的最大匹配位移。越大越宽松，越小越严格。")
    min_confirmed_hits = st.slider("稳定 ID 最少命中帧数", 1, 10, 3, 1, help="新轨迹至少连续命中多少次后才作为稳定 ID 输出。")
    expected_animals = st.number_input("预期动物数量 (0=不限制)", min_value=0, max_value=20, value=0, step=1, help="已知视频里有几只动物时建议填写，可明显降低碎片 ID。")
    detection_buffer = st.slider("额外检测缓冲", 0, 5, 1, 1, help="在预期动物数量之外额外保留多少个高置信检测。")
    st.caption("预期动物数量 > 0 时，只会把最高置信度的少量检测送进 tracker。")

    # v1.0.2 新增: Re-ID 追踪选项
    use_reid_matching = st.checkbox(
        "启用Re-ID身份恢复 (v1.0.2)",
        value=False,
        help="使用外观特征进行个体识别，提高遮挡场景下的身份保持能力。适用于相似外观动物较少的场景。"
    )

    st.markdown("---")
    
    # 显示选项
    st.subheader("显示选项")
    show_bbox = st.checkbox("显示边界框", value=True, help="在预览帧和视频里显示检测框。")
    show_id = st.checkbox("显示追踪ID", value=True, help="在可视化中显示个体 ID。")
    show_trail = st.checkbox("显示轨迹", value=True, help="在预览帧和视频中绘制轨迹尾迹。")
    trail_length = st.slider("轨迹尾迹长度", 15, 300, 60, 15, help="尾迹越长，路径越完整，但画面更容易拥挤。")
    min_track_frames_for_viz = st.slider("轨迹图最短持续帧数", 1, 300, 10, 1, help="低于该持续时间的短命轨迹默认不进入轨迹图。")
    default_max_tracks = int(expected_animals) if int(expected_animals) > 0 else 20
    max_visualized_tracks = st.slider("轨迹图最多显示轨迹数", 1, 50, min(default_max_tracks, 50), 1, help="限制轨迹图中展示的轨迹数量，避免画面爆炸。")

    enable_pose_analysis = False
    pose_model_path = "yolov8n-pose.pt"
    pose_confidence = 0.25
    pose_stride = 1
    show_pose = False
    show_keypoints = False
    show_orientation = False

    pose_engine_type = "yolo"
    # 姿态估计显示选项
    if POSE_AVAILABLE:
        st.markdown("---")
        st.subheader("姿态分析 (支持多引擎)")
        
        pose_engine_options = {
            "📱 YOLOv8-Pose (Top-Down)": "yolo",
            "🧬 DeepLabCut 3.0 superAnimal (Zero-Shot)": "dlc",
            "🕸️ SLEAP (Bottom-Up / PAFs)": "sleap"
        }
        selected_engine_label = st.selectbox(
            "选择姿态推理引擎", 
            list(pose_engine_options.keys()), 
            index=0,
            help="DeepLabCut 3.0 提供跨物种通用关键点；SLEAP Bottom-Up 策略在密集遮挡时更稳定。"
        )
        pose_engine_type = pose_engine_options[selected_engine_label]

        if pose_engine_type == "yolo":
            pose_model_candidates = [model for model in available_models if "pose" in Path(model).name.lower()]
            if pose_model_candidates:
                pose_model_default = pose_model_candidates[0]
                pose_model_path = st.selectbox(
                    "YOLO 姿态模型",
                    pose_model_candidates,
                    index=0,
                    help="用于关键点检测的 YOLO Pose 权重。"
                )
            else:
                pose_model_path = st.text_input(
                    "YOLO 姿态模型路径",
                    value=pose_model_path,
                    help="例如 yolov8n-pose.pt。"
                )
                st.caption("当前未发现本地 pose 权重；您可能需要提前下载。")
        elif pose_engine_type == "dlc":
            pose_model_path = "superanimal_topviewmouse"
            st.info("💡 兼容模式: 已选择 superAnimal_topviewmouse 权重预设。")
            st.caption("提示: 结合 `STCS+vmTracking` 时，DLC 提升身份锚定准确率。")
        elif pose_engine_type == "sleap":
            pose_model_path = "sleap_bottomup"
            st.info("💡 兼容模式: 已选择 SLEAP 自底向上 (Bottom-Up) 策略。")
            st.caption("提示: 即使在两只动物重叠 (遮挡) 的交叉中也能准确分离个体关键点。")
        enable_pose_analysis = st.checkbox(
            "启用姿态分析（更慢）",
            value=False,
            help="会额外做关键点检测，并用于增强站立/理毛等行为统计。"
        )
        pose_confidence = st.slider(
            "姿态置信度阈值",
            0.1,
            1.0,
            0.25,
            0.05,
            help="姿态关键点检测阈值。"
        )
        pose_stride = st.slider(
            "姿态采样步长",
            1,
            8,
            1,
            1,
            help="1 为逐帧姿态分析；更大值可以提速，但会降低姿态行为统计的时间分辨率。"
        )
        show_pose = st.checkbox("显示姿态骨架", value=False, help="在预览帧中叠加姿态骨架。")
        show_keypoints = st.checkbox("显示关键点", value=False, help="显示 nose、前后爪等关键点。")
        show_orientation = st.checkbox("显示身体朝向", value=False, help="显示由鼻子到尾根估计的身体朝向。")
    
    # 追踪模式选择 (v3.0 统一优化)
    st.markdown("---")
    st.subheader("追踪模式")

    tracker_mode_options = [
        "AutoMOT 全自动系统",
        "无标注迁移追踪 / Annotation-free Transfer",
        "AnimalByteTracker 论文级稳定",
        "AnimalMOT SOTA (最低ID Switch)",
        "🟢 快速模式",
        "🔵 标准模式 (推荐)",
        "🟣 旗舰模式 (论文级)",
    ]
    tracker_mode_keys = ['auto', 'unsupervised_transfer', 'animal_bytetrack', 'sota', 'fast', 'standard', 'flagship']

    tracker_mode_selection = st.selectbox(
        "选择追踪模式",
        tracker_mode_options,
        index=0,
        help="快速=实时预览, 标准=日常研究, 旗舰=密集场景/论文发表"
    )
    tracker_mode = tracker_mode_keys[tracker_mode_options.index(tracker_mode_selection)]

    # 模式说明
    mode_descriptions = {
        'auto': (
            "AutoMOT 全自动系统\n\n"
            "• 自动抽样诊断检测阈值\n"
            "• 自动优先选择无标注迁移追踪 / annotation-free transfer tracking\n"
            "• 自动选择追踪引擎和缓冲策略\n"
            "• 运行中监控漏检、ID 膨胀和异常跳变\n\n"
            "适合直接得到稳定结果，再按质量报告微调。"
        ),
        'unsupervised_transfer': (
            "无标注迁移追踪 / Annotation-free Transfer Tracking\n\n"
            "• 受 UDMT 启发：不需要训练标注 / no training annotations required\n"
            "• 首帧自动身份种子 + 自监督外观原型 / first-frame identity seeds + self-supervised prototypes\n"
            "• 局部时序匹配、定位精修、前后向 ID 纠错 / temporal matching, localization refinement, bidirectional ID correction\n"
            "• 面向小鼠、果蝇、线虫、斗鱼等跨物种场景 / cross-species animal tracking\n\n"
            "这是无标注多目标追踪系统，不作为普通标准 MOT 预设。"
        ),
        'fast': (
            "⚡ 实时预览/快速扫描\n\n"
            "引擎: EnhancedTracker (Kalman + 匈牙利匹配 + 光流)\n\n"
            "速度最快，适合初步查看追踪效果"
        ),
        'sota': (
            "AnimalMOT SOTA\n\n"
            "• MOTIP-lite 固定身份字典，优先复用长期 ID\n"
            "• TrackTrack-like 局部优先匹配，减少近邻错配\n"
            "引擎: AnimalMOT\n\n"
            "• ByteTrack 风格低置信二阶段关联\n"
            "• OmniTrax 风格 buffer-and-recover\n"
            "• vmTracking 风格虚拟检测补位\n"
            "• Re-ID/骨架特征作为辅助，不在聚集时强行污染身份"
        ),
        'animal_bytetrack': (
            "AnimalByteTracker 论文级稳定\n\n"
            "• CLAHE + 多阈值 + 凸缺陷分割，处理砂粒背景上的黑鼠粘连\n"
            "• Kalman 物理预测，不依赖 LK 光流纹理，避免 PVMT 在砂粒上漂移\n"
            "• ByteTrack 双阶段 + 接触期速度匹配 + EMA ID 记忆库\n\n"
            "适合 6xWT/同色小鼠/高接触视频。请把预期动物数量填成真实只数。"
        ),
        'standard': (
            "📊 日常研究/多数场景\n\n"
            "引擎: BufferRecoverTracker (OmniTrax 缓冲恢复)\n\n"
            "• 丢失轨迹缓冲恢复，减少 ID 切换\n"
            "• 深度外观嵌入 (128维特征)\n"
            "• 四阶段关联匹配"
        ),
        'flagship': (
            "🏆 论文发表/密集场景/4+动物\n\n"
            "引擎: AdaptiveTrackingPipeline\n\n"
            "• STCS 掩码光流 + vmTracking 虚拟标记\n"
            "• Kalman 平滑 + STCS Stage 2 迭代合并\n"
            "• TrackFilter 噪声清洗\n"
            "• 追踪精度最高，IDF1 95%+"
        ),
    }
    st.info(mode_descriptions[tracker_mode])
    if tracker_mode == 'animal_bytetrack' and int(expected_animals) <= 0:
        st.warning("AnimalByteTracker 需要固定身份槽。请在上方填写真实动物数量，例如 6xWT 填 6。")

    # 默认值（确保所有模式下变量都存在）
    tracker_buffer_duration = 90
    tracker_recovery_thresh = 0.55
    tracker_high_conf_thresh = 0.40
    tracker_match_threshold = 0.85
    tracker_buffer_entry_delay = 15
    tracker_use_identity_dictionary = True
    tracker_identity_match_threshold = 0.72
    tracker_local_match_radius = float(max_distance) * 0.55
    tracker_local_match_margin = 12.0
    tracker_virtual_max_age = 60
    tracker_virtual_max_hits = 120
    animal_bytetrack_threshold = 70
    animal_bytetrack_min_area = 300
    animal_bytetrack_max_area = 50000
    animal_bytetrack_lost_buffer = 240
    transfer_appearance_weight = 0.55
    transfer_motion_weight = 0.35
    transfer_anchor_weight = 0.10
    transfer_match_threshold = 0.72
    transfer_template_update_rate = 0.08
    transfer_localization_shift = 8.0
    transfer_freeze_identity_slots = True
    transfer_bidirectional_refine = True
    auto_profile_samples = 12
    auto_prefer_accuracy = True
    auto_apply_recommended_threshold = True
    enable_online_detection_filter = True
    filter_min_confidence = max(0.03, float(confidence))
    enable_offline_stitching = int(expected_animals) > 0
    use_mask_flow = True
    use_appearance_merger = True
    use_track_filter_option = True
    use_optical_flow = tracker_mode in ('fast', 'flagship')
    use_hungarian = True

    with st.expander("高级追踪参数", expanded=False):
        if tracker_mode == 'auto':
            auto_profile_samples = st.slider(
                "自动诊断抽样帧数",
                4, 40, 12, 2,
                help="开始追踪前自动抽取多少帧评估检测阈值。帧数越多越稳，启动越慢。"
            )
            auto_prefer_accuracy = st.checkbox(
                "优先身份稳定",
                value=True,
                help="启用后 AutoMOT 会优先选择 AnimalByteTracker/AnimalMOT SOTA 和更长遮挡缓冲。"
            )
            auto_apply_recommended_threshold = st.checkbox(
                "自动采用推荐检测阈值",
                value=True,
                help="基于抽样诊断结果自动调整本次追踪的检测阈值。"
            )
            enable_offline_stitching = st.checkbox(
                "分析后离线轨迹缝合",
                value=int(expected_animals) > 0,
                disabled=int(expected_animals) <= 0 or not OFFLINE_STITCH_AVAILABLE,
                help="仅在已知动物数量时启用；把断裂短轨迹缝合回长期 ID。"
            )
        elif tracker_mode == 'unsupervised_transfer':
            transfer_appearance_weight = st.slider(
                "自监督外观权重 / Self-supervised appearance weight",
                0.10, 0.85, 0.55, 0.05,
                help="提高后更依赖视频自身学习到的个体外观；降低后更依赖运动连续性。"
            )
            transfer_motion_weight = st.slider(
                "时序运动权重 / Temporal motion weight",
                0.10, 0.85, 0.35, 0.05,
                help="提高后更保守地按局部运动连续性匹配，适合低纹理或低对比视频。"
            )
            transfer_anchor_weight = st.slider(
                "首帧身份锚权重 / First-frame identity anchor weight",
                0.00, 0.40, 0.10, 0.05,
                help="保持与首帧身份种子的长期一致性，遮挡后可减少身份漂移。"
            )
            transfer_match_threshold = st.slider(
                "迁移匹配阈值 / Transfer match threshold",
                0.35, 1.20, 0.72, 0.01,
                help="越低越严格，越高越容易保持旧 ID。ID 串线时降低，断轨时提高。"
            )
            transfer_template_update_rate = st.slider(
                "原型更新率 / Prototype update rate",
                0.01, 0.30, 0.08, 0.01,
                help="控制自监督外观原型随视频变化的速度。快速变形动物可略增，外观相似时略降。"
            )
            transfer_localization_shift = st.slider(
                "定位精修上限(px) / Localization refinement limit",
                0.0, 24.0, 8.0, 1.0,
                help="允许算法在检测框附近用局部前景质心微调位置。"
            )
            transfer_freeze_identity_slots = st.checkbox(
                "冻结首帧身份槽 / Freeze first-frame identity slots",
                value=True,
                help="启用后不会把后续误检扩张成新身份，适合已知动物数量的无标注实验。"
            )
            transfer_bidirectional_refine = st.checkbox(
                "结束后前后向 ID 纠错 / Bidirectional ID correction after run",
                value=True,
                help="处理完整视频后用前向/反向一致性修复遮挡和短时 ID 漂移。"
            )
            enable_offline_stitching = False
        elif tracker_mode in ('standard', 'sota'):
            tracker_buffer_duration = st.slider(
                "缓冲区持续时间（帧）", 30, 300, 90, 10,
                help="丢失轨迹在缓冲区保留的帧数"
            )
            tracker_recovery_thresh = st.slider(
                "缓冲恢复阈值", 0.30, 0.85, 0.55, 0.05,
                help="越低越容易恢复，越高越严格"
            )
            tracker_high_conf_thresh = st.slider(
                "High/Low 置信分界 (ByteTrack)",
                0.15, 0.80, 0.40, 0.05,
                help="高于该值进入主关联；低于该值仅用于遮挡恢复，不新建 ID。"
            )
            tracker_match_threshold = st.slider(
                "综合匹配阈值",
                0.50, 0.95, 0.85, 0.05,
                help="越低越严格，越高越容易保持旧 ID；ID 串线时降低。"
            )
            tracker_buffer_entry_delay = st.slider(
                "进入缓冲前等待帧数",
                1, 60, 15, 1,
                help="短暂检测丢失时先等待，不立刻移入 buffer，减少闪断造成的 ID churn。"
            )
            if tracker_mode == 'sota':
                tracker_use_identity_dictionary = st.checkbox(
                    "启用固定身份字典 (MOTIP-lite)",
                    value=True,
                    help="把期望动物数映射为长期身份槽，遮挡/漏检后优先复用旧 ID，抑制 ID 膨胀。"
                )
                tracker_identity_match_threshold = st.slider(
                    "身份槽恢复阈值",
                    0.35, 0.95, 0.72, 0.01,
                    help="越高越保守，越低越容易把检测恢复到旧 ID。ID 膨胀时降低，ID 串线时提高。"
                )
                tracker_local_match_radius = st.slider(
                    "局部优先匹配半径",
                    10.0, 200.0, float(max(20.0, max_distance * 0.55)), 5.0,
                    help="TrackTrack-like 局部匹配半径；相邻目标未混淆时优先绑定最近轨迹。"
                )
                tracker_local_match_margin = st.slider(
                    "局部匹配最小间隔",
                    0.0, 80.0, 12.0, 1.0,
                    help="最近轨迹与第二近轨迹至少相差多少像素才直接局部绑定。"
                )
                tracker_virtual_max_age = st.slider(
                    "遮挡预测最大帧数",
                    5, 300, 60, 5,
                    help="检测缺失时继续用卡尔曼/虚拟检测维持 ID 的最长帧数。动物粘连或遮挡较久时增大。"
                )
                tracker_virtual_max_hits = st.slider(
                    "连续虚拟补帧上限",
                    5, 300, 120, 5,
                    help="限制同一 ID 连续预测补帧次数，避免长期 ghost track。"
                )
            enable_online_detection_filter = st.checkbox(
                "在线检测噪声清洗",
                value=True,
                help="进入 tracker 前过滤极小框、贴边噪声和过低置信检测。"
            )
            filter_min_confidence = st.slider(
                "清洗最小置信度",
                0.01, 0.30, float(max(0.03, confidence)), 0.01,
                help="通常应小于或等于检测阈值，避免破坏 ByteTrack 的低置信恢复。"
            )
            enable_offline_stitching = st.checkbox(
                "分析后离线轨迹缝合",
                value=int(expected_animals) > 0,
                disabled=int(expected_animals) <= 0 or not OFFLINE_STITCH_AVAILABLE,
                help="仅在已知动物数量时启用；把断裂短轨迹缝合回长期 ID。"
            )
        elif tracker_mode == 'animal_bytetrack':
            animal_bytetrack_threshold = st.slider(
                "内部阈值检测强度",
                40, 120, 70, 5,
                help="黑鼠/深色动物在浅色垫料上通常用 60-80。漏检时降低，误检多时提高。"
            )
            animal_bytetrack_min_area = st.slider(
                "最小动物轮廓面积",
                50, 3000, 300, 50,
                help="过滤砂粒噪声；小鼠顶视视频通常 200-800 起步。"
            )
            animal_bytetrack_max_area = st.slider(
                "最大粘连轮廓面积",
                5000, 120000, 50000, 1000,
                help="超过该面积的区域不作为单只动物直接使用。"
            )
            animal_bytetrack_lost_buffer = st.slider(
                "EMA 记忆保留时间（帧）",
                60, 420, 240, 10,
                help="遮挡后保留身份记忆的时长；长时间扎堆时增大。"
            )
            enable_online_detection_filter = False
            enable_offline_stitching = st.checkbox(
                "分析后离线轨迹缝合",
                value=int(expected_animals) > 0,
                disabled=int(expected_animals) <= 0 or not OFFLINE_STITCH_AVAILABLE,
                help="仅在已知动物数量时启用；把断裂短轨迹缝合回长期 ID。"
            )
        elif tracker_mode == 'flagship':
            use_mask_flow = st.checkbox("启用掩码光流", value=True, help="STCS 掩码光流形变")
            use_appearance_merger = st.checkbox("启用外观合并 (Stage 2)", value=True, help="STCS 迭代外观合并")
            use_track_filter_option = st.checkbox("启用噪声过滤", value=True, help="四层轨迹噪声清洗")

    # 交互检测设置
    st.subheader("社交交互检测")
    enable_interaction = st.checkbox("启用交互检测", value=True, help="启用后会计算接触、跟随、近距离等社交行为指标。")
    interaction_distance = st.slider("交互距离阈值", 20, 150, 50, 5, help="两只动物中心点小于该距离时，更容易判定为交互。")
    min_duration = st.slider("最小交互持续帧数", 5, 50, 10, 5, help="低于该持续时长的短暂接近会被过滤。")
    behavior_window_seconds = st.slider("行为分段窗口(秒)", 5, 120, 30, 5, help="把整段视频按固定时间窗切分，便于观察行为随时间的变化。")

    with st.expander("参数说明与调参建议", expanded=False):
        st.markdown(
            """
            **检测参数**
            - `imgsz`：控制检测输入尺寸。小目标多时优先提高；速度不够时优先降低到 `416-512`。
            - `vid_stride`：控制是否跳帧。做精细行为分析建议保持 `1`。
            - `置信度阈值`：低阈值保召回，高阈值抑制误检。两只黑鼠常用 `0.10-0.15` 起步。

            **追踪参数**
            - `最大匹配距离`：控制跨帧匹配的空间门限。太大容易串 ID，太小容易断轨。
            - `最大消失帧数`：控制遮挡恢复能力。遮挡多时增大，ghost track 多时减小。
            - `稳定 ID 最少命中帧数`：控制新轨迹确认门槛。已知个体数少时建议 `2-4`。
            - `预期动物数量`：已知场景里只有几只动物时，直接填写能显著抑制碎片 ID。

            **行为参数**
            - `交互距离阈值`：决定多近才算“发生交互”。
            - `最小交互持续帧数`：决定事件必须持续多久才计入统计。
            - `行为分段窗口(秒)`：把整段视频切成固定时间窗，便于比较前中后段行为变化。

            **可视化参数**
            - `轨迹尾迹长度`：影响视频/预览的轨迹长度。
            - `轨迹图最短持续帧数` 与 `最多显示轨迹数`：用于过滤短命误检轨迹，避免轨迹图失真。
            """
        )

# ========================================
# 主页面
# ========================================

st.markdown('<h1 class="main-header">[PAW] AnimalTracker - 小动物行为追踪系统</h1>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>欢迎使用！</h3>
    <p>本系统用于追踪和分析小动物的社交行为模式。</p>
    <p><b>功能：</b></p>
    <ul>
        <li>多动物实时追踪</li>
        <li>社交交互检测</li>
        <li>轨迹可视化</li>
        <li>统计分析报告</li>
    </ul>
    <p><i>基于 STCS 框架理念开发</i></p>
</div>
""", unsafe_allow_html=True)

# ========================================
# 视频输入
# ========================================

st.markdown('<h2 class="sub-header">[VIDEO] 视频输入</h2>', unsafe_allow_html=True)

# 模式选择
mode = st.radio(
    "选择模式",
    ["[FILE] 视频文件", "[CAMERA] 实时摄像头"],
    horizontal=True
)

if mode == "[FILE] 视频文件":
    video_file = st.file_uploader(
        "上传视频文件",
        type=["mp4", "avi", "mov", "mpg", "mpeg", "mkv"],
        help="支持最大 1GB 视频文件"
    )
elif mode == "[CAMERA] 实时摄像头":
    st.info("[INFO] 实时摄像头模式需要本地摄像头支持")
    if st.button("[START] 启动摄像头", type="primary"):
        st.warning("[WARNING] 实时摄像头模式正在开发中...")
        # TODO: 实现实时摄像头功能

if mode == "[FILE] 视频文件" and video_file is not None:
    st.success("[OK] 视频上传成功!")
    
    # 保存视频
    video_path = f"data/videos/{video_file.name}"
    Path("data/videos").mkdir(exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    # 视频信息
    with st.spinner("正在读取视频信息..."):
        try:
            use_buffer_recover = True
            BUFFER_RECOVER_AVAILABLE = True
            use_vmtracker_v2 = False
            VMTRACKER_V2_AVAILABLE = False
            use_cosine_vmtracker = False
            COSINE_VMTRACKER_AVAILABLE = False
            use_stcs_vmtracker = False
            STCS_VMTRACKER_AVAILABLE = False
            use_vm_tracker = False
            VMTRACKER_AVAILABLE = False
            use_enhanced_tracker = False
            use_anti_churn = False
            ANTICHURN_AVAILABLE = False

            processor = VideoProcessor(video_path)
            if processor.open():
                info = processor.get_info()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("帧率", f"{info['fps']:.2f}")
                with col2:
                    st.metric("总帧数", info['frame_count'])
                with col3:
                    st.metric("分辨率", f"{info['width']}x{info['height']}")
                with col4:
                    st.metric("时长(秒)", f"{info['duration']:.2f}")
                
                processor.release()
            else:
                st.error("[ERROR] 无法打开视频文件")
        except Exception as e:
            st.error(f"[ERROR] 处理视频失败: {e}")

    st.markdown('<h2 class="sub-header">[DIAGNOSIS] 检测诊断</h2>', unsafe_allow_html=True)
    analysis_key = f"{Path(video_path).name}|{model_path}|{device}|{imgsz}|{vid_stride}|{half}"
    analysis_result = st.session_state.threshold_analysis.get(analysis_key)

    if st.button("[ANALYZE] 分析推荐检测阈值", key="analyze_thresholds"):
        with st.spinner("正在抽样分析不同阈值下的检测表现..."):
            try:
                analysis_result = analyze_detection_thresholds(
                    video_path,
                    model_path,
                    device=device,
                    imgsz=imgsz,
                    vid_stride=vid_stride,
                    half=half,
                )
                st.session_state.threshold_analysis[analysis_key] = analysis_result
            except Exception as e:
                st.error(f"[ERROR] 检测阈值分析失败: {e}")
                analysis_result = None

    if analysis_result:
        rows = []
        for threshold, summary in analysis_result['summary'].items():
            rows.append({
                'threshold': threshold,
                'avg_detections': round(summary['avg_detections'], 2),
                'max_detections': summary['max_detections'],
                'frame_hit_rate': f"{summary['frame_hit_rate']:.0%}",
                'sample_frames': summary['sample_frames'],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        confidence_summary = analysis_result.get('confidence_summary', {})
        confidence_histogram = analysis_result.get('confidence_histogram', [])
        frame_confidence = analysis_result.get('frame_confidence', [])
        st.info(
            f"推荐阈值: {analysis_result['recommended_threshold']:.2f}。"
            " 规则是尽量在保留检测覆盖率的同时使用更高阈值，减少误报。"
        )
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Sample Predictions", confidence_summary.get('total_predictions', 0))
        with summary_col2:
            st.metric("Mean Confidence", f"{confidence_summary.get('mean_confidence', 0.0):.2f}")
        with summary_col3:
            st.metric("Median Confidence", f"{confidence_summary.get('median_confidence', 0.0):.2f}")
        with summary_col4:
            st.metric("Below Rec. Threshold", f"{confidence_summary.get('below_recommended_ratio', 0.0):.0%}")

        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            if confidence_histogram:
                hist_df = pd.DataFrame(confidence_histogram).set_index('confidence_bin')
                st.caption("Confidence Distribution")
                st.bar_chart(hist_df, use_container_width=True)
        with diag_col2:
            if frame_confidence:
                frame_conf_df = pd.DataFrame(frame_confidence).set_index('frame')
                st.caption("Frame Confidence Trend")
                st.line_chart(
                    frame_conf_df[['mean_confidence', 'max_confidence']],
                    use_container_width=True
                )

    # ========================================
    # 开始追踪
    # ========================================
    
    st.markdown('<h2 class="sub-header">[TRACKING] 行为追踪</h2>', unsafe_allow_html=True)
    
    if st.button("[START] 开始追踪", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            n_animals = int(expected_animals) if int(expected_animals) > 0 else (0 if tracker_mode == 'auto' else 4)
            effective_confidence = float(confidence)
            auto_plan_preview = None
            if tracker_mode == 'auto' and AUTOMATED_MOT_AVAILABLE:
                auto_profile = analysis_result
                if auto_profile is None:
                    status_text.text("[AUTO] 正在抽样诊断检测阈值...")
                    try:
                        auto_profile = analyze_detection_thresholds(
                            video_path,
                            model_path,
                            device=device,
                            imgsz=imgsz,
                            vid_stride=vid_stride,
                            half=half,
                            sample_frames=int(auto_profile_samples),
                        )
                        st.session_state.threshold_analysis[analysis_key] = auto_profile
                    except Exception as exc:
                        st.warning(f"[AUTO] 自动阈值诊断失败，将沿用当前阈值: {exc}")
                        auto_profile = None
                if auto_profile is not None:
                    analysis_result = auto_profile

                plan_profile = auto_profile if auto_apply_recommended_threshold else None
                auto_plan_preview = build_auto_mot_plan(
                    n_animals=n_animals,
                    max_distance=max_distance,
                    max_disappeared=max_disappeared,
                    user_confidence_threshold=confidence,
                    min_confirmed_hits=min_confirmed_hits,
                    detection_buffer=detection_buffer,
                    prefer_accuracy=auto_prefer_accuracy,
                    use_reid=use_reid_matching,
                    video_info=info,
                    detection_profile=plan_profile,
                )
                if auto_apply_recommended_threshold:
                    effective_confidence = auto_plan_preview.confidence_threshold
                status_text.text(
                    "[AUTO] 方案就绪: "
                    f"threshold={effective_confidence:.2f}, "
                    f"engine={auto_plan_preview.tracker_mode}, "
                    f"max_distance={auto_plan_preview.max_distance:.0f}"
                )

            # 初始化检测器
            detector_runtime_config = {}
            if tracker_mode == 'animal_bytetrack':
                detector = None
                detector_runtime_config = {
                    'detector_type': 'AnimalByteTracker internal',
                    'threshold': animal_bytetrack_threshold,
                    'min_area': animal_bytetrack_min_area,
                    'max_area': animal_bytetrack_max_area,
                }
                status_text.text("[INFO] AnimalByteTracker 使用内部 CLAHE/阈值/凸缺陷检测")
            elif use_enhanced_detector:
                # 使用增强检测器
                detector = create_enhanced_detector(
                    model_path=model_path,
                    confidence=effective_confidence,
                    enable_preprocessing=True
                )
                detector_runtime_config = detector.get_runtime_config()
                status_text.text("[INFO] 使用增强检测器 (预处理 + 时序平滑)")
            else:
                # 使用标准检测器
                detector = YOLODetector(
                    model_path,
                    effective_confidence,
                    device=device,
                    imgsz=imgsz,
                    vid_stride=vid_stride,
                    half=half,
                )
                detector_runtime_config = detector.get_runtime_config()
                status_text.text("[INFO] 使用标准检测器")

            # 根据已知个体数限制并发轨迹槽位，抑制误检触发的新 ID 爆炸
            track_slot_limit = None
            if int(expected_animals) > 0:
                track_slot_limit = max(int(expected_animals) + max(int(detection_buffer), 0), 1)

            # 使用统一工厂创建追踪器
            factory_kwargs = {}

            if tracker_mode == 'auto':
                factory_kwargs['video_info'] = info
                factory_kwargs['detection_profile'] = (
                    analysis_result if auto_apply_recommended_threshold else None
                )
                factory_kwargs['user_confidence_threshold'] = effective_confidence
                factory_kwargs['min_confirmed_hits'] = min_confirmed_hits
                factory_kwargs['detection_buffer'] = detection_buffer
                factory_kwargs['prefer_accuracy'] = auto_prefer_accuracy
                if auto_plan_preview is not None:
                    factory_kwargs['plan'] = auto_plan_preview
            elif tracker_mode == 'unsupervised_transfer':
                factory_kwargs['min_confirmed_hits'] = min_confirmed_hits
                factory_kwargs['appearance_weight'] = transfer_appearance_weight
                factory_kwargs['motion_weight'] = transfer_motion_weight
                factory_kwargs['anchor_weight'] = transfer_anchor_weight
                factory_kwargs['match_threshold'] = transfer_match_threshold
                factory_kwargs['template_update_rate'] = transfer_template_update_rate
                factory_kwargs['max_localization_shift'] = transfer_localization_shift
                factory_kwargs['freeze_identity_slots'] = transfer_freeze_identity_slots
                factory_kwargs['enable_bidirectional_refine'] = transfer_bidirectional_refine
            elif tracker_mode in ('standard', 'sota'):
                factory_kwargs['buffer_duration'] = tracker_buffer_duration
                factory_kwargs['recovery_threshold'] = tracker_recovery_thresh
                factory_kwargs['high_conf_thresh'] = tracker_high_conf_thresh
                factory_kwargs['match_threshold'] = tracker_match_threshold
                factory_kwargs['buffer_entry_delay'] = tracker_buffer_entry_delay
                factory_kwargs['dist_thresh'] = max_distance
                factory_kwargs['use_deep_embedding'] = use_reid_matching
                factory_kwargs['min_confirmation_hits'] = min_confirmed_hits
                if tracker_mode == 'sota':
                    factory_kwargs['use_identity_dictionary'] = tracker_use_identity_dictionary
                    factory_kwargs['identity_match_threshold'] = tracker_identity_match_threshold
                    factory_kwargs['local_match_radius'] = tracker_local_match_radius
                    factory_kwargs['local_match_margin'] = tracker_local_match_margin
                    factory_kwargs['virtual_max_age'] = tracker_virtual_max_age
                    factory_kwargs['virtual_max_hits'] = tracker_virtual_max_hits
            elif tracker_mode == 'animal_bytetrack':
                factory_kwargs['threshold'] = animal_bytetrack_threshold
                factory_kwargs['min_area'] = animal_bytetrack_min_area
                factory_kwargs['max_area'] = animal_bytetrack_max_area
                factory_kwargs['lost_buffer'] = animal_bytetrack_lost_buffer
            elif tracker_mode == 'flagship':
                factory_kwargs['use_mask_flow'] = use_mask_flow
                factory_kwargs['use_appearance_merger'] = use_appearance_merger
                factory_kwargs['use_track_filter'] = use_track_filter_option

            if TRACKER_FACTORY_AVAILABLE:
                tracker = create_tracker(
                    mode=tracker_mode,
                    max_distance=max_distance,
                    max_disappeared=max_disappeared,
                    n_animals=n_animals,
                    use_reid=use_reid_matching,
                    **factory_kwargs
                )
                tracker_name = get_tracker_name(tracker_mode)
            else:
                # 回退: 直接用增强追踪器
                tracker = create_enhanced_tracker(
                    max_distance=max_distance,
                    max_disappeared=max_disappeared,
                )
                tracker_name = "EnhancedTracker (回退)"

            status_text.text(f"[INFO] 追踪器就绪: {tracker_name}")

            # 打开视频
            processor = VideoProcessor(video_path)
            processor.open()

            total_frames = processor.frame_count
            all_tracks = {}
            all_detections = {}

            # 初始化可视化器
            visualizer = TrajectoryVisualizer(
                info['width'],
                info['height'],
                max_history_length=info.get('frame_count')
            )
            all_poses = {}
            pose_estimator = None
            pose_visualizer = None
            pose_history = None
            if enable_pose_analysis and POSE_AVAILABLE:
                from pose.estimator import create_pose_estimator
                pose_estimator = create_pose_estimator(
                    model_type=pose_engine_type,
                    confidence_threshold=pose_confidence,
                    device=resolved_device,
                    imgsz=imgsz,
                    half=half and supports_half_precision(resolved_device),
                )
                pose_visualizer = PoseVisualizer()
                pose_history = PoseHistory(max_length=info.get('frame_count') or 100)
            interaction_detector = None
            if enable_interaction:
                interaction_detector = SocialInteractionDetector(
                    interaction_distance=interaction_distance,
                    min_duration=min_duration
                )

            # 处理每一帧
            online_track_filter = None
            if enable_online_detection_filter and TRACK_FILTER_AVAILABLE:
                online_track_filter = create_track_filter(
                    max_speed=max_distance * 1.5,
                    min_confidence=filter_min_confidence,
                    min_frames=max(1, int(min_confirmed_hits)),
                    frame_size=(info['width'], info['height']),
                )
            track_filter_stats = {
                'input_count': 0,
                'output_count': 0,
                'area_filtered': 0,
                'confidence_filtered': 0,
                'border_filtered': 0,
                'speed_filtered': 0,
            }

            for frame_idx, frame in processor.frames_generator():
                if detector is not None and not detector.should_process_frame(frame_idx):
                    progress = (frame_idx + 1) / total_frames
                    progress_bar.progress(progress)
                    continue

                # 检测。AnimalByteTracker 使用内部阈值/凸缺陷检测，避免外部 NMS 合并粘连小鼠。
                if tracker_mode == 'animal_bytetrack':
                    detections = []
                else:
                    detections = detector.detect(frame)

                if tracker_mode not in ('auto', 'animal_bytetrack') and online_track_filter is not None:
                    detections = online_track_filter.filter_detections(
                        detections,
                        frame_size=(info['width'], info['height'])
                    )
                    current_filter_stats = online_track_filter.get_stats()
                    for stat_key, stat_value in current_filter_stats.items():
                        if isinstance(stat_value, (int, float)):
                            track_filter_stats[stat_key] = (
                                track_filter_stats.get(stat_key, 0) + stat_value
                            )
                if tracker_mode in ('auto', 'animal_bytetrack'):
                    tracking_detections = detections
                else:
                    tracking_detections = limit_detections_for_tracking(
                        detections,
                        expected_animals=expected_animals,
                        extra_slots=detection_buffer
                    )

                # 根据追踪器类型调用不同的更新方法
                if use_buffer_recover and BUFFER_RECOVER_AVAILABLE:
                    tracks = tracker.update(frame, tracking_detections)
                elif use_vmtracker_v2 and VMTRACKER_V2_AVAILABLE:
                    tracks = tracker.update(frame, tracking_detections)
                elif use_cosine_vmtracker and COSINE_VMTRACKER_AVAILABLE:
                    tracks = tracker.update(frame, tracking_detections)
                elif use_stcs_vmtracker and STCS_VMTRACKER_AVAILABLE:
                    tracks = tracker.update(frame, tracking_detections)
                elif use_vm_tracker and VMTRACKER_AVAILABLE:
                    tracks = tracker.update(frame, tracking_detections)
                elif use_enhanced_tracker:
                    tracks = tracker.update(frame, tracking_detections)
                elif use_anti_churn and ANTICHURN_AVAILABLE:
                    # 反ID碎片追踪器需要 frame 和 detections
                    tracks = tracker.update(frame, tracking_detections)
                else:
                    # 其他追踪器使用检测中心点
                    detection_centers = [det.center for det in tracking_detections]
                    tracks = tracker.update(detection_centers)

                if hasattr(tracker, 'last_tracking_detections'):
                    tracking_detections = tracker.last_tracking_detections

                track_snapshot = snapshot_tracks(tracks)
                
                # 保存追踪结果
                all_tracks[frame_idx] = track_snapshot
                all_detections[frame_idx] = tracking_detections
                
                # 更新轨迹历史
                visualizer.update_track_history(track_snapshot)

                if pose_estimator is not None and frame_idx % max(int(pose_stride), 1) == 0:
                    frame_poses = pose_estimator.detect_with_tracking(frame, track_snapshot)
                    all_poses[frame_idx] = frame_poses
                    if pose_history is not None:
                        for pose_track_id, pose in frame_poses.items():
                            pose_history.add_pose(pose_track_id, pose)
                elif pose_estimator is not None:
                    all_poses[frame_idx] = {}

                if interaction_detector is not None:
                    interaction_detector.detect_interactions(track_snapshot, frame_idx)
                
                # 更新进度
                progress = (frame_idx + 1) / total_frames
                progress_bar.progress(progress)
            
            processor.release()
            transfer_refine_stats = {}
            if tracker_mode == 'unsupervised_transfer' and hasattr(tracker, 'finalize_tracks'):
                status_text.text("[TRANSFER] 正在执行前后向 ID 纠错 / Running bidirectional ID correction...")
                refined_tracks, transfer_refine_stats = tracker.finalize_tracks(
                    total_frames=total_frames,
                    frame_size=(info['width'], info['height']),
                )
                if refined_tracks:
                    all_tracks = refined_tracks
                    visualizer = TrajectoryVisualizer(
                        info['width'],
                        info['height'],
                        max_history_length=info.get('frame_count')
                    )
                    for _, frame_tracks in sorted(all_tracks.items()):
                        visualizer.update_track_history(frame_tracks)

            offline_stitch_stats = {}
            if enable_offline_stitching and int(expected_animals) > 0 and OFFLINE_STITCH_AVAILABLE:
                all_tracks, offline_stitch_stats = stitch_frame_tracks(
                    all_tracks,
                    n_animals=int(expected_animals),
                    max_distance=max_distance * 3,
                    max_time_gap=max_disappeared * 3,
                )
                visualizer = TrajectoryVisualizer(
                    info['width'],
                    info['height'],
                    max_history_length=info.get('frame_count')
                )
                for _, frame_tracks in sorted(all_tracks.items()):
                    visualizer.update_track_history(frame_tracks)

            auto_annotation_path = None
            auto_label_stats = {}
            publication_figure_path = None
            if tracker_mode == 'auto':
                status_text.text("[AUTO] 正在生成全自动伪标注与论文级轨迹图...")
                auto_target_animals = int(getattr(tracker, 'n_animals', n_animals) or n_animals or 0)
                if hasattr(tracker, 'finalize_auto_annotations') and auto_target_animals > 0:
                    try:
                        auto_annotations, auto_label_stats = tracker.finalize_auto_annotations(
                            total_frames=total_frames,
                            frame_size=(info['width'], info['height']),
                            video_id=Path(video_path).stem,
                        )
                        if auto_annotations:
                            auto_annotation_path = Path("data/auto_annotations") / f"{Path(video_path).stem}_automot_annotations.json"
                            save_annotations(
                                auto_annotation_path,
                                auto_annotations,
                                videos=[{
                                    "id": Path(video_path).stem,
                                    "path": video_path,
                                    "width": info['width'],
                                    "height": info['height'],
                                    "fps": info['fps'],
                                    "frame_count": info['frame_count'],
                                    "source": "automot_pseudo_ground_truth",
                                }],
                            )
                    except Exception as exc:
                        auto_label_stats = {"error": str(exc)}
                        st.warning(f"[AUTO] 自动伪标注生成失败: {exc}")

                try:
                    tracker_stats_for_figure = tracker.get_stats() if hasattr(tracker, 'get_stats') else {}
                    auto_quality_stats = tracker_stats_for_figure.get('auto_mot', {}).get('quality', {})
                    publication_figure_path = Path("data/results") / f"{Path(video_path).stem}_automot_publication_trajectory.png"
                    create_publication_trajectory_figure(
                        all_tracks,
                        info,
                        publication_figure_path,
                        title=f"{Path(video_path).stem} - AutoMOT research trajectory",
                        auto_label_stats=auto_label_stats,
                        quality_stats=auto_quality_stats,
                    )
                except Exception as exc:
                    publication_figure_path = None
                    st.warning(f"[AUTO] 论文级轨迹图生成失败: {exc}")

            tracker_stats_snapshot = tracker.get_stats() if hasattr(tracker, 'get_stats') else {}

            # 保存追踪结果到 session_state
            st.session_state.tracking_results = {
                'all_tracks': all_tracks,
                'all_detections': all_detections,
                'all_poses': all_poses,
                'track_history': visualizer.track_history if hasattr(visualizer, 'track_history') else {},
                'video_info': info,
                'visualizer': visualizer,
                'pose_visualizer': pose_visualizer,
                'pose_history': pose_history.history if pose_history is not None else {},
                'detector': detector,  # 保存detector以访问interactions
                'interaction_detector': interaction_detector,
                'tracker_stats': tracker_stats_snapshot,
                'track_filter_stats': track_filter_stats,
                'offline_stitch_stats': offline_stitch_stats,
                'transfer_refine_stats': transfer_refine_stats,
                'auto_annotation_path': str(auto_annotation_path) if auto_annotation_path else None,
                'auto_label_stats': auto_label_stats,
                'publication_figure_path': str(publication_figure_path) if publication_figure_path else None,
                'tracker_name': tracker_name,
                'model_path': model_path,
                'confidence_threshold': effective_confidence,
                'inference_config': detector_runtime_config,
                'pose_config': {
                    'enabled': bool(pose_estimator is not None),
                    'model_path': pose_model_path,
                    'confidence_threshold': pose_confidence,
                    'pose_stride': int(pose_stride),
                },
                'video_path': video_path,
                'trajectory_video_path': None,
                'annotated_video_path': None,
                'dashboard_video_path': None,
                'cluster_safe_video_path': None,
                'render_jobs': {},
                'export_paths': {},
                'export_bundle_path': None,
                'completed': True
            }

            status_text.text("[OK] 追踪完成!")
            st.success(f"[OK] 成功追踪 {len(all_tracks)} 帧")

        except Exception as e:
            st.error(f"[ERROR] 追踪失败: {e}")
            st.exception(e)

# ========================================
# 显示追踪结果 (如果在 session_state 中有数据)
# ========================================

if st.session_state.tracking_results['completed']:
    results = st.session_state.tracking_results
    all_tracks = results['all_tracks']
    all_detections = results['all_detections']
    all_poses = results.get('all_poses', {})
    info = results['video_info']
    visualizer = results.get('visualizer')
    pose_visualizer = results.get('pose_visualizer')
    if visualizer is None:
        visualizer = TrajectoryVisualizer(
            info['width'],
            info['height'],
            max_history_length=info.get('frame_count')
        )
        results['visualizer'] = visualizer
    if POSE_AVAILABLE and pose_visualizer is None and results.get('pose_config', {}).get('enabled'):
        pose_visualizer = PoseVisualizer()
        results['pose_visualizer'] = pose_visualizer

    track_history = ensure_track_history(results, visualizer)
    detector = results.get('detector', None)
    interaction_detector = results.get('interaction_detector', None)
    tracker_stats = results.get('tracker_stats', {})
    track_filter_stats = results.get('track_filter_stats', {})
    offline_stitch_stats = results.get('offline_stitch_stats', {})
    transfer_refine_stats = results.get('transfer_refine_stats', {})
    auto_label_stats = results.get('auto_label_stats', {})
    auto_annotation_path = results.get('auto_annotation_path')
    publication_figure_path = results.get('publication_figure_path')
    tracker_name = results.get('tracker_name', '未知追踪器')
    video_path = results.get('video_path', video_file)
    detection_counts = {
        frame_idx: len(detections)
        for frame_idx, detections in all_detections.items()
    }
    result_module = st.radio(
        "结果模块",
        ["总览", "行为", "可视化", "导出/组间"],
        horizontal=True,
        key=f"result_module_{Path(video_path).stem}",
    )
    st.caption("结果页按模块懒加载。未打开的模块不会重建图表、行为分析和导出 payload。")

    show_overview = result_module == "总览"
    show_behavior = result_module == "行为"
    show_visualization = result_module == "可视化"
    show_export = result_module == "导出/组间"
    need_frame_analytics = show_overview or show_export
    need_behavior_analytics = show_behavior or show_export
    need_visual_tracks = show_overview or show_visualization

    frame_df = pd.DataFrame()
    track_df = pd.DataFrame()
    if need_frame_analytics:
        frame_df, track_df = get_cached_results_analytics(all_tracks, detection_counts, track_history)

    video_duration_s = float(info.get('duration') or 0.0)
    if video_duration_s <= 0 and info.get('frame_count') and info.get('fps'):
        video_duration_s = float(info['frame_count']) / max(float(info['fps']), 1e-6)

    inference_config = results.get('inference_config', {})
    result_model_path = results.get('model_path', model_path)
    result_confidence = results.get('confidence_threshold', confidence)
    threshold_key = (
        f"{Path(video_path).name}|{result_model_path}|"
        f"{inference_config.get('device_preference', device)}|"
        f"{inference_config.get('imgsz', imgsz)}|"
        f"{inference_config.get('vid_stride', vid_stride)}|"
        f"{inference_config.get('requested_half', half)}"
    )
    threshold_snapshot = st.session_state.threshold_analysis.get(threshold_key, {})

    phase_config = {
        'mode': 'auto',
        'labels': ['early', 'middle', 'late'],
        'boundaries_s': [],
    }
    phase_ui_prefix = f"phase_{Path(video_path).stem}"
    if need_behavior_analytics:
        with st.expander("实验阶段设置", expanded=False):
            phase_mode_label = st.radio(
                "阶段划分方式",
                ["自动三段", "手动三段"],
                horizontal=True,
                key=f"{phase_ui_prefix}_mode"
            )
            if phase_mode_label == "手动三段":
                if video_duration_s <= 1.5:
                    st.warning("当前视频时长过短，暂时回退到自动三段切分。")
                else:
                    label_col1, label_col2, label_col3 = st.columns(3)
                    with label_col1:
                        phase_label_1 = st.text_input("阶段1名称", value="适应期", key=f"{phase_ui_prefix}_label_1")
                    with label_col2:
                        phase_label_2 = st.text_input("阶段2名称", value="刺激期", key=f"{phase_ui_prefix}_label_2")
                    with label_col3:
                        phase_label_3 = st.text_input("阶段3名称", value="恢复期", key=f"{phase_ui_prefix}_label_3")

                    first_boundary_default = round(video_duration_s / 3.0, 1)
                    second_boundary_default = round(video_duration_s * 2.0 / 3.0, 1)
                    max_first_boundary = max(video_duration_s - 1.0, 0.5)
                    max_second_boundary = max(video_duration_s - 0.5, 1.0)

                    boundary_col1, boundary_col2 = st.columns(2)
                    with boundary_col1:
                        first_boundary = st.number_input(
                            "第一阶段结束时间(秒)",
                            min_value=0.5,
                            max_value=float(max_first_boundary),
                            value=float(min(first_boundary_default, max_first_boundary)),
                            step=0.5,
                            key=f"{phase_ui_prefix}_boundary_1"
                        )
                    with boundary_col2:
                        second_boundary = st.number_input(
                            "第二阶段结束时间(秒)",
                            min_value=1.0,
                            max_value=float(max_second_boundary),
                            value=float(min(max(second_boundary_default, first_boundary + 0.5), max_second_boundary)),
                            step=0.5,
                            key=f"{phase_ui_prefix}_boundary_2"
                        )

                    first_boundary = min(float(first_boundary), max_first_boundary)
                    second_boundary = min(max(float(second_boundary), first_boundary + 0.5), max_second_boundary)
                    phase_config = {
                        'mode': 'manual',
                        'labels': [
                            phase_label_1.strip() or "适应期",
                            phase_label_2.strip() or "刺激期",
                            phase_label_3.strip() or "恢复期",
                        ],
                        'boundaries_s': [round(first_boundary, 2), round(second_boundary, 2)],
                    }
                    st.caption(
                        f"{phase_config['labels'][0]}: 0-{phase_config['boundaries_s'][0]:.1f}s | "
                        f"{phase_config['labels'][1]}: {phase_config['boundaries_s'][0]:.1f}-{phase_config['boundaries_s'][1]:.1f}s | "
                        f"{phase_config['labels'][2]}: {phase_config['boundaries_s'][1]:.1f}-{video_duration_s:.1f}s"
                    )
                    st.caption("分段窗口会按时间窗中点归属到对应实验阶段。")
            else:
                st.caption("自动把整段视频均分为前 / 中 / 后三段，适合先快速看整体变化。")

    tracking_config = {
        'max_disappeared': max_disappeared,
        'max_distance': max_distance,
        'min_confirmed_hits': min_confirmed_hits,
        'expected_animals': int(expected_animals),
        'detection_buffer': int(detection_buffer),
        'use_optical_flow': use_optical_flow,
        'use_hungarian_matching': use_hungarian,
        'interaction_enabled': enable_interaction,
        'interaction_distance': interaction_distance,
        'min_duration': min_duration,
        'behavior_window_seconds': behavior_window_seconds,
        'phase_mode': phase_config.get('mode', 'auto'),
        'phase_labels': phase_config.get('labels', []),
        'phase_boundaries_s': phase_config.get('boundaries_s', []),
        'trail_length': trail_length,
        'min_track_frames_for_viz': min_track_frames_for_viz,
        'max_visualized_tracks': max_visualized_tracks,
        'pose_enabled': results.get('pose_config', {}).get('enabled', False),
        'pose_model_path': results.get('pose_config', {}).get('model_path'),
        'pose_confidence_threshold': results.get('pose_config', {}).get('confidence_threshold'),
        'pose_stride': results.get('pose_config', {}).get('pose_stride', 1),
    }
    behavior_analysis = {}
    behavior_summary = {}
    behavior_event_rows = []
    behavior_individual_rows = []
    behavior_pair_rows = []
    behavior_count_rows = []
    behavior_frame_rows = []
    behavior_segment_rows = []
    behavior_phase_rows = []
    behavior_group_stats = {}
    parameter_interpretation = []
    behavior_insights = []
    if need_behavior_analytics:
        normalized_detections = normalize_detections(all_detections)
        normalized_pose_frames = normalize_pose_frames(all_poses)
        phase_config_json = json.dumps(phase_config, ensure_ascii=False, sort_keys=True)
        behavior_analysis = get_cached_behavior_analytics(
            all_tracks,
            normalized_detections,
            normalized_pose_frames,
            track_history,
            info.get('fps') or 10.0,
            interaction_distance,
            min_duration,
            behavior_window_seconds,
            phase_config_json,
        )
        behavior_summary = behavior_analysis.get('summary', {})
        behavior_event_rows = behavior_analysis.get('event_rows', [])
        behavior_individual_rows = behavior_analysis.get('individual_stats', [])
        behavior_pair_rows = behavior_analysis.get('pair_stats', [])
        behavior_count_rows = behavior_analysis.get('behavior_counts', [])
        behavior_frame_rows = behavior_analysis.get('frame_stats', [])
        behavior_segment_rows = behavior_analysis.get('segment_stats', [])
        behavior_phase_rows = behavior_analysis.get('phase_comparison', [])
        behavior_group_stats = behavior_analysis.get('group_stats', {})
        parameter_interpretation = build_parameter_interpretation(
            tracker_name,
            tracking_config=tracking_config,
            inference_config={**inference_config, 'confidence_threshold': result_confidence},
            threshold_analysis=threshold_snapshot,
        )
        behavior_insights = build_behavior_insights(
            behavior_analysis,
            tracker_stats=tracker_stats,
            expected_animals=int(expected_animals),
        )
        behavior_analysis = {
            **behavior_analysis,
            'insights': behavior_insights,
        }

    stable_track_ids = []
    visual_track_history = {}
    visual_tracks = {}
    display_tracks = {}
    if need_visual_tracks:
        stable_track_ids = select_stable_track_ids(
            track_history,
            min_duration_frames=min_track_frames_for_viz,
            max_tracks=max_visualized_tracks
        )
        visual_track_history = filter_track_history(track_history, stable_track_ids)
        visual_tracks = filter_frame_tracks(all_tracks, stable_track_ids)
        display_tracks = build_display_tracks(
            all_tracks,
            track_history,
            expected_animals=expected_animals,
            extra_slots=detection_buffer,
            preferred_track_ids=stable_track_ids,
        )
    render_job_status_panel(results)

    if show_overview:
        # 显示追踪统计
        st.markdown('<h2 class="sub-header">[STATS] 追踪统计</h2>', unsafe_allow_html=True)
        unique_ids = set()
        for frame_tracks in all_tracks.values():
            unique_ids.update(frame_tracks.keys())

        detection_counts_values = [len(d) for d in all_detections.values()]
        avg_detections = sum(detection_counts_values) / len(detection_counts_values) if detection_counts_values else 0
        max_detections = max(detection_counts_values) if detection_counts_values else 0
        avg_tracks_per_frame = (
            sum(len(t) for t in all_tracks.values()) / len(all_tracks)
            if all_tracks else 0
        )
        id_churn = len(unique_ids) / max(avg_tracks_per_frame, 1.0)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("追踪到的个体数", len(unique_ids))
        with col2:
            st.metric("平均每帧个体数", f"{avg_tracks_per_frame:.2f}")
        with col3:
            st.metric("平均检测数", f"{avg_detections:.1f}")
        with col4:
            st.metric("最大检测数", max_detections)
        with col5:
            st.metric("ID Churn代理", f"{id_churn:.2f}")

        if stable_track_ids:
            st.caption(
                f"轨迹可视化当前只显示 {len(stable_track_ids)} 条稳定轨迹："
                f"持续至少 {min_track_frames_for_viz} 帧，最多展示 {max_visualized_tracks} 条。"
            )
            st.caption("单帧预览和结果视频优先显示当前帧的代表性轨迹；轨迹图和动态轨迹视频仍聚焦稳定轨迹。")

        if avg_detections < 2.0:
            st.warning(
                "[WARNING] 当前平均检测数偏低，主要瓶颈更可能在检测器。"
                " 建议先把置信度降到 0.10-0.15，或换更大的 YOLO 模型。"
            )
        elif id_churn > 3.0:
            st.warning(
                "[WARNING] 当前 ID churn 偏高，同一只动物可能频繁产生新 ID。"
                " 建议优先使用 vmTracker，并适当降低 max_distance 或提高检测稳定性。"
            )

        if stable_track_ids and len(unique_ids) > len(stable_track_ids) * 3:
            st.warning(
                f"[WARNING] 历史上累计产生了 {len(unique_ids)} 个 ID，但只有 {len(stable_track_ids)} 条轨迹足够稳定。"
                " 这通常意味着误检或丢轨后不断重建新 ID。"
            )

        if tracker_stats:
            st.markdown('<h2 class="sub-header">[TRACKER] Tracker 诊断</h2>', unsafe_allow_html=True)
            st.caption(f"当前追踪器: {tracker_name}")

            tracker_metric_items = []
            if 'match_rate' in tracker_stats:
                tracker_metric_items.append(("匹配率", f"{tracker_stats['match_rate']:.2%}"))
            if 'id_switches' in tracker_stats:
                tracker_metric_items.append(("ID切换计数", tracker_stats['id_switches']))
            if 'active_tracks' in tracker_stats:
                tracker_metric_items.append(("活跃轨迹缓存", tracker_stats['active_tracks']))
            if 'suppressed_new_tracks' in tracker_stats:
                tracker_metric_items.append(("抑制新ID次数", tracker_stats['suppressed_new_tracks']))
            if 'total_created' in tracker_stats:
                tracker_metric_items.append(("创建ID总数", tracker_stats['total_created']))
            if 'total_recovered' in tracker_stats:
                tracker_metric_items.append(("恢复ID次数", tracker_stats['total_recovered']))
            if 'virtual_injections' in tracker_stats:
                tracker_metric_items.append(("虚拟检测补位", tracker_stats['virtual_injections']))
            if 'skeleton_assisted_matches' in tracker_stats:
                tracker_metric_items.append(("骨架辅助匹配", tracker_stats['skeleton_assisted_matches']))
            if offline_stitch_stats.get('merged_tracks'):
                tracker_metric_items.append(("离线缝合轨迹", offline_stitch_stats['merged_tracks']))
            if track_filter_stats.get('input_count'):
                tracker_metric_items.append(("噪声过滤保留", f"{track_filter_stats.get('output_count', 0)}/{track_filter_stats.get('input_count', 0)}"))
            if 'vm_active' in tracker_stats:
                tracker_metric_items.append(("虚拟标记数", tracker_stats['vm_active']))
            if 'anchor_coverage' in tracker_stats:
                tracker_metric_items.append(("Anchor覆盖率", f"{tracker_stats['anchor_coverage']:.2%}"))

            if tracker_metric_items:
                tracker_cols = st.columns(len(tracker_metric_items))
                for col, (label, value) in zip(tracker_cols, tracker_metric_items):
                    with col:
                        st.metric(label, value)

            auto_mot_stats = tracker_stats.get('auto_mot', {}) if isinstance(tracker_stats, dict) else {}
            if auto_mot_stats:
                st.markdown('<h2 class="sub-header">[AUTO] AutoMOT 自动科研输出</h2>', unsafe_allow_html=True)
                auto_plan = auto_mot_stats.get('plan', {})
                auto_quality = auto_mot_stats.get('quality', {})
                auto_cols = st.columns(5)
                with auto_cols[0]:
                    st.metric("自动目标数", auto_plan.get('target_animals', '-'))
                with auto_cols[1]:
                    st.metric("自动引擎", auto_plan.get('tracker_mode', '-'))
                with auto_cols[2]:
                    st.metric("质量分", f"{auto_quality.get('quality_score', 0.0):.2f}")
                with auto_cols[3]:
                    st.metric("伪标注数", int(auto_label_stats.get('output_annotations', 0)) if auto_label_stats else 0)
                with auto_cols[4]:
                    st.metric("插值标注", int(auto_label_stats.get('interpolated_annotations', 0)) if auto_label_stats else 0)

                if auto_annotation_path:
                    st.caption(f"全自动伪真值标注: {auto_annotation_path}")
                if publication_figure_path:
                    st.caption(f"论文级轨迹图: {publication_figure_path}")

            transfer_stats = {}
            if isinstance(tracker_stats, dict):
                transfer_stats = tracker_stats.get('unsupervised_transfer', {})
                if not transfer_stats:
                    transfer_stats = tracker_stats.get('child_tracker', {}).get('unsupervised_transfer', {})
            if transfer_stats:
                st.markdown('<h2 class="sub-header">[TRANSFER] 无标注迁移追踪 / Annotation-free Transfer</h2>', unsafe_allow_html=True)
                transfer_cols = st.columns(5)
                with transfer_cols[0]:
                    st.metric("身份槽 / ID slots", transfer_stats.get('identity_slots', '-'))
                with transfer_cols[1]:
                    st.metric("自监督更新 / Updates", transfer_stats.get('self_supervised_updates', 0))
                with transfer_cols[2]:
                    st.metric("定位精修 / Refined", transfer_stats.get('localization_refinements', 0))
                with transfer_cols[3]:
                    st.metric("匹配率 / Match rate", f"{transfer_stats.get('match_rate', 0.0):.2%}")
                with transfer_cols[4]:
                    bidirectional = transfer_refine_stats or transfer_stats.get('bidirectional_refine', {})
                    st.metric("双向纠错 / Bidir", "ON" if bidirectional.get('enabled') else "OFF")
                st.caption(
                    "方法来源: 首帧身份种子、自监督外观原型、局部时序匹配、定位精修、前后向 ID 纠错。"
                    " / Method: first-frame identity seeds, self-supervised prototypes, temporal matching, localization refinement, bidirectional ID correction."
                )

        if not frame_df.empty:
            st.markdown('<h2 class="sub-header">[ANALYTICS] 数据趋势</h2>', unsafe_allow_html=True)

            analytics_col1, analytics_col2 = st.columns(2)
            with analytics_col1:
                st.caption("逐帧检测 / 追踪趋势")
                st.line_chart(
                    frame_df.set_index('frame')[['detections', 'tracks']],
                    use_container_width=True
                )
            with analytics_col2:
                st.caption("累计 ID 增长")
                st.line_chart(
                    frame_df.set_index('frame')[['cumulative_ids']],
                    use_container_width=True
                )

            gap_ratio = frame_df['gap_frame'].mean() if 'gap_frame' in frame_df else 0
            analytics_metric_col1, analytics_metric_col2, analytics_metric_col3 = st.columns(3)
            with analytics_metric_col1:
                st.metric("无检测帧占比", f"{gap_ratio:.1%}")
            with analytics_metric_col2:
                st.metric("总轨迹段数", len(track_df) if not track_df.empty else 0)
            with analytics_metric_col3:
                longest_track = int(track_df['duration_frames'].max()) if not track_df.empty else 0
                st.metric("最长轨迹帧数", longest_track)

        if not track_df.empty:
            st.markdown('<h2 class="sub-header">[TRACKS] 轨迹分析</h2>', unsafe_allow_html=True)

            top_tracks = track_df.head(15).set_index('track_id')
            track_chart_col1, track_chart_col2 = st.columns(2)
            with track_chart_col1:
                st.caption("最长轨迹 Top 15")
                st.bar_chart(top_tracks[['duration_frames']], use_container_width=True)
            with track_chart_col2:
                st.caption("路径长度 Top 15")
                st.bar_chart(top_tracks[['path_length_px']], use_container_width=True)

            st.dataframe(track_df.head(20), use_container_width=True)

    if show_overview:
        st.stop()

    if show_behavior and behavior_summary:
        st.markdown('<h2 class="sub-header">[BEHAVIOR] 行为统计</h2>', unsafe_allow_html=True)
        if behavior_summary.get("pose_coverage_ratio", 0.0) > 0:
            st.caption("站立、理毛当前为姿态辅助候选统计；其余交互、跟随、抱团和空间利用率基于轨迹时序关系计算。")
        else:
            st.caption("站立、理毛、回避当前为轨迹/检测框启发式候选统计；交互、跟随、抱团和空间利用率基于轨迹时序关系计算。")

        behavior_metric_cols = st.columns(6)
        with behavior_metric_cols[0]:
            st.metric("交互事件", behavior_summary.get("interaction_events", 0))
        with behavior_metric_cols[1]:
            st.metric("跟随事件", behavior_summary.get("following_events", 0))
        with behavior_metric_cols[2]:
            st.metric("追逐候选", behavior_summary.get("pursuit_events", 0))
        with behavior_metric_cols[3]:
            st.metric("攻击接近候选", behavior_summary.get("aggressive_approach_events", 0))
        with behavior_metric_cols[4]:
            st.metric("抱团事件", behavior_summary.get("huddling_events", 0))
        with behavior_metric_cols[5]:
            st.metric("平均速度(px/s)", behavior_summary.get("avg_speed_px_per_s", 0.0))

        behavior_metric_cols2 = st.columns(6)
        with behavior_metric_cols2[0]:
            st.metric("站立候选", behavior_summary.get("standing_candidate_bouts", 0))
        with behavior_metric_cols2[1]:
            st.metric("理毛候选", behavior_summary.get("grooming_candidate_bouts", 0))
        with behavior_metric_cols2[2]:
            st.metric("逃避候选", behavior_summary.get("escape_events", 0))
        with behavior_metric_cols2[3]:
            st.metric("贴边探索", behavior_summary.get("edge_exploration_bouts", 0))
        with behavior_metric_cols2[4]:
            st.metric("平均覆盖率", f"{behavior_summary.get('mean_arena_coverage_ratio', 0.0):.1%}")
        with behavior_metric_cols2[5]:
            st.metric("平均近邻距离", behavior_summary.get("mean_nearest_neighbor_distance_px", 0.0))

        behavior_metric_cols3 = st.columns(3)
        with behavior_metric_cols3[0]:
            st.metric("平均贴边占比", f"{behavior_summary.get('mean_edge_ratio', 0.0):.1%}")
        with behavior_metric_cols3[1]:
            st.metric("分段窗口数", behavior_summary.get("segment_count", 0))
        with behavior_metric_cols3[2]:
            st.metric("阶段数", behavior_summary.get("phase_count", 0))

        if behavior_summary.get("pose_coverage_ratio", 0.0) > 0:
            pose_metric_cols = st.columns(3)
            with pose_metric_cols[0]:
                st.metric("姿态覆盖率", f"{behavior_summary.get('pose_coverage_ratio', 0.0):.1%}")
            with pose_metric_cols[1]:
                st.metric("平均有效关键点", behavior_summary.get('mean_valid_pose_keypoints', 0.0))
            with pose_metric_cols[2]:
                st.metric("姿态采样步长", results.get('pose_config', {}).get('pose_stride', 1))

        if behavior_count_rows:
            behavior_counts_df = pd.DataFrame(behavior_count_rows)
            st.caption("行为事件分布")
            st.bar_chart(
                behavior_counts_df.set_index("behavior")[["count"]],
                use_container_width=True
            )

        if behavior_frame_rows:
            behavior_frame_df = pd.DataFrame(behavior_frame_rows)
            behavior_chart_col1, behavior_chart_col2 = st.columns(2)
            with behavior_chart_col1:
                st.caption("群体接触 / 距离趋势")
                st.line_chart(
                    behavior_frame_df.set_index("frame")[["close_pair_count", "mean_pair_distance_px"]],
                    use_container_width=True
                )
            with behavior_chart_col2:
                st.caption("空间利用趋势")
                st.line_chart(
                    behavior_frame_df.set_index("frame")[["center_track_ratio", "edge_track_ratio"]],
                    use_container_width=True
                )

        if behavior_group_stats:
            st.caption("群体统计")
            st.dataframe(pd.DataFrame([behavior_group_stats]), use_container_width=True)

        if behavior_segment_rows:
            behavior_segment_df = pd.DataFrame(behavior_segment_rows)
            st.caption(f"分段行为统计（每段 {behavior_window_seconds} 秒）")
            segment_chart_col1, segment_chart_col2 = st.columns(2)
            with segment_chart_col1:
                st.line_chart(
                    behavior_segment_df.set_index("window_label")[["interaction_events", "following_events", "pursuit_events"]],
                    use_container_width=True
                )
            with segment_chart_col2:
                st.line_chart(
                    behavior_segment_df.set_index("window_label")[["mean_pair_distance_px", "escape_events"]],
                    use_container_width=True
                )
            st.dataframe(behavior_segment_df, use_container_width=True)

        if behavior_phase_rows:
            behavior_phase_df = pd.DataFrame(behavior_phase_rows)
            if phase_config.get('mode') == 'manual':
                st.caption(
                    "阶段行为对比（手动实验阶段）: "
                    + " / ".join(phase_config.get('labels', []))
                )
            else:
                st.caption("阶段行为对比（自动前 / 中 / 后）")
            phase_chart_col1, phase_chart_col2 = st.columns(2)
            with phase_chart_col1:
                st.bar_chart(
                    behavior_phase_df.set_index("phase")[["interaction_events", "following_events", "pursuit_events", "escape_events"]],
                    use_container_width=True
                )
            with phase_chart_col2:
                st.bar_chart(
                    behavior_phase_df.set_index("phase")[["mean_pair_distance_px", "mean_active_tracks"]],
                    use_container_width=True
                )
            st.dataframe(behavior_phase_df, use_container_width=True)

        if behavior_individual_rows:
            st.caption("个体行为统计")
            st.dataframe(pd.DataFrame(behavior_individual_rows), use_container_width=True)

        if behavior_pair_rows:
            st.caption("双体交互统计")
            st.dataframe(pd.DataFrame(behavior_pair_rows), use_container_width=True)

        if behavior_event_rows:
            st.caption("行为事件明细")
            st.dataframe(pd.DataFrame(behavior_event_rows).head(100), use_container_width=True)

    if show_behavior and behavior_insights:
        st.markdown('<h2 class="sub-header">[INSIGHT] 行为洞察与见解</h2>', unsafe_allow_html=True)
        for insight in behavior_insights:
            st.info(
                f"**{insight.get('headline', '行为洞察')}**\n\n"
                f"{insight.get('detail', '')}\n\n"
                f"建议：{insight.get('action', '')}"
            )

    if show_behavior and parameter_interpretation:
        st.markdown('<h2 class="sub-header">[GUIDE] 参数解读</h2>', unsafe_allow_html=True)
        st.caption("下面的解释基于你这次实际运行所使用的检测、追踪和行为参数。")
        st.dataframe(pd.DataFrame(parameter_interpretation), use_container_width=True, hide_index=True)

    if show_behavior:
        if not behavior_summary and not behavior_insights and not parameter_interpretation:
            st.info("当前结果还没有可展示的行为统计。")
        st.stop()

    if show_visualization:
        st.markdown('<h2 class="sub-header">[VISUALIZATION] 轨迹可视化</h2>', unsafe_allow_html=True)

        if len(all_tracks) > 0:
            frame_options = sorted(all_tracks.keys())
            selected_frame_index = st.slider(
                "选择帧序号",
                min_value=0,
                max_value=len(frame_options)-1,
                value=0,
                step=1
            )
            selected_frame = frame_options[selected_frame_index]
            frame = load_video_frame(video_path, selected_frame)

            if frame is not None:
                raw_tracks = all_tracks.get(selected_frame, {})
                tracks = display_tracks.get(selected_frame, raw_tracks)
                detections = all_detections.get(selected_frame, [])
                frame_poses = all_poses.get(selected_frame, {})

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("检测 + 追踪")
                    annotated_frame = visualizer.draw_detections(
                        frame, detections,
                        show_bbox=show_bbox,
                        show_id=show_id
                    )
                    annotated_frame = visualizer.draw_tracks(
                        annotated_frame, tracks,
                        show_id=show_id,
                        show_trail=show_trail,
                        trail_length=trail_length
                    )
                    if show_pose and pose_visualizer and frame_poses:
                        annotated_frame = pose_visualizer.draw_poses(
                            annotated_frame,
                            list(frame_poses.values()),
                            show_keypoints=show_keypoints,
                            show_skeleton=True,
                            show_bbox=False,
                            show_id=show_id
                        )
                        if show_orientation:
                            for pose in frame_poses.values():
                                annotated_frame = pose_visualizer.draw_orientation(annotated_frame, pose)
                    st.image(annotated_frame, channels="BGR", width="stretch")
                    if len(raw_tracks) != len(tracks):
                        st.caption(
                            f"当前帧原始追踪数 {len(raw_tracks)}，预览中仅展示 {len(tracks)} 条代表性轨迹，"
                            f"以便与已知个体数 {int(expected_animals)} 对齐。"
                        )

                with col2:
                    st.subheader("信息")
                    positions = []
                    for k, v in list(tracks.items())[:5]:
                        pos_tuple = extract_track_position(v)
                        positions.append(f"ID{k}:{pos_tuple}")

                    info_text = f"""
                    帧号: {selected_frame}
                    追踪数: {len(tracks)}
                    原始追踪数: {len(raw_tracks)}
                    检测数: {len(detections)}
                    姿态数: {len(frame_poses)}
                    位置: {', '.join(positions)}
                    """
                    st.info(info_text)

        if publication_figure_path and Path(publication_figure_path).exists():
            st.markdown('<h2 class="sub-header">[PAPER] 论文级轨迹图</h2>', unsafe_allow_html=True)
            st.image(str(publication_figure_path), width="stretch")
            st.caption(publication_figure_path)

        st.markdown('<h2 class="sub-header">[TRAJECTORY] 轨迹地图</h2>', unsafe_allow_html=True)

        if len(all_tracks) > 0:
            if st.button("[GENERATE] 生成轨迹地图", key="gen_trajectory"):
                with st.spinner("正在生成轨迹地图..."):
                    try:
                        if visual_track_history:
                            trajectory_map = visualizer.create_trajectory_map(
                                visual_track_history,
                                info['width'],
                                info['height']
                            )
                            st.subheader("轨迹地图")
                            st.image(trajectory_map, channels="BGR", width="stretch")
                        else:
                            st.warning("[WARNING] 没有可用的轨迹数据")
                    except Exception as e:
                        st.error(f"[ERROR] 生成轨迹地图失败: {e}")

        st.markdown('<h2 class="sub-header">[ANIMATION] 动态轨迹可视化</h2>', unsafe_allow_html=True)

        if len(all_tracks) > 0:
            if st.button("[RENDER] 生成动态轨迹视频", key="gen_trajectory_video"):
                output_path = Path("data/results") / f"{Path(video_path).stem}_trajectory_animation.mp4"
                submitted = submit_render_job(
                    results,
                    "trajectory_video",
                    "trajectory_video_path",
                    create_result_visualizer(info).create_trajectory_animation,
                    visual_tracks,
                    str(output_path),
                    fps=info.get('fps') or 10.0,
                    trail_length=trail_length,
                    show_id=show_id
                )
                if submitted:
                    st.success("[OK] 动态轨迹视频已提交到后台渲染")
                else:
                    st.info("[INFO] 动态轨迹视频正在后台渲染中")

            render_video_output_panel(
                results,
                "trajectory_video",
                "trajectory_video_path",
                "动态轨迹视频",
                "trajectory_video",
            )

        st.markdown('<h2 class="sub-header">[VIDEO] 结果视频</h2>', unsafe_allow_html=True)

        if len(all_tracks) > 0:
            video_col1, video_col2, video_col3 = st.columns(3)

            with video_col1:
                if st.button("[RENDER] 生成原视频叠加追踪", key="gen_annotated_video"):
                    output_path = Path("data/results") / f"{Path(video_path).stem}_annotated_tracking.mp4"
                    submitted = submit_render_job(
                        results,
                        "annotated_video",
                        "annotated_video_path",
                        create_result_visualizer(info).render_annotated_video,
                        video_path,
                        display_tracks,
                        all_detections,
                        str(output_path),
                        all_poses=all_poses,
                        fps=info.get('fps') or 10.0,
                        show_bbox=show_bbox,
                        show_id=show_id,
                        show_trail=show_trail,
                        trail_length=trail_length,
                        show_pose=show_pose,
                        show_keypoints=show_keypoints,
                        show_orientation=show_orientation,
                    )
                    if submitted:
                        st.success("[OK] 原视频叠加追踪已提交到后台渲染")
                    else:
                        st.info("[INFO] 原视频叠加追踪正在后台渲染中")

                render_video_output_panel(
                    results,
                    "annotated_video",
                    "annotated_video_path",
                    "原视频叠加追踪",
                    "annotated_video",
                )

            with video_col2:
                if st.button("[RENDER] 生成轨迹看板视频", key="gen_dashboard_video"):
                    output_path = Path("data/results") / f"{Path(video_path).stem}_tracking_dashboard.mp4"
                    submitted = submit_render_job(
                        results,
                        "dashboard_video",
                        "dashboard_video_path",
                        create_result_visualizer(info).render_dashboard_video,
                        video_path,
                        display_tracks,
                        all_detections,
                        str(output_path),
                        all_poses=all_poses,
                        fps=info.get('fps') or 10.0,
                        show_bbox=show_bbox,
                        show_id=show_id,
                        show_trail=show_trail,
                        trail_length=trail_length,
                        show_pose=show_pose,
                        show_keypoints=show_keypoints,
                        show_orientation=show_orientation,
                    )
                    if submitted:
                        st.success("[OK] 轨迹看板视频已提交到后台渲染")
                    else:
                        st.info("[INFO] 轨迹看板视频正在后台渲染中")

                render_video_output_panel(
                    results,
                    "dashboard_video",
                    "dashboard_video_path",
                    "轨迹看板视频",
                    "dashboard_video",
                )

            with video_col3:
                if st.button("[RENDER] 生成虚拟标记轨迹看板", key="gen_cluster_safe_video"):
                    output_path = Path("data/results") / f"{Path(video_path).stem}_virtual_marker_dashboard.mp4"
                    submitted = submit_render_job(
                        results,
                        "cluster_safe_video",
                        "cluster_safe_video_path",
                        render_cluster_safe_dashboard,
                        video_path,
                        str(output_path),
                    )
                    if submitted:
                        st.success("[OK] 虚拟标记轨迹看板已提交到后台渲染")
                    else:
                        st.info("[INFO] 虚拟标记轨迹看板正在后台渲染中")

                render_video_output_panel(
                    results,
                    "cluster_safe_video",
                    "cluster_safe_video_path",
                    "虚拟标记轨迹看板",
                    "cluster_safe_video",
                )

    if show_visualization:
        st.stop()

    # ========================================
    # 数据导出
    # ========================================

    st.markdown('<h2 class="sub-header">[EXPORT] 数据导出</h2>', unsafe_allow_html=True)
    export_prefix = Path(video_path).stem
    exporter = DataExporter()
    export_paths = results.setdefault('export_paths', {})
    inference_config = results.get('inference_config', {})
    result_model_path = results.get('model_path', model_path)
    result_confidence = results.get('confidence_threshold', confidence)
    threshold_key = (
        f"{Path(video_path).name}|{result_model_path}|"
        f"{inference_config.get('device_preference', device)}|"
        f"{inference_config.get('imgsz', imgsz)}|"
        f"{inference_config.get('vid_stride', vid_stride)}|"
        f"{inference_config.get('requested_half', half)}"
    )
    threshold_snapshot = st.session_state.threshold_analysis.get(threshold_key, {})
    interactions = (
        interaction_detector.interactions
        if interaction_detector and hasattr(interaction_detector, 'interactions')
        else []
    )
    tracking_config = {
        'max_disappeared': max_disappeared,
        'max_distance': max_distance,
        'min_confirmed_hits': min_confirmed_hits,
        'expected_animals': int(expected_animals),
        'detection_buffer': int(detection_buffer),
        'use_optical_flow': use_optical_flow,
        'use_hungarian_matching': use_hungarian,
        'interaction_enabled': enable_interaction,
        'interaction_distance': interaction_distance,
        'min_duration': min_duration,
        'behavior_window_seconds': behavior_window_seconds,
        'phase_mode': phase_config.get('mode', 'auto'),
        'phase_labels': phase_config.get('labels', []),
        'phase_boundaries_s': phase_config.get('boundaries_s', []),
        'trail_length': trail_length,
        'min_track_frames_for_viz': min_track_frames_for_viz,
        'max_visualized_tracks': max_visualized_tracks,
    }
    analysis_data = build_analysis_payload(
        info,
        all_tracks,
        track_history,
        tracker_stats,
        frame_df,
        track_df,
        interactions,
        tracker_name,
        video_path,
        result_model_path,
        result_confidence,
        threshold_analysis=threshold_snapshot,
        tracking_config=tracking_config,
        inference_config=inference_config,
        behavior_analysis=behavior_analysis,
        parameter_interpretation=parameter_interpretation,
    )
    analysis_data['automot_artifacts'] = {
        'auto_annotation_path': auto_annotation_path,
        'auto_label_stats': auto_label_stats,
        'publication_figure_path': publication_figure_path,
    }
    if auto_annotation_path and Path(auto_annotation_path).exists():
        export_paths['automot_annotations_json'] = auto_annotation_path
    if publication_figure_path and Path(publication_figure_path).exists():
        export_paths['automot_publication_png'] = publication_figure_path

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if st.button("[CSV] 导出追踪数据", key="export_csv_tracks"):
            exporter = DataExporter()
            path = exporter.export_tracks_to_csv(
                all_tracks,
                filename=f"{export_prefix}_tracks.csv"
            )
            export_paths['tracks_csv'] = path
            st.success(f"[OK] 已导出: {path}")

    with col2:
        if st.button("[CSV] 导出交互数据", key="export_csv_interactions"):
            exporter = DataExporter()
            interactions = (
                interaction_detector.interactions
                if interaction_detector and hasattr(interaction_detector, 'interactions')
                else []
            )
            path = exporter.export_interactions_to_csv(
                interactions,
                filename=f"{export_prefix}_interactions.csv"
            )
            export_paths['interactions_csv'] = path
            st.success(f"[OK] 已导出: {path}")

    with col3:
        if st.button("[JSON] 导出完整数据", key="export_json"):
            exporter = DataExporter()
            interactions = (
                interaction_detector.interactions
                if interaction_detector and hasattr(interaction_detector, 'interactions')
                else []
            )
            data = {
                'video_info': info,
                'tracks': all_tracks,
                'track_history': track_history,
                'tracker_stats': tracker_stats,
                'frame_stats': frame_df.to_dict('records') if not frame_df.empty else [],
                'track_stats': track_df.to_dict('records') if not track_df.empty else [],
                'interactions': interactions
            }
            path = exporter.export_to_json(
                analysis_data,
                filename=f"{export_prefix}_analysis.json"
            )
            export_paths['analysis_json'] = path
            st.success(f"[OK] 已导出: {path}")

    with col4:
        if st.button("[TXT] 生成分析报告", key="export_report"):
            exporter = DataExporter()
            interactions = (
                interaction_detector.interactions
                if interaction_detector and hasattr(interaction_detector, 'interactions')
                else []
            )
            path = exporter.export_summary_report(
                all_tracks,
                interactions,
                info,
                filename=f"{export_prefix}_report.txt",
                tracker_stats=tracker_stats,
                frame_stats=frame_df.to_dict('records') if not frame_df.empty else [],
                track_stats=track_df.to_dict('records') if not track_df.empty else [],
                behavior_analysis=behavior_analysis,
                parameter_interpretation=parameter_interpretation,
            )
            export_paths['report_txt'] = path
            st.success(f"[OK] 报告已生成: {path}")

            # 显示报告内容
            with open(path, 'r', encoding='utf-8') as f:
                report_text = f.read()
            st.text(report_text, height=300)

    with col5:
        if st.button("[CSV] 导出逐帧统计", key="export_frame_stats"):
            exporter = DataExporter()
            path = exporter.export_records_to_csv(
                frame_df.to_dict('records') if not frame_df.empty else [],
                filename=f"{export_prefix}_frame_stats.csv"
            )
            export_paths['frame_stats_csv'] = path
            st.success(f"[OK] 已导出: {path}")

    with col6:
        if st.button("[CSV] 导出轨迹统计", key="export_track_stats"):
            exporter = DataExporter()
            path = exporter.export_records_to_csv(
                track_df.to_dict('records') if not track_df.empty else [],
                filename=f"{export_prefix}_track_stats.csv"
            )
            export_paths['track_stats_csv'] = path
            st.success(f"[OK] 已导出: {path}")

    behavior_export_col1, behavior_export_col2, behavior_export_col3 = st.columns(3)
    with behavior_export_col1:
        if st.button("[CSV] 导出行为事件", key="export_behavior_events"):
            path = exporter.export_records_to_csv(
                behavior_event_rows,
                filename=f"{export_prefix}_behavior_events.csv"
            )
            export_paths['behavior_events_csv'] = path
            st.success(f"[OK] 已导出: {path}")

    with behavior_export_col2:
        if st.button("[CSV] 导出个体行为统计", key="export_behavior_individual"):
            path = exporter.export_records_to_csv(
                behavior_individual_rows,
                filename=f"{export_prefix}_behavior_individual.csv"
            )
            export_paths['behavior_individual_csv'] = path
            st.success(f"[OK] 已导出: {path}")

    with behavior_export_col3:
        if st.button("[CSV] 导出双体行为统计", key="export_behavior_pair"):
            path = exporter.export_records_to_csv(
                behavior_pair_rows,
                filename=f"{export_prefix}_behavior_pairs.csv"
            )
            export_paths['behavior_pair_csv'] = path
            st.success(f"[OK] 已导出: {path}")

    behavior_export_col4, behavior_export_col5, behavior_export_col6 = st.columns(3)
    with behavior_export_col4:
        if st.button("[CSV] 导出行为帧统计", key="export_behavior_frame"):
            path = exporter.export_records_to_csv(
                behavior_frame_rows,
                filename=f"{export_prefix}_behavior_frame_stats.csv"
            )
            export_paths['behavior_frame_csv'] = path
            st.success(f"[OK] 已导出: {path}")

    with behavior_export_col5:
        if st.button("[CSV] 导出群体行为统计", key="export_behavior_group"):
            path = exporter.export_records_to_csv(
                [behavior_group_stats] if behavior_group_stats else [],
                filename=f"{export_prefix}_behavior_group_stats.csv"
            )
            export_paths['behavior_group_csv'] = path
            st.success(f"[OK] 已导出: {path}")
    with behavior_export_col6:
        if st.button("[CSV] 导出分段行为统计", key="export_behavior_segment"):
            path = exporter.export_records_to_csv(
                behavior_segment_rows,
                filename=f"{export_prefix}_behavior_segment_stats.csv"
            )
            export_paths['behavior_segment_csv'] = path
            st.success(f"[OK] 已导出: {path}")

    behavior_export_col7 = st.columns(1)[0]
    with behavior_export_col7:
        if st.button("[CSV] 导出阶段行为对比", key="export_behavior_phase"):
            path = exporter.export_records_to_csv(
                behavior_phase_rows,
                filename=f"{export_prefix}_behavior_phase_comparison.csv"
            )
            export_paths['behavior_phase_csv'] = path
            st.success(f"[OK] 已导出: {path}")
    bundle_col1, bundle_col2 = st.columns([1, 2])
    with bundle_col1:
        if st.button("[ZIP] 一键导出全部", key="export_bundle"):
            tracks_path = export_paths.get('tracks_csv') or exporter.export_tracks_to_csv(
                all_tracks,
                filename=f"{export_prefix}_tracks.csv"
            )
            interactions_path = export_paths.get('interactions_csv') or exporter.export_interactions_to_csv(
                interactions,
                filename=f"{export_prefix}_interactions.csv"
            )
            json_path = export_paths.get('analysis_json') or exporter.export_to_json(
                analysis_data,
                filename=f"{export_prefix}_analysis.json"
            )
            report_path = export_paths.get('report_txt') or exporter.export_summary_report(
                all_tracks,
                interactions,
                info,
                filename=f"{export_prefix}_report.txt",
                tracker_stats=tracker_stats,
                frame_stats=frame_df.to_dict('records') if not frame_df.empty else [],
                track_stats=track_df.to_dict('records') if not track_df.empty else [],
                behavior_analysis=behavior_analysis,
                parameter_interpretation=parameter_interpretation,
            )
            frame_stats_path = export_paths.get('frame_stats_csv') or exporter.export_records_to_csv(
                frame_df.to_dict('records') if not frame_df.empty else [],
                filename=f"{export_prefix}_frame_stats.csv"
            )
            track_stats_path = export_paths.get('track_stats_csv') or exporter.export_records_to_csv(
                track_df.to_dict('records') if not track_df.empty else [],
                filename=f"{export_prefix}_track_stats.csv"
            )
            behavior_events_path = export_paths.get('behavior_events_csv') or exporter.export_records_to_csv(
                behavior_event_rows,
                filename=f"{export_prefix}_behavior_events.csv"
            )
            behavior_individual_path = export_paths.get('behavior_individual_csv') or exporter.export_records_to_csv(
                behavior_individual_rows,
                filename=f"{export_prefix}_behavior_individual.csv"
            )
            behavior_pair_path = export_paths.get('behavior_pair_csv') or exporter.export_records_to_csv(
                behavior_pair_rows,
                filename=f"{export_prefix}_behavior_pairs.csv"
            )
            behavior_frame_path = export_paths.get('behavior_frame_csv') or exporter.export_records_to_csv(
                behavior_frame_rows,
                filename=f"{export_prefix}_behavior_frame_stats.csv"
            )
            behavior_group_path = export_paths.get('behavior_group_csv') or exporter.export_records_to_csv(
                [behavior_group_stats] if behavior_group_stats else [],
                filename=f"{export_prefix}_behavior_group_stats.csv"
            )
            behavior_segment_path = export_paths.get('behavior_segment_csv') or exporter.export_records_to_csv(
                behavior_segment_rows,
                filename=f"{export_prefix}_behavior_segment_stats.csv"
            )
            behavior_phase_path = export_paths.get('behavior_phase_csv') or exporter.export_records_to_csv(
                behavior_phase_rows,
                filename=f"{export_prefix}_behavior_phase_comparison.csv"
            )
            group_samples_path = export_paths.get('group_samples_csv')
            group_summary_path = export_paths.get('group_summary_csv')
            group_omnibus_path = export_paths.get('group_omnibus_csv')
            group_pairwise_path = export_paths.get('group_pairwise_csv')

            export_paths.update({
                'tracks_csv': tracks_path,
                'interactions_csv': interactions_path,
                'analysis_json': json_path,
                'report_txt': report_path,
                'frame_stats_csv': frame_stats_path,
                'track_stats_csv': track_stats_path,
                'behavior_events_csv': behavior_events_path,
                'behavior_individual_csv': behavior_individual_path,
                'behavior_pair_csv': behavior_pair_path,
                'behavior_frame_csv': behavior_frame_path,
                'behavior_group_csv': behavior_group_path,
                'behavior_segment_csv': behavior_segment_path,
                'behavior_phase_csv': behavior_phase_path,
            })

            bundle_inputs = {
                f"{export_prefix}_tracks.csv": tracks_path,
                f"{export_prefix}_interactions.csv": interactions_path,
                f"{export_prefix}_analysis.json": json_path,
                f"{export_prefix}_report.txt": report_path,
                f"{export_prefix}_frame_stats.csv": frame_stats_path,
                f"{export_prefix}_track_stats.csv": track_stats_path,
                f"{export_prefix}_behavior_events.csv": behavior_events_path,
                f"{export_prefix}_behavior_individual.csv": behavior_individual_path,
                f"{export_prefix}_behavior_pairs.csv": behavior_pair_path,
                f"{export_prefix}_behavior_frame_stats.csv": behavior_frame_path,
                f"{export_prefix}_behavior_group_stats.csv": behavior_group_path,
                f"{export_prefix}_behavior_segment_stats.csv": behavior_segment_path,
                f"{export_prefix}_behavior_phase_comparison.csv": behavior_phase_path,
            }
            if group_samples_path and Path(group_samples_path).exists():
                bundle_inputs[f"{export_prefix}_group_samples.csv"] = group_samples_path
            if group_summary_path and Path(group_summary_path).exists():
                bundle_inputs[f"{export_prefix}_group_summary_stats.csv"] = group_summary_path
            if group_omnibus_path and Path(group_omnibus_path).exists():
                bundle_inputs[f"{export_prefix}_group_omnibus_stats.csv"] = group_omnibus_path
            if group_pairwise_path and Path(group_pairwise_path).exists():
                bundle_inputs[f"{export_prefix}_group_pairwise_stats.csv"] = group_pairwise_path
            if auto_annotation_path and Path(auto_annotation_path).exists():
                bundle_inputs[f"{export_prefix}_automot_annotations.json"] = auto_annotation_path
            if publication_figure_path and Path(publication_figure_path).exists():
                bundle_inputs[f"{export_prefix}_automot_publication_trajectory.png"] = publication_figure_path

            bundle_path = exporter.export_bundle(bundle_inputs, filename=f"{export_prefix}_exports.zip")
            export_paths['bundle_zip'] = bundle_path
            results['export_bundle_path'] = bundle_path
            st.success(f"[OK] 已生成导出压缩包: {bundle_path}")

    with bundle_col2:
        bundle_path = results.get('export_bundle_path')
        if bundle_path and Path(bundle_path).exists():
            with open(bundle_path, 'rb') as f:
                st.download_button(
                    "Download ZIP Bundle",
                    data=f.read(),
                    file_name=Path(bundle_path).name,
                    mime="application/zip",
                    key="download_bundle_zip"
                )

    manifest_rows = [
        {'artifact': name, 'path': path}
        for name, path in export_paths.items()
        if path and Path(path).exists()
    ]
    if manifest_rows:
        st.caption("Export Manifest")
        st.dataframe(pd.DataFrame(manifest_rows), use_container_width=True, hide_index=True)

    st.markdown('<h2 class="sub-header">[GROUP] 多视频 / 组间比较</h2>', unsafe_allow_html=True)
    st.caption("可把当前结果与本地历史 `*_analysis.json` 一起比较。组间统计默认给出均值、标准差、SEM、Welch t 检验和 Cohen's d。")

    include_current_in_group = st.checkbox("纳入当前结果", value=True, key="group_compare_include_current")
    current_sample_name = Path(video_path).stem
    current_group_default = infer_group_label(current_sample_name)

    local_analysis_paths = discover_analysis_exports()
    selected_group_files = st.multiselect(
        "选择历史分析 JSON",
        options=local_analysis_paths,
        default=[],
        format_func=lambda path: Path(path).name,
        key="group_compare_files",
    )

    comparison_mode_label = st.radio(
        "统计设计",
        ["独立样本", "配对样本"],
        horizontal=True,
        key="group_compare_mode",
    )
    comparison_mode = "paired" if comparison_mode_label == "配对样本" else "independent"
    if comparison_mode == "paired":
        st.caption("配对设计下会使用 pair_id 对齐样本，并执行 paired t / Wilcoxon / Friedman 检验。")
    else:
        st.caption("独立设计下会执行 Welch t / Mann-Whitney / ANOVA / Kruskal-Wallis 检验。")

    comparison_sample_rows = []
    if include_current_in_group:
        current_pair_default = infer_pairing_label(current_sample_name, current_group_default)
        current_cols = st.columns(3 if comparison_mode == "paired" else 2)
        current_pair_id = ""
        with current_cols[0]:
            current_compare_name = st.text_input(
                "当前结果样本名",
                value=current_sample_name,
                key="group_compare_current_name",
            )
        with current_cols[1]:
            current_compare_group = st.text_input(
                "当前结果组别",
                value=current_group_default,
                key="group_compare_current_group",
            )
        if comparison_mode == "paired":
            with current_cols[2]:
                current_pair_id = st.text_input(
                    "当前结果 pair_id",
                    value=current_pair_default,
                    key="group_compare_current_pair_id",
                )
        comparison_sample_rows.append(
            build_sample_row(
                analysis_data,
                sample_name=current_compare_name,
                group_name=current_compare_group,
                pair_id=current_pair_id,
                source_path=str(video_path),
                source_type="current_session",
            )
        )

    if selected_group_files:
        st.caption("为每个历史分析结果指定样本名和实验组。")
    for analysis_path in selected_group_files:
        payload = get_cached_analysis_payload(analysis_path)
        default_sample_name = Path(analysis_path).stem.replace("_analysis", "")
        default_group_name = infer_group_label(default_sample_name)
        default_pair_id = infer_pairing_label(default_sample_name, default_group_name)
        file_cols = st.columns(3 if comparison_mode == "paired" else 2)
        pair_id = ""
        with file_cols[0]:
            sample_name = st.text_input(
                f"{Path(analysis_path).name} 样本名",
                value=default_sample_name,
                key=f"group_sample_{Path(analysis_path).name}",
            )
        with file_cols[1]:
            group_name = st.text_input(
                f"{Path(analysis_path).name} 组别",
                value=default_group_name,
                key=f"group_group_{Path(analysis_path).name}",
            )
        if comparison_mode == "paired":
            with file_cols[2]:
                pair_id = st.text_input(
                    f"{Path(analysis_path).name} pair_id",
                    value=default_pair_id,
                    key=f"group_pair_{Path(analysis_path).name}",
                )
        comparison_sample_rows.append(
            build_sample_row(
                payload,
                sample_name=sample_name,
                group_name=group_name,
                pair_id=pair_id,
                source_path=analysis_path,
                source_type="analysis_json",
            )
        )

    if len(comparison_sample_rows) >= 2:
        comparison_metrics = build_group_comparison(comparison_sample_rows, comparison_mode=comparison_mode)
        sample_comparison_df = pd.DataFrame(comparison_metrics.get("sample_rows", []))
        group_summary_df = pd.DataFrame(comparison_metrics.get("group_summary_rows", []))
        omnibus_df = pd.DataFrame(comparison_metrics.get("omnibus_rows", []))
        pairwise_df = pd.DataFrame(comparison_metrics.get("pairwise_rows", []))
        for note in comparison_metrics.get("notes", []):
            st.warning(note)
        metric_labels = comparison_metrics.get("metric_labels", {})
        metric_options = list(metric_labels.keys())
        selected_metric_keys = st.multiselect(
            "比较指标",
            options=metric_options,
            default=[
                key for key in [
                    "interaction_events",
                    "following_events",
                    "pursuit_events",
                    "escape_events",
                    "avg_speed_px_per_s",
                    "mean_edge_ratio",
                ] if key in metric_options
            ],
            format_func=lambda key: metric_labels.get(key, key),
            key="group_compare_metric_filter",
        )
        if not selected_metric_keys:
            selected_metric_keys = metric_options[:6]

        metric_label_order = [metric_labels[key] for key in selected_metric_keys if key in metric_labels]

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("样本数", len(sample_comparison_df))
        with metric_col2:
            st.metric("实验组数", sample_comparison_df["group"].nunique() if not sample_comparison_df.empty else 0)
        with metric_col3:
            st.metric("总体检验数", len(omnibus_df))

        st.caption("样本级行为概览")
        st.dataframe(sample_comparison_df, use_container_width=True, hide_index=True)

        filtered_group_summary_df = group_summary_df[group_summary_df["metric"].isin(selected_metric_keys)].copy()
        if not filtered_group_summary_df.empty:
            st.caption("组均值 / 方差 / SEM")
            summary_pivot = filtered_group_summary_df.pivot(index="group", columns="metric_label", values="mean")
            summary_pivot = summary_pivot.reindex(columns=[label for label in metric_label_order if label in summary_pivot.columns])
            st.bar_chart(summary_pivot, use_container_width=True)
            st.dataframe(filtered_group_summary_df, use_container_width=True, hide_index=True)

        filtered_omnibus_df = omnibus_df[omnibus_df["metric"].isin(selected_metric_keys)].copy() if not omnibus_df.empty else omnibus_df
        if filtered_omnibus_df is not None and not filtered_omnibus_df.empty:
            st.caption("总体检验")
            st.dataframe(filtered_omnibus_df, use_container_width=True, hide_index=True)

        filtered_pairwise_df = pairwise_df[pairwise_df["metric"].isin(selected_metric_keys)].copy()
        if not filtered_pairwise_df.empty:
            st.caption("成对比较")
            st.dataframe(filtered_pairwise_df, use_container_width=True, hide_index=True)

        comparison_insights = comparison_metrics.get("insights", [])
        if comparison_insights:
            st.caption("组间洞察")
            for insight in comparison_insights:
                st.info(
                    f"**{insight.get('headline', '组间洞察')}**\n\n"
                    f"{insight.get('detail', '')}\n\n"
                    f"建议：{insight.get('action', '')}"
                )

        comparison_export_col1, comparison_export_col2, comparison_export_col3, comparison_export_col4 = st.columns(4)
        with comparison_export_col1:
            if st.button("[CSV] 导出组间样本表", key="export_group_sample_rows"):
                path = exporter.export_records_to_csv(
                    comparison_metrics.get("sample_rows", []),
                    filename=f"{export_prefix}_group_samples.csv"
                )
                export_paths['group_samples_csv'] = path
                st.success(f"[OK] 已导出: {path}")
        with comparison_export_col2:
            if st.button("[CSV] 导出组汇总统计", key="export_group_summary_rows"):
                path = exporter.export_records_to_csv(
                    comparison_metrics.get("group_summary_rows", []),
                    filename=f"{export_prefix}_group_summary_stats.csv"
                )
                export_paths['group_summary_csv'] = path
                st.success(f"[OK] 已导出: {path}")
        with comparison_export_col3:
            if st.button("[CSV] 导出总体检验", key="export_group_omnibus_rows"):
                path = exporter.export_records_to_csv(
                    comparison_metrics.get("omnibus_rows", []),
                    filename=f"{export_prefix}_group_omnibus_stats.csv"
                )
                export_paths['group_omnibus_csv'] = path
                st.success(f"[OK] 已导出: {path}")
        with comparison_export_col4:
            if st.button("[CSV] 导出组间比较", key="export_group_pairwise_rows"):
                path = exporter.export_records_to_csv(
                    comparison_metrics.get("pairwise_rows", []),
                    filename=f"{export_prefix}_group_pairwise_stats.csv"
                )
                export_paths['group_pairwise_csv'] = path
                st.success(f"[OK] 已导出: {path}")
    else:
        st.caption("至少准备 2 个样本后，页面会自动给出组均值、方差、显著性和效应量比较。")

else:
    # 欢迎页面
    st.markdown("""
    ### 如何开始
    
    1. 上传视频文件（MP4, AVI, MOV）
    2. 调整检测和追踪参数
    3. 点击"开始追踪"按钮
    4. 等待处理完成
    5. 查看追踪结果和分析
    
    ### 模式选择
    
    - **视频文件模式**: 上传视频文件进行分析
    - **实时摄像头模式**: 使用摄像头实时追踪
    
    ### 支持的动物
    
    - 小鼠
    - 大鼠  
    - 斑马鱼
    - 其他小型实验动物
    
    ### 系统要求
    
    - Python 3.10+
    - 摄像头或视频文件
    - 推荐 GPU 加速
    """)

# ========================================
# 页脚
# ========================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><b>AnimalTracker - 小动物行为追踪系统</b></p>
    <p><i>基于 STCS 框架理念 | 华中科技大学</i></p>
    <p><small>For research and educational purposes only</small></p>
</div>
""", unsafe_allow_html=True)
