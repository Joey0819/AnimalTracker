"""
追踪诊断工具
快速分析视频追踪问题，定位性能瓶颈
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.detection.detector import YOLODetector
from src.tracking.enhanced_tracker import create_enhanced_tracker
from src.evaluation.metrics import TrackingMetrics, TrackingDebugger, quick_evaluate


def diagnose_video(video_path: str,
                  model_path: str = "yolov8n.pt",
                  confidence: float = 0.25,
                  max_frames: int = 100,
                  output_dir: str = "debug_output"):
    """
    诊断视频追踪问题

    Args:
        video_path: 视频路径
        model_path: 模型路径
        confidence: 置信度阈值
        max_frames: 最大分析帧数
        output_dir: 输出目录
    """
    print("=" * 60)
    print("追踪诊断工具")
    print("=" * 60)
    print(f"视频: {video_path}")
    print(f"模型: {model_path}")
    print(f"置信度: {confidence}")
    print(f"最大帧数: {max_frames}")
    print("-" * 60)

    # 初始化
    detector = YOLODetector(model_path=model_path, confidence_threshold=confidence)
    tracker = create_enhanced_tracker(preset='high_accuracy')
    debugger = TrackingDebugger(output_dir=output_dir)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")

    # 分析变量
    frame_count = 0
    tracking_results = []
    detection_stats = {
        'total_detections': 0,
        'high_conf': 0,    # > 0.7
        'med_conf': 0,     # 0.4 - 0.7
        'low_conf': 0,     # < 0.4
        'confidences': []
    }
    track_stats = {
        'id_history': [],
        'id_switches': 0,
        'max_concurrent': 0
    }

    prev_ids = set()

    print(f"\n开始分析...")
    print("-" * 60)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测
        detections = detector.detect(frame)

        # 统计检测置信度
        for det in detections:
            detection_stats['total_detections'] += 1
            conf = det.confidence
            detection_stats['confidences'].append(conf)

            if conf > 0.7:
                detection_stats['high_conf'] += 1
            elif conf > 0.4:
                detection_stats['med_conf'] += 1
            else:
                detection_stats['low_conf'] += 1

        # 追踪
        tracks = tracker.update(frame, detections)

        # 统计追踪
        current_ids = set(tracks.keys())
        track_stats['id_history.extend(list(current_ids))

        # 检测ID切换
        new_ids = current_ids - prev_ids
        lost_ids = prev_ids - current_ids
        track_stats['id_switches'] += min(len(new_ids), len(lost_ids))

        track_stats['max_concurrent'] = max(track_stats['max_concurrent'], len(current_ids))

        # 保存结果
        tracking_results.append({
            'frame': frame_count,
            'detections': detections,
            'tracks': tracks,
            'detection_count': len(detections),
            'track_count': len(tracks)
        })

        prev_ids = current_ids
        frame_count += 1

        # 进度显示
        if frame_count % 10 == 0:
            print(f"  处理帧: {frame_count}/{max_frames}")

    cap.release()

    # 分析结果
    print("\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)

    # 检测统计
    print("\n[1] 检测分析:")
    avg_det = detection_stats['total_detections'] / frame_count if frame_count > 0 else 0
    print(f"  平均每帧检测数: {avg_det:.2f}")
    print(f"  总检测数: {detection_stats['total_detections']}")

    if detection_stats['confidences']:
        avg_conf = np.mean(detection_stats['confidences'])
        min_conf = np.min(detection_stats['confidences'])
        max_conf = np.max(detection_stats['confidences'])
        print(f"  平均置信度: {avg_conf:.3f}")
        print(f"  置信度范围: [{min_conf:.3f}, {max_conf:.3f}]")

    print(f"\n  置信度分布:")
    print(f"    高 (>0.7): {detection_stats['high_conf']} ({detection_stats['high_conf']/max(detection_stats['total_detections'],1)*100:.1f}%)")
    print(f"    中 (0.4-0.7): {detection_stats['med_conf']} ({detection_stats['med_conf']/max(detection_stats['total_detections'],1)*100:.1f}%)")
    print(f"    低 (<0.4): {detection_stats['low_conf']} ({detection_stats['low_conf']/max(detection_stats['total_detections'],1)*100:.1f}%)")

    # 追踪统计
    print("\n[2] 追踪分析:")
    unique_ids = len(set(track_stats['id_history']))
    print(f"  检测到的唯一ID数: {unique_ids}")
    print(f"  最大同时追踪数: {track_stats['max_concurrent']}")
    print(f"  ID切换次数: {track_stats['id_switches']}")

    if frame_count > 0:
        switch_rate = track_stats['id_switches'] / frame_count
        print(f"  每帧平均切换: {switch_rate:.3f}")

    # 问题诊断
    print("\n[3] 问题诊断:")

    issues = []
    suggestions = []

    # 检测问题
    if avg_det < 2:
        issues.append("⚠️ 检测数量过少")
        suggestions.append("- 降低置信度阈值")
        suggestions.append("- 启用增强检测 (CLAHE预处理)")

    if avg_conf < 0.5:
        issues.append("⚠️ 检测置信度过低")
        suggestions.append("- 检查视频质量（光照、对比度）")
        suggestions.append("- 启用增强检测")

    if detection_stats['low_conf'] / max(detection_stats['total_detections'], 1) > 0.3:
        issues.append("⚠️ 大量低置信度检测")
        suggestions.append("- 考虑提高阈值减少误检")

    # 追踪问题
    if unique_ids > track_stats['max_concurrent'] * 2:
        issues.append("⚠️ ID碎片严重")
        suggestions.append("- 增加 max_disappeared 参数")
        suggestions.append("- 启用 Re-ID 匹配")

    if switch_rate > 0.1:
        issues.append("⚠️ ID切换频繁")
        suggestions.append("- 减小 max_distance 参数")
        suggestions.append("- 启用外观匹配 (Re-ID)")

    if track_stats['max_concurrent'] < avg_det:
        issues.append("⚠️ 追踪数量少于检测数量")
        suggestions.append("- 检查 max_track_slots 参数")

    if not issues:
        print("  ✅ 未发现明显问题")
    else:
        for issue in issues:
            print(f"  {issue}")

    # 建议
    if suggestions:
        print("\n[4] 改进建议:")
        for s in suggestions:
            print(f"  {s}")

    # 生成报告
    print("\n[5] 生成报告:")

    # 基本指标
    metrics = {
        'total_frames': frame_count,
        'num_unique_ids': unique_ids,
        'id_switches': track_stats['id_switches'],
        'avg_tracks_per_frame': track_stats['max_concurrent'],
        'avg_confidence': avg_conf if detection_stats['confidences'] else 0,
        'precision': detection_stats['high_conf'] / max(detection_stats['total_detections'], 1),
    }

    report = debugger.generate_report(metrics, video_path,
                                     f"{output_dir}/diagnosis_report.html")

    print(f"  HTML报告: {output_dir}/diagnosis_report.html")

    # 可视化第一帧
    if tracking_results:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            vis = debugger.visualize_frame(
                frame,
                tracking_results[0]['detections'],
                tracking_results[0]['tracks'],
                [d.confidence for d in tracking_results[0]['detections']]
            )
            cv2.imwrite(f"{output_dir}/frame_0_debug.png", vis)
            print(f"  调试图像: {output_dir}/frame_0_debug.png")

    print("\n" + "=" * 60)
    print("诊断完成！")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="追踪诊断工具")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--model", default="yolov8n.pt", help="模型路径")
    parser.add_argument("--confidence", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--max-frames", type=int, default=100, help="最大分析帧数")
    parser.add_argument("--output", default="debug_output", help="输出目录")

    args = parser.parse_args()

    diagnose_video(
        args.video,
        args.model,
        args.confidence,
        args.max_frames,
        args.output
    )
