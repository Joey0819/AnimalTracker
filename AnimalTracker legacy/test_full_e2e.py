"""
端到端集成测试 v2（生产管线版）
使用生产组件: StaticThresholdDetector + BufferRecoverTracker + STCS合并 → 社交网络 → 见解 → 报告

关键改进 vs v1:
1. 检测器使用 StaticThresholdDetector（不再是 MOG2 背景减除，静止动物也检出）
2. 追踪器使用 BufferRecoverTracker（限制 max_track_slots = N 动物，从源头防止 ID 碎片）
3. 社交网络输入使用合并后的轨迹 ID（不再是 49 个碎片 ID）
"""
import sys
import os
import time
import logging

if hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

# 生产组件导入
from src.detection.threshold_detector import StaticThresholdDetector
from src.tracking.buffer_recover_tracker import BufferRecoverTracker, create_buffer_recover_tracker
from src.tracking.track_filter import TrackFilter
from src.tracking.stcs_appearance_merger import STCSAppearanceMerger
from src.behavior.social_network import SocialNetworkAnalyzer
from src.reporting.insight_generator import InsightGenerator
from src.reporting.report_builder import ReportBuilder


def main():
    video_path = "data/videos/4xPD.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频不存在: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    N_ANIMALS = 4
    MAX_FRAMES = 600

    print(f"{'='*60}")
    print(f"端到端集成测试 v2（生产管线）")
    print(f"{'='*60}")
    print(f"视频: {video_path}")
    print(f"分辨率: {w}x{h}, FPS: {fps}, 总帧数: {total}")
    print(f"测试帧数: {MAX_FRAMES}")
    print(f"{'='*60}\n")

    # --- 初始化生产组件 ---

    # 检测器: 静态阈值（不依赖运动，静止动物也检出）
    detector = StaticThresholdDetector(
        n_animals=N_ANIMALS,
        threshold=70,
        min_area=300,
    )

    # 追踪器: BufferRecoverTracker（限制 max_track_slots = N，防止 ID 膨胀）
    tracker = create_buffer_recover_tracker(
        max_track_slots=N_ANIMALS,
        buffer_duration=90,
        max_age=60,
        recovery_threshold=0.55,
        high_conf_thresh=0.55,
        use_deep_embedding=True,
        embedding_interval=5,
    )

    # 轨迹过滤器
    track_filter = TrackFilter(
        max_speed_px_per_frame=80,
        min_confidence=0.15,
        min_tracklet_frames=5,
        frame_width=w, frame_height=h
    )

    # STCS 外观合并器
    stcs_merger = STCSAppearanceMerger(
        n_animals=N_ANIMALS,
        convergence_ratio=0.90,
        merge_confidence=0.5,
        min_tracklet_length=5,
    )

    # 社交网络分析器
    social_analyzer = SocialNetworkAnalyzer(
        interaction_distance=100,
        min_interaction_duration=5
    )

    # 报告引擎
    insight_gen = InsightGenerator(fps=fps)
    report_builder = ReportBuilder(fps=fps)

    # --- Phase 1: 检测 + 追踪 ---
    print("[Phase 1] 检测 + 追踪（StaticThreshold + BufferRecover）...")
    t0 = time.time()
    frame_idx = 0
    all_track_history = defaultdict(list)
    detection_counts = []
    frame_cache = {}
    prev_ids = set()
    id_switch_count = 0

    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测
        dets = detector.detect(frame)
        detection_counts.append(len(dets))

        # 追踪（传入帧用于嵌入提取）
        tracks = tracker.update(frame, dets)

        # 记录位置历史
        positions = {}
        for tid, track in tracks.items():
            c = track.center
            all_track_history[tid].append(c)
            positions[tid] = c

        # 社交网络在线更新
        if len(positions) >= 2:
            social_analyzer.update(frame_idx, positions)

        # ID 切换检测
        curr_ids = set(tracks.keys())
        if prev_ids and curr_ids:
            new_ids = curr_ids - prev_ids
            if new_ids and len(tracks) <= N_ANIMALS:
                id_switch_count += len(new_ids)
        prev_ids = curr_ids

        # 缓存部分帧用于外观合并
        if frame_idx % 10 == 0:
            frame_cache[frame_idx] = frame.copy()

        frame_idx += 1

        if frame_idx % 100 == 0:
            avg_det = np.mean(detection_counts[-100:]) if detection_counts else 0
            stats = tracker.get_stats()
            print(f"  帧 {frame_idx}/{MAX_FRAMES}, "
                  f"活跃={stats['active_tracks']}, "
                  f"缓冲={stats['buffered_tracks']}, "
                  f"恢复={stats['total_recovered']}, "
                  f"平均检测={avg_det:.1f}")

    cap.release()
    elapsed_tracking = time.time() - t0

    tracker_stats_raw = tracker.get_stats()
    print(f"  追踪完成: {elapsed_tracking:.1f}s ({frame_idx/max(elapsed_tracking,0.1):.0f} fps)")
    print(f"  总创建轨迹={tracker_stats_raw['total_created']}, "
          f"恢复={tracker_stats_raw['total_recovered']}, "
          f"ID切换={id_switch_count}")
    print(f"  历史轨迹数={len(all_track_history)}")

    # --- Phase 2: 轨迹过滤 ---
    print(f"\n[Phase 2] 轨迹过滤...")
    # 合并 active + buffered 以获取完整轨迹集
    all_tracks_for_filter = dict(tracker.active_tracks)
    all_tracks_for_filter.update(tracker.buffered_tracks)
    
    clean_tracks, removed_tracks = track_filter.filter(
        all_tracks_for_filter, frame_size=(w, h))
    filter_stats = track_filter.get_stats()
    print(f"  过滤结果: {filter_stats['input_count']} → "
          f"{filter_stats['output_count']} 轨迹")

    # --- Phase 3: STCS Stage 2 迭代合并 ---
    print(f"\n[Phase 3] STCS Stage 2 迭代外观合并...")
    t1 = time.time()

    def progress_cb(ratio, msg):
        if ratio in (0.0, 1.0) or int(ratio * 10) != int((ratio - 0.01) * 10):
            print(f"  [{ratio:.0%}] {msg}")

    stcs_merger.progress_callback = progress_cb

    if len(clean_tracks) > 1:
        merged_tracks = stcs_merger.iterative_merge(clean_tracks, video_frames=frame_cache)
    else:
        merged_tracks = clean_tracks

    elapsed_merge = time.time() - t1
    print(f"  合并完成: {len(clean_tracks)} → {len(merged_tracks)} 轨迹 "
          f"({elapsed_merge:.1f}s)")

    # --- Phase 4: 社交网络分析 ---
    print(f"\n[Phase 4] 社交网络分析...")
    social_analyzer.finalize()
    social_graph = social_analyzer.build_graph()
    social_metrics = social_analyzer.compute_metrics()
    social_summary = social_analyzer.get_summary()
    print(social_summary)

    # --- Phase 5: 数据见解 ---
    print(f"\n[Phase 5] 生成数据见解...")
    # 使用合并后的轨迹 ID 来筛选历史
    filtered_history = {}
    for tid in merged_tracks:
        if tid in all_track_history:
            filtered_history[tid] = all_track_history[tid]

    # 如果合并后的轨迹少于2条，用全部历史
    if len(filtered_history) < 2:
        filtered_history = dict(all_track_history)

    insights = insight_gen.generate_insights(
        track_history=filtered_history,
        interactions=social_analyzer.interactions,
        social_metrics=social_analyzer.export_to_dict(),
        total_frames=frame_idx
    )

    print(f"  个体数: {insights['summary']['n_animals']}")
    print(f"  平均速度: {insights['summary']['mean_speed_px_s']} px/s")
    print(f"  异常事件: {len(insights['anomalies'])}")
    print(f"  建议数: {len(insights['recommendations'])}")

    # --- Phase 6: 生成报告 ---
    print(f"\n[Phase 6] 生成分析报告...")
    video_info = {
        '文件名': os.path.basename(video_path),
        '分辨率': f'{w}x{h}',
        '帧率': f'{fps} fps',
        '总帧数': total,
        '测试帧数': frame_idx,
    }

    tracker_stats = {
        'active_tracks': len(merged_tracks),
        'total_created': tracker_stats_raw['total_created'],
        'tracklets_merged': len(clean_tracks) - len(merged_tracks),
        'tracks_filtered': len(removed_tracks),
        'id_recoveries': tracker_stats_raw['total_recovered'],
        'id_switches': id_switch_count,
        'frame_count': frame_idx,
    }

    report = report_builder.build_report(
        insights=insights,
        tracker_stats=tracker_stats,
        social_summary=social_summary,
        video_info=video_info,
        tracker_name="StaticThresholdDetector + BufferRecoverTracker + STCS Stage2"
    )

    # 保存报告
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "e2e_test_report.md"
    report_builder.save_report(report, str(report_path))

    json_path = output_dir / "e2e_test_insights.json"
    report_builder.save_json(insights, str(json_path))

    social_csv_path = output_dir / "e2e_social_interactions.csv"
    social_analyzer.export_interactions_csv(str(social_csv_path))

    print(f"\n{'='*60}")
    print(f"测试完成!")
    print(f"{'='*60}")
    print(f"  追踪耗时: {elapsed_tracking:.1f}s ({frame_idx/max(elapsed_tracking,0.1):.0f} fps)")
    print(f"  合并耗时: {elapsed_merge:.1f}s")
    print(f"  轨迹数: 创建 {tracker_stats_raw['total_created']} → "
          f"过滤后 {len(clean_tracks)} → 合并后 {len(merged_tracks)}")
    print(f"  ID 恢复次数: {tracker_stats_raw['total_recovered']}")
    print(f"  ID 切换次数: {id_switch_count}")
    print(f"  社交交互: {len(social_analyzer.interactions)} 次")
    print(f"  异常事件: {len(insights['anomalies'])} 个")
    print(f"\n输出文件:")
    print(f"  报告: {report_path}")
    print(f"  见解: {json_path}")
    print(f"  社交: {social_csv_path}")

    # 预览报告
    print(f"\n{'='*60}")
    print(f"报告预览 (前 2000 字符):")
    print(f"{'='*60}")
    print(report[:2000])


if __name__ == "__main__":
    main()
