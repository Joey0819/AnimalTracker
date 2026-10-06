"""
render_stitched_video.py — 两阶段渲染管线
==========================================
Pass 1: MaskFlowVMTracker 全帧追踪 (掩码光流 + 虚拟标记)
Pass 2: OfflineStitcher 离线缝合 → 强制收敛到 N_ANIMALS
Pass 3: 视频渲染输出
"""
import os
import cv2
import numpy as np
import time
from collections import defaultdict

from src.tracking.mask_flow_vm_tracker import MaskFlowVMTracker
from src.tracking.post_processor import OfflineStitcher
from src.tracking.id_corrector import TrajectoryIDCorrector
from generate_tracking_video_v6 import draw_overlay


def main():
    video_path = "data/videos/6xWT.mp4"
    output_path = "data/results/tracking_v8_maskflow_6xWT.mp4"
    N_ANIMALS = 6

    if not os.path.exists(video_path):
        print(f"[ERROR] 视频不存在: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用新的掩码光流虚拟标记追踪器
    tracker = MaskFlowVMTracker(
        n_animals=N_ANIMALS,
        threshold=70,
        min_area=300,
        max_area=50000,
        mask_iou_weight=0.50,
        marker_weight=0.30,
        shape_weight=0.20,
        match_threshold=0.70,
        max_disappeared=45
    )

    print("=" * 60)
    print("第一阶段: MaskFlowVM 全帧追踪 (掩码光流 + 虚拟标记)")
    print("=" * 60)

    track_history = defaultdict(dict)

    t0 = time.time()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 一步完成: 检测 + 分割 + 关联
        tracks = tracker.update(frame)

        for tid, trk in tracks.items():
            if trk.time_since_update == 0:
                track_history[tid][frame_idx] = {
                    'bbox': trk.bbox.copy(),
                    'center': trk.smoothed_center,
                    'hits': trk.hits
                }

        frame_idx += 1
        if frame_idx % 1000 == 0:
            active = sum(1 for t in tracks.values()
                         if t.time_since_update == 0)
            print(f"  [Pass 1] {frame_idx}/{total} frames | "
                  f"Active: {active}/{N_ANIMALS} | "
                  f"Total tracks created: {tracker._stats['total_created']}")

    elapsed_1 = time.time() - t0
    stats = tracker.get_stats()
    print(f"第一阶段完成: 用时 {elapsed_1:.1f}s, "
          f"生成了 {len(track_history)} 条原始轨迹。")
    print(f"  Stats: {stats}")

    print(f"\n{'=' * 60}")
    print(f"第二阶段: 离线缝合 (Offline Stitching)")
    print(f"{'=' * 60}")

    stitcher = OfflineStitcher(
        n_animals=N_ANIMALS,
        max_distance=300.0,
        max_time_gap=300
    )

    simple_hist = {
        tid: {f: data['center'] for f, data in hist.items()}
        for tid, hist in track_history.items()
    }
    stitched_simple = stitcher.stitch(simple_hist)
    print(f"离线缝合完成: 剩余活跃轨迹数 -> {len(stitched_simple)}")

    # ================================================================
    # Stage 2.5: UDMT/TrackTrack 启发的轨迹 ID 修正
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"第 2.5 阶段: 轨迹速度异常检测 + ID 交换修正 (UDMT-inspired)")
    print(f"{'=' * 60}")

    corrector = TrajectoryIDCorrector(
        velocity_z_threshold=3.0,
        swap_window=5,
        min_track_length=30,
        max_swap_distance=200
    )
    stitched_simple = corrector.correct(stitched_simple)
    print(f"ID 修正完成: 最终轨迹数 -> {len(stitched_simple)}")

    # 把拼接后的状态回写
    final_history = defaultdict(dict)
    for tid, hist in stitched_simple.items():
        for f in hist.keys():
            original_data = None
            for orig_tid, orig_hist in track_history.items():
                if f in orig_hist and orig_hist[f]['center'] == hist[f]:
                    original_data = orig_hist[f]
                    break
            if original_data:
                final_history[tid][f] = original_data

    print(f"\n{'=' * 60}")
    print(f"第三阶段: 结果渲染与写入...")
    print(f"{'=' * 60}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    class MockTrack:
        def __init__(self, tid, bbox, center, history):
            self.id = tid
            self.bbox = bbox
            self.smoothed_center = center
            self.center = (int(center[0]), int(center[1]))
            self.time_since_update = 0
            self.history = history

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_tracks = {}
        for tid, hist in final_history.items():
            if frame_idx in hist:
                data = hist[frame_idx]
                t_hist = []
                for i in range(max(0, frame_idx - 120), frame_idx + 1):
                    if i in hist:
                        t_hist.append(hist[i]['center'])
                current_tracks[tid] = MockTrack(
                    tid, data['bbox'], data['center'], t_hist)

        viz = draw_overlay(frame, current_tracks, frame_idx, total,
                           fps_video, 0, 0, None, N_ANIMALS)
        out.write(viz)

        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"  [Pass 3] {frame_idx}/{total} frames rendered...")

    out.release()
    cap.release()

    print(f"\n{'=' * 60}")
    print(f"终极版生成完毕! 文件保存至: {output_path}")
    print(f"算法: MaskFlowVM (Watershed + Optical Flow + Virtual Markers)")


if __name__ == "__main__":
    main()
