"""
run_animalmot_x.py — AnimalMOT-X 科研级追踪平台一键运行
========================================================
融合 9 篇前沿论文技术的统一入口
"""
import os
import cv2
import numpy as np
import time
from collections import defaultdict

from src.tracking.animalmot_x import AnimalMOTX
from generate_tracking_video_v6 import draw_overlay


def run(video_path: str, output_path: str, n_animals: int):
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频不存在: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = AnimalMOTX(n_animals=n_animals, threshold=70)

    print("=" * 60)
    print(f"AnimalMOT-X 科研级追踪 | {n_animals} 只动物 | {total} 帧")
    print("技术: CLAHE+Watershed+VirtualMarkers+OpticalFlow+ByteTrack")
    print("      +TrackTrack+MOTIP+CBAM+UDMT | 9篇前沿论文融合")
    print("=" * 60)

    # ===================== Pass 1: 全帧追踪 =====================
    print("\n[Pass 1] 全帧在线追踪...")
    t0 = time.time()
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tracker.update(frame)
        fidx += 1
        if fidx % 1000 == 0:
            s = tracker.get_stats()
            print(f"  {fidx}/{total} | Active: {s['active']}/{n_animals} "
                  f"| Created: {s['created']} | Recovered: {s['recovered']}")

    t1 = time.time()
    stats = tracker.get_stats()
    print(f"[Pass 1] 完成 ({t1 - t0:.0f}s) | "
          f"轨迹: {len(tracker.full_history)} | Stats: {dict(stats)}")

    # ===================== Pass 2: 后处理 =====================
    print("\n[Pass 2] 离线缝合 + UDMT ID 修正...")
    corrected = tracker.post_process()
    print(f"[Pass 2] 最终轨迹数: {len(corrected)}")

    # ===================== Pass 3: 渲染 =====================
    print(f"\n[Pass 3] 视频渲染...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 回写 bbox 数据
    full_hist = tracker.full_history
    final_hist = defaultdict(dict)
    for tid, frames in corrected.items():
        for f, pos in frames.items():
            for orig_tid, orig_h in full_hist.items():
                if f in orig_h and orig_h[f]['center'] == pos:
                    final_hist[tid][f] = orig_h[f]
                    break

    class MockTrack:
        def __init__(self, tid, bbox, center, history):
            self.id = tid
            self.bbox = bbox
            self.smoothed_center = center
            self.center = (int(center[0]), int(center[1]))
            self.time_since_update = 0
            self.history = history

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current = {}
        for tid, hist in final_hist.items():
            if fidx in hist:
                d = hist[fidx]
                trail = [hist[i]['center']
                         for i in range(max(0, fidx - 120), fidx + 1)
                         if i in hist]
                current[tid] = MockTrack(tid, d['bbox'], d['center'], trail)

        viz = draw_overlay(frame, current, fidx, total, fps, 0, 0, None,
                           n_animals)
        out.write(viz)
        fidx += 1
        if fidx % 2000 == 0:
            print(f"  {fidx}/{total} rendered...")

    out.release()
    cap.release()
    print(f"\n{'=' * 60}")
    print(f"✅ 完成! {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run("data/videos/6xWT.mp4",
        "data/results/tracking_motx_6xWT.mp4",
        n_animals=6)
