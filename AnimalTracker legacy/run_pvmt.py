"""
run_pvmt.py — 持久虚拟标记追踪器 (PVMT) 入口
===============================================
范式: "Track Always, Detect to Correct"
标记通过 LK 光流逐帧传播, 检测只做校正, 聚集中标记独立存活
"""
import os, cv2, time
from collections import defaultdict
from src.tracking.persistent_marker_tracker import PersistentVirtualMarkerTracker
from generate_tracking_video_v6 import draw_overlay


def run(video_path, output_path, n_animals):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = PersistentVirtualMarkerTracker(
        n_animals=n_animals, n_markers=15, threshold=70)

    print("=" * 60)
    print(f"PVMT 持久虚拟标记追踪 | {n_animals} 只 | {total} 帧")
    print("范式: Track Always, Detect to Correct")
    print("核心: LK稀疏光流标记传播 + 前后向一致性验证")
    print("=" * 60)

    t0 = time.time()
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tracker.update(frame)
        fidx += 1
        if fidx % 500 == 0:
            s = tracker.get_stats()
            active = sum(1 for m in tracker.markers.values()
                         if m.alive_count >= 4)
            corr = s.get('corrections', 0)
            heal = s.get('healed', 0)
            print(f"  {fidx}/{total} | Active: {active}/{n_animals} "
                  f"| Created: {s.get('created',0)} "
                  f"| Corrections: {corr} | Healed: {heal}")

    t1 = time.time()
    print(f"\n[Pass 1] 完成 ({t1-t0:.0f}s) | "
          f"轨迹: {len(tracker.full_history)} | {tracker.get_stats()}")

    print("\n[Pass 2] 后处理...")
    corrected = tracker.post_process()
    print(f"最终轨迹: {len(corrected)}")

    print("\n[Pass 3] 渲染...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    full = tracker.full_history
    final = defaultdict(dict)
    for tid, frames in corrected.items():
        for f, pos in frames.items():
            for ot, oh in full.items():
                if f in oh and oh[f]['center'] == pos:
                    final[tid][f] = oh[f]
                    break

    class MT:
        def __init__(s, tid, bbox, center, hist):
            s.id = tid; s.bbox = bbox; s.smoothed_center = center
            s.center = (int(center[0]), int(center[1]))
            s.time_since_update = 0; s.history = hist

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur = {}
        for tid, hist in final.items():
            if fidx in hist:
                d = hist[fidx]
                trail = [hist[i]['center']
                         for i in range(max(0, fidx-120), fidx+1)
                         if i in hist]
                cur[tid] = MT(tid, d['bbox'], d['center'], trail)
        out.write(draw_overlay(frame, cur, fidx, total, fps, 0, 0,
                               None, n_animals))
        fidx += 1
        if fidx % 2000 == 0:
            print(f"  {fidx}/{total} rendered...")

    out.release(); cap.release()
    print(f"\n{'='*60}\n✅ {output_path}\n{'='*60}")


if __name__ == "__main__":
    run("data/videos/6xWT.mp4",
        "data/results/tracking_pvmt_6xWT.mp4", 6)
