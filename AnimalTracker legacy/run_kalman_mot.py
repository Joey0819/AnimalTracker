"""
run_kalman_mot.py — KalmanMOT: Kalman + MOG2 + ByteTrack
"""
import os, cv2, time
from collections import defaultdict
from src.tracking.kalman_mot import KalmanMOT
from generate_tracking_video_v6 import draw_overlay

def run(video_path, output_path, n_animals):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = KalmanMOT(n_animals=n_animals)

    print("=" * 60)
    print(f"KalmanMOT | {n_animals} 只 | {total} 帧")
    print("MOG2背景建模 + Kalman滤波 + ByteTrack双阶段 + Watershed")
    print("=" * 60)

    t0 = time.time()
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        tracker.update(frame)
        fidx += 1
        if fidx % 1000 == 0:
            s = tracker.get_stats()
            print(f"  {fidx}/{total} | Active: {s['active']}/{n_animals}"
                  f" | Created: {s.get('created',0)}"
                  f" | Recovered: {s.get('recovered',0)}")

    print(f"\n[Pass 1] 完成 ({time.time()-t0:.0f}s) | {tracker.get_stats()}")
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
                    final[tid][f] = oh[f]; break

    class MT:
        def __init__(s, tid, bbox, center, hist):
            s.id=tid; s.bbox=bbox; s.smoothed_center=center
            s.center=(int(center[0]),int(center[1]))
            s.time_since_update=0; s.history=hist

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cur = {}
        for tid, hist in final.items():
            if fidx in hist:
                d = hist[fidx]
                trail = [hist[i]['center']
                         for i in range(max(0,fidx-120),fidx+1)
                         if i in hist]
                cur[tid] = MT(tid, d['bbox'], d['center'], trail)
        out.write(draw_overlay(frame, cur, fidx, total, fps, 0, 0,
                               None, n_animals))
        fidx += 1
        if fidx % 2000 == 0: print(f"  {fidx}/{total} rendered...")

    out.release(); cap.release()
    print(f"\n{'='*60}\n✅ {output_path}\n{'='*60}")

if __name__ == "__main__":
    run("data/videos/6xWT.mp4",
        "data/results/tracking_kalman_6xWT.mp4", 6)
