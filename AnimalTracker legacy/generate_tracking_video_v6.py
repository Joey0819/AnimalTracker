"""
可视化轨迹视频 v6.0 — AnimalMOT (自适应卡尔曼平滑 + 骨架 Re-ID)

对比:
  v4: SimpleTracker (6 ID switches, 锯齿轨迹)
  v5: BufferRecover + Skeleton (5 ID switches, 锯齿轨迹)
  v6: AnimalMOT (目标 ≤3 ID switches, 平滑轨迹)

核心改进:
  1. 每个 AnimalTrack 内置自适应卡尔曼 → 静止不抖/运动跟随
  2. 速度门控 80px/frame → 异常检测被拒绝
  3. 骨架嵌入直接传入匹配代价 → 形态指纹辅助关联
"""
import sys
import os
import time

if hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np

N_ANIMALS = 6

TRACK_COLORS = [
    (0, 255, 127),    # 春绿
    (255, 105, 180),  # 热粉
    (30, 144, 255),   # 道奇蓝
    (255, 215, 0),    # 金色
    (0, 255, 255),    # 青色
    (255, 69, 0),     # 橙红
]

SKELETON_EDGES = [
    (0, 3), (1, 0), (2, 0), (3, 4), (4, 5),
    (5, 6), (3, 7), (3, 8), (5, 9), (5, 10),
]

KEYPOINT_COLORS = [
    (0, 0, 255), (255, 0, 255), (255, 0, 255), (0, 255, 0),
    (0, 255, 255), (0, 200, 200), (200, 200, 0),
    (255, 128, 0), (255, 128, 0), (128, 0, 255), (128, 0, 255),
]


def get_color(tid):
    return TRACK_COLORS[tid % len(TRACK_COLORS)]


def draw_skeleton(overlay, kpts, color):
    if kpts is None:
        return
    for i, j in SKELETON_EDGES:
        if (i < len(kpts) and j < len(kpts) and
                kpts[i][2] > 0.3 and kpts[j][2] > 0.3):
            cv2.line(overlay,
                     (int(kpts[i][0]), int(kpts[i][1])),
                     (int(kpts[j][0]), int(kpts[j][1])),
                     color, 1, cv2.LINE_AA)
    for idx, kp in enumerate(kpts):
        if kp[2] > 0.3:
            c = KEYPOINT_COLORS[idx] if idx < len(KEYPOINT_COLORS) else color
            cv2.circle(overlay, (int(kp[0]), int(kp[1])), 3, c, -1, cv2.LINE_AA)


def draw_overlay(frame, tracks, frame_idx, total_frames, fps,
                 id_switches, recoveries, pose_results=None, n_animals_target=6):
    overlay = frame.copy()

    # 关联 pose → track
    for tid, track in tracks.items():
        if track.time_since_update > 0:
            continue

        color = get_color(tid)

        # 平滑轨迹尾巴 (使用 track.history → 已经是卡尔曼平滑后的)
        pts = track.history
        trail_len = 120
        start = max(1, len(pts) - trail_len)
        for i in range(start, len(pts)):
            # 校验: 如果两个相邻点的距离过大(缝合点跳变), 断开连线
            dist = np.linalg.norm(np.array(pts[i]) - np.array(pts[i-1]))
            if dist > 60:  # 一帧内动物不可能移动超过60像素
                continue
            
            a = (i - start) / (len(pts) - start) if len(pts) > start else 1
            thickness = max(1, int(a * 3))
            cv2.line(overlay,
                     (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]), int(pts[i][1])),
                     color, thickness, cv2.LINE_AA)

        # Bbox
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # ID 标签
        label = f"#{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(overlay, (x1, y1-th-10), (x1+tw+4, y1), color, -1)
        cv2.putText(overlay, label, (x1+2, max(20, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # 中心点 (平滑后)
        cv2.circle(overlay, track.center, 4, (255, 255, 255), -1)

        # 骨架
        if pose_results:
            tc = np.array(track.center, dtype=float)
            best_pr = None
            best_d = 80
            for pr in pose_results:
                d = np.linalg.norm(tc - np.array(pr.center))
                if d < best_d:
                    best_d = d
                    best_pr = pr
            if best_pr is not None:
                draw_skeleton(overlay, best_pr.keypoints, color)

    # 信息面板
    panel = overlay.copy()
    cv2.rectangle(panel, (0, 0), (550, 95), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.75, overlay, 0.25, 0, overlay)

    active = sum(1 for t in tracks.values() if t.time_since_update == 0)
    lines = [
        f"Frame {frame_idx}/{total_frames}  |  {fps:.0f} FPS",
        f"Active: {active}/{n_animals_target}  |  ID Switches: {id_switches}  |  Recoveries: {recoveries}",
        f"Tracker: AnimalMOT (Adaptive Kalman + Skeleton Re-ID)",
    ]
    for i, line in enumerate(lines):
        cv2.putText(overlay, line, (10, 22 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)

    return overlay


def main():
    video_path = "data/videos/6xWT.mp4"
    output_path = "data/results/tracking_v6_6xWT_full.mp4"

    if not os.path.exists(video_path):
        print(f"[ERROR] 视频不存在: {video_path}")
        return

    from src.detection.threshold_detector import StaticThresholdDetector
    from src.tracking.animalmot import AnimalMOT, create_animalmot
    from src.pose.pose_engine import GeometricPoseEngine

    geo_pose = GeometricPoseEngine()

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    MAX_FRAMES = total

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (w, h))

    detector = StaticThresholdDetector(n_animals=N_ANIMALS, threshold=70)
    tracker = create_animalmot(n_animals=N_ANIMALS)

    t0 = time.time()
    frame_idx = 0
    prev_ids = set()
    id_switch_count = 0

    print(f"{'='*60}")
    print(f"生成 v6 AnimalMOT (自适应卡尔曼 + 骨架 Re-ID)")
    print(f"视频: {video_path} ({w}x{h}, {fps_video:.0f}fps)")
    print(f"输出: {output_path}")
    print(f"{'='*60}")

    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 检测
        dets = detector.detect(frame)

        # 2. 骨架估计 → 附加到检测
        det_bboxes = [d.bbox for d in dets if hasattr(d, 'bbox')]
        pose_results = geo_pose.detect_from_bboxes(frame, det_bboxes) if det_bboxes else None

        # 附加骨架嵌入
        if pose_results:
            for pr in pose_results:
                if pr.skeleton_embedding is None:
                    continue
                pc = np.array(pr.center, dtype=float)
                best_det = None
                best_dist = 80.0
                for d in dets:
                    dc = np.array(d.center, dtype=float) if hasattr(d, 'center') else np.array(
                        [(d.bbox[0]+d.bbox[2])/2, (d.bbox[1]+d.bbox[3])/2])
                    dist = np.linalg.norm(pc - dc)
                    if dist < best_dist:
                        best_dist = dist
                        best_det = d
                if best_det is not None:
                    best_det.skeleton_embedding = pr.skeleton_embedding

        # 3. 追踪 (传入 frame 用于 CNN 外观提取 + 虚拟检测注入)
        tracks = tracker.update(dets, frame=frame)

        # 4. ID 切换计数
        curr_ids = {tid for tid, t in tracks.items() if t.time_since_update == 0}
        if prev_ids and curr_ids:
            new_ids = curr_ids - prev_ids
            if new_ids and len(tracks) <= N_ANIMALS:
                id_switch_count += len(new_ids)
        prev_ids = curr_ids

        # 5. 绘制
        stats = tracker.get_stats()
        elapsed = time.time() - t0
        fps_now = (frame_idx + 1) / max(elapsed, 0.01)

        viz = draw_overlay(frame, tracks, frame_idx, MAX_FRAMES, fps_now,
                           id_switch_count, stats['total_recovered'],
                           pose_results)
        out.write(viz)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  [{frame_idx}/{MAX_FRAMES}] "
                  f"active={len(curr_ids)}, "
                  f"switches={id_switch_count}, "
                  f"recoveries={stats['total_recovered']}, "
                  f"virtual={stats.get('virtual_injections', 0)}, "
                  f"{fps_now:.0f}fps")

    cap.release()
    out.release()
    elapsed = time.time() - t0

    final = tracker.get_stats()
    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"{'='*60}")
    print(f"  输出: {output_path}")
    print(f"  总耗时: {elapsed:.1f}s ({frame_idx/elapsed:.0f} fps)")
    print(f"  ID 切换总数: {id_switch_count}")
    print(f"  ID 恢复次数: {final['total_recovered']}")
    print(f"  虚拟检测注入: {final.get('virtual_injections', 0)}")
    print(f"  创建轨迹数: {final['total_created']}")
    print(f"  最大并发: {final['max_concurrent']}")
    print(f"  CNN Re-ID: {'Yes' if final.get('reid_available') else 'No'}")


if __name__ == "__main__":
    main()
