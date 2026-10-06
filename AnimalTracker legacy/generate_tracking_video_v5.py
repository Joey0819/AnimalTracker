"""
可视化轨迹视频生成器 v5.0 (骨架 Re-ID 增强版)

对比 v4:
  v4: StaticThresholdDetector + SimpleTracker (纯位置匹配)
  v5: StaticThresholdDetector + BufferRecoverTracker + Skeleton Re-ID (形态指纹)

改进点:
  1. BufferRecoverTracker 四阶段匹配 (IoU + 外观 + 运动 + 骨架)
  2. 骨架几何嵌入: 20维旋转/缩放不变形态特征
  3. Buffer-Recover: 丢失轨迹在缓冲区等待恢复，而非立即删除
  4. 可视化叠加骨架关键点 (当 pose engine 可用时)
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

N_ANIMALS = 4

# ========================================================================
# 调色板 (与 v4 一致，便于视觉对比)
# ========================================================================
TRACK_COLORS = [
    (0, 255, 127),    # 春绿
    (255, 105, 180),  # 热粉
    (30, 144, 255),   # 道奇蓝
    (255, 215, 0),    # 金色
    (0, 255, 255),    # 青色
    (255, 69, 0),     # 橙红
]

# 骨架连接 (用于可视化)
SKELETON_EDGES = [
    (0, 3), (1, 0), (2, 0), (3, 4), (4, 5),
    (5, 6), (3, 7), (3, 8), (5, 9), (5, 10),
]

KEYPOINT_COLORS = [
    (0, 0, 255),     # nose - 红
    (255, 0, 255),   # left_ear - 品红
    (255, 0, 255),   # right_ear - 品红
    (0, 255, 0),     # neck - 绿
    (0, 255, 255),   # spine_mid - 黄
    (0, 200, 200),   # spine_base
    (200, 200, 0),   # tail_base - 青
    (255, 128, 0),   # left_forepaw - 橙
    (255, 128, 0),   # right_forepaw - 橙
    (128, 0, 255),   # left_hindpaw - 紫
    (128, 0, 255),   # right_hindpaw - 紫
]


def get_color(tid):
    return TRACK_COLORS[tid % len(TRACK_COLORS)]


def draw_skeleton(overlay, kpts, color, alpha=0.7):
    """在帧上绘制骨架关键点和连接"""
    if kpts is None:
        return

    # 绘制骨骼连接
    for i, j in SKELETON_EDGES:
        if (i < len(kpts) and j < len(kpts) and
                kpts[i][2] > 0.3 and kpts[j][2] > 0.3):
            pt1 = (int(kpts[i][0]), int(kpts[i][1]))
            pt2 = (int(kpts[j][0]), int(kpts[j][1]))
            cv2.line(overlay, pt1, pt2, color, 1, cv2.LINE_AA)

    # 绘制关键点
    for idx, kp in enumerate(kpts):
        if kp[2] > 0.3:
            kp_color = KEYPOINT_COLORS[idx] if idx < len(KEYPOINT_COLORS) else color
            cv2.circle(overlay, (int(kp[0]), int(kp[1])), 3, kp_color, -1, cv2.LINE_AA)


def draw_overlay(frame, tracks, frame_idx, fps, total_frames,
                 id_switch_count, recoveries, tracker_name,
                 pose_results=None, skeleton_weight=0.0):
    """绘制追踪可视化叠加层"""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # 构建 pose → 位置的索引，用于关联 track 与 pose
    pose_map = {}
    if pose_results:
        for pr in pose_results:
            pose_map[pr.center] = pr

    for tid, track in tracks.items():
        if track.time_since_update > 0:
            continue

        color = get_color(tid)

        # 轨迹尾巴
        pts = list(track.history) if hasattr(track, 'history') else []
        trail_len = 100
        for i in range(max(1, len(pts) - trail_len), len(pts)):
            a = (i - max(0, len(pts) - trail_len)) / min(len(pts), trail_len)
            thickness = max(1, int(a * 3))
            p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)

        # Bbox
        bbox = track.bbox if hasattr(track, 'bbox') else None
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # ID 标签
            label = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(overlay, (x1, y1-th-10), (x1+tw+4, y1), color, -1)
            cv2.putText(overlay, label, (x1+2, max(20, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # 中心点
        center = track.center
        cv2.circle(overlay, (int(center[0]), int(center[1])), 4, (255, 255, 255), -1)

        # 绘制骨架 (如果有 pose 结果)
        if pose_results:
            best_pose = None
            best_dist = 60  # 最大关联距离
            tc = np.array(center, dtype=float)
            for pr in pose_results:
                pc = np.array(pr.center, dtype=float)
                d = np.linalg.norm(tc - pc)
                if d < best_dist:
                    best_dist = d
                    best_pose = pr
            if best_pose is not None:
                draw_skeleton(overlay, best_pose.keypoints, color)

    # ---- 信息面板 ----
    panel_h = 115
    panel_overlay = overlay.copy()
    cv2.rectangle(panel_overlay, (0, 0), (520, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(panel_overlay, 0.75, overlay, 0.25, 0, overlay)

    active = sum(1 for t in tracks.values()
                 if hasattr(t, 'time_since_update') and t.time_since_update == 0)

    lines = [
        f"Frame {frame_idx}/{total_frames}  |  {fps:.1f} FPS",
        f"Active: {active}/{N_ANIMALS}  |  ID Switches: {id_switch_count}  |  Recoveries: {recoveries}",
        f"Tracker: {tracker_name}",
        f"Skeleton Re-ID weight: {skeleton_weight:.2f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(overlay, line, (10, 22 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)

    return overlay


def main():
    video_path = "data/videos/4xPD.mp4"
    output_path = "data/results/tracking_v5_skeleton_reid.mp4"
    MAX_FRAMES = 600

    if not os.path.exists(video_path):
        print(f"[ERROR] 视频不存在: {video_path}")
        return

    # ── 导入生产组件 ──
    from src.detection.threshold_detector import StaticThresholdDetector
    from src.tracking.buffer_recover_tracker import (
        BufferRecoverTracker, create_buffer_recover_tracker)

    # 姿态引擎: 优先 DLC/SLEAP，否则用 GeometricPoseEngine (轮廓椭圆拟合)
    # YOLO-Pose 是人体模型，对小鼠无效，故不使用
    from src.pose.pose_engine import GeometricPoseEngine
    geo_pose = GeometricPoseEngine()
    pose_engine_name = "Geometric (contour/ellipse)"

    # 尝试加载 DLC/SLEAP
    dlc_engine = None
    try:
        from src.pose.pose_engine import DLCSuperAnimalEngine
        dlc_engine = DLCSuperAnimalEngine()
        if dlc_engine.available:
            pose_engine_name = "DLC superAnimal"
            print(f"[OK] 姿态引擎: DLC 3.0 superAnimal")
        else:
            dlc_engine = None
    except Exception:
        pass

    sleap_engine = None
    if dlc_engine is None:
        try:
            from src.pose.pose_engine import SLEAPEngine
            sleap_engine = SLEAPEngine()
            if sleap_engine.available:
                pose_engine_name = "SLEAP Bottom-Up"
                print(f"[OK] 姿态引擎: SLEAP")
            else:
                sleap_engine = None
        except Exception:
            pass

    if dlc_engine is None and sleap_engine is None:
        print(f"[OK] 姿态引擎: {pose_engine_name} (DLC/SLEAP 未安装)")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ── 初始化 ──
    detector = StaticThresholdDetector(n_animals=N_ANIMALS, threshold=70)
    tracker = create_buffer_recover_tracker(
        max_track_slots=N_ANIMALS,
        buffer_duration=90,
        max_age=60,
        recovery_threshold=0.55,
        high_conf_thresh=0.55,
        use_deep_embedding=False,    # 关闭人体深度嵌入，用骨架嵌入替代
        embedding_interval=5,
    )

    tracker_name = f"BufferRecover + Skeleton Re-ID ({pose_engine_name})"

    t0 = time.time()
    frame_idx = 0
    prev_ids = set()
    id_switch_count = 0

    print(f"{'='*60}")
    print(f"生成 v5 骨架 Re-ID 增强版")
    print(f"视频: {video_path} ({w}x{h}, {fps:.0f}fps, {total} frames)")
    print(f"输出: {output_path}")
    print(f"{'='*60}")

    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 检测
        dets = detector.detect(frame)

        # 2. 骨架估计 (在追踪之前，以便嵌入传入匹配代价)
        pose_results = None
        det_bboxes = [d.bbox for d in dets if hasattr(d, 'bbox')]

        if dlc_engine is not None:
            pose_results = dlc_engine.detect(frame)
        elif sleap_engine is not None:
            pose_results = sleap_engine.detect(frame)
        if not pose_results and det_bboxes:
            pose_results = geo_pose.detect_from_bboxes(frame, det_bboxes)

        # 将骨架嵌入附加到对应的检测对象上
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

        # 3. 追踪 (检测已附骨架嵌入，会传入匹配代价函数)
        tracks = tracker.update(frame, dets)

        # 4. 注册骨架嵌入到 Re-ID 匹配器 (用于后续帧的缓冲恢复)
        if pose_results:
            for pr in pose_results:
                if pr.skeleton_embedding is None:
                    continue
                pc = np.array(pr.center, dtype=float)
                best_tid = -1
                best_dist = 80.0
                for tid, trk in tracks.items():
                    tc = np.array(trk.center, dtype=float)
                    d = np.linalg.norm(pc - tc)
                    if d < best_dist:
                        best_dist = d
                        best_tid = tid
                if best_tid >= 0:
                    tracker.register_skeleton_embedding(
                        best_tid, pr.skeleton_embedding)

        # 4. ID 切换计数
        curr_ids = {tid for tid, t in tracks.items()
                    if t.time_since_update == 0}
        if prev_ids and curr_ids:
            new_ids = curr_ids - prev_ids
            if new_ids and len(tracks) <= N_ANIMALS:
                id_switch_count += len(new_ids)
        prev_ids = curr_ids

        # 5. 绘制
        stats = tracker.get_stats()
        viz = draw_overlay(
            frame, tracks, frame_idx, fps, MAX_FRAMES,
            id_switch_count, stats['total_recovered'],
            tracker_name, pose_results,
            skeleton_weight=tracker._skeleton_weight
        )
        out.write(viz)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{frame_idx}/{MAX_FRAMES}] "
                  f"active={len(curr_ids)}, "
                  f"switches={id_switch_count}, "
                  f"recoveries={stats['total_recovered']}, "
                  f"skel_matches={stats.get('skeleton_assisted_matches', 0)}, "
                  f"{frame_idx/elapsed:.0f}fps")

    cap.release()
    out.release()
    elapsed = time.time() - t0

    final_stats = tracker.get_stats()
    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"{'='*60}")
    print(f"  输出: {output_path}")
    print(f"  总耗时: {elapsed:.1f}s ({frame_idx/elapsed:.0f} fps)")
    print(f"  ID 切换总数: {id_switch_count}")
    print(f"  ID 恢复次数: {final_stats['total_recovered']}")
    print(f"  骨架辅助匹配: {final_stats.get('skeleton_assisted_matches', 0)}")
    print(f"  创建轨迹数: {final_stats['total_created']}")
    print(f"  最大并发: {final_stats['max_concurrent']}")
    print(f"  骨架权重: {tracker._skeleton_weight:.2f}")


if __name__ == "__main__":
    main()
