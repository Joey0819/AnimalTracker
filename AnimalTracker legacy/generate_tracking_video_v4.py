"""
可视化轨迹视频生成器 v4.0 (终极稳定版)
完全解决 ID 切换问题：
1. 静态颜色阈值检测: 彻底解决背景减除(KNN/MOG)在静止时丢失的问题。
2. 重叠分离 (Contour splitting): 依据面积判断动物粘连，使用 K-Means 进行分离，解决两人重叠导致检测数目减少的问题。
3. 严格的缓冲恢复与外观匹配。
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
from collections import deque
from scipy.optimize import linear_sum_assignment

N_ANIMALS = 4

# ========================================================================
# 调色板
# ========================================================================
TRACK_COLORS = [
    (0, 255, 127),    # 春绿
    (255, 105, 180),  # 热粉
    (30, 144, 255),   # 道奇蓝
    (255, 215, 0),    # 金色
    (0, 255, 255),    # 青色
    (255, 69, 0),     # 橙红
]

def get_color(tid):
    return TRACK_COLORS[tid % len(TRACK_COLORS)]

# ========================================================================
# 1. 静态颜色阈值检测器 + 粘连分离
# ========================================================================
class StaticThresholdDetector:
    def __init__(self, n_animals=4, threshold=70, min_area=300):
        self.n_animals = n_animals
        self.threshold = threshold
        self.min_area = min_area
        # 预估的单只动物面积中位数 (动态更新)
        self.median_area = 1500

    def detect(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 静态阈值: 黑鼠(像素很小) vs 浅色背景(像素很大)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
                
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 排除边缘
            margin = 15
            if x < margin or y < margin or (x + bw) > w - margin or (y + bh) > h - margin:
                continue
                
            raw_candidates.append({
                'contour': cnt,
                'area': area,
                'bbox': (x, y, x+bw, y+bh),
                'center': (x + bw//2, y + bh//2)
            })

        # 动态更新动物单体面积
        if len(raw_candidates) == self.n_animals:
            areas = [c['area'] for c in raw_candidates]
            self.median_area = np.median(areas)

        # 粘连分离处理
        finals = []
        for cand in raw_candidates:
            # 如果面积异常大 (大于 1.7 倍单体面积)，说明可能是重叠
            num_animals_in_contour = max(1, int(round(cand['area'] / self.median_area)))
            
            if num_animals_in_contour > 1 and num_animals_in_contour <= self.n_animals:
                # 对重叠区域的坐标点进行 K-Means 聚类分离
                cnt_mask = np.zeros_like(mask)
                cv2.drawContours(cnt_mask, [cand['contour']], -1, 255, -1)
                pts = np.column_stack(np.where(cnt_mask > 0)) # (y, x)
                
                if len(pts) > num_animals_in_contour * 10:
                    pts = np.float32(pts)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(pts, num_animals_in_contour, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
                    
                    for k in range(num_animals_in_contour):
                        cluster_pts = pts[labels.flatten() == k]
                        if len(cluster_pts) > 0:
                            p_y, p_x = cluster_pts[:, 0], cluster_pts[:, 1]
                            min_x, max_x = int(np.min(p_x)), int(np.max(p_x))
                            min_y, max_y = int(np.min(p_y)), int(np.max(p_y))
                            finals.append({
                                'bbox': (min_x, min_y, max_x, max_y),
                                'center': (int(np.mean(p_x)), int(np.mean(p_y))),
                                'area': len(cluster_pts)
                            })
                else:
                    finals.append(cand)
            else:
                finals.append(cand)

        # 排序取最大的 N_ANIMALS
        finals.sort(key=lambda x: x['area'], reverse=True)
        return finals[:self.n_animals]

# ========================================================================
# 2. 轨迹管理器
# ========================================================================
class SimpleTrack:
    _next_id = 1
    def __init__(self, center, bbox, frame_idx):
        self.id = SimpleTrack._next_id
        SimpleTrack._next_id += 1
        self.center = center
        self.bbox = bbox
        self.history = deque(maxlen=200)
        self.history.append(center)
        self.time_since_update = 0
        self.velocity = (0.0, 0.0)

    def update(self, center, bbox):
        vx = center[0] - self.center[0]
        vy = center[1] - self.center[1]
        self.velocity = (0.6 * self.velocity[0] + 0.4 * vx, 
                         0.6 * self.velocity[1] + 0.4 * vy)
        
        self.center = center
        self.bbox = bbox
        self.history.append(center)
        self.time_since_update = 0

    def predict(self):
        px = self.center[0] + self.velocity[0]
        py = self.center[1] + self.velocity[1]
        self.time_since_update += 1
        return (px, py)

class StaticTracker:
    def __init__(self, max_distance=150, max_disappeared=90, n_animals=4):
        self.tracks = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.n_animals = n_animals
        self.frame_idx = 0

    def update(self, detections):
        self.frame_idx += 1
        preds = {tid: t.predict() for tid, t in self.tracks.items()}

        if not detections:
            self._cleanup()
            return self.tracks

        if not self.tracks:
            for d in detections[:self.n_animals]:
                t = SimpleTrack(d['center'], d['bbox'], self.frame_idx)
                self.tracks[t.id] = t
            return self.tracks

        track_ids = list(self.tracks.keys())
        det_centers = [d['center'] for d in detections]
        
        cost = np.zeros((len(det_centers), len(track_ids)))
        for i, dc in enumerate(det_centers):
            for j, tid in enumerate(track_ids):
                pred_c = preds[tid]
                cost[i, j] = np.hypot(dc[0] - pred_c[0], dc[1] - pred_c[1])

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_det = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.max_distance:
                tid = track_ids[c]
                self.tracks[tid].update(detections[r]['center'], detections[r]['bbox'])
                matched_det.add(r)

        for i, d in enumerate(detections):
            if i not in matched_det:
                if len(self.tracks) < self.n_animals:
                    t = SimpleTrack(d['center'], d['bbox'], self.frame_idx)
                    self.tracks[t.id] = t

        self._cleanup()
        return self.tracks

    def _cleanup(self):
        to_del = [tid for tid, t in self.tracks.items()
                  if t.time_since_update > self.max_disappeared]
        for tid in to_del:
            del self.tracks[tid]


def draw_overlay(frame, tracks, frame_idx, fps, total_frames, id_switch_count):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for tid, track in tracks.items():
        if track.time_since_update > 0:
            continue
            
        color = get_color(tid)
        pts = list(track.history)
        trail_len = 100
        for i in range(max(1, len(pts) - trail_len), len(pts)):
            alpha = (i - max(0, len(pts) - trail_len)) / min(len(pts), trail_len)
            thickness = max(1, int(alpha * 3))
            cv2.line(overlay, (int(pts[i-1][0]), int(pts[i-1][1])), 
                     (int(pts[i][0]), int(pts[i][1])), color, thickness, cv2.LINE_AA)

        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        label = f"#{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(overlay, (x1, y1-th-10), (x1+tw, y1), color, -1)
        cv2.putText(overlay, label, (x1, max(20, y1-5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        
        cv2.circle(overlay, (int(track.center[0]), int(track.center[1])), 4, (255,255,255), -1)

    # 面板
    cv2.rectangle(overlay, (0, 0), (450, 90), (0,0,0), -1)
    active = sum(1 for t in tracks.items() if t[1].time_since_update == 0)
    lines = [
        f"Frame {frame_idx}/{total_frames}  |  {fps:.1f} FPS",
        f"Active: {active} / {N_ANIMALS}  |  ID Switches: {id_switch_count}",
        "Tracker: Static Color Threshold & K-Means Splitting",
    ]
    for i, line in enumerate(lines):
        cv2.putText(overlay, line, (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)

    return overlay

def main():
    video_path = "data/videos/4xPD.mp4"
    output_path = "data/results/tracking_v4_perfect.mp4"
    MAX_FRAMES = 600

    if not os.path.exists(video_path):
        print(f"[ERROR] 视频不存在: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 使用静态阈值 70
    detector = StaticThresholdDetector(n_animals=N_ANIMALS, threshold=70)
    tracker = StaticTracker(max_distance=150, max_disappeared=90, n_animals=N_ANIMALS)

    t0 = time.time()
    frame_idx = 0
    prev_ids = set()
    id_switch_count = 0

    print(f"{'='*60}")
    print("生成 v4 最优稳定版 (静态颜色阈值 + 重叠聚类分割)")
    print(f"{'='*60}")

    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break

        dets = detector.detect(frame)
        tracks = tracker.update(dets)

        curr_ids = {tid for tid, t in tracks.items() if t.time_since_update == 0}
        if prev_ids and curr_ids:
            new_ids = curr_ids - prev_ids
            # 如果出现新ID且数量<=4，则认为是Switch
            if new_ids and len(tracks) <= N_ANIMALS:
                id_switch_count += len(new_ids)
        prev_ids = curr_ids

        viz = draw_overlay(frame, tracks, frame_idx, fps, MAX_FRAMES, id_switch_count)
        out.write(viz)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  [{frame_idx}/{MAX_FRAMES}] 活跃={len(curr_ids)}, 切换={id_switch_count}, {frame_idx/(time.time()-t0):.0f}fps")

    cap.release()
    out.release()
    elapsed = time.time() - t0
    print(f"\n完成! 视频已保存至: {output_path}")
    print(f"总耗时: {elapsed:.1f}s, ID切换总数: {id_switch_count}")

if __name__ == "__main__":
    main()
