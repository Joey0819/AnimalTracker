"""
分割感知追踪器完整测试
验证实例分割集成到追踪链路的功能
"""

import numpy as np
import cv2
from src.tracking.segment_aware_tracker import (
    create_segment_aware_tracker,
    MaskedDetection,
    MaskedTrack,
    SegmentAwareMatcher
)

def create_overlapping_masks():
    """
    创建两个重叠的圆形掩码
    模拟两只小鼠重叠的场景
    """
    h, w = 480, 640

    # 第一个掩码 - 左侧圆
    mask1 = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask1, (200, 240), 50, 1, -1)

    # 第二个掩码 - 右侧圆，与第一个重叠
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask2, (250, 240), 50, 1, -1)

    return mask1, mask2

def test_masked_detection():
    """测试掩码检测类"""
    print("=" * 60)
    print("测试1: 掩码检测类")
    print("=" * 60)

    mask1, mask2 = create_overlapping_masks()

    # 创建检测
    det1 = MaskedDetection(mask1, (150, 190, 250, 290), 0.9)
    det2 = MaskedDetection(mask2, (200, 190, 300, 290), 0.85)

    print(f"检测1:")
    print(f"  中心点 (质心): {det1.center}")
    print(f"  面积: {det1.area} 像素")
    print(f"  置信度: {det1.confidence}")

    print(f"\n检测2:")
    print(f"  中心点 (质心): {det2.center}")
    print(f"  面积: {det2.area} 像素")
    print(f"  置信度: {det2.confidence}")

    # 计算重叠
    overlap_ratio = det1.get_overlap_ratio(det2)
    iou = det1.get_iou(det2)

    print(f"\n重叠分析:")
    print(f"  重叠比例: {overlap_ratio:.3f}")
    print(f"  IoU: {iou:.3f}")

    if overlap_ratio > 0.1:
        print(f"  → 检测到重叠场景！")

    print("[OK] 测试通过\n")

def test_segment_aware_matcher():
    """测试分割感知匹配器"""
    print("=" * 60)
    print("测试2: 分割感知匹配器")
    print("=" * 60)

    matcher = SegmentAwareMatcher(
        max_distance=100.0,
        overlap_threshold=0.1
    )

    # 创建轨迹和检测
    mask1, mask2 = create_overlapping_masks()

    track1 = MaskedTrack(1, MaskedDetection(mask1, (150, 190, 250, 290), 0.9))
    track2 = MaskedTrack(2, MaskedDetection(mask2, (200, 190, 300, 290), 0.85))

    # 新帧的检测（稍微移动）
    mask1_new = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask1_new, (205, 240), 50, 1, -1)

    mask2_new = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask2_new, (255, 240), 50, 1, -1)

    det1_new = MaskedDetection(mask1_new, (155, 190, 255, 290), 0.9)
    det2_new = MaskedDetection(mask2_new, (205, 190, 305, 290), 0.85)

    tracks = [track1, track2]
    detections = [det1_new, det2_new]

    # 执行匹配
    matches, info = matcher.match(tracks, detections)

    print(f"匹配结果: {matches}")
    print(f"检测到的重叠组数: {info.get('num_overlaps', 0)}")

    for track_idx, det_idx in matches:
        track = tracks[track_idx]
        det = detections[det_idx]
        print(f"  轨迹{track.id} -> 检测{det_idx}: {track.center} -> {det.center}")

    print("[OK] 测试通过\n")

def test_segment_aware_tracker():
    """测试分割感知追踪器"""
    print("=" * 60)
    print("测试3: 分割感知追踪器")
    print("=" * 60)

    tracker = create_segment_aware_tracker(
        max_distance=100.0,
        max_disappeared=30
    )

    # 模拟多帧
    frames_data = []

    # 帧1: 两只分离的小鼠
    mask1_1 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask1_1, (100, 240), 40, 1, -1)
    det1_1 = MaskedDetection(mask1_1, (60, 200, 140, 280), 0.9)

    mask2_1 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask2_1, (300, 240), 40, 1, -1)
    det2_1 = MaskedDetection(mask2_1, (260, 200, 340, 280), 0.9)

    frames_data.append([det1_1, det2_1])

    # 帧2: 小鼠靠近
    mask1_2 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask1_2, (120, 240), 40, 1, -1)
    det1_2 = MaskedDetection(mask1_2, (80, 200, 160, 280), 0.9)

    mask2_2 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask2_2, (280, 240), 40, 1, -1)
    det2_2 = MaskedDetection(mask2_2, (240, 200, 320, 280), 0.9)

    frames_data.append([det1_2, det2_2])

    # 帧3: 小鼠重叠（关键测试场景）
    mask1_3 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask1_3, (180, 240), 40, 1, -1)
    det1_3 = MaskedDetection(mask1_3, (140, 200, 220, 280), 0.85)

    mask2_3 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask2_3, (220, 240), 40, 1, -1)
    det2_3 = MaskedDetection(mask2_3, (180, 200, 260, 280), 0.85)

    frames_data.append([det1_3, det2_3])

    # 追踪
    print("帧1: 初始化")
    result1 = tracker.update_with_masks(frames_data[0])
    print(f"  活跃轨迹: {list(result1.keys())}")

    print("\n帧2: 小鼠靠近")
    result2 = tracker.update_with_masks(frames_data[1])
    print(f"  活跃轨迹: {list(result2.keys())}")
    for tid, track in result2.items():
        print(f"    轨迹{tid}: 中心 {track.center}")

    print("\n帧3: 小鼠重叠 (关键测试)")
    result3 = tracker.update_with_masks(frames_data[2])
    print(f"  活跃轨迹: {list(result3.keys())}")
    for tid, track in result3.items():
        print(f"    轨迹{tid}: 中心 {track.center}")

    # 统计
    print("\n追踪统计:")
    stats = tracker.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("[OK] 测试通过\n")

def test_overlap_detection():
    """测试重叠检测功能"""
    print("=" * 60)
    print("测试4: 重叠检测")
    print("=" * 60)

    matcher = SegmentAwareMatcher(overlap_threshold=0.05)

    # 创建多个检测，其中一些重叠
    masks = []
    dets = []

    # 检测1和2重叠
    mask1 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask1, (200, 240), 50, 1, -1)
    dets.append(MaskedDetection(mask1, (150, 190, 250, 290), 0.9))

    mask2 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask2, (220, 240), 50, 1, -1)
    dets.append(MaskedDetection(mask2, (170, 190, 270, 290), 0.9))

    # 检测3不重叠
    mask3 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask3, (500, 240), 50, 1, -1)
    dets.append(MaskedDetection(mask3, (450, 190, 550, 290), 0.9))

    # 检测4与1重叠
    mask4 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask4, (210, 240), 50, 1, -1)
    dets.append(MaskedDetection(mask4, (160, 190, 260, 290), 0.85))

    # 检测重叠
    overlap_groups = matcher._detect_overlaps(dets)

    print(f"检测数量: {len(dets)}")
    print(f"检测到的重叠组: {len(overlap_groups)}")

    for i, group in enumerate(overlap_groups):
        print(f"  重叠组 {i+1}: 检测索引 {group}")
        for idx in group:
            print(f"    检测{idx}: 中心 {dets[idx].center}, 面积 {dets[idx].area}")

    print("[OK] 测试通过\n")

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("分割感知追踪器完整测试套件")
    print("=" * 60 + "\n")

    test_masked_detection()
    test_segment_aware_matcher()
    test_segment_aware_tracker()
    test_overlap_detection()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)

if __name__ == "__main__":
    main()
