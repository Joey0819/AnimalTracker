"""
Re-ID集成匹配器测试
验证外观特征集成到匹配阶段的功能
"""

import numpy as np
import cv2
from src.tracking.reid_integrated_matcher import create_reid_matcher
from src.tracking.hungarian_matcher import Track
from src.detection.detector import Detection

def test_reid_matcher():
    """测试Re-ID集成匹配器"""
    print("=" * 60)
    print("Re-ID集成匹配器测试")
    print("=" * 60)

    # 创建匹配器
    matcher = create_reid_matcher(
        feature_method='combined',
        appearance_weight=0.4,
        adaptive_weighting=True
    )

    print(f"[OK] 匹配器创建成功")
    print(f"  - 外观权重: {matcher.appearance_weight}")
    print(f"  - 自适应权重: {matcher.adaptive_weighting}")
    print(f"  - Re-ID阈值: {matcher.reid_threshold}")

    # 创建模拟轨迹和检测
    print("\n[Test1] 创建模拟轨迹和检测")

    # 模拟外观图像
    mock_crop_1 = np.random.randint(100, 150, (128, 64, 3), dtype=np.uint8)  # 亮色个体
    mock_crop_2 = np.random.randint(50, 100, (128, 64, 3), dtype=np.uint8)   # 暗色个体
    mock_crop_3 = np.random.randint(100, 150, (128, 64, 3), dtype=np.uint8)  # 与1相似

    # 创建轨迹类
    class MockTrack:
        def __init__(self, track_id, center, bbox):
            self.id = track_id
            self.center = center
            self.bbox = bbox

    class MockDetection:
        def __init__(self, center, bbox, confidence=0.9):
            self.center = center
            self.bbox = bbox
            self.confidence = confidence

    # 注册轨迹外观
    tracks = [
        MockTrack(1, (100, 100), (80, 80, 120, 120)),
        MockTrack(2, (200, 200), (180, 180, 220, 220)),
    ]

    # 注册外观
    matcher.register_track(1, mock_crop_1)
    matcher.register_track(2, mock_crop_2)

    print(f"  注册了 {len(matcher.appearance_models)} 个外观模型")

    # 创建检测
    detections = [
        MockDetection((105, 102), (85, 82, 125, 122)),  # 接近轨迹1
        MockDetection((205, 202), (185, 182, 225, 222)),  # 接近轨迹2
        MockDetection((300, 300), (280, 280, 320, 320)),  # 新检测
    ]

    detection_crops = [mock_crop_1, mock_crop_2, mock_crop_3]

    print(f"  创建了 {len(detections)} 个检测")

    # 执行匹配
    print("\n[Test2] 执行Re-ID集成匹配")
    matches, info = matcher.match(tracks, detections, detection_crops)

    print(f"  匹配结果: {matches}")
    print(f"  匹配数量: {len(matches)}")

    # 解析匹配
    for track_idx, det_idx in matches:
        track = tracks[track_idx]
        det = detections[det_idx]
        similarity = matcher.get_appearance_similarity(track.id, detection_crops[det_idx])
        print(f"  轨迹{track.id} -> 检测{det_idx}: 位置{track.center}->{det.center}, 相似度{similarity:.3f}")

    # 测试外观相似度
    print("\n[Test3] 外观相似度测试")
    sim_1_1 = matcher.get_appearance_similarity(1, mock_crop_1)
    sim_1_2 = matcher.get_appearance_similarity(1, mock_crop_2)
    sim_1_3 = matcher.get_appearance_similarity(1, mock_crop_3)

    print(f"  轨迹1 vs crop1 (同个体): {sim_1_1:.3f}")
    print(f"  轨迹1 vs crop2 (不同个体): {sim_1_2:.3f}")
    print(f"  轨迹1 vs crop3 (相似个体): {sim_1_3:.3f}")

    # 测试冲突解决
    print("\n[Test4] 冲突解决测试")
    # 创建冲突：两个轨迹竞争同一检测
    conflicting_tracks = [
        MockTrack(1, (100, 100), (80, 80, 120, 120)),
        MockTrack(3, (102, 98), (82, 78, 122, 122)),  # 接近的轨迹
    ]
    conflicting_detections = [
        MockDetection((105, 100), (85, 80, 125, 120)),
    ]
    conflicting_crops = [mock_crop_1]

    resolved = matcher.resolve_conflicts(conflicting_tracks, conflicting_detections, conflicting_crops)
    print(f"  冲突匹配结果: {resolved}")
    print(f"  冲突解决次数: {matcher.stats['conflicts_resolved']}")

    # 获取统计
    print("\n[Test5] 统计信息")
    stats = matcher.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("[OK] Re-ID集成匹配器测试完成！")
    print("=" * 60)

def test_enhanced_tracker_with_reid():
    """测试带Re-ID的增强追踪器"""
    print("\n" + "=" * 60)
    print("带Re-ID的增强追踪器测试")
    print("=" * 60)

    from src.tracking.enhanced_tracker import create_enhanced_tracker

    # 创建追踪器
    tracker = create_enhanced_tracker(
        preset='high_accuracy',
        use_reid_matching=True
    )

    print(f"[OK] 追踪器创建成功")
    print(f"  - 使用Re-ID匹配: {tracker.use_reid_matching}")
    print(f"  - 使用光流: {tracker.use_optical_flow}")
    print(f"  - 使用匈牙利匹配: {tracker.use_hungarian_matching}")

    if tracker.reid_matcher:
        print(f"  - Re-ID匹配器: {type(tracker.reid_matcher).__name__}")
        print(f"  - 外观权重: {tracker.reid_matcher.appearance_weight}")

    # 模拟帧更新
    print("\n[Test] 模拟帧更新")

    class MockDetection:
        def __init__(self, x, y, w=40, h=40, conf=0.9):
            self.bbox = (x, y, x + w, y + h)
            self.center = (x + w//2, y + h//2)
            self.confidence = conf

    # 创建模拟帧
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 第一帧：初始化
    detections_1 = [
        MockDetection(100, 100),
        MockDetection(200, 200),
    ]
    tracks_1 = tracker.update(frame, detections_1)
    print(f"  帧1: 检测{len(detections_1)}个, 追踪{len(tracks_1)}个轨迹")

    # 第二帧：移动
    detections_2 = [
        MockDetection(105, 102),
        MockDetection(205, 202),
    ]
    tracks_2 = tracker.update(frame, detections_2)
    print(f"  帧2: 检测{len(detections_2)}个, 追踪{len(tracks_2)}个轨迹")

    # 第三帧：新增
    detections_3 = [
        MockDetection(110, 104),
        MockDetection(210, 204),
        MockDetection(300, 300),
    ]
    tracks_3 = tracker.update(frame, detections_3)
    print(f"  帧3: 检测{len(detections_3)}个, 追踪{len(tracks_3)}个轨迹")

    # 统计
    stats = tracker.get_stats()
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("[OK] 带Re-ID的增强追踪器测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_reid_matcher()
    test_enhanced_tracker_with_reid()
