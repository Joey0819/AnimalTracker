"""
综合测试 - 验证所有优化后的功能
Test Full Pipeline with All Optimizations

测试内容:
1. 增强检测 (EnhancedYOLODetector)
2. Re-ID集成匹配 (ReIDIntegratedMatcher)
3. 分割感知追踪 (SegmentAwareTracker)
4. 姿态感知追踪 (PoseAwareTracker)
"""

import numpy as np
import cv2
import sys

def test_all_modules():
    """测试所有模块导入"""
    print("=" * 60)
    print("模块导入测试")
    print("=" * 60)

    modules = []

    # 检测模块
    try:
        from src.detection.detector import YOLODetector, Detection
        modules.append(("YOLODetector", "OK"))
    except Exception as e:
        modules.append(("YOLODetector", f"FAIL: {e}"))

    try:
        from src.detection.enhanced_detector import EnhancedYOLODetector, create_enhanced_detector
        modules.append(("EnhancedYOLODetector", "OK"))
    except Exception as e:
        modules.append(("EnhancedYOLODetector", f"FAIL: {e}"))

    # 追踪模块
    try:
        from src.tracking.tracker import MultiObjectTracker
        modules.append(("MultiObjectTracker", "OK"))
    except Exception as e:
        modules.append(("MultiObjectTracker", f"FAIL: {e}"))

    try:
        from src.tracking.enhanced_tracker import EnhancedTracker, create_enhanced_tracker
        modules.append(("EnhancedTracker", "OK"))
    except Exception as e:
        modules.append(("EnhancedTracker", f"FAIL: {e}"))

    # Re-ID模块
    try:
        from src.tracking.reid_integrated_matcher import ReIDIntegratedMatcher, create_reid_matcher
        modules.append(("ReIDIntegratedMatcher", "OK"))
    except Exception as e:
        modules.append(("ReIDIntegratedMatcher", f"FAIL: {e}"))

    # 分割模块
    try:
        from src.tracking.segment_aware_tracker import (
            SegmentAwareTracker, create_segment_aware_tracker,
            MaskedDetection, MaskedTrack
        )
        modules.append(("SegmentAwareTracker", "OK"))
    except Exception as e:
        modules.append(("SegmentAwareTracker", f"FAIL: {e}"))

    # 姿态模块
    try:
        from src.tracking.pose_aware_tracker import (
            PoseAwareMarkerGenerator, create_pose_aware_tracker,
            PoseAwareMarker, PoseKeypoint
        )
        modules.append(("PoseAwareTracker", "OK"))
    except Exception as e:
        modules.append(("PoseAwareTracker", f"FAIL: {e}"))

    # 分割器
    try:
        from src.segmentation.segmenter import InstanceSegmenter, InstanceMask
        modules.append(("InstanceSegmenter", "OK"))
    except Exception as e:
        modules.append(("InstanceSegmenter", f"FAIL: {e}"))

    # 打印结果
    for name, status in modules:
        status_str = "[OK]" if status == "OK" else "[FAIL]"
        print(f"  {status_str} {name}")
        if status != "OK":
            print(f"       {status}")

    ok_count = sum(1 for _, s in modules if s == "OK")
    print(f"\n总计: {ok_count}/{len(modules)} 模块正常")

    return ok_count == len(modules)

def test_enhanced_detector():
    """测试增强检测器"""
    print("\n" + "=" * 60)
    print("增强检测器测试")
    print("=" * 60)

    try:
        from src.detection.enhanced_detector import create_enhanced_detector

        detector = create_enhanced_detector(
            model_path="yolov8n.pt",
            confidence=0.25,
            enable_preprocessing=True
        )

        print("[OK] 增强检测器创建成功")
        print(f"  预处理启用: {detector.enable_preprocessing}")
        print(f"  时序平滑启用: {detector.enable_temporal_smooth}")
        print(f"  历史长度: {detector.history_len}")

        # 测试图像增强
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        enhanced = detector.preprocess_frame(test_frame)

        print(f"[OK] 图像预处理测试通过")
        print(f"  输入形状: {test_frame.shape}")
        print(f"  输出形状: {enhanced.shape}")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

def test_reid_matcher():
    """测试Re-ID匹配器"""
    print("\n" + "=" * 60)
    print("Re-ID集成匹配器测试")
    print("=" * 60)

    try:
        from src.tracking.reid_integrated_matcher import create_reid_matcher
        from src.tracking.hungarian_matcher import Track

        matcher = create_reid_matcher(
            feature_method='combined',
            adaptive_weighting=True
        )

        print("[OK] Re-ID匹配器创建成功")
        print(f"  外观权重: {matcher.appearance_weight}")
        print(f"  自适应权重: {matcher.adaptive_weighting}")

        # 创建模拟数据
        mock_crop = np.random.randint(100, 150, (128, 64, 3), dtype=np.uint8)

        class MockTrack:
            def __init__(self, track_id, center):
                self.id = track_id
                self.center = center
                self.bbox = (center[0]-20, center[1]-20, center[0]+20, center[1]+20)

        tracks = [MockTrack(1, (100, 100))]
        matcher.register_track(1, mock_crop)

        print(f"[OK] 外观模型注册成功")
        print(f"  活跃模型数: {len(matcher.appearance_models)}")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

def test_segment_tracker():
    """测试分割感知追踪器"""
    print("\n" + "=" * 60)
    print("分割感知追踪器测试")
    print("=" * 60)

    try:
        from src.tracking.segment_aware_tracker import (
            create_segment_aware_tracker, MaskedDetection
        )

        tracker = create_segment_aware_tracker(max_distance=100.0)

        print("[OK] 分割感知追踪器创建成功")

        # 创建模拟掩码
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 50, 1, -1)

        det = MaskedDetection(mask, (270, 190, 370, 290), 0.9)

        print(f"[OK] 掩码检测创建成功")
        print(f"  中心 (质心): {det.center}")
        print(f"  面积: {det.area}")

        # 测试追踪
        result = tracker.update_with_masks([det])
        print(f"[OK] 追踪更新成功")
        print(f"  活跃轨迹: {list(result.keys())}")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

def test_pose_tracker():
    """测试姿态感知追踪器"""
    print("\n" + "=" * 60)
    print("姿态感知追踪器测试")
    print("=" * 60)

    try:
        from src.tracking.pose_aware_tracker import (
            create_pose_aware_tracker, PoseKeypoint, PoseAwareMarker
        )

        generator = create_pose_aware_tracker(
            confidence_threshold=0.5,
            min_visible_keypoints=2
        )

        print("[OK] 姿态感知标记生成器创建成功")
        print(f"  置信度阈值: {generator.confidence_threshold}")
        print(f"  最少关键点: {generator.min_visible_keypoints}")

        # 创建模拟关键点
        keypoints = {
            'nose': PoseKeypoint('nose', (100, 100), 0.85),
            'left_ear': PoseKeypoint('left_ear', (90, 95), 0.82),
            'right_ear': PoseKeypoint('right_ear', (110, 95), 0.78),
        }

        marker = PoseAwareMarker(1, keypoints, 0)

        print(f"[OK] 姿态感知标记创建成功")
        print(f"  中心: {marker.center}")
        print(f"  稳定关键点数: {len(marker.get_stable_keypoints())}")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

def test_integration():
    """测试集成功能"""
    print("\n" + "=" * 60)
    print("集成功能测试")
    print("=" * 60)

    try:
        from src.tracking.enhanced_tracker import create_enhanced_tracker

        # 创建带所有功能的增强追踪器
        tracker = create_enhanced_tracker(
            preset='high_accuracy',
            use_optical_flow=True,
            use_hungarian_matching=True,
            use_reid_matching=True
        )

        print("[OK] 完整功能增强追踪器创建成功")
        print(f"  光流估计: {tracker.use_optical_flow}")
        print(f"  匈牙利匹配: {tracker.use_hungarian_matching}")
        print(f"  Re-ID匹配: {tracker.use_reid_matching}")

        if tracker.reid_matcher:
            print(f"  Re-ID匹配器: {type(tracker.reid_matcher).__name__}")

        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("AnimalTracker 综合测试套件")
    print("版本: v1.0.3")
    print("=" * 60)

    results = []

    # 运行测试
    results.append(("模块导入", test_all_modules()))
    results.append(("增强检测器", test_enhanced_detector()))
    results.append(("Re-ID匹配器", test_reid_matcher()))
    results.append(("分割追踪器", test_segment_tracker()))
    results.append(("姿态追踪器", test_pose_tracker()))
    results.append(("集成功能", test_integration()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    ok_count = sum(1 for _, p in results if p)
    print(f"\n总计: {ok_count}/{len(results)} 测试通过")

    if ok_count == len(results):
        print("\n[OK] 所有测试通过！系统已就绪。")
        return 0
    else:
        print("\n[FAIL] 部分测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
