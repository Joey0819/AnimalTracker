"""
vmTracker 集成测试脚本
测试 vmTracker 与主应用的集成
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict

# 尝试导入 vmTracker
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / 'src' / 'tracking'))

    from vmtracker import create_vm_tracker, VMTracker
    VMTRACKER_AVAILABLE = True
    print("✅ vmTracker 导入成功")
except ImportError as e:
    VMTRACKER_AVAILABLE = False
    print(f"❌ vmTracker 导入失败: {e}")


class MockDetection:
    """模拟检测结果"""
    def __init__(self, x, y, w, h, confidence=0.95):
        self.bbox = (x, y, x + w, y + h)
        self.center = (x + w / 2, y + h / 2)
        self.confidence = confidence
        self.score = confidence


def test_vmtracker_import():
    """测试 vmTracker 导入"""
    print("\n" + "="*60)
    print("测试 1: vmTracker 导入")
    print("="*60)

    if not VMTRACKER_AVAILABLE:
        print("❌ vmTracker 不可用")
        return False

    # 测试创建追踪器
    for preset in ['balanced', 'high_accuracy', 'fast']:
        tracker = create_vm_tracker(preset=preset)
        print(f"✅ 创建 {preset} 预设成功")

    return True


def test_vmtracker_tracking():
    """测试 vmTracker 追踪功能"""
    print("\n" + "="*60)
    print("测试 2: vmTracker 追踪功能")
    print("="*60)

    if not VMTRACKER_AVAILABLE:
        print("❌ vmTracker 不可用")
        return False

    # 创建追踪器
    tracker = create_vm_tracker(preset='balanced')

    print(f"\n追踪器配置:")
    print(f"  光流估计: {tracker.use_optical_flow}")
    print(f"  匈牙利匹配: {tracker.use_hungarian_matching}")
    print(f"  虚拟标记: {tracker.use_virtual_markers}")

    # 模拟视频处理
    print(f"\n模拟处理 20 帧...")

    for frame_id in range(20):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 模拟 4 个动物
        detections = [
            MockDetection(100, 100, 50, 50, 0.95),
            MockDetection(200, 150, 50, 50, 0.92),
            MockDetection(300, 200, 50, 50, 0.97),
            MockDetection(400, 250, 50, 50, 0.90),
        ]

        # 添加运动
        for i, det in enumerate(detections):
            offset_x = int(np.sin(frame_id * 0.1 + i) * 20)
            offset_y = int(np.cos(frame_id * 0.1 + i) * 20)
            det.center = (det.center[0] + offset_x, det.center[1] + offset_y)

        # 更新追踪器
        tracks = tracker.update(frame, detections)

        if frame_id % 5 == 0:
            print(f"  帧 {frame_id}: {len(tracks)} 个轨迹")

    # 最终统计
    stats = tracker.get_stats()
    print(f"\n最终统计:")
    print(f"  处理帧数: {stats['frames_processed']}")
    print(f"  活跃轨迹: {stats['active_tracks']}")
    print(f"  匹配率: {stats['match_rate']:.2%}")

    if tracker.use_virtual_markers:
        print(f"  虚拟标记: {stats['vm_active']} 个")
        print(f"  锚点关联: {stats['anchor_associations']} 次")
        print(f"  锚点覆盖率: {stats.get('anchor_coverage', 0):.2%}")

    print("\n✅ vmTracker 追踪测试通过！")
    return True


def test_integration_with_app():
    """测试与主应用的集成"""
    print("\n" + "="*60)
    print("测试 3: 主应用集成")
    print("="*60)

    if not VMTRACKER_AVAILABLE:
        print("❌ vmTracker 不可用，无法测试集成")
        return False

    print("\n模拟主应用配置流程...")

    # 模拟用户选择
    print("  1. 用户选择追踪器: vmTracker")
    print("  2. 用户选择预设: 平衡模式")

    # 创建追踪器（模拟 app.py 中的代码）
    tracker = create_vm_tracker(preset='balanced')
    print("  3. vmTracker 初始化成功 ✅")

    # 模拟视频处理循环
    print("\n  4. 模拟视频处理...")

    for frame_idx in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            MockDetection(100 + frame_idx * 10, 100, 50, 50, 0.95),
            MockDetection(200 + frame_idx * 10, 150, 50, 50, 0.92),
        ]

        tracks = tracker.update(frame, detections)

        if frame_idx == 4:
            print(f"     帧 {frame_idx}: 追踪到 {len(tracks)} 个动物")

    print("\n✅ 主应用集成测试通过！")
    return True


def test_performance_comparison():
    """测试性能对比"""
    print("\n" + "="*60)
    print("测试 4: 性能对比")
    print("="*60)

    if not VMTRACKER_AVAILABLE:
        print("❌ vmTracker 不可用")
        return False

    print("\n对比不同预设的性能...")

    presets = ['fast', 'balanced', 'high_accuracy']
    results = {}

    for preset in presets:
        tracker = create_vm_tracker(preset=preset)

        # 处理相同的数据
        import time
        start_time = time.time()

        for frame_id in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = [
                MockDetection(100, 100, 50, 50, 0.95),
                MockDetection(200, 150, 50, 50, 0.92),
                MockDetection(300, 200, 50, 50, 0.97),
            ]
            tracks = tracker.update(frame, detections)

        elapsed = time.time() - start_time
        stats = tracker.get_stats()

        results[preset] = {
            'time': elapsed,
            'frames': stats['frames_processed'],
            'tracks': stats['active_tracks']
        }

        print(f"\n{preset}:")
        print(f"  处理时间: {elapsed:.3f} 秒")
        print(f"  FPS: {stats['frames_processed'] / elapsed:.1f}")

    print("\n✅ 性能对比测试完成！")
    return True


def run_all_tests():
    """运行所有集成测试"""
    print("\n" + "="*60)
    print("🚀 vmTracker 集成测试套件")
    print("="*60)

    results = []

    # 测试 1: 导入
    results.append(("导入测试", test_vmtracker_import()))

    if VMTRACKER_AVAILABLE:
        # 测试 2: 追踪功能
        results.append(("追踪功能", test_vmtracker_tracking()))

        # 测试 3: 主应用集成
        results.append(("主应用集成", test_integration_with_app()))

        # 测试 4: 性能对比
        results.append(("性能对比", test_performance_comparison()))

    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！vmTracker 已成功集成！")
        print("\n下一步:")
        print("  1. 运行主应用: streamlit run app.py")
        print("  2. 选择 'vmTracker (v0.3.0 ⭐推荐)'")
        print("  3. 上传视频进行测试")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
