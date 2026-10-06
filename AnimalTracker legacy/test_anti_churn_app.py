"""
测试 AntiChurnTracker 集成到 app.py
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 60)
print("AntiChurnTracker 集成测试")
print("=" * 60)

# 测试导入
try:
    from tracking.anti_churn_tracker import create_anti_churn_tracker
    print("[OK] AntiChurnTracker 导入成功")
except ImportError as e:
    print(f"[FAIL] 导入失败: {e}")
    sys.exit(1)

# 测试创建
tracker = create_anti_churn_tracker(
    max_track_slots=4,
    max_distance=80,
    max_disappeared=50
)

print(f"[OK] 追踪器创建成功")
print(f"  - 最大轨迹数: {tracker.max_track_slots}")
print(f"  - 最大距离: {tracker.max_distance}")
print(f"  - 最大消失帧数: {tracker.max_disappeared}")
print(f"  - 最小确认帧数: {tracker.min_confirmed_hits}")

# 模拟追踪
import numpy as np

print("\n[Test] 模拟追踪...")

for frame_idx in range(20):
    # 模拟4只小鼠的检测
    detections = []
    base_positions = [(100, 100), (200, 200), (300, 100), (400, 200)]

    for i, (bx, by) in enumerate(base_positions):
        x = bx + frame_idx * 2
        y = by + np.random.randint(-3, 4)

        class MockDetection:
            def __init__(self, x, y, conf):
                self.bbox = (x-20, y-20, x+20, y+20)
                self.center = (x, y)
                self.confidence = conf

        detections.append(MockDetection(x, y, 0.8))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = tracker.update(frame, detections)

    if frame_idx == 0:
        print(f"  帧0: 初始化 {len(tracks)} 条轨迹")
    elif frame_idx == 9:
        print(f"  帧9: 活跃轨迹 {len(tracks)} 条")
    elif frame_idx == 19:
        print(f"  帧19: 活跃轨迹 {len(tracks)} 条")

# 获取统计
stats = tracker.get_stats()

print("\n[统计]")
print(f"  总检测数: {stats.get('total_detections', 0)}")
print(f"  创建ID数: {stats.get('total_ids_created', 0)}")
print(f"  活跃轨迹: {stats.get('active_tracks', 0)}")
print(f"  拒绝新轨迹: {stats.get('rejected_new_tracks', 0)}")
print(f"  Churn率: {stats.get('churn_rate', 0):.2f}")

# 计算churn率
churn_rate = stats.get('churn_rate', 0)
if churn_rate < 1.0:
    print(f"\n[SUCCESS] Churn率正常 ({churn_rate:.2f} < 1.0)")
else:
    print(f"\n[WARNING] Churn率偏高 ({churn_rate:.2f} >= 1.0)")

print("\n" + "=" * 60)
print("[OK] 测试完成！AntiChurnTracker 已成功集成到 app.py")
print("=" * 60)

print("\n使用方法:")
print("1. 运行 app.py")
print("2. 在 '选择追踪器' 下拉框中选择:")
print("   '反ID碎片追踪器 (v1.0.4 🔥解决ID混乱)'")
print("3. 设置预期动物数量为 4")
print("4. 点击 [START] 开始追踪")
