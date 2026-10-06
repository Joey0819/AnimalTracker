"""
简化版 AI 行为识别演示
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import math

print("="*80)
print("AnimalTracker v0.2.0 - AI 行为识别功能演示")
print("="*80)

# 模拟特征提取
class SimpleBehaviorRecognizer:
    """简化版行为识别器"""

    BEHAVIOR_TYPES = {
        'exploration': '探索',
        'resting': '休息',
        'moving': '移动',
        'interaction': '互动',
        'high_speed': '快速移动',
        'low_speed': '慢速移动'
    }

    def __init__(self):
        self.speed_history = []

    def recognize(self, tracks, frame_idx):
        """
        识别当前帧的行为

        Args:
            tracks: 轨迹字典
            frame_idx: 帧号

        Returns:
            behavior: 行为类型
        """
        if not tracks:
            return '其他'

        # 计算平均速度
        speeds = []
        for track_id, track in tracks.items():
            if hasattr(track, 'velocity') and track.velocity is not None:
                speed = np.linalg.norm(track.velocity)
                speeds.append(speed)

        if not speeds:
            return '其他'

        avg_speed = np.mean(speeds)

        # 简单规则
        if avg_speed < 0.3:
            return '休息'
        elif avg_speed < 1.0:
            return '探索'
        elif avg_speed < 2.0:
            return '移动'
        elif avg_speed >= 3.0:
            return '快速移动'
        else:
            return '其他'

# 测试识别器
recognizer = SimpleBehaviorRecognizer()

print("\n支持的交互类型:")
behavior_list = list(recognizer.BEHAVIOR_TYPES.items())[:6]
for i, (key, name) in behavior_list:
    print(f"  {key}. {name}")

# 模拟测试
print("\n模拟不同行为模式:")

# 场景1：休息
print("\n场景 1: 休息（0帧）")
tracks_rest = {
    1: type('Track', (), {'id': 1, 'center': (100, 100), 'velocity': np.array([0, 0])}),
    2: type('Track', (), {'id': 2, 'center': (200, 200), 'velocity': np.array([0, 0])})
}
behavior = recognizer.recognize(tracks_rest, 0)
print(f"  识别结果: {behavior}")

# 场景2：探索
print("\n场景 2: 探索（10帧）")
tracks_explore = {
    1: type('Track', (), {'id': 1, 'center': (100, 100), 'velocity': np.array([0.5, 0.2])}),
    2: type('Track', (), {'id': 2, 'center': (200, 200), 'velocity': np.array([0.6, 0.3])}),
}
behavior = recognizer.recognize(tracks_explore, 0)
print(f"  识别结果: {behavior}")

# 场景3：快速移动
print("\n场景 3: 快速移动（20帧）")
tracks_fast = {
    1: type('Track', (), {'id': 1, 'center': (100, 100), 'velocity': np.array([5.0, 3.0])}),
    2: type('Track', (), {'id': 2, 'center': (200, 200), 'velocity': np.array([4.0, 2.0])})
}
behavior = recognizer.recognize(tracks_fast, 0)
print(f"  识别结果: {behavior}")

print("\n" + "="*80)
print("AI 行为识别功能演示完成！")
print("\n功能说明:")
print("  ✓ 行为特征提取")
print("  ✓ 速度识别")
print("  ✓ 行为分类")
print("  ✓ 模式识别")
print("\n下一步:")
print("  • 集成到主应用")
print("  • 添加更多特征")
print("  • 训练 LSTM 模型")
print("="*80)
