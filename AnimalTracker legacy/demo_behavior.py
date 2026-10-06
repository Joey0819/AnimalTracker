"""
独立的行为识别演示 - 无复杂依赖
"""

print("="*80)
print("AnimalTracker v0.2.0 - AI 行为识别功能")
print("="*80)

import numpy as np

class SimpleBehaviorRecognizer:
    """简化版行为识别器"""

    def __init__(self):
        self.speed_history = []

    def recognize(self, avg_speed):
        """根据平均速度识别行为"""
        if avg_speed < 0.3:
            return '休息 (Resting)'
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

print("\n支持的行为类型:")
behaviors = [
    '休息',
    '探索',
    '移动',
    '快速移动'
]

for i, behavior in enumerate(behaviors, 1):
    print(f"  {i}. {behavior}")

# 测试场景
print("\n测试不同行为:")

# 场景1：休息
print(f"\n场景 1: 休息 (速度≈0)")
behavior = recognizer.recognize(0.1)
print(f"  识别结果: {behavior}")

# 场景2：探索
print(f"\n场景 2: 探索 (速度≈0.5)")
behavior = recognizer.recognize(0.5)
print(f"  识别结果: {behavior}")

# 场景3：移动
print(f"\n场景 3: 移动 (速度≈1.5)")
behavior = recognizer.recognize(1.5)
print(f"  识别结果: {behavior}")

# 场景4：快速移动
print(f"\n场景 4: 快速移动 (速度≈3.5)")
behavior = recognizer.recognize(3.5)
print(f"  识别结果： {behavior}")

print("\n" + "="*80)
print("AI 行为识别功能演示完成！")
print("\n核心原理:")
print("  1. 提取速度特征")
print("   2. 分析运动模式")
print("   3. 匹配行为模式")
print("\n扩展功能:")
print("  • 15种行为类型")
print("  • 序列模式识别")
print("  • 社交网络分析")
print("\n预期效果:")
print("  • 自动识别行为状态")
print("  • 识别行为序列")
print("  • 适合日常研究使用")

print("\n" + "="*80)
