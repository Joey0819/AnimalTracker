"""
AI 行为识别功能演示
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("="*80)
print("AnimalTracker v0.2.0 - AI 行为识别功能")
print("="*80)

try:
    from src.behavior.behavior_recognizer import BehaviorRecognizer, create_behavior_recognizer

    # 创建识别器
    print("\n[1/2] 创建行为识别器...")

    # 规则版本（快速）
    recognizer_rule = create_behavior_recognizer(
        num_classes=15,
        sequence_length=10,
        use_lstm=False
    )

    print(f"✓ 支持 {len(recognizer_rule.behavior_types)} 种行为类型")
    print(f"✓ 序列长度: {recognizer_rule.sequence_length} 帧")
    print(f"✓ 使用规则分类（快速版本）")

    # 模拟追踪数据
    print("\n[2/2] 模拟追踪数据...")

    mock_history = []

    # 场景1：慢速探索行为
    print("  场景 1: 慢速探索（10帧）")
    for frame_idx in range(10):
        tracks = {}
        for track_id in range(3):
            speed = 0.3 + frame_idx * 0.05
            angle = frame_idx * 0.1
            tracks[track_id] = type('Track', (), {
                'id': track_id,
                'center': (100 + track_id * 50 + speed * 10,
                         200 + track_id * 30),
                'velocity': np.array([speed * math.cos(angle), speed * math.sin(angle)])
            })()
        mock_history.append(tracks)

    # 识别行为
    behaviors = recognizer_rule.recognize(mock_history)

    print(f"\n识别结果:")
    for beh in behaviors[:3]:
        print(f"  - {beh['behavior_type']}")
        print(f"    置信度: {beh['confidence']:.2f}")
        print(f"    平均速度: {beh['avg_speed']:.2f}")

    print("\n" + "="*80)
    print("AI 行为识别功能完成！")
    print("\n支持的完整功能:")
    print("  ✓ 行为特征提取")
    print("  ✓ 规则分类器（快速）")
    print("  ✓ LSTM 分类器（需要训练）")
    print("  ✓ 15种行为类型")
    print("\n预期效果:")
    print("  • 自动识别行为模式")
    print("  • 识别行为序列")
    print("  • 适合日常研究使用")

except Exception as e:
    print(f"FAILED - {e}")
    import traceback
    traceback.print_exc()

print("="*80)
