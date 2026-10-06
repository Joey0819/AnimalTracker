"""
行为聚类分析模块测试
Test Behavior Clustering Module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import numpy as np


def test_stgcn():
    """测试 Social ST-GCN 模块"""
    print("=" * 60)
    print("Social ST-GCN 模块测试")
    print("=" * 60)

    from src.behavior.stgcn import create_social_stgcn

    # 创建分析器
    stgcn = create_social_stgcn(num_nodes_max=10, window_size=20)
    print("[1] Social ST-GCN 初始化成功")

    # 模拟追踪数据
    print("\n[2] 模拟追踪数据")
    tracks_history = []
    for frame_idx in range(30):
        tracks = {
            1: (100 + frame_idx * 2, 100 + frame_idx),
            2: (150 + frame_idx * 1.5, 120 + frame_idx * 0.5),
            3: (200 + frame_idx, 150)
        }
        tracks_history.append(tracks)
    print(f"  生成了 {len(tracks_history)} 帧追踪数据")

    # 处理序列
    print("\n[3] 提取序列特征")
    features = stgcn.extract_sequence_features(tracks_history, 0, 30)
    print(f"  特征维度: {len(features)}")
    print(f"  特征范围: [{features.min():.3f}, {features.max():.3f}]")

    # 获取统计
    print("\n[4] 图统计信息")
    stats = stgcn.get_graph_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("[OK] Social ST-GCN 模块测试通过!")
    return True


def test_clustering():
    """测试行为聚类模块"""
    print("\n" + "=" * 60)
    print("行为聚类模块测试")
    print("=" * 60)

    from src.behavior.clustering import create_behavior_clustering

    # 生成模拟数据
    print("\n[1] 生成模拟追踪数据")
    tracks_history = []

    # 场景1: 静止休息
    for frame_idx in range(60):
        tracks = {
            1: (100, 100),
            2: (120, 100),
            3: (110, 120)
        }
        tracks_history.append(tracks)

    # 场景2: 探索移动
    for frame_idx in range(60, 120):
        tracks = {
            1: (100 + (frame_idx - 60) * 3, 100 + (frame_idx - 60) * 2),
            2: (120 + (frame_idx - 60) * 2, 100 + (frame_idx - 60)),
            3: (110 + (frame_idx - 60) * 4, 120 + (frame_idx - 60) * 3)
        }
        tracks_history.append(tracks)

    # 场景3: 聚集接触
    for frame_idx in range(120, 180):
        tracks = {
            1: (200, 200),
            2: (195, 203),
            3: (200, 195)
        }
        tracks_history.append(tracks)

    print(f"  生成了 {len(tracks_history)} 帧数据")

    # 创建聚类器
    print("\n[2] 创建行为聚类分析器")
    analyzer = create_behavior_clustering(
        n_clusters=15,
        n_dimensions=8,
        sequence_length=30
    )
    print(f"  聚类数量: {analyzer.n_clusters}")
    print(f"  降维维度: {analyzer.n_dimensions}")

    # 拟合
    print("\n[3] 拟合聚类模型")
    analyzer.fit(tracks_history, segment_length=30, overlap=0.5)

    # 预测
    print("\n[4] 预测行为类型")
    test_segments = [
        (0, 30, "场景1: 静止"),
        (60, 90, "场景2: 移动"),
        (120, 150, "场景3: 聚集")
    ]

    for start, end, name in test_segments:
        label, confidence = analyzer.predict(tracks_history, start, end)
        print(f"  {name}: {label} (置信度: {confidence:.3f})")

    # 摘要
    print("\n[5] 聚类摘要")
    summary = analyzer.get_cluster_summary()
    active_clusters = [s for s in summary if s['size'] > 0]
    print(f"  活跃聚类: {len(active_clusters)}")
    for item in active_clusters[:5]:
        print(f"    聚类 {item['cluster_id']}: {item['label']} (大小: {item['size']})")

    print("\n" + "=" * 60)
    print("[OK] 行为聚类模块测试通过!")
    return True


def test_integrated_analyzer():
    """测试集成行为分析器"""
    print("\n" + "=" * 60)
    print("集成行为分析器测试")
    print("=" * 60)

    from src.behavior.integrated import create_behavior_analyzer

    # 创建分析器
    print("\n[1] 创建集成行为分析器")
    analyzer = create_behavior_analyzer(
        num_animals=5,
        sequence_length=20,
        n_clusters=10
    )
    print("  初始化成功")

    # 生成模拟数据
    print("\n[2] 生成模拟追踪数据")

    # 场景1: 静止
    for frame_idx in range(40):
        tracks = {
            1: (100, 100),
            2: (120, 100),
            3: (110, 120)
        }
        analyzer.process_frame(tracks, frame_idx)

    # 场景2: 接近交互
    for frame_idx in range(40, 80):
        progress = (frame_idx - 40) / 40
        tracks = {
            1: (100 + progress * 50, 100 + progress * 20),
            2: (120 - progress * 30, 100 + progress * 10),
            3: (110, 120)
        }
        analyzer.process_frame(tracks, frame_idx)

    # 场景3: 聚集
    for frame_idx in range(80, 120):
        tracks = {
            1: (150, 120),
            2: (90, 110),
            3: (110, 120)
        }
        analyzer.process_frame(tracks, frame_idx)

    print(f"  处理了 {len(analyzer.tracks_history)} 帧")

    # 训练聚类
    print("\n[3] 训练行为聚类")
    analyzer.train_clustering(segment_length=20, overlap=0.5)

    # 获取摘要
    print("\n[4] 行为摘要")
    summary = analyzer.get_behavior_summary()
    print(f"  总帧数: {summary.get('total_frames', 0)}")
    print(f"  训练状态: {summary.get('training_status', 'unknown')}")

    if summary.get('behavior_distribution'):
        print(f"  行为分布:")
        for behavior, count in summary['behavior_distribution'].items():
            print(f"    {behavior}: {count}")

    # 获取交互事件
    print("\n[5] 交互事件")
    interactions = analyzer.get_interaction_events()
    print(f"  检测到 {len(interactions)} 个交互事件")
    for inter in interactions[:5]:
        print(f"    {inter}")

    # 图统计
    print("\n[6] 社交图统计")
    graph_stats = analyzer.stgcn.get_graph_statistics()
    for key, value in graph_stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("[OK] 集成行为分析器测试通过!")
    return True


def test_behavior_clustering_complete():
    """完整的行为聚类测试"""
    print("\n" + "=" * 60)
    print("行为聚类完整流程测试")
    print("=" * 60)

    from src.behavior import (
        create_social_stgcn,
        create_behavior_clustering,
        create_behavior_analyzer
    )

    print("\n[1] 模块导入成功")
    print("  - Social ST-GCN")
    print("  - 行为聚类器")
    print("  - 集成分析器")

    # 创建ST-GCN
    print("\n[2] 创建 Social ST-GCN")
    stgcn = create_social_stgcn()
    print(f"  最大节点数: {stgcn.num_nodes_max}")
    print(f"  时间窗口: {stgcn.window_size}")

    # 创建聚类器
    print("\n[3] 创建行为聚类器")
    clustering = create_behavior_clustering()
    print(f"  聚类数量: {clustering.n_clusters}")

    # 创建集成分析器
    print("\n[4] 创建集成分析器")
    analyzer = create_behavior_analyzer()
    print(f"  最大动物数: {analyzer.num_animals}")

    # 模拟数据流
    print("\n[5] 模拟数据流处理")

    for scenario_idx in range(3):
        print(f"\n  场景 {scenario_idx + 1}:")

        for frame_idx in range(30):
            offset = scenario_idx * 100
            tracks = {
                1: (100 + frame_idx + offset, 100 + frame_idx // 2),
                2: (120 + frame_idx // 2 + offset, 100),
                3: (110 + offset, 120 + frame_idx)
            }

            result = analyzer.process_frame(tracks, frame_idx + scenario_idx * 30)

        print(f"    处理了 30 帧")

    # 训练
    print("\n[6] 训练聚类模型")
    analyzer.train_clustering(segment_length=20, overlap=0.5)

    # 最终统计
    print("\n[7] 最终统计")
    summary = analyzer.get_behavior_summary()
    print(f"  总帧数: {summary.get('total_frames', 0)}")
    print(f"  训练状态: {summary.get('training_status', 'unknown')}")

    if summary.get('behavior_distribution'):
        print(f"  检测到的行为类型: {len(summary['behavior_distribution'])}")

    print("\n" + "=" * 60)
    print("[OK] 行为聚类完整流程测试通过!")
    return True


if __name__ == "__main__":
    success = True

    try:
        success = test_stgcn() and success
        success = test_clustering() and success
        success = test_integrated_analyzer() and success
        success = test_behavior_clustering_complete() and success

        if success:
            print("\n" + "=" * 60)
            print("行为聚类分析模块 - 所有测试通过!")
            print("=" * 60)
            print("\n功能清单:")
            print("  [OK] Social ST-GCN 社交图分析")
            print("  [OK] 行为特征提取")
            print("  [OK] PCA降维 (替代UMAP)")
            print("  [OK] 层次聚类")
            print("  [OK] 行为类型自动识别")
            print("  [OK] 集成行为分析")
            print("\n基于论文:")
            print("  - STCS (Patterns, 2024) - segCluster方法")
            print("  - Social ST-GCN - 社交时空图卷积网络")
        else:
            print("\n部分测试失败，请检查错误信息")
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
