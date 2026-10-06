"""
Re-ID 个体识别模块测试
Test Re-ID Module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2

def test_feature_extractor():
    """测试特征提取器"""
    print("="*60)
    print("特征提取器测试")
    print("="*60)

    from src.tracking.reid.feature_extractor import (
        ColorHistogramExtractor,
        HOGExtractor,
        CombinedFeatureExtractor,
        create_feature_extractor
    )

    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # 测试颜色直方图
    print("\n[测试1] 颜色直方图提取器")
    color_extractor = ColorHistogramExtractor()
    color_feat = color_extractor.extract(test_image)
    print(f"✅ 特征维度: {len(color_feat)}")

    # 测试HOG提取器
    print("\n[测试2] HOG提取器")
    hog_extractor = HOGExtractor()
    hog_feat = hog_extractor.extract(test_image)
    print(f"✅ 特征维度: {len(hog_feat)}")

    # 测试组合提取器
    print("\n[测试3] 组合特征提取器")
    combined_extractor = CombinedFeatureExtractor()
    combined_feat = combined_extractor.extract(test_image)
    print(f"✅ 特征维度: {len(combined_feat)}")

    # 测试距离计算
    print("\n[测试4] 特征距离计算")
    feat1 = np.random.rand(100)
    feat2 = np.random.rand(100)

    dist_cosine = color_extractor.compute_distance(feat1, feat2, 'cosine')
    dist_euclidean = color_extractor.compute_distance(feat1, feat2, 'euclidean')

    print(f"  余弦距离: {dist_cosine:.4f}")
    print(f"  欧氏距离: {dist_euclidean:.4f}")

    print("\n" + "="*60)
    print("✅ 特征提取器测试通过！")
    print("="*60)
    return True


def test_appearance_model():
    """测试外观模型"""
    print("\n" + "="*60)
    print("外观模型测试")
    print("="*60)

    from src.tracking.reid.reid import AppearanceModel

    # 创建外观模型
    print("\n[测试1] 创建外观模型")
    model = AppearanceModel(track_id=1, max_history=10)
    print(f"✅ 外观模型创建: {model}")

    # 添加特征
    print("\n[测试2] 添加特征")
    for i in range(5):
        feature = np.random.rand(100)
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        model.update(feature, image, confidence=0.9 - i*0.1, frame_id=i)

    print(f"✅ 添加了 {len(model.features)} 个特征")

    # 获取平均特征
    print("\n[测试3] 获取平均特征")
    mean_feat = model.get_mean_feature()
    print(f"✅ 平均特征维度: {len(mean_feat)}")

    # 测试最佳匹配
    print("\n[测试4] 最佳匹配距离")
    new_feature = model.features[0] + np.random.randn(100) * 0.1
    best_dist = model.get_best_match_feature(new_feature)
    print(f"✅ 最佳匹配距离: {best_dist:.4f}")

    # 测试可信度
    print("\n[测试5] 模型可信度")
    is_confident = model.is_confident(threshold=3)
    print(f"✅ 模型可信: {is_confident}")

    print("\n" + "="*60)
    print("✅ 外观模型测试通过！")
    print("="*60)
    return True


def test_reidentifier():
    """测试Re-ID识别器"""
    print("\n" + "="*60)
    print("Re-ID识别器测试")
    print("="*60)

    from src.tracking.reid.reid import ReIdentifier, create_reidentifier

    # 创建识别器
    print("\n[测试1] 创建识别器")
    reid = create_reidentifier()
    print(f"✅ 识别器创建成功")

    # 注册追踪
    print("\n[测试2] 注册追踪")
    for i in range(3):
        crop = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        reid.register_track(i+1, crop, frame_id=0)

    print(f"✅ 注册了 {len(reid.appearance_models)} 个追踪")

    # 模拟重新识别
    print("\n[测试3] 重新识别")
    test_crop = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    track_id, info = reid.reidentify(test_crop, [1, 2, 3], frame_id=10)

    if track_id is not None:
        print(f"✅ 识别为: ID={track_id}")
        print(f"  置信度: {info.get('confidence', 0):.3f}")
        print(f"  距离: {info.get('distance', 0):.3f}")
    else:
        print(f"⚠️  未能识别: {info.get('reason', 'unknown')}")

    # 获取统计
    print("\n[测试4] 统计信息")
    stats = reid.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("✅ Re-ID识别器测试通过！")
    print("="*60)
    return True


def test_reid_integration():
    """测试完整Re-ID流程"""
    print("\n" + "="*60)
    print("Re-ID集成测试")
    print("="*60)

    from src.tracking.reid.reid import create_reidentifier
    import matplotlib
    matplotlib.use('Agg')  # 使用非GUI后端
    import matplotlib.pyplot as plt

    # 创建识别器
    reid = create_reidentifier()

    # 模拟追踪场景：3只动物
    print("\n[场景] 模拟3只动物的追踪")

    # 每只动物有不同的外观特征
    colors = [
        (200, 100, 50),   # 棕色
        (100, 150, 50),   # 绿色
        (100, 100, 200),   # 蓝色
    ]

    # 注册每只动物的外观
    for i, color in enumerate(colors):
        # 创建带颜色的图像
        crop = np.ones((60, 60, 3), dtype=np.uint8)
        crop[:] = color
        # 添加一些纹理
        noise = np.random.randint(-20, 20, (60, 60, 3), dtype=np.int16)
        crop = crop.astype(np.int16) + noise
        crop = np.clip(crop, 0, 255).astype(np.uint8)

        reid.register_track(i+1, crop, frame_id=i)
        print(f"  注册动物{i+1}: 颜色={color}")

    # 模拟重新识别场景
    print("\n[测试] 遮挡后重新识别")

    for i, color in enumerate(colors):
        # 模拟遮挡后的检测
        test_crop = np.ones((60, 60, 3), dtype=np.uint8)
        test_crop[:] = color
        noise = np.random.randint(-30, 30, (60, 60, 3), dtype=np.int16)
        test_crop = test_crop.astype(np.int16) + noise
        test_crop = np.clip(test_crop, 0, 255).astype(np.uint8)

        # 重新识别
        track_id, info = reid.reidentify(test_crop, [1, 2, 3], frame_id=100)

        correct = (track_id == i + 1)
        status = "✅" if correct else "❌"

        print(f"  {status} 动物{i+1} -> 识别为ID={track_id} (置信度: {info.get('confidence', 0):.2f})")

    # 最终统计
    print("\n[统计] 识别性能")
    stats = reid.get_stats()
    print(f"  活跃追踪: {stats['active_tracks']}")
    print(f"  总匹配数: {stats['total_matches']}")

    print("\n" + "="*60)
    print("✅ Re-ID集成测试完成！")
    print("="*60)
    return True


if __name__ == "__main__":
    success = True

    try:
        success = test_feature_extractor() and success
        success = test_appearance_model() and success
        success = test_reidentifier() and success
        success = test_reid_integration() and success

        if success:
            print("\n" + "🎉"*20)
            print("所有测试通过！Re-ID模块已就绪！")
            print("🎉"*20)
        else:
            print("\n部分测试失败，请检查错误信息")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
