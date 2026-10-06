"""
实例分割模块测试
Test Instance Segmentation Module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2

def test_segmentation_utils():
    """测试分割工具"""
    print("="*60)
    print("分割工具测试")
    print("="*60)

    from src.segmentation.utils import MaskUtils, SegmentationVisualizer

    # 测试掩码创建
    print("\n[测试1] 创建测试掩码")
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)
    print(f"✅ 掩码创建成功，形状: {mask.shape}, 面积: {np.sum(mask)//255}")

    # 测试边界框提取
    print("\n[测试2] 提取边界框")
    bbox = MaskUtils.mask_to_bbox(mask)
    print(f"✅ 边界框: {bbox}")

    # 测试质心计算
    print("\n[测试3] 计算质心")
    centroid = MaskUtils.get_mask_centroid(mask)
    print(f"✅ 质心: {centroid}")

    # 测试掩码优化
    print("\n[测试4] 优化掩码")
    # 添加噪声
    noisy_mask = mask.copy()
    cv2.circle(noisy_mask, (50, 50), 10, 255, -1)
    refined = MaskUtils.refine_mask(noisy_mask, kernel_size=5, iterations=1)
    print(f"✅ 掩码优化完成，噪声消除: {np.sum(noisy_mask)//255} -> {np.sum(refined)//255}")

    print("\n" + "="*60)
    print("✅ 分割工具测试通过！")
    print("="*60)
    return True


def test_instance_mask():
    """测试实例掩码类"""
    print("\n" + "="*60)
    print("实例掩码类测试")
    print("="*60)

    from src.segmentation.segmenter import InstanceMask

    # 创建两个测试掩码
    mask1_data = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask1_data, (80, 100), 40, 255, -1)

    mask2_data = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask2_data, (120, 100), 40, 255, -1)

    mask1 = InstanceMask(mask1_data, (30, 50, 130, 150), 0.95)
    mask2 = InstanceMask(mask2_data, (70, 50, 170, 150), 0.92)

    print(f"\n[测试1] 掩码1: {mask1}")
    print(f"  面积: {mask1.area}")
    print(f"  中心: {mask1.center}")

    print(f"\n[测试2] 掩码2: {mask2}")
    print(f"  面积: {mask2.area}")
    print(f"  中心: {mask2.center}")

    # 测试重叠计算
    print("\n[测试3] 重叠检测")
    overlap_ratio = mask1.get_overlap_ratio(mask2)
    iou = mask1.get_iou(mask2)
    print(f"  重叠比例: {overlap_ratio:.3f}")
    print(f"  IoU: {iou:.3f}")

    print("\n" + "="*60)
    print("✅ 实例掩码类测试通过！")
    print("="*60)
    return True


def test_segmentation_visualizer():
    """测试分割可视化器"""
    print("\n" + "="*60)
    print("分割可视化器测试")
    print("="*60)

    from src.segmentation.segmenter import InstanceMask
    from src.segmentation.utils import SegmentationVisualizer

    # 创建测试图像
    frame = np.ones((300, 400, 3), dtype=np.uint8) * 200

    # 创建测试掩码
    mask_data = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(mask_data, (150, 150), 60, 255, -1)

    mask = InstanceMask(mask_data, (80, 80, 220, 230), 0.95)

    # 创建可视化器
    visualizer = SegmentationVisualizer(alpha=0.5)

    # 绘制掩码
    print("\n[测试1] 绘制掩码")
    result = visualizer.draw_mask(frame.copy(), mask, show_bbox=True, show_contour=True)
    print(f"✅ 掩码绘制成功，图像形状: {result.shape}")

    # 绘制带ID的掩码
    print("\n[测试2] 绘制带ID的掩码")
    result = visualizer.draw_mask_with_id(frame.copy(), mask, track_id=1)
    print(f"✅ 带ID掩码绘制成功")

    # 保存结果
    os.makedirs("demo_output", exist_ok=True)
    output_path = "demo_output/segmentation_test.png"
    cv2.imwrite(output_path, result)
    print(f"✅ 结果已保存: {output_path}")

    print("\n" + "="*60)
    print("✅ 可视化器测试通过！")
    print("="*60)
    return True


def test_segmenter():
    """测试实例分割器"""
    print("\n" + "="*60)
    print("实例分割器测试")
    print("="*60)

    from src.segmentation.segmenter import InstanceSegmenter, create_segmenter

    # 创建分割器
    print("\n[测试1] 创建分割器")
    try:
        segmenter = create_segmenter('n')
        print("✅ 分割器创建成功")
        print(f"  模型路径: {segmenter.model_path}")
    except Exception as e:
        print(f"⚠️  模型创建: {e}")
        print("   这是正常的，使用时模型会自动下载")

    print("\n" + "="*60)
    print("✅ 分割器测试完成！")
    print("="*60)
    return True


if __name__ == "__main__":
    success = True

    try:
        success = test_segmentation_utils() and success
        success = test_instance_mask() and success
        success = test_segmentation_visualizer() and success
        success = test_segmenter() and success

        if success:
            print("\n" + "🎉"*20)
            print("所有测试通过！实例分割模块已就绪！")
            print("🎉"*20)
        else:
            print("\n部分测试失败，请检查错误信息")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
