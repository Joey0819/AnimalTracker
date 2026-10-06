"""
快速测试脚本 - 测试核心功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2

print("="*80)
print("AnimalTracker v0.2.0 - 核心模块快速测试")
print("="*80)

# 测试光流估计
print("\n[1/2] 测试光流估计...")
try:
    # 创建简单测试
    prev_img = np.zeros((100, 100, 3), dtype=np.uint8)
    prev_img[40:60, 40:60] = 255

    curr_img = np.zeros((100, 100, 3), dtype=np.uint8)
    curr_img[42:62, 42:62] = 255

    # 计算光流
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    print(f"OK - 光流形状: {flow.shape}, 范围: [{flow.min():.2f}, {flow.max():.2f}]")

except Exception as e:
    print(f"FAILED - {e}")

# 测试匈牙利匹配
print("\n[2/2] 测试匈牙利匹配...")
try:
    from scipy.optimize import linear_sum_assignment

    # 创建代价矩阵
    cost_matrix = np.array([
        [10, 25, 15],
        [20, 10, 30],
        [15, 20, 10]
    ])

    # 匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    print(f"OK - 匹配结果: {list(zip(row_ind, col_ind))}, 总代价: {total_cost}")

except Exception as e:
    print(f"FAILED - {e}")

print("\n" + "="*80)
print("核心模块测试完成!")
print("="*80)
