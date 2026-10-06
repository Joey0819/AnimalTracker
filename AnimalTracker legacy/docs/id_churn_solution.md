# ID Churn 问题解决方案

## 问题诊断

您的追踪结果：
- 历史累计 24 个 ID
- 只有 4 条轨迹稳定
- **ID Churn 率** = (24-4)/4 = 5倍（正常应该 < 1.5倍）

## 根本原因

| 原因 | 说明 |
|------|------|
| **检测不稳定** | 同一只动物在不同帧置信度波动大 |
| **max_distance 过大** | 导致错误的跨帧匹配 |
| **max_disappeared 过小** | 轨迹过快被删除，被迫重建 |
| **虚拟标记阈值过高** | 没有生成足够的稳定标记 |

## 解决方案

### 方案1: 使用反ID Churn追踪器 (推荐)

新的 `AntiChurnTracker` 专门针对此问题优化：

```python
from src.tracking.anti_churn_tracker import create_anti_churn_tracker

tracker = create_anti_churn_tracker(
    max_track_slots=4,      # 预期动物数量
    max_distance=80.0,      # 降低匹配距离
    max_disappeared=50,     # 增加保留时间
    min_confirmed_hits=3    # 确认门槛
)
```

### 方案2: 调整现有追踪器参数

如果继续使用 `EnhancedTracker`，调整以下参数：

| 参数 | 建议值 | 原默认值 | 说明 |
|------|--------|----------|------|
| `max_distance` | 60-80 | 100 | 减少错误匹配 |
| `max_disappeared` | 40-60 | 30 | 延长轨迹保留 |
| `min_confirmed_hits` | 3-5 | 3 | 增加确认门槛 |
| `预期动物数量` | 4 | 0 | 限制最大轨迹数 |

### 方案3: 启用Re-ID匹配

```python
tracker = create_enhanced_tracker(
    preset='high_accuracy',
    use_reid_matching=True  # 关键：启用Re-ID
)
```

## 参数调优指南

### 快速诊断参数

运行诊断工具查看当前问题：

```bash
python diagnose_tracking.py your_video.mp4 --confidence 0.25 --max-frames 100
```

查看输出：
- `平均每帧检测数`: 应该接近动物数量
- `ID切换次数`: 越低越好
- `置信度分布`: 大量低置信度需要调整

### 参数调整顺序

1. **第一步**: 降低 `max_distance`
   ```
   100 → 80 → 60 → 50 (如果ID切换仍然频繁)
   ```

2. **第二步**: 增加 `max_disappeared`
   ```
   30 → 50 → 70 (如果轨迹经常断裂)
   ```

3. **第三步**: 设置 `预期动物数量`
   ```
   在UI中设置: 预期动物数量 = 4
   这会强制限制最大轨迹数
   ```

4. **第四步**: 启用增强检测
   ```
   勾选 "启用增强检测" 选项
   这会稳定检测结果
   ```

## 在UI中使用

1. 打开 `app.py`
2. 在侧边栏设置：
   - 预期动物数量: `4`
   - 最大匹配距离: `60` (从100降低)
   - 最大消失帧数: `50` (从30增加)
   - 启用增强检测: `✓`
   - 追踪器: `vmTracker` 或 `增强追踪器`

3. 重新运行追踪

## 预期效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 历史ID数 | 24 | ~6 |
| 稳定轨迹数 | 4 | 4 |
| ID Churn率 | 5.0 | < 1.0 |
| IDF1 | ~60% | ~90% |

## 如果问题仍然存在

1. **检查视频质量**
   - 光照是否充足
   - 动物是否被遮挡
   - 对比度是否足够

2. **尝试不同的检测置信度**
   - 降低到 0.15 获取更多检测
   - 或提高到 0.35 减少误检

3. **联系开发者**
   - 保存 `debug_output` 目录中的诊断报告
   - 提供视频样本片段
