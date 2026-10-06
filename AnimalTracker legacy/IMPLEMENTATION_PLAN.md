# 多动物社交追踪一站式平台 — 实施计划

## 目标

基于三个开源仓库的核心算法，将现有 AnimalTracker 升级为 **采集 → 追踪 → 分析 → 数据见解 → AI报告** 的一站式平台。

## 三仓库技术提取摘要

| 仓库 | 可提取的核心技术 | 用途 |
|------|-----------------|------|
| **STCS** (`backend/app/algorithm/`) | FlowTracker 掩码光流追踪 + Merger 冲突团迭代合并 + OSNet 分类器 Re-ID | 高密度场景 ID 稳定性 (Stage 1+2) |
| **OmniTrax** (`utils/track/`) | Buffer-and-Recover Kalman 追踪 + 代价矩阵填充防强制错配 + 多类别投票 | 低误检鲁棒追踪 + 物种分类 |
| **vmTracking** (`08_evaluation/`) | `evaluateMetrics.m` / `findBestMatchingBlock.m` 标准化评估流程 | 追踪精度量化基准 |

## 已完成的工作

- [x] STCS FlowTracker IoM 公式修正 (max 而非 min)
- [x] STCS cost matrix 对齐 (union bbox)
- [x] STCS TrackletMerger 重写 (NetworkX clique 检测)
- [x] vmTracking 评估器翻译 (`vmtracking_evaluator.py`)
- [x] 基础 Kalman 增强 (`enhanced_kalman.py`)
- [x] 自适应管线雏形 (`adaptive_pipeline.py`)

---

## Phase 1: 补齐 STCS Stage 2 迭代合并循环 (🔴 最高优先级)

> **这是 STCS 能达到 95%+ IDF1 的关键，目前完全缺失。**

### [NEW] `src/tracking/stcs_appearance_merger.py`
实现 STCS `merger.py` 中的迭代 train→classify→merge 循环：
1. `init_tracks_kernel()` — 从最大冲突团初始化骨架轨迹
2. `extract_crop_features()` — 从视频帧裁剪动物图块，提取外观特征
3. `classify_tracklets()` — 用特征向量对未分配碎片打分
4. `merge_by_appearance()` — 高置信碎片合并到骨架轨迹
5. `iterative_merge()` — 主循环: 重复上述步骤直到已分配帧 ≥ 95%
6. `process_dumped()` — 回收被过滤的极短碎片

使用已有的 `reid_module.py` 中 `SimpleReID`(颜色直方图) 或 `OSNetReID`(ResNet50特征) 替代 MMPretrain。

### [MODIFY] `src/tracking/adaptive_pipeline.py`
- 将 `finalize_and_merge()` 对接新的 `STCSAppearanceMerger`
- 增加 `merge_progress_callback` 供 UI 显示合并进度

---

## Phase 2: 集成 OmniTrax Buffer-and-Recover 机制 (🟡 高优先级)

### [NEW] `src/tracking/buffer_recover_tracker.py`
移植 OmniTrax `tracker.py` 的核心创新：
1. **代价矩阵填充**: 添加 N 列占位符列 (`dist_thresh`)，避免远距离检测被强制匹配到已有轨迹
2. **Buffer-and-Recover**: 轨迹未被检测时，Kalman 继续预测位置（而非立即删除），`max_frames_to_skip` 帧后才清除
3. **多类别投票**: 每个轨迹记录历史检测类别，终止时取多数投票确定最终类型
4. **可选 Kalman 开关**: `use_kf=True/False`，无 KF 时退化为纯匹配

### [MODIFY] `app.py`
- 在追踪器选择菜单中新增 "OmniTrax Buffer-Recover" 选项
- 暴露 `dist_thresh`, `max_frames_to_skip`, `use_kf` 参数

---

## Phase 3: 轨迹预过滤与噪声清洗 (🟡 高优先级)

### [NEW] `src/tracking/track_filter.py`
对齐 STCS `Merger.filter_tracks()`:
1. **面积过滤**: 移除 mask 面积异常（< 5% 或 > 95% 分位数）的检测
2. **速度过滤**: 移除帧间位移超过物理极限的跳变点
3. **置信度过滤**: 移除平均置信度低于阈值的超短碎片 (< 3 帧)
4. **边界过滤**: 移除贴合画面边缘的误检

---

## Phase 4: 社交行为分析模块增强 (🟢 中优先级)

### [MODIFY] `src/behavior/detector.py`
增强社交交互检测:
1. **接近-远离事件**: 基于轨迹间距离变化率，检测接近/远离/跟随行为
2. **聚集事件**: 当 ≥3 个体在阈值半径内持续 N 帧，记录聚集事件
3. **追逐事件**: 基于运动方向一致性 + 距离收敛，检测追逐行为
4. **社交网络图**: 构建个体间交互频次的加权无向图

### [NEW] `src/behavior/social_network.py`
1. 从追踪结果构建个体间 **交互频次矩阵**
2. 计算社交网络指标：度中心性、聚类系数、社群检测
3. 输出 networkx 图对象，支持可视化

---

## Phase 5: 数据见解与 AI 报告 (🟢 中优先级)

### [NEW] `src/reporting/insight_generator.py`
自动化数据见解引擎:
1. **统计摘要**: 每个个体的活动量、速度分布、空间偏好
2. **异常检测**: 识别运动模式突变（突然静止/突然加速）
3. **时间段对比**: 将视频按时间段切分，对比行为变化
4. **社交动态**: 谁与谁互动最多、社交网络随时间变化趋势

### [NEW] `src/reporting/report_builder.py`
AI 报告生成器:
1. 汇总所有分析结果生成结构化 Markdown 报告
2. 包含关键图表（轨迹热力图、社交网络图、活动时间线）
3. 提供可导出的 PDF/HTML 报告

### [MODIFY] `app.py`
新增 Streamlit 页面:
- **📊 数据见解** — 统计摘要、异常检测、社交网络可视化
- **📝 AI报告** — 一键生成综合分析报告

---

## 实施顺序

| 阶段 | 预计文件数 | 优先级 | 说明 |
|------|-----------|--------|------|
| Phase 1 | 2 | 🔴 最高 | STCS 迭代合并是追踪精度的核心 |
| Phase 2 | 2 | 🟡 高 | OmniTrax buffer-recover 直接减少 ID 碎片 |
| Phase 3 | 1 | 🟡 高 | 噪声清洗是后续分析的前提 |
| Phase 4 | 2 | 🟢 中 | 社交行为分析增强 |
| Phase 5 | 3 | 🟢 中 | 报告生成 |

## 验证计划

### 自动化测试
```bash
python -c "from src.tracking.stcs_appearance_merger import STCSAppearanceMerger; print('OK')"
python -c "from src.tracking.buffer_recover_tracker import create_buffer_recover_tracker; print('OK')"
python -c "from src.tracking.track_filter import TrackFilter; print('OK')"
python src/tracking/vmtracking_evaluator.py
```

### 集成验证
```bash
streamlit run app.py
```
