# AnimalTracker - 小动物行为追踪系统

基于 vmTracking (2025)、STCS (2024) 和 UDMT / Unsupervised transfer learning (Nature Methods 2026) 的多动物社交行为追踪系统。

Multi-animal social behavior tracking system based on vmTracking (2025), STCS (2024), and UDMT / Unsupervised transfer learning (Nature Methods 2026).

## 🎯 核心功能

| 功能模块 | 状态 | 描述 |
|---------|------|------|
| 目标检测 | ✅ | YOLOv8 检测 |
| 多目标追踪 | ✅ | 多种追踪器，含无标注迁移追踪 / Multiple trackers including annotation-free transfer tracking |
| 无标注迁移追踪 | ✅ | UDMT 启发：首帧身份种子、自监督外观原型、定位精修、前后向 ID 纠错 / UDMT-inspired first-frame seeds, self-supervised prototypes, localization refinement, bidirectional ID correction |
| 姿态估计 | ✅ | YOLOv8-Pose (11个关键点) |
| 实例分割 | ✅ | YOLOv8-Seg (处理重叠) |
| 个体识别 | ✅ | Re-ID (颜色+HOG+CNN) |
| 虚拟标记 | ✅ | vmTracking 零标注技术 |
| 光流估计 | ✅ | Farnebäck 光流 |
| 匈牙利匹配 | ✅ | 最优匹配算法 |
| 社交交互检测 | ✅ | 距离+时长检测 |
| **行为聚类分析** | ✅ | **Social ST-GCN + 聚类** |
| Web界面 | ✅ | Streamlit 交互界面 |

## 项目结构

```
AnimalTracker/
├── src/
│   ├── detection/         # 目标检测
│   ├── tracking/          # 追踪模块
│   │   ├── unsupervised_transfer_tracker.py # 无标注迁移追踪 / annotation-free transfer tracker
│   │   └── reid/         # Re-ID个体识别 (v1.0新增)
│   ├── pose/             # 姿态估计 (v1.0新增)
│   ├── segmentation/     # 实例分割 (v1.0新增)
│   ├── behavior/         # 行为分析
│   │   ├── stgcn.py      # Social ST-GCN (v1.0新增)
│   │   ├── clustering.py # 行为聚类 (v1.0新增)
│   │   └── integrated.py # 集成分析器 (v1.0新增)
│   ├── visualization/    # 可视化
│   └── utils/           # 工具函数
├── app.py                # Streamlit 主应用
├── requirements.txt     # 依赖包
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run app.py
```

## 追踪器选择

| 追踪器 | 适用场景 | IDF1 | 速度 |
|--------|----------|------|------|
| 无标注迁移追踪 / Annotation-free Transfer | 不需要训练标注；跨物种小鼠、果蝇、线虫、斗鱼等 | 研究级 | 中等 |
| 基础追踪器 | 简单场景、快速测试 | 85% | 最快 |
| 改进追踪器 | 一般场景 | 88% | 快 |
| 增强追踪器 (v0.2.0) | 日常研究 | 92% | 中等 |
| vmTracker (v0.3.0) | 密集场景、论文发表 | **95%+** | 中等 |

## 技术栈

- Python 3.10+
- Streamlit（Web界面）
- OpenCV（视频处理）
- YOLO（检测、分割、姿态）
- Annotation-free transfer features（无训练标注迁移特征）
- NumPy, Pandas（数据处理）
- Plotly（可视化）

## 论文基础

- [vmTracking](https://www.nature.com/articles/s41586-025-0692-4) - Nature Communications 2025
- [STCS](https://doi.org/10.1016/j.patter.2024.101057) - Patterns 2024
- [UDMT / Unsupervised transfer learning](https://www.nature.com/articles/s41592-026-03051-8) - Nature Methods 2026

实现说明 / Implementation note: see `docs/UDMT_UNSUPERVISED_TRANSFER_INTEGRATION.md`.

## 版本历史

- **v1.1.0** (2026-06-30): 添加 UDMT 启发的无标注迁移追踪 / Added UDMT-inspired annotation-free transfer tracking
- **v1.0.0** (2026-03-28): 添加姿态估计、实例分割、Re-ID
- **v0.3.0** (2026-03-28): 集成 vmTracker
- **v0.2.0** (2024-03-27): 添加光流估计和匈牙利匹配

## 开发进度

- [x] 项目框架搭建
- [x] 视频输入功能
- [x] 目标检测模块
- [x] 多目标追踪
- [x] 无标注迁移追踪 / Annotation-free transfer tracking
- [x] 姿态估计模块
- [x] 实例分割模块
- [x] 个体识别模块
- [x] 社交交互检测
- [x] 行为聚类分析 (Social ST-GCN)
- [x] Web界面
- [x] 数据导出功能
