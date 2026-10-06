# DenseTrack-AI

**Foundation-Model AutoLabel MOT for Laboratory Animal Tracking**

零人工训练标注的多动物追踪与自动伪标注系统。基础安装可用 OpenCV 背景减除 + 几何姿态跑通 baseline；安装可选后端后，可接入 Norfair、Grounding DINO、SAM 2、DLC SuperAnimal 和外部 UDMT 研究后端做成熟开源 MOT、开放词汇检测、视频级分割传播、零样本姿态估计和无监督迁移追踪。

## 核心特性

- 🎯 **零标注追踪** — 基础 baseline 无需训练数据，可选 Foundation Model 后端
- 🔬 **科学级审计** — 每条轨迹、每个连接点都有置信度评分和审计报告
- 🐭 **小鼠优先** — 昆虫、斑马鱼为后续扩展目标
- 🔄 **自适应 ReID** — 有足够高置信 tracklet 时才训练，避免无样本误学习
- 📊 **多格式导出** — MOT Challenge / COCO / DLC / CSV + 可视化叠加视频
- 🧰 **成熟开源优先** — 通用 MOT 可直接使用 BSD-3 Norfair 后端；项目特定身份稳定层仅在成熟库不能覆盖时启用
- 🎞️ **干净轨迹回放** — 低置信或大跳变时自动断开短尾迹，避免把 ID 跳变画成长乱线
- 🧷 **长时身份连续性** — 接触冻结、合并框拒绝、硬重获门控、正反向动态规划与闭集拼接共同降低 ID switch 风险
- 🧱 **分层离散几何** — 将归一化位置、尺度和极坐标速度编码为两阶段残差 token，抑制坐标抖动对关联的放大
- 🧪 **UDMT 外部后端** — 可对接无训练标注的 UDMT 研究工作流，按许可证独立安装运行
- 🤖 **自动初始化** — 自动生成第一帧/早期帧个体 seed points，减少或替代 UDMT 式人工点击
- 🍎 **Mac 原生双向追踪** — `mac_adaptive` 后端自动选择两端种子，融合正向/反向轨迹并自适应更新轻量外观模板
- ⚠️ **身份风险审计** — 自动输出逐帧 `identity_risk.json/csv`，把正反向冲突优先送入稀疏 GT 审核包
- 📦 **UDMT 自动衔接包** — 自动输出官方工作区可读取的 `start_pos_array.txt`，首帧识别成功时无需人工点击
- 🧭 **无监督行为发现** — 从轨迹自动聚类 recurring motifs，输出 embedding 与逐个体时间线
- 🧩 **姿态增强模体** — 自动读取几何姿态；安装可选依赖后可切换 UMAP + HDBSCAN
- 🧪 **透明基础指标优先** — 受控效应与剂量比较默认输出路程、速度、区域停留、进入次数和时间箱；复杂表征作为发现层
- 🧬 **BEAST3D 启发视觉 SSL** — 从无标注轨迹、姿态、身份风险和可选多视角相机清单导出可迁移表征
- 🐬 **Dolph2Vec 启发音频 SSL** — 从未标注 WAV 中提取物种特异声学单元和哨声候选，用于通信行为审计
- 🔗 **长时全局拼接** — `long_term` DAG 路径覆盖使用稳健端点速度和离散几何连接遮挡后的轨迹片段
- 🧬 **接触事件审计** — 逐事件合并/拆分记录、前 K 个入→出身份假设、最优-次优 margin 与标签反转段，直接驱动稀疏 GT 审核优先级
- 🖥️ **双语研究面板** — 一键生成中文 / English 轨迹回放、自动初始化、行为模体与下载面板
- 🌐 **本地 Web 工作台** — 上传视频后可选零点击自动初始化或 UDMT 风格首帧点击辅助，后台追踪、回传已审核 GT、生成稀疏指标并打开结果面板
- ✅ **科研指标验证** — 风险优先抽取稀疏帧并预填 MOT；人工审核后计算 HOTA / DetA / AssA / MOTA / IDF1 / ID switches

## 长时身份方法边界

默认的“科研长时身份”配置参考 [HieDG](https://arxiv.org/abs/2607.00494) 的位置、尺度、极坐标速度分解与两阶段残差量化思想，并参考 [WaspMOT](https://arxiv.org/abs/2607.08729) 的闭集长序列协议和空间轨迹片段拼接。当前实现是无需训练的确定性本地适配：它不包含 HieDG 的 DETR、MLP 或学习码本，也不声称复现论文结果。

每次运行输出 `identity_continuity.json`、`identity_stitches.csv` 和 `identity_risk.json/csv`。其中稳定度、双向一致率和风险分数只用于无标注审计；只有基于人工审核 GT 的 HOTA、AssA、IDF1 和 ID switch 才能用于科研准确率结论。
HOTA 家族计算遵循 [TrackEval](https://github.com/JonathonLuiten/TrackEval) 的 0.05–0.95 阈值与全局关联定义。
算法、配置和验证边界详见 [`docs/LONG_TERM_IDENTITY.md`](docs/LONG_TERM_IDENTITY.md)。

## 行为汇总策略

[2026 bioRxiv 方法比较](https://www.biorxiv.org/content/10.64898/2026.07.20.739643v1)
在 314 段小鼠录像、八个药物-剂量条件中比较参数聚合、Keypoint-MoSeq、VAME、
SimBA 和 A-SOiD。论文报告的受控药物效应与剂量区分场景中，透明基础聚合与复杂
机器学习汇总没有显著性能差异。DenseTrack 因此把 `experiment_metrics.json` 中的
路程、速度、区域停留、进入次数、追踪覆盖和时间箱设为默认解释层；无监督模体和
SSL 表征主要用于开放式行为发现、社会交互、长期身份或大规模行为图谱。

该证据不直接解决身份关联，也不证明复杂表征在所有任务中无收益。详细策略见
[`docs/BEHAVIOR_SUMMARY_POLICY.md`](docs/BEHAVIOR_SUMMARY_POLICY.md)。

## 快速开始

### 安装

```bash
# 基础安装（背景减除检测 + 几何姿态 + 追踪）
pip install -e .

# 安装 Grounding DINO 检测后端
pip install -e ".[grounding-dino]"

# 安装 SAM 2 视频分割
pip install -e ".[sam2]"

# 安装 DLC SuperAnimal 姿态估计
pip install -e ".[dlc]"

# 安装高级无监督行为发现（UMAP + HDBSCAN）
pip install -e ".[behavior]"

# 一次安装所有后端
pip install -e ".[full]"
```

### 使用

```bash
# 推荐入口：零点击执行追踪、行为分析、模体发现和双语可视化报告
densetrack complete cage_video.mp4 \
  --config configs/4xPD_mac_adaptive.yaml \
  --n-animals 4 \
  --output results/cage_complete

# 启动本地 Web 工作台：默认使用科研长时身份配置
# 上传视频后可选零点击自动初始化，或首帧逐只点击后再处理
densetrack serve --workspace results/web_workspace --host 127.0.0.1 --port 8765

# 运行基础 AutoLabel MOT baseline（无需可选模型后端）
densetrack run cage_video.mp4 --n-animals 5 --species mouse

# 黑色小鼠 + 浅色垫料 + 固定俯视相机：推荐用固定数量追踪配置
densetrack run cage_video.mp4 --config configs/mice_black_topview_fixed.yaml --n-animals 4

# Mac 本地增强路径：零点击自动初始化 + 轻量外观模板 + 双向一致性融合
densetrack complete 4xPD.mp4 \
  --config configs/4xPD_mac_adaptive.yaml \
  --n-animals 4 \
  --output results/4xPD_mac_adaptive_complete

# 通用成熟开源 MOT 路径：复用 Norfair tracker（BSD-3）
pip install -e ".[norfair]"
densetrack run cage_video.mp4 \
  --config configs/mice_topview_norfair.yaml \
  --n-animals 5

# 使用 Foundation Model 检测后端（需先安装并配置权重）
densetrack run cage_video.mp4 --detector grounding_dino --n-animals 5

# 使用自定义配置
densetrack run cage_video.mp4 --config configs/mice_topview.yaml --n-animals 5

# 指定检测后端
densetrack run cage_video.mp4 --detector background_sub --n-animals 5

# 评估追踪结果（需要标注数据）
densetrack evaluate predictions.mot ground_truth.mot

# 从已完成 run 重新导出格式
densetrack export results/cage_001 --format sleap

# 基于轨迹做规则行为统计（无需额外标注）
densetrack analyze results/cage_001

# 无行为标注聚类 recurring motifs
densetrack discover-behaviors results/cage_001 --clusters 6

# 从已有追踪结果导出 BEAST3D 启发的无训练标注视觉 SSL 表征
densetrack extract-representations results/cage_001 \
  --dim 8 \
  --window-frames 15 \
  --sample-every 3

# 多视角实验可提供相机清单；当前会审计是否具备 3D SSL 预训练条件
densetrack extract-representations results/multiview_run \
  --camera-manifest configs/cameras/session_001.yaml

# 从未标注 WAV 中导出 Dolph2Vec 启发的物种特异声学单元
densetrack analyze-audio-ssl dolphin_recording.wav \
  --species dolphin \
  --codebook-size 12 \
  --whistle-min-frequency 1000

# 使用姿态特征和高级 UMAP + HDBSCAN 路径
densetrack discover-behaviors results/cage_001 \
  --pose-csv results/cage_001/poses.csv \
  --feature-mode pose \
  --embedding umap \
  --clustering hdbscan \
  --hdbscan-min-cluster-size 200 \
  --hdbscan-min-samples 2

# 使用 ILP 全局轨迹优化器
densetrack run cage_video.mp4 --config configs/mice_topview.yaml --optimizer ilp

# 对碎片化长序列使用闭集离散几何拼接
densetrack run cage_video.mp4 --config configs/mice_topview.yaml --optimizer long_term

# 为已有运行结果生成独立双语 HTML 研究面板
densetrack build-report results/cage_001 --auto-init-dir results/cage_auto_init

# 查看可用行为学实验模板
densetrack list-experiment-templates

# 从已有 tracks.mot 生成 Tracking Master 风格区域/时间箱指标
densetrack analyze-experiment results/cage_001 \
  --template open_field \
  --time-bin-s 60

# 完整流程中指定实验范式；输出 experiment_metrics.json/csv 并刷新双语面板
densetrack complete cage_video.mp4 \
  --config configs/4xPD_mac_adaptive.yaml \
  --n-animals 4 \
  --experiment-template three_chamber_social \
  --output results/cage_social

# 自动生成初始化点，替代第一帧人工点击
densetrack auto-init cage_video.mp4 --config configs/mice_black_topview_fixed.yaml --n-animals 4

# 生成 UDMT 外部衔接包：默认只扫描首帧，输出官方工作区兼容的 start_pos_array.txt
densetrack udmt-handoff cage_video.mp4 \
  --config configs/mice_black_topview_fixed.yaml \
  --n-animals 4 \
  --output results/cage_udmt_handoff

# 为科学验证准备 100 帧风险优先预填审核包
densetrack prepare-validation results/cage_complete --frames 100

# 在结果面板中打开 validation_review/review.html，拖动框修正后下载 reviewed_gt.mot
# 本地 Web 工作台可直接上传该文件，确认审核后生成稀疏 HOTA / AssA / MOTA / IDF1 报告

# 仅在人工修正并确认 GT 后计算稀疏验证指标
densetrack evaluate-validation results/cage_complete/validation_review --confirm-reviewed

# 检查外部 UDMT 研究后端（UDMT 不打包进本项目）
densetrack udmt-check --udmt-home /path/to/UDMT --udmt-python /path/to/udmt/bin/python --accept-license

# 导入 UDMT 派生的 MOT-like 结果并转成 DenseTrack 标准输出
densetrack udmt-import path/to/udmt_tracks.mot --output results/udmt_import

# 查看视频信息
densetrack info --video cage_video.mp4

# Phase 0 基准测试
python scripts/benchmark.py --video cage_video.mp4 --n-animals 5
```

### Python API

```python
from densetrack.config import PipelineConfig
from densetrack.pipeline import Pipeline

# 加载配置
config = PipelineConfig.from_yaml("configs/mice_topview.yaml")
config.video_path = "cage_video.mp4"
config.n_animals = 5

# 运行流水线
pipeline = Pipeline(config)
results = pipeline.run()

# 结果
trajectories = results["trajectories"]
report = results["confidence_report"]
```

## 架构

```
原始视频
  → 开放词汇/传统检测 (Grounding DINO / 背景减除 / 暗目标分割)
  → 视频级分割传播 (SAM 2)
  → 零样本姿态估计 (DLC SuperAnimal / 几何特征)
  → 轨迹片段生成 (ByteTrack / 固定数量追踪 / Mac 原生自适应双向追踪)
  → 自监督 ReID 学习
  → 分层离散几何与全视频身份平滑 (位置/尺度/极坐标速度)
  → 全局轨迹图优化 (贪心/匈牙利/ILP/长时闭集路径覆盖)
  → 置信度审计报告
  → 多格式导出 (MOT/COCO/CSV + 叠加视频)
  → 无监督行为模体发现 (质心/几何姿态，可选 UMAP + HDBSCAN)
  → 行为学实验范式指标 (区域停留/进入/潜伏期/时间箱)
  → BEAST3D 启发视觉 SSL 表征 (轨迹/姿态/身份风险/多视角就绪审计)
  → Dolph2Vec 启发音频 SSL 表征 (WAV 声学单元/哨声候选)
  → 双语可视化研究面板
```

## 项目结构

```
densetrack/
├── __init__.py          # 包初始化
├── models.py            # 核心数据模型（Detection, Tracklet, Trajectory...）
├── config.py            # Pydantic 配置模型
├── pipeline.py          # 流水线编排
├── cli.py               # Typer CLI 入口
├── detection/           # 检测模块
│   ├── base.py              # BaseDetector 抽象接口
│   ├── grounding_dino.py    # Grounding DINO 适配器
│   ├── sam2_propagator.py   # SAM 2 视频分割传播
│   ├── background_sub.py    # 背景减除检测（纯 OpenCV）
│   └── dark_object.py       # 黑色动物/浅色背景暗目标分割
├── pose/                # 姿态估计
│   ├── base.py              # BasePoseEstimator 抽象接口
│   ├── dlc_superanimal.py   # DLC SuperAnimal 适配器
│   └── geometric.py         # 几何特征提取（纯 OpenCV）
├── tracking/            # 追踪模块
│   ├── kalman_filter.py     # Kalman 滤波器
│   ├── matching.py          # IoU 匹配 + 匈牙利算法
│   ├── norfair_tracker.py    # 成熟开源 Norfair MOT 适配器
│   ├── tracklet_generator.py # 本地 ByteTrack-style fallback
│   ├── fixed_count_tracker.py # 已知动物数量的固定 ID 追踪
│   ├── mac_adaptive_tracker.py # Mac 原生零点击外观自适应双向追踪
│   └── global_optimizer.py  # 全局轨迹图优化
├── reid/                # 重识别模块
│   ├── features.py          # ResNet18 特征提取
│   └── self_supervised.py   # 自监督 ReID 训练
├── initialization/      # 自动初始化
│   └── auto_seed.py         # 自动生成个体 seed points
├── external/            # 外部研究后端适配
│   └── udmt.py              # UDMT 环境检查、自动 seed 衔接包与结果导入
├── behavior/            # 行为统计
│   ├── rule_based.py        # 活动比例、接触矩阵、追逐候选
│   └── unsupervised.py      # B-SOiD 风格无监督 motif 聚类
├── experiments/         # 实验范式模板与指标
│   └── templates.py         # Tracking Master 风格区域/时间箱指标导出
├── representation/      # 无标注视觉表征
│   └── visual_ssl.py        # BEAST3D 启发轨迹/姿态 SSL 嵌入
├── audio/               # 动物通信音频表征
│   └── bioacoustic_ssl.py   # Dolph2Vec 启发声学单元与哨声候选
├── reporting/           # 研究结果展示
│   └── static_dashboard.py  # 双语独立 HTML 面板 + 浏览器兼容预览
├── workflows/           # 端到端工作流
│   └── complete_local.py    # 零点击视频到报告一键流程
├── webapp.py             # 双语本地上传、异步处理与结果下载工作台
├── validation/          # 科学验证准备
│   └── review_package.py    # 风险优先稀疏 GT 预填审核包 + 审核确认评估
├── confidence/          # 置信度审计
│   ├── auditor.py           # 置信度评分
│   └── report.py            # 报告生成
├── export/              # 导出模块
│   ├── mot_format.py        # MOT Challenge 格式
│   ├── coco_format.py       # COCO 格式
│   ├── dlc_format.py        # DLC CSV 格式
│   └── overlay_video.py     # 可视化叠加视频
└── utils/               # 工具函数
    ├── video.py             # 视频 I/O
    ├── geometry.py          # 几何计算
    └── metrics.py           # MOT 评估指标
```

## 置信度系统

每条轨迹和每个连接点都有置信度评分：

| 等级 | 分数 | 含义 | 建议 |
|------|------|------|------|
| 🟢 高 | ≥ 0.8 | 可直接使用 | 自动接受 |
| 🟡 中 | 0.5-0.8 | 建议抽查验证 | 快速审查 |
| 🟠 低 | 0.2-0.5 | 需要人工审查 | 仔细检查 |
| 🔴 不确定 | < 0.2 | 身份可能错误 | 必须审查 |

## 许可证

- 核心代码：Apache-2.0
- Grounding DINO 权重：请查看模型卡
- SAM 2：Apache-2.0
- DLC SuperAnimal 权重：研究/非商用限制
- UDMT：外部研究后端；公开仓库许可证为 academic/non-profit non-commercial research use only，不能默认打包进商业发行
- BEAST3D / Dolph2Vec：当前只作为论文方法信号和工程设计参考；本项目没有复制其源码或权重，也不声称完整复现其神经网络训练流程

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码检查
ruff check densetrack/

# 类型检查
mypy densetrack/
```
