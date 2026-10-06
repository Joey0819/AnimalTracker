# 长时身份连续性 / Long-Term Identity Continuity

DenseTrack 的 `research_long_term` 配置面向固定动物数量、长视频、相似外观、频繁交叉和遮挡后的身份恢复。目标是减少轨迹碎片和 ID switch，同时保留可审计的失败边界。

## 方法 / Method

### HieDG 启发的离散几何

每个候选关联分别编码：

- 归一化中心位置 `(cx/W, cy/H)`
- 归一化框尺度 `(w/W, h/H)`
- 极坐标速度 `(magnitude, direction)`

每个二维分量经过固定的两阶段残差量化：第一阶段选择粗网格，第二阶段量化粗网格内残差。默认每阶段每维 8 个 bin，因此每个二维阶段对应 64 个离散码。方向使用圆周距离，低速时忽略不稳定的方向角。

这保留了 [HieDG](https://arxiv.org/abs/2607.00494) 的几何分解和 coarse-to-fine 离散化原则，但不包含论文的 Deformable DETR、MLP、学习码本、视觉 dropout curriculum 或端到端 ID 分类器。DenseTrack 实现是无需训练的确定性适配，不是论文复现。

### WaspMOT 启发的长时拼接

`global_optimizer.method=long_term` 将局部 tracklet 视为有向无环图节点，并在已知动物数量约束下求解路径覆盖。连接分数结合：

- 多帧中位数端点速度，抑制单帧抖动
- 静止端点与弹道预测的双假设空间距离
- 分层离散几何相似度
- 外观、体型、朝向、姿态和时间间隔

该设计参考 [WaspMOT](https://arxiv.org/abs/2607.08729) 的闭集长序列设定与空间轨迹片段拼接结果。WaspMOT 官方仓库当前尚未发布论文中的完整拼接实现，因此 DenseTrack 没有声称复用或复现其代码。

### 遮挡恢复保护

`mac_adaptive` 路径同时使用正向和反向追踪，并遵循“宁可缺测，不制造身份”的约束：

- 接触区的一对一匹配 margin 过低时冻结身份，不更新外观模板
- 一个大轮廓同时覆盖多个预测中心时，将其视为合并目标并拒绝单 ID 分配
- 遮挡重获半径按缺失帧数的平方根增长，并设置硬距离上限
- Viterbi 状态允许拒绝一个或全部检测候选，不再强制把伪检分配给某个 ID
- 超物理速度的观测锚点降级为预测，同一 ID 的缺失段仅由可靠锚点插值或受限外推
- 帧间跳变使用非封顶二次惩罚，避免一次全局身份置换换取后续局部证据

全视频动态规划结合双向证据、离散几何和长期运动历史选择身份路径。候选状态数有上限，避免动物数量增大时出现阶乘级内存增长。

### 长视频执行 / Long-Sequence Execution

动态规划先缓存每个身份的历史稳健速度，并为相邻帧预计算小型“上一检测 → 当前检测”代价表，再用矩阵完成候选状态转移。该实现保持相同目标函数，避免为每对候选排列重复编码离散几何。

在本机完整处理 `4xPD.mp4`（9,363 帧、4 只动物、820×800、15 FPS）的
2026-07-28 实测结果为：

- 检测与姿态：33.7 秒
- 长时身份追踪：83.5 秒
- 全流程与可视化导出：142.9 秒，65.54 FPS

该耗时是单机工程基准，不是跨硬件性能承诺。

### 接触事件审计层 / Contact-Event Audit Layer

`contact_events.enabled=true`（默认开启）在追踪完成后输出 `contact_events.json` 和
`contact_events.csv`，把每一次接触/重叠片段变成可审计事件：

- 参与者 ID、帧范围、合并测量帧（至少一个参与者由预测承载，或被另一参与者框高 IoU 包含）
- 事件类型：`measurement_ambiguous`（存在合并测量）或 `proximity_only`（仅近距离，全程观测连续）
- 拆分边界的前 K 个“入 → 出”身份假设：用接触前稳健速度做恒速外推，加上体型对数比惩罚，
  代价经 softmax 转成相对分数；报告最优与次优的 margin 和 confident / ambiguous / uncertain / unresolved 等级
- `tracker_matches_best`：最优假设与追踪器实际续接是否一致

边界：假设概率是固定打分函数的无标注相对分数，未经过 GT 校准，不是身份准确率。
原始不一致事件会沿持续标签反转级联（同一对动物后续每次接触都会显示为交换最优），
因此审核单位是 `inversion_episodes`（时间间隔 ≤ `episode_gap_frames` 且参与者重叠的
不一致事件链），而不是原始事件数。稀疏 GT 审核包会用 `frame_review_priorities`
提高不确定接触帧的采样权重。

## 科研验证 / Scientific Validation

无标注输出 `identity_stability_score`、双向一致率、离散 token 一致率和 `identity_risk` 仅用于发现需要审核的帧，不能作为准确率。

`confidence_report.json` 中的高/中/低分只表示检测覆盖与轨迹连接置信度，不表示身份准确率。身份正确性必须由人工审核 GT 的指标报告确认。

人工审核 GT 后，`densetrack evaluate` 和审核工作流计算：

- HOTA、DetA、AssA、LocA
- MOTA、MOTP、IDF1
- ID switches、fragmentations、precision、recall

HOTA 家族遵循 [TrackEval](https://github.com/JonathonLuiten/TrackEval) 的 0.05–0.95 IoU 阈值、全局 alignment 和 association Jaccard 定义。稀疏审核指标只适用于被审核帧；正式论文结论应使用完整、多视频、预注册的测试集。

## 回归证据 / Regression Evidence

功能测试包含一个两只动物交叉、10 帧轨迹中断和单帧端点速度污染的闭集合成场景：

| 方法 | ID switches | AssA |
| --- | ---: | ---: |
| Legacy ILP endpoint motion | 2 | 33.3% |
| Long-term discrete geometry | 0 | 100.0% |

这是确定性的算法回归用例，不代表真实动物数据集上的泛化性能。真实效果必须通过人工审核 GT 报告。

同一 `4xPD.mp4` 的导出 MOT 连续性审计如下。它只检查物理跳变，不等同于人工 GT 上的 ID switch：

| 指标 | 2026-07-27 旧版 | 2026-07-28 身份保护版 |
| --- | ---: | ---: |
| 最大单帧位移 | 755.413 px | 134.575 px |
| `>180 px` 位移次数 | 1,242 | 0 |
| `>300 px` 位移次数 | 732 | 0 |
| `>500 px` 位移次数 | 231 | 0 |
| 无标注大跳变风险段 | 未输出 | 0 |
| 可靠观测数/帧 | 3.52/4 | 2.78/4 |

可靠观测下降是有意的保守取舍：可疑接触、合并和远端重获改为预测段，不再冒充真实观测。

## 主要配置 / Key Configuration

```yaml
tracklet:
  backend: mac_adaptive
  adaptive_discrete_geometry: true
  adaptive_geometry_coarse_bins: 8
  adaptive_geometry_fine_bins: 8
  adaptive_geometry_knn: 5
  adaptive_velocity_window: 7
  adaptive_reacquisition_min_margin: 0.035
  adaptive_reacquisition_distance_growth: 5.0
  adaptive_max_reacquisition_distance: 200.0
  adaptive_contact_distance: 75.0
  adaptive_contact_min_margin: 0.075
  adaptive_merged_area_ratio: 1.65
  adaptive_smoothing_geometry_weight: 0.50
  adaptive_smoothing_velocity_weight: 0.65
  adaptive_smoothing_jump_penalty: 8.0
  adaptive_smoothing_max_states: 128

global_optimizer:
  method: long_term
  closed_set_stitching: true
  long_term_geometry_weight: 0.25
  long_term_velocity_window: 7
```

每次完整运行会保存 `identity_continuity.json`、`identity_stitches.csv`、`identity_risk.json`、`identity_risk.csv`、`contact_events.json` 和 `contact_events.csv`，用于方法溯源、拼接检查、接触事件审计和风险优先审核。已有运行目录可用 `densetrack contact-events <run_dir>` 以不可变输入重算接触事件层。
