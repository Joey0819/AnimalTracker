# 行为汇总策略 / Behavior Summary Policy

## 证据边界

[Comparison of multiple video tracking-based behavioral summary approaches for compound discrimination](https://www.biorxiv.org/content/10.64898/2026.07.20.739643v1)
是 2026 年 bioRxiv 预印本。研究在 314 段自由活动小鼠录像、八个药物-剂量条件中，
比较参数聚合、Keypoint-MoSeq、VAME、SimBA 和 A-SOiD 五类汇总方法。作者报告：
在该受控实验的药物效应检测和剂量区分任务中，透明基础聚合与复杂机器学习汇总
没有显著性能差异。

这不是“复杂表征无效”的普遍结论，也不是多目标身份关联算法证据。

## DenseTrack 默认分层

### 第一层：透明参数聚合

受控处理效应、剂量区分和中小规模临床前研究默认解释：

- 每 ID 路程和平均速度
- 区域停留时间和进入次数
- 固定时间箱统计
- 追踪覆盖率
- 研究者绘制并校准的区域

这些结果写入 `experiment_metrics.json`、`experiment_zone_metrics.csv` 和
`experiment_time_bins.csv`。

### 第二层：复杂发现表征

以下问题再启用无监督模体、姿态增强聚类或 SSL：

- 开放式行为发现
- 社会交互结构
- 长期身份与遮挡恢复
- 大规模行为图谱
- 跨视角或跨物种表征迁移

复杂层用于提出和组织候选行为，不默认替代第一层的可解释统计。

## 身份质量边界

透明行为汇总仍依赖可靠轨迹。`identity_risk.json` 中的大跳变、双向分歧和预测比例
用于无标注审计；HOTA、AssA、IDF1 和 ID switch 必须基于研究者审核 GT。对于接触、
合并轮廓和远端重获，系统优先输出预测段并降低观测覆盖，而不是强制分配一个身份。
