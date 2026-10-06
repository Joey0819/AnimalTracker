# UDMT-Inspired Annotation-Free Transfer Tracking / UDMT 启发的无标注迁移追踪

## What Changed / 更新内容

AnimalTracker now includes a dedicated annotation-free transfer tracking mode:

AnimalTracker 现在新增了独立的无标注迁移追踪模式：

- UI mode: `无标注迁移追踪 / Annotation-free Transfer`
- Backend: `src/tracking/unsupervised_transfer_tracker.py`
- Factory mode: `unsupervised_transfer`
- AutoMOT accuracy-first routing: prefers `unsupervised_transfer`

## Method Mapping / 方法映射

This implementation is inspired by the Nature Methods 2026 UDMT paper, but does not vendor or copy UDMT source code.

该实现受 2026 年 Nature Methods UDMT 论文启发，但不复制、不内置 UDMT 源码。

| UDMT idea | AnimalTracker implementation |
| --- | --- |
| No training annotation | Uses first-frame detections as identity seeds and learns appearance prototypes from the video itself |
| Bidirectional closed-loop tracking | Runs end-of-video bidirectional identity repair through the existing consistency module |
| Spatiotemporal feature extraction | Uses lightweight color, texture, shape, and temporal-motion transfer features for local matching |
| Localization refining | Refines each detection around the local foreground centroid before matching |
| Bidirectional ID correction | Fuses forward/reverse identity evidence after processing |
| Automatic parameter tuning | AutoMOT routes accuracy-first jobs into this mode and tunes threshold/distance/buffer parameters |

## User-Facing Behavior / 用户可见行为

- Frontend/UI: the tracking mode selector exposes `无标注迁移追踪 / Annotation-free Transfer`.
- Backend: the tracker keeps fixed identity slots seeded from the first confident frame, then updates prototypes without manual labels.
- Results: the overview page shows bilingual metrics for identity slots, self-supervised updates, localization refinements, match rate, and bidirectional correction state.
- Exports/analytics: refined frame tracks flow into the existing behavior analytics, visualization, and export modules.

## Scope Boundary / 能力边界

This is a practical AnimalTracker algorithm update for annotation-free multi-object tracking. It is not a full reproduction of UDMT's ST-Net training stack or GUI.

这是 AnimalTracker 的无标注多目标追踪算法更新，不是完整复现 UDMT 的 ST-Net 训练栈或原始 GUI。

The UDMT repository license is restricted to academic or non-profit noncommercial research use, so AnimalTracker implements compatible concepts independently instead of bundling UDMT code.

UDMT 仓库许可证限制为学术或非营利非商业研究用途，因此 AnimalTracker 独立实现兼容思想，而不是打包 UDMT 代码。
