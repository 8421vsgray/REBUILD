# REBUILD 实验协议

本文档记录 REBUILD 当前阶段的评估协议。CorreGen 原始实现由 Git 标签
`corregen-baseline-20261020` 保存；本文档描述 REBUILD 中逐步显式化后的接口，不能反向解释为 CorreGen 原版行为。

## 数据与采样

- REBUILD 只研究 Noisy Correspondence（NC）；训练集和评估集使用同一批经过 MR 置换与标准化的直推式聚类样本。
- REBUILD 不支持 Corruption Ratio（CR）特征损坏，不添加训练专用高斯特征噪声，也不在评估阶段恢复 MR=0 数据。
- 训练和评估分别创建 `RandomSampler`，二者采样器独立。
- 两个 DataLoader 均使用 `drop_last=False`，因此最后一个不足整批的 batch 仍会被评估。
- 评估结果按数据集样本索引写回全局特征与标签数组，不能按 batch 位置覆盖。

## Batch Size

- `batch_size` 控制训练 DataLoader。
- `eval_batch_size` 控制评估 DataLoader。
- `eval_batch_size` 未在 YAML 或命令行显式设置时，默认取 `batch_size`。
- 正式 REBUILD 配置的训练和评估 batch size 上限均为 `512`；超过该上限的正式训练入口会直接拒绝。
- 正式数据集 YAML 显式设置 `batch_size: 512` 和 `eval_batch_size: 512`，避免把评估大小留给隐式默认值。
- 在 `512` 上限内，受控诊断可以显式设置不同的 `eval_batch_size`，并且必须记录在日志中。
- 两个批大小均限制在对应数据集长度以内，但不丢弃样本。
- 评估 DataLoader 不得在源码中硬编码为 512。

`B=1024` 仅保留为 DROT/投影器压力诊断规模，不作为正式训练、评估或结果报告的
batch size。模型配置中的隐藏层宽度 `1024` 与 batch size 无关，不应混淆。

## 评估与指标

当前基线评估保留 CorreGen 的特征提取、batch 内跨视图匹配、全数据集特征汇总、余弦 K-means 以及 ACC/NMI/ARI/F/CAR 指标定义。评估 batch 的候选集合会影响 batch 内匹配结果，因此改变 `eval_batch_size` 必须作为协议变化单独记录。

正式实验至少保存：

- 实际训练和评估 batch size；
- 数据集、视图顺序、MR、随机种子和实际输入维度；
- Git Commit SHA 与工作区状态；
- 评估采样方式、K-means 随机状态和各项原始指标。

## 当前阶段的边界

当前协议只保留 MR 作为数据扰乱变量。Sinkhorn、Partial OT 和 GMM 仍属于尚未完成的独立算法迁移任务，不能从本协议文字推断它们已经删除。
