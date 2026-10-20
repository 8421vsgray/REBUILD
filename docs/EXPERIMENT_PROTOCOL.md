# REBUILD 实验协议

本文档记录 REBUILD 当前阶段的评估协议。CorreGen 原始实现由 Git 标签
`corregen-baseline-20261020` 保存；本文档描述 REBUILD 中逐步显式化后的接口，不能反向解释为 CorreGen 原版行为。

## 数据与采样

- 训练集和评估集使用同一批直推式聚类样本；是否有特征污染由数据加载配置决定。
- 训练和评估分别创建 `RandomSampler`，二者采样器独立。
- 两个 DataLoader 均使用 `drop_last=False`，因此最后一个不足整批的 batch 仍会被评估。
- 评估结果按数据集样本索引写回全局特征与标签数组，不能按 batch 位置覆盖。

## Batch Size

- `batch_size` 控制训练 DataLoader。
- `eval_batch_size` 控制评估 DataLoader。
- `eval_batch_size` 未在 YAML 或命令行显式设置时，默认取 `batch_size`。
- 显式设置 `eval_batch_size` 时，允许与训练 batch 不同，并且必须记录在日志中。
- 两个批大小均限制在对应数据集长度以内，但不丢弃样本。
- 评估 DataLoader 不得在源码中硬编码为 512。

## 评估与指标

当前基线评估保留 CorreGen 的特征提取、batch 内跨视图匹配、全数据集特征汇总、余弦 K-means 以及 ACC/NMI/ARI/F/CAR 指标定义。评估 batch 的候选集合会影响 batch 内匹配结果，因此改变 `eval_batch_size` 必须作为协议变化单独记录。

正式实验至少保存：

- 实际训练和评估 batch size；
- 数据集、视图顺序、MR/CR、随机种子；
- Git Commit SHA 与工作区状态；
- 评估采样方式、K-means 随机状态和各项原始指标。

## 当前阶段的边界

本协议提交只修复评估 batch size 的显式配置和样本覆盖测试，不移除 CorreGen 的 Sinkhorn、Partial OT 或 GMM。后续算法迁移必须使用独立 Commit，并在完成数值测试后更新本文档。
