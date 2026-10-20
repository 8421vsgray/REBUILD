这些应该直接从 REBUILD 中删除，而不是保留成一堆 `enable_gmm: false`、`use_sinkhorn: false` 的开关。

我核查了 CorreGen 官方源码，它确实包含三种训练状态：

| CorreGen 原有训练状态  | 机制                       | REBUILD 处理             |
| ---------------- | ------------------------ | ---------------------- |
| 前期 Rectification | 对角线监督                    | 保留，改名为 Diagonal Warmup |
| 中期对应估计           | Entropic OT / Partial OT | 删除，改为 GW + DROT        |
| 后期聚类引导           | GMM 密度估计与加权 OT           | 整体删除                   |

你的 REBUILD 应该只有两个阶段：

阶段 1：Diagonal Warmup

跨视图和视图内均采用对角线点级监督，不执行 GW，不调用 DROT。

阶段 2：GW Structural Supervision

视图内继续使用对角线监督；跨视图通过对角线初始化的 GW 扩散生成软伪标签，投影交由 DROT 执行。

在源码中，主要需要处理的位置也很明确：

- `model.py`：删除 `ot_torch` 引用和原始 `joint_dist_estimation()`，替换成 GW 伪标签求解。
- `engine_train.py`：删除 `calculate_dist()`、GMM 更新及 `InfoBuffer` 的使用。
- `eval.py`：删除 `update_infoBuffer()` 和 `GaussianMixture` 导入，但保留正式评估使用的 K-means。
- `main_train.py`：删除 `start_guide_epoch`、`p`、`m`、`beta`、`reg`、`rho` 等已经失去用途的配置，并移除 `InfoBuffer` 初始化。
- `ot_torch/`：确认没有其他引用后，可以整个删除并单独提交。
- `utils.py`：删除不再使用的 `InfoBuffer` 类。

有个地方特别注意：删 GMM 不等于删 K-means。 后者仍然负责聚类评估，ACC/NMI/ARI 还需要它。

## 直接追加给 Codex 的提示词

# REBUILD 追加任务：彻底移除 CorreGen 遗留算法

## 任务定位

在完成 CorreGen 原版封存和基线审计之后，对 REBUILD 执行一次明确的**算法迁移与遗留代码清理**。

这是实质性的算法改造，不是纯命名重构。必须独立记录每次修改，不能宣称与 CorreGen 完全行为等价。

REBUILD 不需要 CorreGen 的三阶段训练、不需要 Sinkhorn、不需要 Partial OT、不需要 GMM 聚类引导。

最终必须成为只有两个训练阶段的 STRIDE 实现：

### 第一阶段：Diagonal Warmup

当 `epoch < diagonal_epochs`：

- 视图内监督使用对角矩阵。
- 跨视图监督使用对角矩阵。
- 不调用 GW。
- 不调用 DROT。
- 不使用 GMM 或任何密度加权。

### 第二阶段：GW Structural Supervision

当 `epoch >= diagonal_epochs`：

- 视图内监督始终保持对角线。
- 跨视图监督使用 GW 生成的软伪标签。
- GW 耦合从 (I/B) 初始化。
- 有限步更新后调用 DROT 进行欧式投影。
- 伪标签采用 (P=BT) 转换为行随机监督矩阵。
- 不使用 Sinkhorn、Partial OT、GMM、样本密度加权。
- 两个跨视图方向分别计算和记录。

不允许保留任何隐藏的第三训练阶段。

## 一、移除旧 OT 路径

审查 `model.py`，删除原始 CorreGen 的：

- `from ot_torch import Partial_ot, entropic_wasserstein`
- `joint_dist_estimation()` 中的 Sinkhorn 与 Partial OT 分支。
- 不再使用的 `self.reg`、`self.rho`、`self.beta`。
- 原始 OT 的视图内非对角线软监督混合逻辑。
- 所有旧 OT 超参数及调用链。

用新的 `PseudoLabelSolver` 替代跨视图对应估计过程。

不要修改原始最大似然损失的数学定义。

先完成替代实现及测试，再删除旧文件。

确认项目没有残余依赖后，使用 `git rm` 删除 `ot_torch/`。

注意：DROT 自己的 `rho` 是迭代步长参数，与 CorreGen 的 Partial OT `rho` 含义不同。新的 DROT 配置应使用明确名称，例如 `drot_rho`，避免旧参数被错误复用。

## 二、彻底删除 GMM

审查并移除以下依赖链：

**`main_train.py`**

- 删除 `start_guide_epoch`。
- 删除 GMM 参数 `p`、`m`。
- 删除 `InfoBuffer` 初始化。
- 删除传入 `train_one_epoch()` 的旧密度缓存参数。

**`engine_train.py`**

- 删除 `calculate_dist()`。
- 删除根据 epoch 启用 GMM 的逻辑。
- 删除 `update_infoBuffer()` 调用。
- 删除对样本密度 `dist` 的引用。

**`eval.py`**

- 删除 `GaussianMixture` 导入。
- 删除 `update_infoBuffer()`。
- 删除 GMM 密度估计和更新代码。

**`utils.py`**

- 删除确认不再被使用的 `InfoBuffer` 类。

**`torch_clustering/`**

- 保留评估所需的 `PyTorchKMeans` 和相关依赖。
- 检查 `torch_clustering/__init__.py` 对 GaussianMixture 和其他旧模块的导入。
- 只有在确认没有调用关系之后，才删除对应 GMM 实现文件或导入。
- 不允许因删除 GMM 而破坏 K-means 评估。

## 三、统一成两阶段训练配置

新的 YAML 只需要一项训练阶段切换参数：

`diagonal_epochs: 100`

具体数值必须由 YAML 指定，不要硬编码，也不要自动继承 CorreGen 的 `start_rectify_epoch` 默认值。

训练状态只能是：

- `diagonal_warmup`
- `gw_supervision`

删除旧的 `rectify`、`start_guide_epoch` 等阶段控制逻辑。

不要引入 `stage3`、`clustering_guidance` 或相应的启用开关。

在配置校验中明确禁止已经废弃的配置字段，包括：

- `start_rectify_epoch`
- `start_guide_epoch`
- `beta`
- `reg`
- 原 CorreGen Partial OT 的 `rho`
- `p`
- `m`

若发现旧字段，应明确报错并指出迁移方式，不允许静默忽略。

## 四、保留真正需要的 CorreGen 组件

保留并验证：

- 在线编码器。
- EMA 目标编码器。
- 跨视图 Decoder。
- 最大似然损失。
- AdamW 和学习率调度器。
- 数据集加载与噪声对应生成。
- 跨视图推理匹配与特征融合。
- K-means 聚类评估。
- ACC、NMI、ARI、F、CAR 指标。

除非另有独立任务，不改变上述模块的数学行为。

同时修正评估 Batch Size 硬编码的问题：

`eval_batch_size` 默认等于 YAML 中的 `batch_size`，支持显式覆盖，但 REBUILD
正式训练和评估均不得超过 `512`。不能在源码中固定写成 `512` 来替代配置；
正式 YAML 应显式记录 `batch_size: 512` 与 `eval_batch_size: 512`。

## 五、删除后必须执行的验证

编写自动化测试，证明：

1. warmup 阶段不调用 DROT 或 GW。
2. GW 阶段实际调用投影接口。
3. 训练过程只有两个阶段。
4. 视图内监督始终为对角矩阵。
5. GW 耦合初始化为 (I/B)。
6. 生成的伪标签非负，行和约等于 1。
7. 两个跨视图监督方向均正确执行。
8. 原最大似然损失的计算保持一致。
9. 模型能完成前向传播与反向传播。
10. 训练流程不再依赖 `InfoBuffer` 或 GMM。
11. 正式评估仍能调用 K-means。
12. 评估样本完整覆盖，索引和标签对应正确。
13. YAML 的 Batch Size 确实控制 DataLoader。
14. 已废弃配置字段会被拒绝。
15. DROT 的参数不与旧 Partial OT 参数混淆。

使用静态搜索检查残余引用：

```
git grep -n -E 'start_guide_epoch|start_rectify_epoch|InfoBuffer|update_infoBuffer|calculate_dist|GaussianMixture|Partial_ot|entropic_wasserstein|joint_dist_estimation'
```

预期在活跃训练源码中不再出现上述旧调用。

历史文档、审计文档及基线说明中允许保留这些名称。

## 六、Git 提交顺序

不要一次性删除全部内容后才提交。

建议提交顺序：

1. `test: establish corregen baseline contracts`
2. `refactor: introduce two-stage training interface`
3. `feat: replace corregen ot with gw pseudolabel solver`
4. `refactor: remove gmm guidance and density buffers`
5. `refactor: remove unused sinkhorn and partial ot`
6. `refactor: remove obsolete configuration fields`
7. `test: verify two-stage stride training`
8. `docs: record migration from corregen to rebuild`

必要时拆分成更多 Commit。

每个提交必须是自洽的：不留下悬空导入、不破坏已有测试，不允许出现长期无法运行的半迁移状态。

如果某阶段无法通过测试，应报告阻塞问题，不得声称完成。

## 七、最终验收

必须能够明确证明：

- REBUILD 中没有 CorreGen 的 GMM 训练路径。
- 没有 Entropic OT 或 Partial OT 求解路径。
- 没有隐藏的第三阶段。
- 没有残余 GMM 密度缓存。
- 两阶段训练由 YAML 控制。
- DROT 是 GW 唯一启用的正式欧式投影后端。
- K-means 评估仍然可用。
- Git 历史能够追溯旧算法被何时、为何替代或删除。

请执行实际修改、测试和 Git 提交，并提供 Commit SHA 与验收结果。

**不要为了保留 CorreGen 的原貌而留下无用算法。原版已经由 Git 基线 Tag 保存，REBUILD 应该只保留 STRIDE 真正需要的功能。**

还有个工程判断：你现在不必追求“删除后还能复现 CorreGen 的 ACC”。

因为一旦把 CorreGen 的 Sinkhorn 和 GMM 替换成 GW，你已经改变了算法。旧版能通过 Git Tag 恢复即可，新版应该验证自己的数学正确性，而不是强行保持旧算法结果。

我更建议最终 REBUILD 的训练核心能简化到一个非常清楚的逻辑：

```
if epoch < diagonal_epochs:    labels = [identity, identity]else:    labels = gw_pseudolabel_solver(...)loss_intra = intra_view_loss(identity)loss_inter = cross_view_loss(labels)loss = loss_intra + loss_inter
```

如果最后训练主逻辑能简洁到这个程度，同时 DROT 和 GW 在外部独立验证过，那才是你想要的 REBUILD。
