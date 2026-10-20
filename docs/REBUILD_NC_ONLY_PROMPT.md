# REBUILD 补充任务：彻底删除 CR 特征损坏机制

REBUILD 只研究 Noisy Correspondence（NC），不研究 CorreGen 的 Corruption Ratio（CR）特征损坏任务。必须彻底移除 CR 相关配置、数据生成、分支逻辑和实验路径命名，而不是保留 `c_ratio: 0` 或 `enable_cr: false`。

## 一、删除 CR 配置

在 `main_train.py` 及所有配置解析模块中：

- 删除 `c_ratio` 参数和默认值；
- 删除实验输出目录中的 `cr` 字段；
- 删除 CR 参数验证逻辑；
- 删除 YAML 中的 `c_ratio`；
- 用户传入 `c_ratio` 时必须明确报错，不能静默忽略。

## 二、删除 CR 数据生成逻辑

在 `dataset_loader.py` 中删除 `c_ratio` 分支、`np.random.normal()` 特征噪声、`noisy_data_X` 以及训练集/原始评估集之间的 CR 选择分支。同步修改 `load_mat()`、`load_dataset()` 返回接口和所有调用者，不留下悬空参数或失效解包。

正常的数据标准化和数据审计所需的原始样本信息必须保留。

## 三、严格保留 MR

MR 是 REBUILD 的核心实验变量，必须保留：

- `m_ratio`，必要时再单独进行语义命名规范化；
- YAML 指定的视图顺序、固定 Anchor/被扰乱视图和样本身份信息；
- MR 置换后的类别标记，仅用于评估和离线验证；
- 原始 MR 随机置换协议、随机种子和置换索引；
- MR=0、0.2、0.5、0.8 实验支持。

删除 CR 后，相同数据和 MR 配置必须产生与 CorreGen 在 `c_ratio=0` 条件下相同的 MR 置换结果。删除 CR 不得顺便改变 MR 的随机数或算法。

## 四、训练与评估数据

CR 删除后，训练和评估使用同一份经过 MR 置换和标准化的数据版本；两者仍可使用独立 DataLoader、Batch Size 和采样顺序。评估不能恢复为 MR=0 的完全对齐数据，也不能添加额外噪声。

## 五、必须新增或改写测试

至少验证：

1. `m_ratio=0` 时不存在样本身份扰乱。
2. `m_ratio>0` 时只扰乱指定视图。
3. Anchor 数据保持不变。
4. 特征损坏代码完全消失。
5. 删除 CR 前后等价条件下的 MR 置换结果一致。
6. 训练集与评估集的数据身份协议一致。
7. `c_ratio` 配置触发明确错误。
8. 实验输出目录不再包含 CR 字段。
9. ACC/NMI/ARI/F/CAR 接口保持正常。
10. DROT、GW 和伪标签生成不依赖 CR 参数。

## 六、Git 提交要求

本任务独立于 GMM、Sinkhorn 删除和 DROT 接入。建议至少拆分为：

- `refactor(data): remove corruption ratio augmentation`
- `test(data): verify mismatch-only protocol`
- `docs: document NC-only evaluation protocol`

每个提交前运行相关测试并记录 Commit SHA。

## 七、最终目标

REBUILD 的数据协议只保留一个主要扰乱维度：Mismatch Ratio（MR）。所有 YAML、训练入口和数据加载器中，不应再出现 CR 有效代码路径；也不应声称支持 CorreGen 原论文的 NC+CR 全部实验设置。
