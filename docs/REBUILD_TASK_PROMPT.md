Exit code: 0
Wall time: 7.2 seconds
Output:
我核实了三个仓库的接口。这次 REBUILD 最重要的原则，是让 Codex 建立一个可验证、可回滚、可追溯的工程，而不是再生成一份自称“干净复现”的代码。

三个源码仓库分别是：

- [CorreGen 官方原版](https://github.com/XLearning-SCU/2026-ICLR-CorreGen?utm_source=chatgpt.com)：作为 REBUILD 的真正代码起点，保留其 Git 历史。
- [你的 STRIDE](https://github.com/8421vsgray/STRIDE?utm_source=chatgpt.com)：作为方法实现参考，不直接复制整个工程。
- [Regularized-DROT](https://github.com/WangZesen/Regularized-DROT?utm_source=chatgpt.com)：作为独立 GPU 投影求解器，放在 `solver/Regularized-DROT/`，固定其上游 Commit。

我还确认了两个必须写进任务书的细节：CorreGen 原版将评估 Batch Size 写死为 `512`；DROT 的二次正则系数会乘以矩阵两个维度之和，因此不能把 GW 投影候选矩阵直接作为代价输入。

下面这份提示词可以直接交给拥有终端、Git 和 GPU 环境的 Codex。它不仅要求完成 REBUILD，还规定了阶段验收、严格的数值对照、Git 提交纪律和多种子实验隔离。

# Codex 总任务：REBUILD — 从 CorreGen 可信重建 STRIDE

## 一、你的身份与最终目标

你是一名高级 PyTorch 科研工程师、GPU 数值计算工程师和科研代码审计员。

我要创建一个全新的科研项目，名称为 **REBUILD**。

REBUILD 的目标不是简单整理旧代码，而是从 CorreGen 官方原版出发，逐步重建 STRIDE 的数学机制，并集成 Regularized-DROT 提供的二次正则最优传输求解器，建立一套**每次修改均可追溯、每个关键模块可以独立测试、每次实验能够准确复现**的工程。

请实际执行任务，而不是只生成实施建议或项目骨架。

最终必须实现：

1. 从 CorreGen 官方源码创建新的 Git 项目。
2. 完整阅读 STRIDE 的实际实现，整理数学机制和模块接口。
3. 保留并验证 CorreGen 基线的行为。
4. 设计可替换的伪标签生成和欧式投影接口。
5. 将 Regularized-DROT 克隆到本项目的 `solver/` 目录。
6. 编译、运行并验证 DROT 的 PyTorch CUDA 接口。
7. 实现 DROT 二次正则 OT 与 STRIDE 欧式投影目标的严格数学转换。
8. 实现 STRIDE 的对角线初始化、有限步 GW 扩散和点级伪标签监督。
9. 修正评估 Batch Size 硬编码问题，统一 YAML 配置和评估协议。
10. 建立独立测试、数值基准、实验隔离与 Git 版本追溯体系。
11. 至少跑通小规模端到端训练及评估；具备条件时进一步完成正式实验。
12. 最终形成可以直接进行科研实验的 REBUILD 工程，而不是一个仅能通过空测试的演示项目。

**重要：正确性、可追溯性和可复现性优先于 ACC、代码美观与运行速度。**

---

## 二、三个源码仓库

### A. CorreGen 官方源码：工程起点

[https://github.com/XLearning-SCU/2026-ICLR-CorreGen](https://github.com/XLearning-SCU/2026-ICLR-CorreGen)

首先克隆 CorreGen 官方原版，将项目根目录命名为 `REBUILD`。

必须：

- 保留 CorreGen 的原始 Git 历史。
- 记录克隆时的完整 Commit SHA。
- 将原版标记为可追溯的基线 Tag。
- 创建独立的 REBUILD 开发分支。
- 保留 CorreGen 上游 Remote，但后续不得未经授权直接向官方仓库推送。
- 不允许通过删除 `.git` 重新初始化来伪造一份所谓的干净工程。
- 不允许覆盖已经存在的 REBUILD 或其他本地项目目录。

不要在克隆后立即大规模格式化、重命名或重构 CorreGen。

先理解，再修改。

### B. STRIDE：方法实现参考

[https://github.com/8421vsgray/STRIDE](https://github.com/8421vsgray/STRIDE)

完整阅读相关源码，尤其是：

- `model.py`
- `projection.py`
- `engine_train.py`
- `eval.py`
- `main_train.py`
- `dataset_loader.py`
- YAML 配置及其他被依赖的文件

阅读时要重点研究：

1. 在线编码器与 EMA 目标编码器。
2. 对角线预热阶段。
3. 跨视图预测表示的构造。
4. 基于视图内关系矩阵的 GW 梯度。
5. 初始耦合矩阵 (T_0=I/B)。
6. 有限步投影梯度更新。
7. 软耦合矩阵转换为点级监督目标。
8. 跨视图交叉熵监督。
9. 推理阶段跨视图匹配与融合。
10. 现有评估协议、CAR 定义和数据扰乱机制。

请生成 `docs/STRIDE_IMPLEMENTATION_AUDIT.md`，记录：

- STRIDE 的数学机制。
- 每个数学公式在旧源码中的对应位置。
- 需要移植的核心行为。
- 旧版本中存在但当前版本不一定包含的机制。
- 可能影响结果的工程行为。
- 必须独立验证、不能盲目相信的部分。

特别注意：**当前公开 STRIDE 源码并不必然等于全部历史实验所使用的源码。**

不得从当前版本推断历史结果一定采用过某项机制。

不要直接复制整个 STRIDE 工程，也不要未经验证就移植其现有 Newton 投影器。

### C. Regularized-DROT：独立 GPU 求解器

[https://github.com/WangZesen/Regularized-DROT](https://github.com/WangZesen/Regularized-DROT)

该仓库必须放在 REBUILD 中：

`solver/Regularized-DROT/`

建议使用 Git Submodule，将上游 Commit SHA 固定到父项目中，保持第三方源码的完整性和来源可追溯性。

可以在 `solver/` 下另外实现自己的 PyTorch 适配器。

不要将 DROT 当成一份可以任意修改的匿名代码。

检查其许可证、依赖、编译要求、测试说明以及 CUDA 兼容性。保留必要的版权与引用信息。

除非有充分理由，不修改第三方 CUDA Kernel。

---

## 三、Git 管理：全过程强制执行

这是本项目的最高优先级工程规则之一。

### 3.1 每完成一个独立修改，立即提交 Commit

采用语义化提交信息：

- `chore: initialize rebuild from corregen`
- `docs: audit stride implementation`
- `test: add baseline evaluation checks`
- `refactor: introduce pseudolabel solver interface`
- `chore: pin regularized drot dependency`
- `feat: add drot euclidean projection adapter`
- `test: verify euclidean projection optimality`
- `feat: implement gw pseudolabel diffusion`
- `fix: make evaluation batch size configurable`
- `test: verify cross-view label alignment`

Commit 不要求数量少，但要求含义明确。

一次 Commit 尽量只解决一个问题。

不要将数据协议、GW 算法、损失函数、评估实现混合修改。

### 3.2 每个阶段必须建立验收记录

每个阶段完成后：

1. 运行相关测试。
2. 检查 `git diff --check`。
3. 检查 `git status`。
4. 提交该阶段代码。
5. 记录 Commit SHA。
6. 更新 `docs/REBUILD_CHANGELOG.md`。
7. 记录当前有哪些测试通过、哪些失败、哪些尚未执行。

禁止未经测试就宣布某个核心模块已经正确。

### 3.3 Git 不负责记录大规模训练产物

所有大规模日志、checkpoint、缓存、临时文件和数据集必须与源码分离。

每次实验记录：

- 完整 Git Commit SHA。
- 是否有未提交的源码改动。
- 实际生效的 YAML 配置。
- 数据文件 SHA256。
- 随机种子。
- Python、PyTorch、CUDA、GPU 型号。
- DROT 固定版本。
- 训练启动时间。
- 评估协议版本。
- 模型权重文件路径及哈希。
- 每项评估指标。

扫多个随机种子时，不允许自动修改训练源码或配置文件。

每个种子的输出目录必须隔离，并禁止静默覆盖已有 checkpoint。

---

## 四、先让 CorreGen 原版成为可验证基线

在引入 STRIDE 或 DROT 之前，先检查 CorreGen 原版的运行环境。

重点验证：

1. 原版数据加载能正确工作。
2. 训练与评估的样本身份对应正确。
3. 模型能运行完整的前向和反向传播。
4. 原版评估逻辑可以独立调用。
5. 相同数据、相同 checkpoint、相同评估协议能够得到稳定的评估结果。

如果具备所需数据与 GPU，执行最小规模的训练冒烟测试。

将结果记录为基线，不要求复现论文全部数字后才能继续开发，但必须如实标记验证程度。

不要以 STRIDE 的结果是否优于 CorreGen 作为此阶段正确性的标准。

---

## 五、修复评估协议：Batch Size 必须从 YAML 读取

CorreGen 原版 `main_train.py` 的评估 DataLoader 存在固定 `batch_size=512` 的设计。

REBUILD 不允许继续写死评估 Batch Size。

要求：

- 默认 `eval_batch_size = batch_size`。
- 当 YAML 明确设置 `eval_batch_size` 时，允许独立指定。
- 所有配置优先级必须明确。
- 实际使用的训练与评估 Batch Size 都必须记录在日志和配置快照中。
- 不允许在某个文件中偷偷覆盖 YAML。
- 不允许因 Batch Size 改变而意外丢失评估样本。
- 训练和评估的 DataLoader 必须明确区分。

同时验证：

- 评估全样本覆盖。
- 不重复、不漏评估。
- 特征和真实标签的索引严格一致。
- Batch 内跨视图匹配索引正确。
- CAR 的定义与计算符合设计。
- ACC、NMI、ARI、F 的计算正确。
- K-means 随机状态可控制。
- 评估随机数不应在无意间改变后续训练的数据排列。
- 不同评估 Batch Size 下可能产生的性能差异必须真实记录。

**注意：使用相同 YAML 的 Batch Size，并不自动意味着不同方法公平。**

必须同时核查 CorreGen、STRIDE 及其他基线的视图选择、噪声生成、样本覆盖、特征融合和聚类协议。

不要假设结果应该保持不变，也不要用测试集 ACC 来选择最有利的评估 Batch Size。

---

## 六、设计可替换的伪标签求解架构

先设计接口，再接入 DROT。

建议将数学过程拆成三个独立层次。

### A. Relation Builder

根据两个视图的特征生成结构关系矩阵。

例如：

[\
D_1=1-\hat Z_1\hat Z_1^\top\
]

[\
D_2=1-\hat Z_2\hat Z_2^\top.\
]

具体归一化、对称化和数值截断行为应参考 STRIDE 源码，并且由测试确认。

### B. Transport Projector

统一欧式投影接口：

`project(X, marginal, ...) -> (T, diagnostics)`

其中：

- 输入 `X` 是欧式投影的候选矩阵，而不是 OT 代价矩阵。
- `marginal` 表示每行、每列的目标边缘质量。
- 返回的 `T` 必须满足非负性和边缘约束。
- 返回值必须包含真实可核查的诊断指标。
- 必须能够明确表示成功、失败、迭代耗尽及数值异常。

此接口未来可以接入：

- `drot`
- `newton`
- `cvxpy_reference`（小规模测试专用）

不要让训练模块直接依赖 DROT CUDA 的具体参数与内部实现。

### C. Pseudo-label Solver

根据关系矩阵、初始化方式、步长、GW 迭代步数和投影后端生成伪标签。

建议接口：

`generate(D1, D2, config) -> (P, diagnostics)`

确保：

- 可以独立运行。
- 不依赖编码器内部状态。
- 不负责评估。
- 不直接计算交叉熵。
- 可在小型人工矩阵上做单元测试。

---

## 七、DROT 与欧式投影的数学统一

这是本任务最重要的数值部分。

不能只让 CUDA 函数返回一个矩阵就认定接入成功。

### 7.1 STRIDE 的欧式投影问题

STRIDE 需要求解：

\underset{T\in\mathcal U}{\arg\min}\
\frac12|T-X|\_F^2\
]

其中

[\
\mathcal U=\
\left{\
T\ge0:\
T\mathbf1=\frac1B\mathbf1,\
T^\top\mathbf1=\frac1B\mathbf1\
\right}.\
]

展开目标函数：

\frac12|T|\_F^2\
-\langle X,T\rangle\
+\frac12|X|\_F^2.\
]

最后一项与 (T) 无关。

因此这与二次正则 OT 等价。

### 7.2 Regularized-DROT 的优化目标

官方仓库定义：

[\
\min\_{T\in\mathcal U}\
\langle C,T\rangle+\
\frac{(m+n)\alpha}{2}|T|\_F^2.\
]

其 PyTorch CUDA 接口为：

`reg_drot.quadratic_drot(cost, p, q, rho, r_weight, max_iter, eps)`

对于 (m=n=B)，设置：

[\
p=q=\frac1B\mathbf1.\
]

为了遵循 DROT 对非负归一化 Cost 的要求，对任意非常数输入矩阵 (X)，设置：

[\
s=\max(X)-\min(X)>0\
]

[\
C=\frac{\max(X)-X}{s}\
]

[\
\boxed{\
\texttt{r_weight}=\frac{1}{2Bs}\
}\
]

由此可得：

[\
\langle C,T\rangle+\
B\cdot\texttt{r_weight}|T|\_F^2\
]

与原始欧式投影目标仅相差与 (T) 无关的常数及正比例缩放，因此理论最优解相同。

请自行结合 DROT 源码和论文核对以上系数，不要只照抄公式。

`rho` 是 DROT 的迭代参数，不是 GW 步长，也不是 `r_weight`，不允许混淆。

### 7.3 常数矩阵与数值退化情况

当 (X) 所有元素完全相等时，欧式投影是均匀耦合：

[\
T\_{ij}=\frac{1}{B^2}.\
]

此时避免除以零，直接使用解析结果。

对于几乎常数、存在极端值、数值范围很小或很大的矩阵：

- 必须测试 `float32` 精度稳定性。
- 不允许未经验证就将近似常数矩阵随意映射为均匀耦合。
- 不允许静默裁剪或重缩放导致优化目标发生改变。
- 检查非有限输入及非法边缘值。
- 对超出支持范围的情况，明确报错或标记失败。

---

## 八、先独立验证 DROT，而不是直接投入训练

必须先完成 DROT 自身安装、运行、数值验证和速度测试。

### 8.1 原生运行验证

检查 CUDA Toolkit、编译器、PyTorch 与 GPU 兼容性。

尝试构建官方 PyTorch CUDA 扩展，并运行官方 `pytorch/test.py`。

注意：

- 官方实现只支持 CUDA `float32`。
- 官方文档说明的环境是其已测试环境，不意味着其他版本一定兼容。
- 不得通过未经解释的大范围升级或降级破坏现有训练环境。
- 如编译失败，保存完整日志并定位原因。
- 原生测试失败时不得宣称 DROT 已成功接入。

### 8.2 独立欧式投影验证

建立 `tests/test_drot_projection.py`。

至少测试：

- 2×2、8×8、16×16、32×32 随机矩阵。
- 对角线附近的输入。
- 已经满足边缘约束的矩阵。
- 常数矩阵。
- 大量负值的矩阵。
- 稀疏有效集输入。
- 强烈非均匀的矩阵。
- 大动态范围矩阵。
- B=512、1024 的真实规模矩阵。

必须检查：

[\
\min T\_{ij}\ge-\varepsilon\_{\mathrm{num}}\
]

[\
\max_i\left|\
\sum_j T\_{ij}-\frac1B\
\right|\
]

[\
\max_j\left|\
\sum_i T\_{ij}-\frac1B\
\right|.\
]

还必须验证目标函数：

[\
f(T)=\frac12|T-X|\_F^2.\
]

### 8.3 使用 CVXPY 建立独立数值参考

对小规模矩阵，用 CVXPY 的独立凸优化求解器计算同一个欧式投影问题。

比较：

1. DROT 与参考解的矩阵差异。
2. 原始目标函数差异。
3. 边缘约束残差。
4. 非负性。
5. KKT 驻点与互补条件。
6. 是否出现求解器返回结果但实际不满足约束。

不能将当前 STRIDE 的自研 Newton 直接当作唯一真值。

若可用，可以额外和 Newton 交叉比较，但最终必须有独立数值参考。

对于 DROT 的绝对残差 `eps` 与 STRIDE 的相对残差要求，必须建立清晰的转换或统一的验收指标。

不要直接把同一个数值传给两种不同含义的容差参数。

### 8.4 DROT 失败机制

投影适配器必须：

- 返回明确的状态信息。
- 重新计算实际输出的可行性残差。
- 避免将未收敛结果静默标记为成功。
- 不允许将 NaN 或 Inf 耦合传给下游损失。
- 不允许默认用 Newton 替换 DROT 结果，却在日志里仍声称使用 DROT。
- 可以保留上一可行耦合作为显式失败回退，但必须记录。

DROT 当前接口没有提供耦合或对偶变量 Warm Start，不要虚构此功能。

### 8.5 GPU 性能测试

建立 `benchmarks/benchmark_projection.py`。

分别测试：

- B=128
- B=256
- B=512
- B=1024

测试矩阵包括：

- 普通随机矩阵。
- 接近对角线的矩阵。
- 高稀疏性耦合对应的候选矩阵。
- 从 STRIDE 实际 GW 更新中采样的候选矩阵。

必须进行 CUDA 预热，计时前后进行必要的 CUDA 同步。

记录：

- 平均耗时。
- 中位耗时。
- P95 耗时。
- 实际迭代次数（如可获取）。
- 可行性残差。
- 目标函数误差。
- 失败率。
- GPU 硬件与环境。

必须在**相同精度条件**下比较不同求解器。

速度快但输出不满足投影约束，不算性能胜利。

---

## 九、实现 STRIDE 的有限步 GW 伪标签

在 DROT 单独验证通过后，才开始真正接入 GW。

按照 STRIDE 的数学机制：

[\
T_0=\frac1B I.\
]

对于对称关系矩阵，其当前实现中的结构梯度形式为：

[\
G(T)=\
-2\left(\
D_1TD_2^\top+\
D_1^\top TD_2\
\right).\
]

候选更新：

[\
X_k=T_k-\eta G(T_k).\
]

随后：

[\
T\_{k+1}=\Pi\_{\mathcal U}(X_k).\
]

请确认该梯度对应的具体 GW 目标和数学假设，并通过有限差分或自动微分参考实现验证梯度。

不能只看形状与 Loss 是否下降。

要求：

- 保留对角线初始化。
- 实现可配置的有限 GW 步数。
- 使用独立的投影接口。
- 检查每一步耦合的可行性。
- 保证软标签不参与反向传播。
- 将满足边缘质量 (1/B) 的耦合转成行随机伪标签：

[\
P=BT.\
]

- 检查 (P) 的每行和是否约等于 1。
- 明确区分两个跨视图监督方向。
- 分别记录两个方向的投影状态，避免统计被第二次调用覆盖。
- 检查 GW 步长、投影迭代次数、数值精度等配置能够完整追溯。

当前任务先保留 STRIDE 的主要数学逻辑，不主动增加新损失、复杂门控或额外创新。

历史上的先验锁定机制应当独立记录，在明确其对应代码版本后，再作为可开关的独立模块迁移，不要默认混入基础实现。

---

## 十、损失函数和训练流程

原则：优先沿用 CorreGen 经核实的训练行为，只有在实现 STRIDE 所必需时才修改。

特别检查 CorreGen 的最大似然损失采用的 Softmax 归一化方式。

不要擅自把全矩阵 Softmax 改为逐行 Softmax，或者把原始目标直接换成标准 `F.cross_entropy()`。

如果要修改：

1. 给出数学理由。
2. 新增独立测试。
3. 单独 Commit。
4. 标记为算法变更。
5. 不能和 DROT 集成放在同一次修改中。

STRIDE 的核心训练应包含：

- 对角线预热。
- 在线编码器和 EMA 目标编码器。
- 跨视图预测表示。
- GW 生成的软监督标签。
- 点级跨视图监督。
- 合理的失败回退和诊断日志。

需要检查训练中所有样本是否按照预期参与优化。

---

## 十一、建议的工程结构

请结合实际源码合理调整，但保留明确的模块边界。

建议结构：

- `REBUILD/`
  - `config/`：全部 YAML 配置。
  - `solver/`
    - `Regularized-DROT/`：固定 Commit 的官方第三方仓库。
    - `drot_adapter.py`：DROT 与欧式投影目标的适配层。
    - `projection_interface.py`：投影器统一接口。
    - `reference_projection.py`：小规模参考求解器。
  - `pseudolabel/`
    - `relations.py`
    - `gw.py`
    - `generator.py`
  - `tests/`
    - `test_data_protocol.py`
    - `test_drot_projection.py`
    - `test_gw_gradient.py`
    - `test_pseudolabel.py`
    - `test_eval.py`
    - `test_training_smoke.py`
  - `benchmarks/`
    - `benchmark_projection.py`
  - `docs/`
    - `STRIDE_IMPLEMENTATION_AUDIT.md`
    - `REBUILD_CHANGELOG.md`
    - `SOLVER_VALIDATION.md`
    - `EXPERIMENT_PROTOCOL.md`
    - `REPRODUCIBILITY.md`
  - 现有 CorreGen 训练、模型、数据和评估文件，按最小必要范围逐步修改。

不要为了迎合目录结构而盲目搬运 CorreGen 全部模块。

不要引入没有实际必要的抽象层。

---

## 十二、实验隔离与随机种子扫描

REBUILD 必须支持多种子实验，但每个实验都要具有独立身份。

建立安全的运行目录策略：

`runs/<commit>/<dataset>/<noise_ratio>/<seed>/<run_id>/`

每个实验拥有独立的：

- 配置快照。
- 环境信息。
- 训练日志。
- checkpoint。
- 评估结果。
- 运行状态记录。

规则：

- 已存在实验不得静默覆盖。
- 不能让多个种子写入相同的 `checkpoint_200.pt`。
- 不能让运行中的配置变化影响其他种子。
- 不允许为了扫种子自动编辑源码文件。
- 同一组实验应当固定同一个 Commit。
- 如果工作区有未提交的源码修改，默认拒绝正式实验，除非显式以调试模式运行并完整记录差异。
- 不能只记录均值，应当保留各个种子的原始结果及标准差。

必要时编写独立实验启动器，但其职责只限于配置解析、启动、隔离和记录，不负责动态修改算法。

---

## 十三、分阶段推进与提交要求

严格按照以下顺序执行，不允许为了加快进度而跳过验收。

**阶段 0：初始化 REBUILD**

克隆 CorreGen，记录原始 SHA，建立开发分支和基线 Tag，提交工程来源说明。

**阶段 1：审计与基线**

阅读 STRIDE，完成方法审计；验证 CorreGen 原始行为，建立基线测试。

**阶段 2：评估和配置**

消除评估 Batch Size 硬编码，建立统一 YAML 协议和样本索引测试。

**阶段 3：伪标签求解接口**

建立 Relation、Transport Projector 和 Pseudo-label Generator 的独立接口，不引入新求解器行为。

**阶段 4：DROT 集成**

以 Git Submodule 方式引入 DROT，固定版本，编译官方扩展并通过官方测试。

**阶段 5：数值正确性验证**

实现严格目标转换、CVXPY 对照、KKT 检查和数值精度报告。

此阶段未通过之前，不允许把 DROT 作为正式训练后端。

**阶段 6：GW 集成**

实现对角线初始化、有限步结构更新、投影、软标签归一化及独立单元测试。

**阶段 7：STRIDE 训练**

接入完整训练链路，验证前向、反向、EMA、监督方向和日志。

**阶段 8：正式评估与可复现性**

验证推理过程、跨视图匹配、K-means 及完整实验追溯。

**阶段 9：性能测试与验收**

对 DROT 进行 GPU Benchmark，并在具备数据和算力的条件下运行指定实验。

每个阶段完成后至少提交一次 Git Commit。

如果某个阶段失败，应保存真实失败证据，先定位原因，而不是修改无关模块绕开测试。

---

## 十四、完成标准

只有满足以下条件，才能把 REBUILD 标记为完成：

- CorreGen 原版来源与 Commit 可追溯。
- STRIDE 实现细节已经形成审计文档。
- DROT 源码作为固定版本的第三方依赖被管理。
- PyTorch CUDA 扩展在目标环境中成功构建并通过官方基础测试。
- DROT 可以求解经过数学转换的 STRIDE 欧式投影问题。
- 小规模投影结果通过独立参考求解器验证。
- 大规模投影达到明确记录的精度标准。
- GW 梯度和伪标签归一化通过测试。
- 对角线初始化有效。
- 损失函数与训练实现符合已记录的数学定义。
- 评估 Batch Size 不再被硬编码为 512。
- YAML 配置与实际生效参数一致。
- 多种子运行不会覆盖彼此产物。
- 实验可追溯至 Commit、数据、配置和环境。
- 有完整的数值精度报告与 GPU 性能报告。
- 至少一个实际数据集的训练、checkpoint 保存和评估链路完整跑通。
- Git 中有真实、分阶段的开发和测试记录。

不要用“代码已经生成”“应该可以运行”“理论上支持 GPU”代替实际验证。

没有 GPU、数据集、CUDA 编译器或其他必要资源时，明确标注尚未完成的验证，不得伪造测试通过信息。

不以超过某个 ACC 作为正确性验收条件；科学性能结论必须由正式受控实验得出。

---

## 十五、最终执行要求

请直接开始 REBUILD 的实际建设，而不是只输出一个计划。

执行时遵守：

1. 先检查目标路径和现有 Git 状态。
2. 克隆 CorreGen 官方仓库。
3. 保存基线 Commit。
4. 阅读 STRIDE 真实源码并记录发现。
5. 建立统一的伪标签与投影接口。
6. 克隆并固定 Regularized-DROT。
7. 构建、运行、验证 DROT。
8. 完成数学转换与数值测试。
9. 逐步接入 GW、训练、评估。
10. 在每个独立阶段进行 Git Commit。
11. 遇到问题先定位、先验证，不得大范围无依据改写。
12. 不破坏原有 STRIDE 仓库，也不覆盖用户已有实验文件。

每完成一个阶段，报告：

- 已完成的实际工作。
- 修改文件列表。
- Commit SHA。
- 测试结果。
- 尚未验证的风险。
- 下个阶段的具体工作。

**最终目标是让 REBUILD 成为一个每项实验结论都能找到代码依据、数值证据和运行记录的科研工程。**

如果必须在“更高 ACC”和“更高可信度”之间优先选择，现阶段优先可信度。

开始执行。

我特意把整个项目的核心验收集中到了阶段 5：DROT 数值正确性。因为这是最有可能出现“CUDA 正常运行、矩阵看起来合理，但求解目标实际上没有统一”的地方。

还有两个细节值得坚持：

第一，DROT 最好使用 Git Submodule，而不是把几百个第三方文件混进自己的源码历史。 这样既满足你要在 `solver/` 目录中保留源码的要求，也能明确知道它来自哪个上游版本。

第二，REBUILD 的第一次重要里程碑，不应该是某个数据集超过 CorreGen，而应该是：任意给定一个 GW 候选矩阵 \\(X\\)，DROT 能够产生经过独立验证的正确欧式投影。

这一点如果建立起来，你以后就不必一直纠结自研 Newton 是否写对了。
