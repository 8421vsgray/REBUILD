# Codex 追加任务：REBUILD 命名体系重构

## 一、任务目标

在 REBUILD 工程中，对从 CorreGen 继承的代码进行系统性的命名规范化。

本任务属于 **Naming Refactor（纯命名重构）**，不属于算法开发。

目标是建立清晰、统一、可维护的 REBUILD 命名体系，去除 CorreGen 原有代码中含糊、不一致或过度依赖论文作者个人习惯的命名。

本次任务必须全程使用 Git Commit，确保每项重命名能够追溯和回滚。

严禁在命名重构期间顺便修改训练算法、超参数、损失函数、数据协议、评估逻辑或任何数学计算。

## 二、执行时机与 Git 约束

在 CorreGen 原版基线已经封存、相关基础测试已经建立后执行。

开始前必须：

1. 检查当前 Git 分支和 Commit。
2. 检查 `git status`，确保工作区干净。
3. 确认 CorreGen 原始 Commit 和基线 Tag 已保存。
4. 记录当前命名体系与入口文件。
5. 为此次命名重构创建独立分支或明确的开发阶段。

如果工作区存在未提交的用户修改，不得擅自覆盖、清理或合并。

命名重构必须使用独立 Commit，不允许与删除无用功能、DROT 接入或 STRIDE 算法迁移混合提交。

## 三、统一命名规范

全部新增和重构代码遵循 Python PEP 8 规范。

| 对象 | 规范 |
| --- | --- |
| Python 文件和目录 | `snake_case` |
| 普通函数 | `snake_case` |
| 局部变量 | `snake_case` |
| 类名称 | `PascalCase` |
| 常量 | `UPPER_SNAKE_CASE` |
| 私有方法 | `_snake_case` |
| YAML 配置字段 | `snake_case` |
| 测试文件 | `test_<module>.py` |

名称必须表达实际语义，而不是为了缩短代码而使用模糊缩写。

重点审查以下候选名称，但不允许机械替换：

- `OursModel`：当前基线应改为 `CorreGenBaselineModel`；只有真正完成 STRIDE 机制迁移后才使用 `StrideModel`。
- `cl`：如果表示交叉熵或最大似然目标，应改成表达具体目标的名称。
- `param`：根据实际含义区分 `solver_config`、`training_config` 和模型超参数。
- `tag`：如果包含训练阶段状态，应改为 `training_phase` 等明确名称。
- `train_time`：如果表示独立训练次数，应改为 `num_runs`。
- `indexs`：改为 `indices` 或更准确的语义名称。
- `z`、`p`、`z_t`：在模块边界和公共接口优先使用有语义的名称；内部数学变量可以在公式明确时保留。

对于 `args`、`x`、`y`、`i` 等在简单局部范围内含义明确的变量，不必为了统一而强制改名。

数学变量名称必须与真实含义一致，不能混淆 transport plan、soft label、cross-view prediction 和 embedding。

## 四、建立命名映射文档

在任何重命名实施之前创建 `docs/NAMING_CONVENTIONS.md`，至少包括：

1. REBUILD 的统一命名规则。
2. CorreGen 原始名称。
3. REBUILD 新名称。
4. 重命名原因。
5. 涉及的文件。
6. 是否影响外部接口。
7. 是否影响 YAML 或 checkpoint 兼容性。

建议使用如下映射表：

| 原始名称 | 建议名称 | 说明 |
| --- | --- | --- |
| `OursModel` | `CorreGenBaselineModel` | 明确当前仍为基线模型 |
| `train_time` | `num_runs` | 表示独立实验次数 |
| `indexs` | `indices` | 修正不规范拼写 |
| `tag` | `training_phase` | 仅在语义一致时采用 |
| `param` | `solver_config` | 仅在确实表示求解配置时采用 |

必须先完成命名映射分析，再实施重构。

## 五、分批修改，不允许全局盲替换

### 阶段 A：类与函数命名

修改含糊的类名和函数名，同步更新所有导入、构造调用、测试和文档，保持原始算法行为。

完成后运行测试并提交：

`refactor(naming): clarify model and function names`

### 阶段 B：内部变量命名

整理含糊的局部变量、参数名称与返回值名称。对于位置参数、关键字参数和被外部引用的字段，必须检查调用兼容性；不得仅凭文本匹配批量替换。

完成后运行测试并提交：

`refactor(naming): clarify training and solver variables`

### 阶段 C：文件和模块命名

审查 Python 模块、脚本和目录名称。需要重命名的文件使用 `git mv`，保持 Git 历史可追踪。

检查 Python import、相对路径、训练启动命令、YAML 路径、测试发现机制、shell 脚本和文档示例。不要为了整洁而一次性重新分层全部文件。

完成后运行测试并提交：

`refactor(naming): standardize module names`

### 阶段 D：配置命名

统一 YAML 中语义不清晰或命名不一致的字段。YAML 是外部接口，不是普通局部变量。

要求：

- 更新所有读取与验证逻辑。
- 为配置字段创建明确的 Schema 或配置校验。
- 避免旧字段静默失效。
- 对需要兼容的旧字段显式声明映射并检测冲突。
- 不得借重命名改变默认参数数值。

完成后运行测试并提交：

`refactor(config): standardize configuration names`

### 阶段 E：日志和输出命名

统一日志字段、统计名称和输出文件命名，明确区分训练损失、GW 更新次数、投影成功率、投影失败率、耦合非对角质量以及 ACC/NMI/ARI/F/CAR。

不能只改日志名称而改变统计定义。

完成后提交：

`refactor(logging): standardize experiment diagnostics`

## 六、严格保持行为等价

每一批命名重构必须通过回归测试，至少检查：

1. 相同输入下模型前向输出一致。
2. 相同权重和输入下损失数值一致。
3. 固定随机状态下梯度一致。
4. 训练参数和优化器配置一致。
5. 数据顺序与样本索引一致。
6. 评估结果一致。
7. YAML 解析后的实际配置一致。

GPU 数值运算使用适当数值容差，并记录容差选择理由。比较时必须使用相同权重、输入和随机状态，避免把随机性误判为重构错误。

如果重命名导致 checkpoint 的 `state_dict` 键名变化，必须提供兼容性迁移方案或者保持权重键名不变。

禁止为了让重构测试通过而修改算法。发现已有 Bug 时先记录，不在本任务中混合修复。

## 七、不要修改第三方原始源码

`solver/Regularized-DROT/` 是独立第三方依赖，不得对其执行 REBUILD 统一重命名，不得为了符合本项目习惯修改第三方 CUDA 函数、文件、类名或参数名。需要命名适配时，放在 REBUILD 自己的 Adapter 中。

CorreGen 的原始 Git 历史、基线 Tag 以及历史审计材料不应被改写。

## 八、删除与重命名必须分开

如果发现不再需要的 CorreGen 模块，将其登记为清理候选并分析依赖；不要在 Naming Refactor 中直接删除。删除必须另开 Cleanup 任务，以独立 Commit 记录原因。

## 九、阶段验收

每完成一个阶段必须：

1. 执行相关测试。
2. 检查 Git Diff。
3. 检查是否存在不相关修改。
4. 确认配置和模型行为没有改变。
5. 执行 `git diff --check`。
6. 提交独立 Commit。
7. 记录 Commit SHA。
8. 更新 `docs/NAMING_CONVENTIONS.md`。
9. 更新 `docs/REBUILD_CHANGELOG.md`。

严禁将多个阶段堆积到最后才进行一次巨大提交。如果测试没有成功运行，必须明确记录尚未验证的部分。

## 十、最终完成标准

命名重构完成后：

- 不再存在明显含糊的项目级命名。
- CorreGen 原始版本仍然能够通过 Git 恢复。
- 所有 import 和配置引用保持一致。
- 训练与评估入口正常工作。
- 相关测试全部通过。
- 关键计算行为与重构前一致。
- Git 历史完整记录命名变化。
- 未混入算法修改。
- DROT 等第三方依赖没有被意外修改。
- 项目能够在此基础上继续集成 STRIDE。

最后输出：

1. 原始到新命名的完整映射。
2. Commit 列表及对应 SHA。
3. 修改文件清单。
4. 运行过的测试及结果。
5. 仍存在的兼容性风险。
6. REBUILD 后续模块应遵守的命名约定。

本任务的目标是让 REBUILD 拥有清晰的命名体系，而不是创造一个计算行为已经改变、却被称为“纯重构”的新模型。

在完成这些检查后，继续按照 REBUILD 总任务推进 DROT 和 STRIDE 的集成。

不要急于把所有 `CorreGen` 字样改成 `STRIDE`。在算法迁移完成之前，基线模型应叫 `CorreGenBaselineModel`；等 GW 结构监督真正接入并通过验证后，再建立 `StrideModel`。每次命名重构最多涉及一个相对独立的模块，避免形成难以回滚的巨大 Commit。
