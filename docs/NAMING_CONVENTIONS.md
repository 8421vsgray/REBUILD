# REBUILD 命名约定

本文档记录 REBUILD 的命名重构映射。命名重构不改变数学计算、训练行为、数据协议或评估定义；算法迁移必须另行提交。

## 规则

- Python 文件、目录、函数、局部变量使用 `snake_case`。
- 类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。
- 私有方法使用 `_snake_case`。
- YAML 字段使用表达语义的 `snake_case`。
- 测试文件使用 `test_<module>.py`。
- 公共模块边界优先使用语义名称；简单局部范围内的 `args`、`x`、`y`、`i` 不强制改名。
- transport plan、soft label、cross-view prediction 和 embedding 必须保持概念区分。

## 当前映射

| CorreGen 原始名称 | REBUILD 名称 | 原因 | 外部接口/兼容性 |
| --- | --- | --- | --- |
| `OursModel` | `CorreGenBaselineModel` | 当前代码仍是 CorreGen 基线，不应提前称为 STRIDE | 计划保留兼容别名；`state_dict` 键不变 |
| `train_time` | `num_runs` | 表示独立训练次数 | YAML/CLI 改名需显式迁移，不能静默失效 |
| `indexs` | `indices` | 修正拼写并表达样本索引 | 仅内部变量，无 checkpoint 影响 |
| `InfoBuffer` | 待算法迁移删除 | 不是命名重构目标，属于 GMM 清理任务 | 不在命名阶段删除 |
| `tag` | `training_phase` | 仅在字段语义完全一致时采用 | 需要同步调用者 |
| `param` | `solver_config` 或 `training_config` | 根据实际内容区分配置类型 | 需要检查关键字调用 |
| `z` / `p` / `z_t` | 语义化 embedding 名称 | 区分 online、prediction 和 target 表示 | 内部公式变量可保留 |

## 阶段记录

- 阶段 A：类与函数命名；先将 `OursModel` 明确为 `CorreGenBaselineModel`。
- 阶段 B：内部变量命名；不与算法删除混合。
- 阶段 C：文件和模块命名；使用 `git mv` 并检查所有 import。
- 阶段 D：配置命名；建立显式旧字段迁移/冲突检查，不改变默认值。
- 阶段 E：日志和输出命名；只改变字段名称，不改变统计定义。

每个阶段必须有独立 Commit、回归测试、`git diff --check` 和 changelog 记录。

## Checkpoint 兼容性

PyTorch `state_dict` 键名由模块属性决定，不应因类名变化而改变。REBUILD 仍需保留旧完整模型 checkpoint 的兼容策略；如果兼容别名不足以加载旧对象，必须提供显式迁移脚本并记录风险，不得静默失败。

在 STRIDE 的 GW 结构监督真正接入并通过验证前，不能把基线类命名为 `StrideModel`。
