# 嵌套明细错位归属：三种机制的可复现实验

三个脚本互相独立，直接用仓库自带 venv 运行，**不改动库代码**：

```
.venv/bin/python repro/scenario_a_check_with_reentry.py   # check_with 重入同一 Validator
.venv/bin/python repro/scenario_b_child_parent_path.py    # 子验证器带错父路径
.venv/bin/python repro/scenario_c_handler_stale_state.py  # 同一错误处理器残留上一行状态
.venv/bin/python repro/trace_first_divergence.py          # 统一插桩，定位第一处归属分歧
```

每个脚本同时打印四份证据：输入 document、原始 `v._errors`（ValidationError 对象）、
`document_error_tree`、流式 handler 的回调时序，以及最终 `v.errors` 显示。

## 结论速查

| 机制 | 第一处分歧位置 | 正常返回时还可信的导出行 | 内层抛异常时 | 仅靠格式化能否补救 |
|---|---|---|---|---|
| A 重入 | 嵌套 `validate()` 触发的 `__init_processing` 把 `_errors`/两棵树整体替换（validator.py:628-631），合法错误被丢弃；随后 rogue 错误继承未被重置的 `document_path`（validator.py:287） | 仅顶层"clients 有错"成立；叶子的行/字段归属全不可信（其 field/value 在输入 document 中根本不存在） | 根 `_errors`/树为空；只有被外部持有的子验证器对象 `_errors` 里还有成形错误，公共 API 取不到 | 不能。正确的错误对象已被销毁，只能重新验证 |
| B 错父路径 | `_get_child_validator` 生成子验证器那一刻路径已错（validator.py:346-351），早于任何错误构造；错误只是忠实记录了错路径 | code/message/value/schema_path 可信，唯独 document_path 的行号不可信；两份不同原行输入渲染出逐字相同的输出 | 父级不导入任何错误；孤儿子验证器上的错误同样带着错路径 | 不能。路径在 ValidationError 构造时固化，handler 只读取它 |
| C handler 残留 | 错误构造与两棵树全部正确；分歧发生在第二次运行的 `start()` 没有清空旧状态（errors.py:545-546 的 clear 被绕过） | `_errors`、两棵错误树完全可信，只有 handler 渲染出的 `v.errors` 被污染 | 上游真值依旧完好（异常前已登记的错误不受影响） | 能。`_error` 中先建树后 emit（validator.py:272-278），用会在 start() 重置的新 handler 重新渲染原始错误即可还原 |

## 三种机制的判别特征

- A：叶子错误的 `value`/字段名在输入 document 里不存在（如 `rogue_field`）。
- B：叶子的 `value` 确实在 document 中，但位于 document_path 所指索引之外的另一行。
- C：`v._errors` 为空、`document_error_tree` 为空，`v.errors` 却非空——正确实现下不可能出现。
