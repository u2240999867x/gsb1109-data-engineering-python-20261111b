# TinyDB 同外部号 upsert 覆盖取证报告

## 1. 问题与 `upsert` 的三条代码路径

`tinydb/table.py:583` 的 `Table.upsert` 先尝试 `update`，再依据返回 ID 是否为空决定是否 `insert`，共三种可观测路径：

| 路径 | 调用形态 | 触发条件 | 返回 ID | 对主数据的影响 |
|---|---|---|---|---|
| 普通 dict 条件匹配 | `upsert(dict, where(ext_id) == 'X-1')` | 一条或多条行满足条件 | **所有**匹配行的 ID | 对每个匹配行做 `dict.update` 式字段合并，可能一次覆盖多行 |
| Document 指定 doc_id | `upsert(Document(payload, doc_id=2))` | 该 doc_id 已存在 | `[2]` | 仅合并该行；doc_id 不存在时落入下一种 |
| 零匹配后插入 | 上述两种的兜底分支 | 无任何匹配 | 新分配（或 Document 自带）的 ID | 新增一行，无旧值可言 |

关键风险在第一行：文档明确 “This will update *all* documents matching the query”。当多个商品共用同一外部号时，单条源变更的返回 ID 列表会出现多个元素——**日志虽然记下了这些 ID（受影响行可定位），却没有记录每行操作前的字段值（旧值丢失）**。`update` 用的是合并语义，被 payload 覆盖的字段不留下任何历史痕迹；“原先缺该键”与“原先是任何旧值”在终表里完全不可区分。

## 2. 两段观测等价、覆盖内容不同的历史

主数据表中 doc_id 1、2 共用外部号 `X-1`。旧日志只有三条源变更、返回 ID 与最终行：

```
step0  cond  ext_id==X-1, payload={status:on, cat:B}      -> ids (1, 2)
step1  Document doc_id=2, payload={note:z}                 -> ids (2,)
step2  cond  ext_id==X-9, payload={ext_id:X-9,status:on}  -> ids (3,)
终表:
  1: {ext_id:X-1, sku:s1, status:on, cat:B}
  2: {ext_id:X-1, sku:s2, status:on, cat:B, note:z}
  3: {ext_id:X-9, status:on, source:web}
```

两段历史（`forensics/scenarios.py` 的 `SEED_A`/`SEED_B`）重放后**终表逐字节相同、三步返回 ID 序列完全相同**，但 step0/step1 实际覆盖的旧值不同：

| 单元格 | 历史 A 旧值 | 历史 B 旧值 | 终值 | 日志能否区分 |
|---|---|---|---|---|
| 行1.status | `off`（被覆盖） | `off`（被覆盖） | `on` | 否 |
| 行1.cat | `A`（被覆盖） | `B`（未变，不在 overwritten 中） | `B` | **否** |
| 行2.status | `off`（被覆盖） | `on`（未变） | `on` | **否** |
| 行2.cat | `A`（被覆盖） | `A`（被覆盖） | `B` | 否 |
| 行2.note（step1 前） | `m`（被覆盖） | `n`（被覆盖） | `z` | **否** |

历史 B 中行1.cat、行2.status 根本没有发生写入，修复范围比历史 A 小，但仅凭返回 ID 和终表无法识别这一点。

## 3. 有限值域上的完整枚举（不是单条猜测）

给定旧字段值域（含“缺键”这一可能，用 `<absent>` 表示）：

- `status ∈ {on, off, <absent>}`，`cat ∈ {A, B, <absent>}`，`note ∈ {m, n, <absent>}`
- 每行 3 个域字段 → 27 种组合；两行共 **729 个候选种子状态**。

`enumerate_seed_states` 穷举全部 729 个种子，用真实 TinyDB 逐个重放，再以 `matches_log`（返回 ID 序列 + 终表 + 源变更三重比对）过滤，得到 **243 段与现有日志兼容的历史**（243 = 3⁵：行1 的三个域字段 × 行2 的两个域字段自由；行1 若带 `note` 会在终表残留，与日志矛盾，故被排除）。

跨全部兼容历史汇总，每个受影响字段的**所有**可能原值：

| 步骤 | 行.字段 | 候选旧值全集 | 结论 |
|---|---|---|---|
| 0 | 行1.status | `{off, on, <absent>}` | 无法恢复 |
| 0 | 行1.cat | `{A, B, <absent>}` | 无法恢复 |
| 0 | 行2.status | `{off, on, <absent>}` | 无法恢复 |
| 0 | 行2.cat | `{A, B, <absent>}` | 无法恢复 |
| 0 | 行2.note | `{m, n, <absent>}` | 无法恢复（step1 才被覆盖，但 step0 时可能已存在） |
| 1 | 行2.note | `{m, n, <absent>}` | 无法恢复 |
| 0/1 | 行1/行2.ext_id | `{X-1}` | 可确定（条件键，种子固定） |
| 0/1 | 行1/行2.sku | `{s1}` / `{s2}` | 可确定（从不出现在任何 payload 中，终值即旧值） |
| 1 | 行2.status / cat | `{on}` / `{B}` | 可确定（step0 已将其钉成终值） |
| 2 | 行3 | — | 插入路径，无旧值；受影响行就是返回的 `3` |

**凭现有日志能确定的**：每一步走了哪条路径、触及哪些行（返回 ID 即权威受影响行集合）、插入行无旧值、以及从未被任何 payload 写过的字段（如 `sku`）和条件键 `ext_id` 的旧值。
**无法恢复的**：任何被 payload 合并过的字段，其旧值或“原先缺键”都可能——必须以候选集合而非单点猜测交付。

### 对照：路径本身即可判定旧值的窄化情形

若源变更改用 `where('status') == 'on'` 过滤且日志记录返回 `(1,)`（更新而非插入），则行1 操作前必然 `status=='on'`：反事实种子 `status='off'` 重放会走 insert 路径、返回新 ID `(2,)`，与日志矛盾。枚举确认该场景 9 个种子中只有 3 个兼容，且 `status` 候选收敛为 `{on}`（可确定），而未被条件约束的 `cat` 仍是 `{A, B, <absent>}`。这说明可判定性来自“条件字段 + 更新/插入分支”携带的信息，而非默认存在。

## 4. 区分两段历史的最小追加观测

`minimal_discriminating_observation` 比较两段历史在目标步骤前的快照，差异单元格为：

- 行1.cat：A=`A` vs B=`B`
- 行2.status：A=`off` vs B=`on`
- 行2.note：A=`m` vs B=`n`

**最小追加观测 = 其中任意一个单元格的操作前标量值**（一次外部系统快照、一条 binlog/CDC 前镜像或一次备份核对均可）。例如只补到“step0 前行2.status 为 on”，即排除历史 A、锁定历史 B；观测任一项都足以二选一，无需全表前镜像。

## 5. 带旧值重放与逆补丁回修复范围

`replay` 在真实 TinyDB（`MemoryStorage`）上执行，并逐步保存每行 before/after 快照：

- `repair_scope(step)` 给出该步精确修复足迹：更新行的被覆盖字段旧值、新增字段、以及插入路径的新 doc_id；
- `inverse_patches` 按**后写先回**顺序生成逆补丁——插入步生成 `remove`，更新步生成整行 `restore`；
- `rollback` 用“先删后整行插入”执行 restore（而非 merge），保证“原先缺键、后来新增”的字段被真正撤回；测试 `test_rollback_withdraws_fields_added_to_previously_absent_keys` 覆盖这一点。

两段历史回滚后均与其种子状态逐字段一致（脚本输出 `回滚后与该历史种子状态一致: True`）。实践含义：在追加观测锁定真实历史之前，修复范围只能按候选集合表达（如行2.note 回滚目标 ∈ {m, n, 删除该键}）；观测一旦补齐，即可从对应历史的逆补丁中取出确定性的回滚脚本，避免按错误猜测覆盖更多主数据。

## 6. 复现方式

```bash
.venv/bin/python -m forensics            # 打印完整取证报告
.venv/bin/python -m pytest tests/test_forensics.py -o addopts="" -q
```

代码布局：

- `forensics/model.py`：三路径重放、种子枚举、候选旧值汇总、最小判别观测、逆补丁与回滚；
- `forensics/scenarios.py`：双历史场景、729 种子枚举、窄化对照场景与报告输出；
- `tests/test_forensics.py`：13 项测试（全量 240 passed / 1 skipped，无回归）。
