"""Concrete forensic scenario: two histories sharing one legacy log.

Master data holds products keyed by an external ID ``ext_id``. Rows 1 and 2
were imported twice under the same external ID, so a condition-based upsert
can fan out to both. The legacy importer logged only each source change, the
IDs returned by ``upsert`` and the final row snapshots.

Two histories (``A`` and ``B``) are built so that:

* the final table is byte-identical,
* every returned-ID sequence is identical,
* but the fields actually overwritten on rows 1 and 2 differ.

A full enumeration over the finite old-value domains then recovers *every*
seed state compatible with the log, not just the two example histories.
"""

from __future__ import annotations

from pprint import pp
from typing import Mapping

from tinydb.table import Document

from .model import (
    CondChange,
    DocumentChange,
    LogEntry,
    ReplayResult,
    candidate_old_values,
    enumerate_seed_states,
    inverse_patches,
    matches_log,
    minimal_discriminating_observation,
    repair_scope,
    replay,
    rollback,
)
from .model import EXT_KEY, TABLE_NAME, _fresh_table

ABSENT = "<absent>"

DOMAINS: dict[str, tuple[object, ...]] = {
    "status": ("on", "off", ABSENT),
    "cat": ("A", "B", ABSENT),
    "note": ("m", "n", ABSENT),
}

ROW_IDS = (1, 2)
FIXED_FIELDS = {
    1: {EXT_KEY: "X-1", "sku": "s1"},
    2: {EXT_KEY: "X-1", "sku": "s2"},
}

CHANGES = (
    CondChange("X-1", {EXT_KEY: "X-1", "status": "on", "cat": "B"}),
    DocumentChange(2, {"note": "z"}),
    CondChange("X-9", {EXT_KEY: "X-9", "status": "on", "source": "web"}),
)

OBSERVED_FINAL: dict[int, dict] = {
    1: {EXT_KEY: "X-1", "sku": "s1", "status": "on", "cat": "B"},
    2: {EXT_KEY: "X-1", "sku": "s2", "status": "on", "cat": "B", "note": "z"},
    3: {EXT_KEY: "X-9", "status": "on", "source": "web"},
}

LEGACY_LOG = (
    LogEntry(0, CHANGES[0], (1, 2)),
    LogEntry(1, CHANGES[1], (2,)),
    LogEntry(2, CHANGES[2], (3,)),
)

SEED_A: dict[int, dict] = {
    1: {EXT_KEY: "X-1", "sku": "s1", "status": "off", "cat": "A"},
    2: {EXT_KEY: "X-1", "sku": "s2", "status": "off", "cat": "A", "note": "m"},
}

SEED_B: dict[int, dict] = {
    1: {EXT_KEY: "X-1", "sku": "s1", "status": "off", "cat": "B"},
    2: {EXT_KEY: "X-1", "sku": "s2", "status": "on", "cat": "A", "note": "n"},
}


def _strip_absent(seed: Mapping[int, Mapping]) -> dict[int, dict]:
    built: dict[int, dict] = {}
    for doc_id, row in seed.items():
        built[doc_id] = {k: v for k, v in row.items() if v != ABSENT}
    return built


def run_pair() -> tuple[ReplayResult, ReplayResult]:
    history_a = replay(SEED_A, CHANGES)
    history_b = replay(SEED_B, CHANGES)
    return history_a, history_b


def run_full_enumeration() -> list[ReplayResult]:
    """Replay every domain-allowed seed and keep the log-compatible ones."""

    compatible: list[ReplayResult] = []
    for raw_seed in enumerate_seed_states(ROW_IDS, DOMAINS, FIXED_FIELDS):
        seed = _strip_absent(raw_seed)
        result = replay(seed, CHANGES)
        if matches_log(result, LEGACY_LOG, OBSERVED_FINAL):
            compatible.append(result)
    return compatible


def run_narrowing_scenario() -> dict[str, object]:
    """A narrower case where the returned ID pins one old value.

    The source change filters on ``where('status') == 'on'`` rather than on
    the external key. The legacy log records returned IDs ``(1,)`` (an update,
    not an insert), so row 1 *must* have had ``status == 'on'`` beforehand:
    a seed with ``status`` off or absent would have produced an insert with a
    fresh ID, contradicting the log. Meanwhile ``cat`` stays unconstrained.
    """

    domains = {"status": ("on", "off", ABSENT), "cat": ("A", "B", ABSENT)}
    change = CondChange(
        "on", {"cat": "B"}, cond_field="status"
    )
    entries = [LogEntry(0, change, (1,))]
    final = {1: {EXT_KEY: "X-7", "sku": "s7", "status": "on", "cat": "B"}}

    compatible: list[ReplayResult] = []
    for raw_seed in enumerate_seed_states(
        (1,), domains, {1: {EXT_KEY: "X-7", "sku": "s7"}}
    ):
        seed = _strip_absent(raw_seed)
        result = replay(seed, [change])
        if matches_log(result, entries, final):
            compatible.append(result)

    report = candidate_old_values(compatible, 0)
    counterfactual = replay(
        {1: {EXT_KEY: "X-7", "sku": "s7", "status": "off", "cat": "A"}},
        [change],
    )
    return {
        "compatible_histories": len(compatible),
        "old_value_report": report,
        "counterfactual_path": counterfactual.traces[0].path,
        "counterfactual_returned": counterfactual.traces[0].returned_ids,
    }


def print_report() -> None:
    history_a, history_b = run_pair()
    compatible = run_full_enumeration()

    print("=" * 72)
    print("旧导入日志（仅含源变更、upsert 返回 ID、最终行）")
    print("=" * 72)
    for entry in LEGACY_LOG:
        print(f"  step {entry.step}: {entry.change!r} -> ids={entry.returned_ids}")
    print("  终表:")
    pp(OBSERVED_FINAL, indent=4)

    print("\n" + "=" * 72)
    print("路径判定（依据 tinydb.table.Table.upsert 三分支）")
    print("=" * 72)
    for trace in history_a.traces:
        print(f"  step {trace.step}: {trace.path:<14} 返回 {trace.returned_ids}")

    print("\n" + "=" * 72)
    print("历史 A / B 的观测等价性")
    print("=" * 72)
    same_final = history_a.final_table == history_b.final_table == OBSERVED_FINAL
    same_ids = (
        history_a.returned_sequences
        == history_b.returned_sequences
        == [e.returned_ids for e in LEGACY_LOG]
    )
    print(f"  终表完全相同: {same_final}")
    print(f"  返回 ID 序列相同: {same_ids}")

    print("\n两段历史实际被覆盖的字段（日志未记录的部分）:")
    for label, result in (("A", history_a), ("B", history_b)):
        print(f"  历史 {label}:")
        for trace in result.traces:
            if trace.overwritten:
                for doc_id, old in trace.overwritten.items():
                    print(f"    step {trace.step} 行 {doc_id} 旧值 {old}")
            elif trace.path == "insert":
                print(f"    step {trace.step} 新插入行 {trace.returned_ids[0]}（无旧值）")

    print("\n" + "=" * 72)
    print(f"有限值域枚举：729 个种子中有 {len(compatible)} 段历史与现有日志兼容")
    print("=" * 72)
    for step in range(len(CHANGES)):
        report = candidate_old_values(compatible, step)
        print(f"\n  step {step} 路径={report['paths']} "
              f"受影响行={report['affected_doc_ids']}")
        for doc_id, fields in report["by_doc_id"].items():
            for name, info in fields.items():
                flag = "可唯一确定" if info["determined"] else "无法恢复"
                print(
                    f"    行 {doc_id}.{name}: 候选旧值={info['possible_values']}"
                    f"  -> {flag}"
                )

    print("\n" + "=" * 72)
    print("可唯一确定 vs 无法恢复")
    print("=" * 72)
    print("  可确定: 每一步触及哪些行（返回 ID 即受影响行）及其路径；")
    print("          step2 为零匹配插入，行3不存在旧值；")
    print("          ext_id（条件键）与 sku（从不出现在任何 payload 中）")
    print("          终表可见即旧值；step1 时行2 的 status/cat 已被 step0 钉死。")
    print("  无法恢复: 被 payload 覆盖的 status/cat，以及 step1 被覆盖的 note，")
    print("            旧值或'原先缺键'都可能，merge 不留痕迹。")

    narrow = run_narrowing_scenario()
    print("\n" + "=" * 72)
    print("对照：条件路径本身即可判定旧值的窄化场景")
    print("=" * 72)
    print(f"  兼容历史数: {narrow['compatible_histories']}")
    report = narrow["old_value_report"]
    for name, info in report["by_doc_id"][1].items():
        flag = "可唯一确定" if info["determined"] else "无法恢复"
        print(f"    行1.{name}: 候选旧值={info['possible_values']} -> {flag}")
    print(
        "  反事实验证: 若旧 status='off'，同一条变更走 "
        f"{narrow['counterfactual_path']} 路径、返回 {narrow['counterfactual_returned']}，"
        "与日志 (1,) 矛盾。"
    )

    probes = minimal_discriminating_observation(history_a, history_b, 0)
    print("\n" + "=" * 72)
    print("区分历史 A / B 的最小追加观测")
    print("=" * 72)
    print("  任取以下单个单元格的操作前值即可判别（最小观测 = 1 个标量）:")
    for probe in probes:
        print(
            f"    行{probe['doc_id']}.{probe['field']}: "
            f"A={probe['history_a_old']!r} vs B={probe['history_b_old']!r}"
        )

    print("\n" + "=" * 72)
    print("带旧值重放 -> 逆补丁 -> 回修复范围")
    print("=" * 72)
    for label, result in (("A", history_a), ("B", history_b)):
        print(f"\n  历史 {label} 各步修复范围:")
        for step in range(len(CHANGES)):
            print(f"    {repair_scope(result, step)}")
        table = _fresh_table()
        for doc_id, row in result.final_table.items():
            table.insert(Document(row, doc_id=doc_id))
        rollback(table, inverse_patches(result))
        restored = {doc.doc_id: dict(doc) for doc in table.all()}
        seed = SEED_A if label == "A" else SEED_B
        print(f"  历史 {label} 回滚后与该历史种子状态一致: {restored == seed}")


if __name__ == "__main__":
    print_report()
