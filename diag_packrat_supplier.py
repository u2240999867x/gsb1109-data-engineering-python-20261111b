# -*- coding: utf-8 -*-
"""
诊断：enable_packrat() 之后，输出行完全一样、累计总量却不同。

关键前提（见 pyparsing/core.py:1354）：parse_string() 每次入口都会调用
ParserElement.reset_cache()，所以 packrat HIT 只会发生在【同一次解析内部】
的回溯/前瞻里，不会跨 parse_string 复用。本脚本的所有 HIT 都是单次解析内
由 MatchFirst 回溯或 FollowedBy 前瞻制造的同位置二次尝试。

三类机理：
  M1 失败分支先产生外部计数：fail action 在 _parseNoCache 异常路径执行
     （core.py:957-958），异常被 _parseCache 缓存（core.py:1123-1125）后，
     同 key 重试走 HIT 直接 raise 缓存异常（core.py:1137-1145），fail action
     不再执行。
  M2 成功子式被缓存复用：成功结果以 value[1].copy() 缓存（core.py:1128），
     HIT 路径直接返回副本（core.py:1147-1157），parse action 不再执行，
     动作里累加的指标少一次。
  M3 同名字段在最终结果中被替换：同名 resultsName 的全部 occurrence 都保留
     在 _tokdict 里（results.py:258-265），但 modal 命名下 __getitem__ 只返回
     最后一个（results.py:253-256），as_dict()/持久化行看不到被遮蔽条目，
     而它的 parse action 已经累加过指标。
"""
from __future__ import annotations

import pyparsing as pp
from pyparsing.core import ParserElement, ParseBaseException

TRACE: list[tuple] = []


def loc_info(instring: str, loc: int) -> str:
    return (
        f"第{pp.lineno(loc, instring)}行第{pp.col(loc, instring)}列"
        f" 原行={pp.line(loc, instring)!r}"
    )


_orig_parse_cache = ParserElement._parseCache


def traced_parse_cache(self, instring, loc, do_actions=True, callPreParse=True):
    tag = getattr(self, "diag_tag", type(self).__name__)
    key = (self, instring, loc, callPreParse, do_actions)
    with ParserElement.packrat_cache_lock:
        cached = ParserElement.packrat_cache.get(key)
        hit = cached is not ParserElement.packrat_cache.not_in_cache
    TRACE.append(("ENTER", tag, loc_info(instring, loc), "HIT" if hit else "MISS"))
    try:
        out = _orig_parse_cache(self, instring, loc, do_actions, callPreParse)
    except ParseBaseException as exc:
        TRACE.append(
            ("EXIT ", tag, loc_info(instring, loc),
             "HIT-RAISE" if hit else "MISS-RAISE", f"{type(exc).__name__}@{exc.loc}")
        )
        raise
    TRACE.append(
        ("EXIT ", tag, loc_info(instring, loc), "HIT-OK" if hit else "MISS-OK",
         f"endloc={out[0]}")
    )
    return out


ParserElement._parseCache = traced_parse_cache


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def filtered_trace(tags: set[str]) -> None:
    for row in TRACE:
        if row[1] in tags:
            print("  " + " | ".join(str(x) for x in row))


# ================================================================ M1
def build_m1():
    """PO 条目用圆括号、合同条目用花括号；PO 左括号挂 fail action 做外部计数。
    每行先做 FollowedBy 前瞻校验，再真实解析，制造同位置二次尝试。"""
    counters = {"external_fail": 0}
    event_lines: list[str] = []

    def fail_po_open(s, loc, expr, err):
        counters["external_fail"] += 1
        info = loc_info(s, loc)
        event_lines.append(info)
        TRACE.append(("FAIL-ACTION", "po_open", info, "external_fail++"))

    po_open = pp.Suppress("(").set_fail_action(fail_po_open)
    po_open.diag_tag = "po_open"
    po_item = pp.Group(po_open + pp.Word(pp.alphas) + pp.Suppress(")"))
    po_item.diag_tag = "po_item"

    cn_item = pp.Group(
        pp.Suppress("{") + pp.Suppress("X") + pp.Word(pp.alphas) + pp.Suppress("}")
    )
    cn_item.diag_tag = "cn_item"

    choice = pp.MatchFirst([po_item, cn_item])
    choice.diag_tag = "choice"
    # 前瞻 + 真实解析：同一 loc 对 choice（及其 po_item 失败分支）尝试两次
    line = pp.Group(pp.FollowedBy(choice) + choice)
    line.diag_tag = "line"
    grammar = pp.OneOrMore(line)
    grammar.diag_tag = "grammar"
    return grammar, counters, event_lines


m1_text = "( A )\n{ X B }\n{ X C }\n"


def m1_runner():
    grammar, counters, event_lines = build_m1()
    result = grammar.parse_string(m1_text)
    return result, counters, event_lines


banner("M1 失败分支先产生外部计数")
for packrat in (False, True):
    ParserElement.disable_memoization()
    TRACE.clear()
    if packrat:
        ParserElement.enable_packrat(cache_size_limit=None)
    r1, c1, ev1 = m1_runner()
    print(f"\n--- packrat={'ON' if packrat else 'OFF'} 输出={r1.as_list()} "
          f"external_fail={c1['external_fail']} cache_stats={ParserElement.packrat_cache_stats[:]}")
    print("  fail action 实际触发的原行：")
    for line in ev1:
        print("    -", line)
    print("  缓存相关轨迹（choice / po_item / po_open，HIT 在 choice 节点短路）：")
    filtered_trace({"choice", "line", "po_item", "po_open"})


# ================================================================ M2
def build_m2():
    """alt1 在 '=' 处失败后 alt2 重试：common 键元素在同一 loc 被尝试两次。"""
    totals = {"key_runs": 0, "qty_sum": 0}

    common = pp.Word(pp.alphas)
    common.diag_tag = "common"

    def common_action(tokens):
        totals["key_runs"] += 1
        TRACE.append(("PARSE-ACTION", "common", f"token={tokens[0]!r}",
                      f"key_runs={totals['key_runs']}"))

    common.set_parse_action(common_action)

    qty = pp.Word(pp.nums)
    qty.diag_tag = "qty"

    def qty_action(tokens):
        totals["qty_sum"] += int(tokens[0])
        TRACE.append(("PARSE-ACTION", "qty", f"token={tokens[0]!r}",
                      f"qty_sum={totals['qty_sum']}"))

    qty.set_parse_action(qty_action)

    alt1 = pp.Group(common + pp.Suppress("=") + qty)
    alt1.diag_tag = "alt1_PO"
    alt2 = pp.Group(common + pp.Suppress(":") + qty)
    alt2.diag_tag = "alt2_contract"
    choice = pp.MatchFirst([alt1, alt2])
    choice.diag_tag = "choice"
    return choice, totals


m2_text = "ABC: 7"

banner("M2 成功子式被缓存复用")
m2_snapshots = {}
for packrat in (False, True):
    ParserElement.disable_memoization()
    TRACE.clear()
    if packrat:
        ParserElement.enable_packrat(cache_size_limit=None)
    grammar, totals = build_m2()
    r2 = grammar.parse_string(m2_text)
    m2_snapshots[packrat] = (r2.as_list(), dict(totals))
    print(f"\n--- packrat={'ON' if packrat else 'OFF'} 输出={r2.as_list()} "
          f"指标={totals} cache_stats={ParserElement.packrat_cache_stats[:]}")
    filtered_trace({"common", "choice", "alt1_PO", "alt2_contract"})
    for row in TRACE:
        if row[0] == "PARSE-ACTION":
            print("  " + " | ".join(row))

print(f"\nM2 对比：输出相同={m2_snapshots[False][0] == m2_snapshots[True][0]}，"
      f"key_runs OFF={m2_snapshots[False][1]['key_runs']} "
      f"vs ON={m2_snapshots[True][1]['key_runs']}")


# ================================================================ M3
def build_m3():
    """两个条目同名 entry：modal 命名下持久化的 dict 只留最后一个 occurrence。"""
    counters = {"po_events": 0, "cn_events": 0, "qty_sum": 0}

    def make_piece(tag, marker, counter_key):
        marker_expr = pp.Literal(marker)
        marker_expr.diag_tag = f"{tag}_marker"

        def marker_action(tokens):
            counters[counter_key] += 1
            TRACE.append(("PARSE-ACTION", f"{tag}_marker",
                          f"token={tokens[0]!r}", f"{counter_key}={counters[counter_key]}"))

        marker_expr.set_parse_action(marker_action)

        q = pp.Word(pp.nums)
        q.diag_tag = f"{tag}_qty"

        def qty_action(tokens):
            counters["qty_sum"] += int(tokens[0])
            TRACE.append(("PARSE-ACTION", f"{tag}_qty", f"token={tokens[0]!r}",
                          f"qty_sum={counters['qty_sum']}"))

        q.set_parse_action(qty_action)
        # 动作挂在叶子上；Group + 同名 modal 命名不影响叶子动作的 tokens
        return pp.Group(marker_expr + q)("entry")

    po_piece = make_piece("po", "PO", "po_events")
    cn_piece = make_piece("cn", "CN", "cn_events")
    grammar = pp.Group(po_piece + cn_piece)
    grammar.diag_tag = "row"
    return grammar, counters


m3_text = "PO 5 CN 7"

banner("M3 同名字段在最终结果中被替换")
for packrat in (False, True):
    ParserElement.disable_memoization()
    TRACE.clear()
    if packrat:
        ParserElement.enable_packrat(cache_size_limit=None)
    grammar, counters = build_m3()
    r3 = grammar.parse_string(m3_text)
    row = r3[0]
    print(f"\n--- packrat={'ON' if packrat else 'OFF'}")
    print("  str(row)        =", str(row))
    print("  row.as_list()   =", row.as_list())
    print("  row.as_dict()   =", row.as_dict(), "  <- 持久化/打印行只暴露最后一个 entry")
    print("  row['entry']    =", row["entry"], "（modal __getitem__ 取最后 occurrence）")
    print("  row._all_names  =", row._all_names)
    print("  _tokdict 全量 occurrence（被遮蔽的条目仍在）：")
    for name, occurrences in row._tokdict.items():
        for occ in occurrences:
            print(f"    name={name!r} offset={occ.offset} value={occ.result.as_list()!r}")
    print("  动作侧累计指标   =", counters)
    for event in TRACE:
        if event[0] == "PARSE-ACTION":
            print("  " + " | ".join(event))


# ================================================================ 两组无法区分的历史
banner("历史快照对比：只保存 (最终 ParseResults 视图, 总量)")

# 组一：无回溯文法 vs 带回溯且 packrat 缓存复用，最终视图与总量相同
ParserElement.disable_memoization()
direct_common = pp.Word(pp.alphas)
direct_tot = {"key_runs": 0}
direct_common.set_parse_action(lambda t: direct_tot.__setitem__(
    "key_runs", direct_tot["key_runs"] + 1))
g_direct = pp.Group(direct_common + pp.Suppress(":") + pp.Word(pp.nums))
h1 = g_direct.parse_string(m2_text)
snapshot_h1 = (h1.as_list(), direct_tot["key_runs"])

ParserElement.disable_memoization()
ParserElement.enable_packrat(cache_size_limit=None)
g_back, back_tot = build_m2()
h2 = g_back.parse_string(m2_text)
snapshot_h2 = (h2.as_list(), back_tot["key_runs"])

print("\n组一（输入 ABC: 7，只存输出行 + key_runs）")
print("  H1 无回溯文法单次解析   =", snapshot_h1)
print("  H2 回溯文法+packrat ON  =", snapshot_h2,
      f" cache_stats={ParserElement.packrat_cache_stats[:]}")
print("  两个历史快照是否相等    =", snapshot_h1 == snapshot_h2)
print("  但 H2 解析期 common 实际被尝试 2 次（1 次 MISS + 1 次 HIT），H1 只尝试 1 次")

# 组二：单行双条目 vs 两个批次行，最后持久化行与总量相同
def make_counter_grammar(counters):
    """每个 marker/数量叶子都在动作里累加 counters，并返回带动作的同名条目。"""

    def mk(marker, key):
        marker_expr = pp.Literal(marker).set_parse_action(
            lambda t: counters.__setitem__(key, counters[key] + 1)
        )
        qty_expr = pp.Word(pp.nums).set_parse_action(
            lambda t: counters.__setitem__(
                "qty_sum", counters["qty_sum"] + int(t[0])
            )
        )
        return pp.Group(marker_expr + qty_expr)("entry")

    return pp.OneOrMore(pp.MatchFirst([mk("PO", "po_events"),
                                       mk("CN", "cn_events")]))


ParserElement.disable_memoization()
g3, c3 = build_m3()
h3 = g3.parse_string("PO 5 CN 7")
snapshot_h3 = (h3[0].as_dict(), c3["qty_sum"])
h3_occurrences = [o.result.as_list() for o in h3[0]._tokdict["entry"]]
h3_lines = 1

ParserElement.disable_memoization()
c4 = {"po_events": 0, "cn_events": 0, "qty_sum": 0}
g4 = make_counter_grammar(c4)
g4.parse_string("PO 5")   # 批次 1：第一源行
h4 = g4.parse_string("CN 7")  # 批次 2：第二源行，持久化的是最后一行
snapshot_h4 = (h4.as_dict(), c4["qty_sum"])
h4_lines = 2

print("\n组二（只持久化最后一行的 dict 视图 + qty_sum 总量）")
print("  H3 单批次行 'PO 5 CN 7'  =", snapshot_h3, " 源行数 =", h3_lines)
print("  H4 两批次行 PO5 -> CN7   =", snapshot_h4, " 源行数 =", h4_lines)
print("  两个历史快照是否相等     =", snapshot_h3 == snapshot_h4)
print("  H3 被 dict 视图遮蔽的 occurrence =", h3_occurrences)
print("  H4 实际动作事件 po/cn =", c4["po_events"], c4["cn_events"],
      " vs H3 =", c3["po_events"], c3["cn_events"])
