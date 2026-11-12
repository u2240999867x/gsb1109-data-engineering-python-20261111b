"""修订单清洗写操作的列影响认证分类器。

三类判定：
  CERTIFIED-LOCAL    只影响声明的逻辑列（原地改写且目标不被任何其它列可达）
  ALIAS-PROPAGATING  沿 Python 别名传播，或主动安装新的共享引用
  UNCERTIFIABLE      循环图 / 不稳定或会抛异常的 afilter，无法静态认证

判定只依赖身份图（id() 可达性），不依赖值相等性——JSON 相等的两棵树
可能具有完全不同的身份图。
"""
import copy
import json
from collections.abc import MutableMapping, MutableSequence

import dpath
from dpath import segments

CONTAINER = (MutableMapping, MutableSequence)
LOCAL = "CERTIFIED-LOCAL"
PROPAGATING = "ALIAS-PROPAGATING"
UNCERTIFIABLE = "UNCERTIFIABLE"


def _children(node):
    if isinstance(node, MutableMapping):
        yield from node.values()
    elif isinstance(node, MutableSequence):
        yield from node


def identity_graph(root):
    """返回 (id->对象 的可达表, 是否存在回边)。DAG 共享不算循环。"""
    seen, gray, cyclic = {}, set(), False

    def dfs(node):
        nonlocal cyclic
        if not isinstance(node, CONTAINER):
            return
        nid = id(node)
        if nid in gray:
            cyclic = True
            return
        if nid in seen:
            return
        gray.add(nid)
        seen[nid] = node
        for child in _children(node):
            dfs(child)
        gray.discard(nid)

    dfs(root)
    return seen, cyclic


def reachable_ids(root):
    table, cyclic = identity_graph(root)
    return set(table), cyclic


def installs_mutable(src):
    """merge 实际安装的是 walk 出的值；只要其中含容器就会留下 src 引用。"""
    return any(isinstance(v, CONTAINER) for _p, v in segments.walk(src))


def filter_stable(row, afilter, leaves_only):
    """对同一输入两次求值必须一致且不抛异常。"""
    if afilter is None:
        return True, None
    for _path, value in segments.walk(row):
        if leaves_only and not segments.leaf(value):
            continue
        try:
            if afilter(value) != afilter(value):
                return False, f"过滤器对 {value!r} 两次求值结果不一致"
        except Exception as exc:
            return False, f"过滤器对 {value!r} 抛 {type(exc).__name__}: {exc}"
    return True, None


def _matched_paths(row, glob):
    gl = dpath._split_path(glob)
    return [p for p, _ in segments.walk(row) if segments.match(p, gl)]


def _declared_column(glob):
    parts = dpath._split_path(glob)
    return parts[0] if parts else None


def _leaks_to_other_columns(row, obj, declared, columns):
    """obj 的 id 是否出现在声明列之外任一逻辑列的可达集合中。"""
    oid = id(obj)
    for col in columns:
        if col == declared:
            continue
        ids, _ = reachable_ids(row[col])
        if oid in ids:
            return col
    return None


def classify(row, op, glob, *, columns=("colA", "colB"), value=None,
             src=None, afilter=None):
    """返回 (判定, 理由)。"""
    _, cyclic = identity_graph(row)
    if cyclic:
        return UNCERTIFIABLE, "对象图含回边：walk 无 visited 集合，会无限递归，且 JSON 不可导出"

    declared = _declared_column(glob)

    if op == "search":
        ok, why = filter_stable(row, afilter, leaves_only=False)
        if not ok:
            return UNCERTIFIABLE, why
        return LOCAL, "只读，不改源树；但非 yielded 结果树按引用装配，下游改写会回流"

    if op == "merge":
        ok, why = filter_stable(src, afilter, leaves_only=False)
        if not ok:
            return UNCERTIFIABLE, "merge 内部 search(src,'**') 对全部节点调 afilter：" + why
        target = dpath.get(row, glob)
        leak = _leaks_to_other_columns(row, target, declared, columns)
        if leak is not None:
            return PROPAGATING, f"合并目标被列 {leak} 别名可达，递归 merger 原地改共享容器"
        if installs_mutable(src):
            return PROPAGATING, "merge 不深拷贝 src：缺失路径整体安装 src 容器引用，ADDITIVE 列表 target += found 保留 src 元素"
        return LOCAL, "src 的 walk 值均为不可变叶子，且合并目标不被其它列可达"

    leaves_only = op in ("set", "delete")
    ok, why = filter_stable(row, afilter, leaves_only=True)
    if not ok:
        return UNCERTIFIABLE, why

    paths = _matched_paths(row, glob)
    if op == "set" and isinstance(value, CONTAINER) and len(paths) > 1:
        return PROPAGATING, f"同一可变 value 被装入 {len(paths)} 个命中点，制造新的孪生别名"

    # set/delete 的物理写都发生在父容器上（parent[key] = value / del parent[key]），
    # 因此传播判据是“父容器是否被声明列之外的逻辑列别名可达”，而非叶子本身。
    for path in paths:
        parent = segments.get(row, path[:-1]) if len(path) > 1 else row
        leak = _leaks_to_other_columns(row, parent, declared, columns)
        if leak is not None:
            kind = "父容器原地替换键" if op == "set" else "父容器原地删键/置 None"
            return PROPAGATING, f"{path} 的{kind}，而该父容器被列 {leak} 别名可达"
    return LOCAL, f"{len(paths)} 个命中点的父容器均不被其它逻辑列可达"


def column_snapshot(row, columns):
    return {c: json.dumps(row[c], sort_keys=True) for c in columns}


def affected_columns(before, after):
    return sorted(c for c in before if before[c] != after[c])


def shared_row():
    nested = {"name": "acme", "addr": {"city": "OLD", "zip": "0000"}}
    return {"id": 1, "colA": nested, "colB": nested}


def independent_row():
    nested = {"name": "acme", "addr": {"city": "OLD", "zip": "0000"}}
    return {"id": 1, "colA": copy.deepcopy(nested), "colB": copy.deepcopy(nested)}


def run_batch(row, batch):
    for op, glob, kw in batch:
        verdict, reason = classify(row, op, glob, **kw)
        before = column_snapshot(row, ("colA", "colB"))
        if op == "search":
            dpath.search(row, glob, afilter=kw.get("afilter"))
        elif op == "set":
            dpath.set(row, glob, kw["value"], afilter=kw.get("afilter"))
        elif op == "delete":
            dpath.delete(row, glob, afilter=kw.get("afilter"))
        elif op == "merge":
            dpath.merge(dpath.get(row, glob), kw["src"], afilter=kw.get("afilter"))
        actual = affected_columns(before, column_snapshot(row, ("colA", "colB")))
        print(f"{op:7} {glob:18} -> {verdict:18} 实际受影响列={actual}")
        print(f"        理由: {reason}")


class _Alternating:
    """模拟不稳定过滤器：连续两次调用必然给出不同结论。"""

    def __init__(self):
        self.n = 0

    def __call__(self, _):
        self.n += 1
        return self.n % 2 == 0


flaky_filter = _Alternating()


if __name__ == "__main__":
    print("#" * 78)
    print("# 混合小批 A：别名历史（colA 与 colB 共享嵌套对象）")
    print("#" * 78)
    batch = [
        ("search", "/colA/**", {}),
        ("set", "/colA/addr/city", {"value": "NEW"}),
        ("delete", "/colA/addr/zip", {}),
        ("merge", "/colA", {"src": {"note": "ok"}}),
        ("set", "/colA/name", {"value": "X", "afilter": lambda v: isinstance(v, str)}),
    ]
    run_batch(shared_row(), batch)

    print("\n" + "#" * 78)
    print("# 混合小批 B：独立历史（值树相同，身份图不同）")
    print("#" * 78)
    run_batch(independent_row(), batch)

    print("\n" + "#" * 78)
    print("# 无法认证的两类：循环图与不稳定过滤器")
    print("#" * 78)
    cyc = {"colA": {"self": None}, "colB": {}}
    cyc["colA"]["self"] = cyc
    print("循环图 set 判定:", classify(cyc, "set", "/colA/x", value=1))

    row = {"colA": {"v1": 10, "v2": 20}, "colB": {}}
    print("随机过滤器 set 判定:",
          classify(row, "set", "/colA/*", value=0, afilter=flaky_filter))

    def leaf_only_ok(x):
        return int(x) > 15

    row2 = {"colA": {"v1": 10, "v2": {"deep": 1}}, "colB": {}}
    print("同一过滤器：set 判定 =",
          classify(row2, "set", "/colA/**", value=9, afilter=leaf_only_ok)[0],
          "| search 判定 =",
          classify(row2, "search", "/**", afilter=leaf_only_ok))

    print("\n" + "#" * 78)
    print("# 两条历史：导出 JSON 完全相同，同一清理后受影响列不同")
    print("#" * 78)
    h_alias, h_indep = shared_row(), independent_row()
    export_alias = json.dumps(h_alias, sort_keys=True)
    print("清理前 JSON 相同?", export_alias == json.dumps(h_indep, sort_keys=True))
    print("但身份图不同? colA is colB:",
          h_alias["colA"] is h_alias["colB"],
          "vs", h_indep["colA"] is h_indep["colB"])

    def cleanup(row):
        dpath.set(row, "/colA/addr/city", "NEW")
        dpath.delete(row, "/colA/addr/zip")
        dpath.merge(row["colA"], {"note": "ok"})

    before_a = column_snapshot(h_alias, ("colA", "colB"))
    before_i = column_snapshot(h_indep, ("colA", "colB"))
    cleanup(h_alias)
    cleanup(h_indep)
    print("别名历史 受影响列:", affected_columns(before_a, column_snapshot(h_alias, ("colA", "colB"))))
    print("独立历史 受影响列:", affected_columns(before_i, column_snapshot(h_indep, ("colA", "colB"))))
