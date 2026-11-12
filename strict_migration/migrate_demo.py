"""STRICT 表迁移判据演示。

用两批原始输入（CSV / JSONL）经 sqlite-utils 的真实导入路径生成"成品库"，
再按单元格的实际持久值（typeof + quote）把数据划入三个分区，并对主键/外键
做"迁移后判等"的实体合并检查。

运行：.venv/bin/python strict_migration/migrate_demo.py
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlite_utils import Database  # noqa: E402
from sqlite_utils.utils import rows_from_file, Format  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "ledgers")
WORK = os.path.join(HERE, "_work")

INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1

# 三个分区
LOSSLESS = "可无损迁移"
NUMERIC = "仅数值等价"
UNRECOVERABLE = "无法还原原输入"


def _exceeds_int64(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and not (
        INT64_MIN <= value <= INT64_MAX
    )


def _numeric_equal(a: object, b: object) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fa) and math.isfinite(fb) and fa == fb


class StrictProbe:
    """引擎权威探针：把一个持久值真插进各类型的 STRICT 表，按落库结果分区。

    - 插入即被拒（或 NaN 被绑成 NULL、大整数进 REAL 被舍入）-> 无法还原
    - 插入后 quote() 与源完全一致（同存储类、同位序列）       -> 可无损迁移
    - 插入成功但存储类/表示变化、数值仍精确相等               -> 仅数值等价
    """

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        con.execute("CREATE TEMP TABLE _p_int(v INTEGER) STRICT")
        con.execute("CREATE TEMP TABLE _p_real(v REAL) STRICT")
        con.execute("CREATE TEMP TABLE _p_text(v TEXT) STRICT")
        con.execute("CREATE TEMP TABLE _p_blob(v BLOB) STRICT")
        self._tables = {
            "INTEGER": "_p_int",
            "REAL": "_p_real",
            "TEXT": "_p_text",
            "BLOB": "_p_blob",
        }

    def classify(self, storage: str, value, target: str) -> tuple[str, object]:
        if value is None:
            return LOSSLESS, None  # 可空列下 NULL 原样落库；NaN 变来的 NULL 在成品库无法识别
        if target == "ANY":
            return LOSSLESS, value  # STRICT ANY 原样保留存储类
        if target == "REAL":
            if isinstance(value, float) and not math.isfinite(value):
                return UNRECOVERABLE, None  # inf 落库为 9e999，NaN 被绑成 NULL
            if storage == "integer" and abs(int(value)) > 2**53:
                return UNRECOVERABLE, None  # int64 进 IEEE-754 double 必然丢精度
        table = self._tables[target]
        source_quote = self.con.execute("SELECT quote(?)", (value,)).fetchone()[0]
        self.con.execute(f"DELETE FROM {table}")
        try:
            self.con.execute(f"INSERT INTO {table} VALUES (?)", (value,))
        except sqlite3.Error:
            return UNRECOVERABLE, None
        stored_value, stored_type, stored_quote = self.con.execute(
            f"SELECT v, typeof(v), quote(v) FROM {table}"
        ).fetchone()
        if stored_quote == source_quote:
            return LOSSLESS, stored_value
        if stored_type == "null":
            return UNRECOVERABLE, None  # NaN -> NULL 的静默篡改
        if _numeric_equal(value, stored_value):
            return NUMERIC, stored_value
        return UNRECOVERABLE, stored_value  # 接受但数值已变（如 int64 进 REAL 被舍入）


def load_finished_db(
    name: str, fmt: Format, parents_file: str, children_file: str
) -> tuple[str, dict]:
    """按 sqlite-utils 规范路径（insert_all 一次性推断列类型）生成非 STRICT 成品库。

    整批 OverflowError 时隔离超 int64 的行后重试，模拟真实 ETL 的"坏行旁路"，
    并返回台账对账信息：台账行数、入库行数、被旁路行。
    """
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f"{name}.db")
    if os.path.exists(path):
        os.remove(path)
    db = Database(path)
    report: dict[str, dict] = {}

    for table, fp in (("parents", parents_file), ("children", children_file)):
        with open(fp, "rb") as fh:
            rows = list(rows_from_file(fh, format=fmt)[0])
        pk = "biz_id" if table == "parents" else "order_id"
        ledger_count = len(rows)
        overflow = [r for r in rows if any(_exceeds_int64(v) for v in r.values())]
        loadable = [r for r in rows if not any(_exceeds_int64(v) for v in r.values())]
        try:
            db[table].insert_all(loadable, pk=pk)
        except OverflowError as exc:  # 兜底：理论上已被上面的预筛拦截
            report[table] = {"error": str(exc)}
            continue
        db.conn.commit()
        report[table] = {
            "ledger_rows": ledger_count,
            "loaded_rows": db[table].count,
            "sidelined": [(str(r.get(pk)), {k: v for k, v in r.items() if _exceeds_int64(v)})
                          for r in overflow],
        }
    db.close()
    return path, report


def show_persisted_identity(con: sqlite3.Connection, table: str, key: str) -> None:
    print(f"\n=== 同 typeof + 同持久值，不同业务编号（{table}）===")
    groups: dict[tuple, list] = {}
    for biz, v in con.execute(f'SELECT "{key}", code FROM "{table}"'):
        storage = con.execute("SELECT typeof(?)", (v,)).fetchone()[0]
        quote = con.execute("SELECT quote(?)", (v,)).fetchone()[0]
        groups.setdefault((storage, quote), []).append(biz)
    for (storage, quote), bizs in groups.items():
        print(f"  持久签名 ({storage}, {quote}) 对应业务编号 {bizs}")


def scan_columns(
    con: sqlite3.Connection, probe: StrictProbe, table: str, targets: dict[str, str]
) -> None:
    print(f"\n=== 表 {table}：按持久值逐列分区 ===")
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    for col in cols:
        target = targets.get(col, "ANY")
        buckets: dict[str, int] = {}
        samples: dict[str, list] = {}
        for (v,) in con.execute(f'SELECT "{col}" FROM "{table}"'):
            storage = con.execute("SELECT typeof(?)", (v,)).fetchone()[0]
            zone, _ = probe.classify(storage, v, target)
            buckets[zone] = buckets.get(zone, 0) + 1
            samples.setdefault(zone, []).append((storage, v))
        detail = ", ".join(f"{k}={v}" for k, v in sorted(buckets.items()))
        print(f"  {col:<12} 目标 {target:<7} : {detail}")
        for zone, seen in samples.items():
            uniq = sorted({(s, repr(v)) for s, v in seen})
            print(f"      {zone}: {uniq[:4]}")


def key_merge_gate(
    con: sqlite3.Connection, probe: StrictProbe, table: str, key: str, target: str
) -> None:
    """迁移后判等分桶：同一目标键值对应多个源实体即为合并风险。"""
    print(f"\n=== 实体合并闸门：{table}.{key} -> STRICT {target} ===")
    buckets: dict[object, list] = {}
    rejected = []
    for rid, (v, storage) in enumerate(
        con.execute(f'SELECT "{key}", typeof("{key}") FROM "{table}"'), start=1
    ):
        zone, target_key = probe.classify(storage, v, target)
        if zone == UNRECOVERABLE:
            rejected.append((rid, storage, v))
            continue
        buckets.setdefault(target_key, []).append((rid, storage, v))
    total = sum(len(v) for v in buckets.values()) + len(rejected)
    merged = {k: rows for k, rows in buckets.items() if len(rows) > 1}
    print(
        f"  源行数={total} 目标实体数={len(buckets)} "
        f"被拒行={len(rejected)} 发生合并的键={len(merged)}"
    )
    for k, rows in list(merged.items())[:5]:
        print(f"  合并: 目标键 {k!r} <- {rows}")
    for row in rejected[:5]:
        print(f"  拒绝: {row}")


def fk_join_check(
    con: sqlite3.Connection, probe: StrictProbe, target: str
) -> None:
    """父键迁移后，子表外键按迁移判等是否仍能唯一命中父实体。"""
    print(f"\n=== 外键核对：children.parent_code -> parents.code（父键目标 {target}）===")
    parents = con.execute("SELECT DISTINCT code, typeof(code) FROM parents").fetchall()
    parent_buckets: dict[object, list] = {}
    for v, storage in parents:
        zone, tk = probe.classify(storage, v, target)
        if zone == UNRECOVERABLE:
            continue
        parent_buckets.setdefault(tk, []).append(v)
    for oid, pc, storage in con.execute(
        "SELECT order_id, parent_code, typeof(parent_code) FROM children"
    ):
        zone, tk = probe.classify(storage, pc, target)
        if zone == UNRECOVERABLE:
            print(f"  {oid}: 外键 {pc!r}({storage}) 无法转为 {target}，迁移后悬空")
            continue
        hits = parent_buckets.get(tk, [])
        status = "唯一命中" if len(hits) == 1 else f"歧义/悬空，候选={hits}"
        print(f"  {oid}: 持久外键 {pc!r}({storage}) -> 目标键 {tk!r}，{status}")


def strict_build_attempt(
    con: sqlite3.Connection, table: str, targets: dict[str, str], key: str
) -> None:
    """真实建 STRICT 表并搬数，让数据库自己报出合并/类型错误。"""
    print(f"\n=== 实建 STRICT 表验证：{table}（键 {key}）===")
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    defs = ", ".join(
        f'"{c}" {targets.get(c, "ANY")}' + (" PRIMARY KEY" if c == key else "")
        for c in cols
    )
    new = f"{table}_strict"
    con.execute(f"DROP TABLE IF EXISTS {new}")
    con.execute(f"CREATE TABLE {new} ({defs}) STRICT")
    quoted_cols = ", ".join(f'"{c}"' for c in cols)
    ok = fail = 0
    for row in con.execute(f'SELECT {quoted_cols} FROM "{table}"'):
        try:
            con.execute(
                f"INSERT INTO {new} VALUES ({','.join('?' * len(cols))})", row
            )
            ok += 1
        except sqlite3.Error as exc:
            fail += 1
            print(
                f"  写入失败 {dict(zip(cols, row))}: "
                f"{type(exc).__name__}: {str(exc)[:60]}"
            )
    count = con.execute(f"SELECT count(*) FROM {new}").fetchone()[0]
    print(f"  成功 {ok} 行，失败 {fail} 行；STRICT 表实有 {count} 行")


def main() -> None:
    csv_db, csv_report = load_finished_db(
        "csv_batch",
        Format.CSV,
        os.path.join(LEDGER, "batch_csv", "parents.csv"),
        os.path.join(LEDGER, "batch_csv", "children.csv"),
    )
    nl_db, nl_report = load_finished_db(
        "nl_batch",
        Format.NL,
        os.path.join(LEDGER, "batch_nl", "parents.jsonl"),
        os.path.join(LEDGER, "batch_nl", "children.jsonl"),
    )

    print("########## 批次一：CSV 导入（一切先落 TEXT）##########")
    con = sqlite3.connect(csv_db)
    probe = StrictProbe(con)
    print("台账对账:")
    for table, info in csv_report.items():
        print(" ", table, info)
    print(
        "schema:\n"
        + con.execute(
            "SELECT sql FROM sqlite_master WHERE name='parents'"
        ).fetchone()[0]
    )
    show_persisted_identity(con, "parents", "biz_id")
    targets = {"biz_id": "TEXT", "code": "INTEGER", "qty": "INTEGER"}
    scan_columns(con, probe, "parents", targets)
    key_merge_gate(con, probe, "parents", "code", "INTEGER")
    fk_join_check(con, probe, "INTEGER")
    strict_build_attempt(con, "parents", targets, "biz_id")
    con.close()

    print("\n\n########## 批次二：JSONL 导入（insert_all 把混合列推断为 REAL）##########")
    con = sqlite3.connect(nl_db)
    probe = StrictProbe(con)
    print("台账对账（导入旁路与舍入证据）:")
    for table, info in nl_report.items():
        print(" ", table, info)
    print(
        "schema:\n"
        + con.execute(
            "SELECT sql FROM sqlite_master WHERE name='parents'"
        ).fetchone()[0]
    )
    show_persisted_identity(con, "parents", "biz_id")
    print("\n  -- 按成品库真实存储选目标：code REAL / qty REAL --")
    scan_columns(con, probe, "parents", {"biz_id": "TEXT", "code": "REAL", "qty": "REAL"})
    print("\n  -- 若强行把 code 收敛到 STRICT INTEGER --")
    scan_columns(con, probe, "parents", {"biz_id": "TEXT", "code": "INTEGER", "qty": "REAL"})
    print("\n  -- 台账 vs 成品库：NL-005 在导入时已被 REAL 舍入（损坏发生在迁移之前）--")
    for row in con.execute(
        "SELECT biz_id, code, typeof(code), quote(code) FROM parents WHERE biz_id='NL-005'"
    ):
        print("  成品库:", row, " 台账原值: 9007199254740993")
    key_merge_gate(con, probe, "parents", "code", "INTEGER")
    fk_join_check(con, probe, "INTEGER")
    strict_build_attempt(
        con, "parents", {"biz_id": "TEXT", "code": "REAL", "qty": "REAL"}, "biz_id"
    )

    print("\n=== NaN 证据：成品库只看得到 NULL，无法区分真空值与 NaN ===")
    for row in con.execute(
        "SELECT biz_id, qty, typeof(qty), quote(qty) FROM parents"
    ):
        print(" ", row)
    con.close()

    print("\n\n########## 合成键表演示：单行可转 ≠ 全表可迁 ##########")
    con = sqlite3.connect(":memory:")
    probe = StrictProbe(con)
    con.execute("CREATE TABLE legacy(code, biz TEXT PRIMARY KEY) WITHOUT ROWID")
    con.executemany(
        "INSERT INTO legacy VALUES(?,?)",
        [(1, "E1"), ("1", "E2"), ("01", "E3"), (1.0, "E4"), (True, "E5")],
    )
    key_merge_gate(con, probe, "legacy", "code", "INTEGER")
    print("\n  -- 以业务列 biz 做主键：5 行全部可写，逐行都合法 --")
    strict_build_attempt(con, "legacy", {"code": "INTEGER", "biz": "TEXT"}, "biz")
    print("\n  -- 改以 code 做 STRICT 主键：实体吞并由数据库直接拒绝 --")
    strict_build_attempt(con, "legacy", {"code": "INTEGER", "biz": "TEXT"}, "code")


if __name__ == "__main__":
    main()
