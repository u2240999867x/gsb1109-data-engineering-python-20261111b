"""
CachingMiddleware 提交边界回归测试。

背景：清洗批次写入 TinyDB，外层套了 CachingMiddleware。
调用者只看到 insert 的返回/异常，但数据是否落盘取决于
缓存计数是否达到 WRITE_CACHE_SIZE 或显式 flush()。

四个行级观测点：
1. 写入调用者：doc_id 与异常类型；
2. 缓存对象：cache 内容与 _cache_modified_count；
3. 底层 storage：实际 write 次数、事件与文件内容；
4. 新进程重开：全新进程重开 JSON 文件验证磁盘状态。

两类失败向调用者抛同一个异常 CommitError：
- pre_write：落盘前失败，数据只在缓存，磁盘没有；
- post_write：fsync 已完成但确认丢失，磁盘已更新，
  调用者却收到异常。

仅凭异常无法区分二者，必须比对缓存状态与新进程
看到的磁盘状态。
"""

import json
import os
import subprocess
import sys
import textwrap
from itertools import product

import pytest

from tinydb import TinyDB, where
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage, MemoryStorage

CACHE_SIZE = 3


class CommitError(RuntimeError):
    """两类失败对外暴露的同一种异常。"""


class ProbedJSONStorage(JSONStorage):
    """带行级观测与故障注入点的 JSONStorage。

    观测事件（按实例记到 self.log）：
      - before_write：进入底层 write，句柄写入前；
      - after_persist：fsync+truncate 完成、方法返回前；
      - read：读取/重开事件。
    故障点由实例属性控制：
      - fail_on_write='pre_write'：写入前抛 CommitError；
      - fail_on_write='post_write'：持久化后抛 CommitError。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []
        self.write_calls = 0
        self.fail_on_write = None

    def read(self):
        self.log.append(('read', None))
        return super().read()

    def write(self, data):
        self.write_calls += 1
        seq = self.write_calls
        self.log.append(('before_write', seq))

        if self.fail_on_write == 'pre_write':
            # 写前失败：一个字节都没写，磁盘保持旧内容。
            raise CommitError(
                f'pre-write failure on storage write #{seq}'
            )

        self._handle.seek(0)
        serialized = json.dumps(data, **self.kwargs)
        self._handle.write(serialized)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.truncate()

        # 此刻新状态已在磁盘上，可被另一进程读到。
        self.log.append(('after_persist', seq))

        if self.fail_on_write == 'post_write':
            # 已持久但确认丢失：收异常，磁盘却已更新。
            raise CommitError(
                f'ack lost after persist on write #{seq}'
            )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / 'batch.db')


def _make_db(path, cache_size=CACHE_SIZE):
    middleware = CachingMiddleware(ProbedJSONStorage)
    middleware.WRITE_CACHE_SIZE = cache_size
    db = TinyDB(path, storage=middleware)
    return db, middleware


def _raw_file_docs(path):
    """绕过缓存直接读磁盘，返回默认表文档 dict。"""
    if os.path.getsize(path) == 0:
        return {}
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    return payload.get('_default', {})


def _write_events(storage):
    """只看底层 write 事件，过滤 read 噪声。"""
    return [event for event in storage.log if event[0] != 'read']


def _cache_docs(middleware):
    """观测点 2：缓存中默认表的文档。"""
    if middleware.cache is None:
        return None
    return middleware.cache.get('_default', {})


def _read_in_subprocess(path):
    """全新进程重开数据库，模拟崩溃后进程重开。"""
    code = textwrap.dedent(
        """
        import json, sys
        from tinydb import TinyDB
        from tinydb.middlewares import CachingMiddleware
        from tinydb.storages import JSONStorage

        with TinyDB(sys.argv[1],
                    storage=CachingMiddleware(JSONStorage)) as db:
            json.dump(db.all(), sys.stdout)
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', code, path],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# 1. 四个观测点的基线：正常路径上每一层看到什么
# ---------------------------------------------------------------------------

def test_four_observation_points_baseline(db_path):
    db, middleware = _make_db(db_path)
    storage = middleware.storage

    # 观测点 1（调用者）：insert 正常返回 doc_id
    doc_id = db.insert({'biz_key': 'k1'})
    assert doc_id == 1

    # 观测点 2（缓存）：数据在缓存，计数=1，尚未刷盘
    assert middleware._cache_modified_count == 1
    assert _cache_docs(middleware) == {
        '1': {'biz_key': 'k1'}
    }

    # 观测点 3（storage）：阈值未到，底层无 write
    assert storage.write_calls == 0
    assert _write_events(storage) == []

    # 显式 flush 触发一次底层写入
    middleware.flush()
    assert storage.write_calls == 1
    assert _write_events(storage) == [
        ('before_write', 1), ('after_persist', 1)
    ]
    assert middleware._cache_modified_count == 0

    db.close()

    # 观测点 4（新进程）：另一进程从磁盘读到同一记录
    seen = _read_in_subprocess(db_path)
    assert seen == [{'biz_key': 'k1'}]


# ---------------------------------------------------------------------------
# 2. 写前失败：缓存保留、磁盘未变
# 同进程可凭缓存恢复，新进程重开则不行
# ---------------------------------------------------------------------------

def test_pre_write_failure_keeps_cache_and_disk_untouched(db_path):
    db, middleware = _make_db(db_path)
    storage = middleware.storage
    storage.fail_on_write = 'pre_write'

    # 先攒阈值-1 条（计数 1、2），都在缓存
    db.insert({'biz_key': 'k1'})
    db.insert({'biz_key': 'k2'})
    assert storage.write_calls == 0

    # 第 3 条达阈值，自动 flush 时故障
    with pytest.raises(CommitError, match='pre-write failure'):
        db.insert({'biz_key': 'k3'})

    # 观测点 2：计数先增到 3，flush 失败未重置；
    # 缓存仍持全部 3 条——同进程唯一恢复依据。
    assert middleware._cache_modified_count == CACHE_SIZE
    assert set(_cache_docs(middleware)) == {'1', '2', '3'}

    # 观测点 3：底层 write 落盘前失败，文件仍为空
    assert storage.write_calls == 1
    assert _write_events(storage) == [('before_write', 1)]
    assert _raw_file_docs(db_path) == {}

    # 同进程恢复：清故障后 flush，缓存落盘，可安全重发
    storage.fail_on_write = None
    middleware.flush()
    assert storage.write_calls == 2
    assert set(_raw_file_docs(db_path)) == {'1', '2', '3'}

    db.close()

    # 观测点 4：新进程只看到 flush 后落盘的数据
    keys = {doc['biz_key'] for doc in _read_in_subprocess(db_path)}
    assert keys == {'k1', 'k2', 'k3'}


def test_pre_write_failure_without_flush_invisible_to_new_process(db_path):
    """flush 成功前进程崩溃，缓存消失，新进程看到漏记。"""
    db, middleware = _make_db(db_path)
    middleware.storage.fail_on_write = 'pre_write'

    db.insert({'biz_key': 'k1'})
    db.insert({'biz_key': 'k2'})
    with pytest.raises(CommitError):
        db.insert({'biz_key': 'k3'})

    # 不 flush，模拟进程死亡：丢弃内存缓存，只留文件
    del db, middleware

    # 新进程重开：三条全丢，缓存无法跨进程恢复
    assert _read_in_subprocess(db_path) == []


# ---------------------------------------------------------------------------
# 3. 已持久但确认丢失：磁盘已更新，盲目重发会重复
# ---------------------------------------------------------------------------

def test_post_write_failure_persisted_but_caller_sees_error(db_path):
    db, middleware = _make_db(db_path)
    storage = middleware.storage
    storage.fail_on_write = 'post_write'

    db.insert({'biz_key': 'k1'})
    db.insert({'biz_key': 'k2'})

    # 第 3 条触发自动 flush：已 fsync，但确认丢失
    with pytest.raises(CommitError, match='ack lost'):
        db.insert({'biz_key': 'k3'})

    # 与写前失败同异常，但观测点 3/4 结论相反：
    # flush 在 write 正常返回后才清零；这里 write 抛
    # 异常，计数同样停在 3、缓存同样 3 条——缓存层
    # 无法区分两种失败。
    assert middleware._cache_modified_count == CACHE_SIZE
    assert set(_cache_docs(middleware)) == {'1', '2', '3'}

    # 观测点 3：after_persist 证明数据已 fsync
    assert storage.write_calls == 1
    assert _write_events(storage) == [
        ('before_write', 1), ('after_persist', 1)
    ]
    assert set(_raw_file_docs(db_path)) == {'1', '2', '3'}

    # 清故障后关闭：close 再 flush（计数仍为 3），
    # 全量重写同样 3 行，内容幂等，干净退出。
    storage.fail_on_write = None
    db.close()

    # 观测点 4：新进程确认三条都在。若整批重发会拿
    # 到新 doc_id（4、5、6），产生重复业务键。
    keys = {doc['biz_key'] for doc in _read_in_subprocess(db_path)}
    assert keys == {'k1', 'k2', 'k3'}


def test_same_exception_requires_disk_observation(db_path):
    """同异常下，最小判据是文件/新进程里有没有数据。"""
    pre_path = db_path
    post_path = db_path + '.post'

    db_pre, mw_pre = _make_db(pre_path)
    mw_pre.storage.fail_on_write = 'pre_write'
    db_pre.insert({'biz_key': 'k1'})
    db_pre.insert({'biz_key': 'k2'})
    with pytest.raises(CommitError):
        db_pre.insert({'biz_key': 'k3'})

    db_post, mw_post = _make_db(post_path)
    mw_post.storage.fail_on_write = 'post_write'
    db_post.insert({'biz_key': 'k1'})
    db_post.insert({'biz_key': 'k2'})
    with pytest.raises(CommitError):
        db_post.insert({'biz_key': 'k3'})

    # 调用者视角：异常、缓存计数、缓存内容完全一致
    assert mw_pre._cache_modified_count == \
        mw_post._cache_modified_count
    assert _cache_docs(mw_pre) == _cache_docs(mw_post)

    # 最小判据：绕过缓存读磁盘（等价新进程重开）
    assert _raw_file_docs(pre_path) == {}
    assert set(_raw_file_docs(post_path)) == {'1', '2', '3'}

    mw_pre.storage.fail_on_write = None
    mw_post.storage.fail_on_write = None
    db_pre.close()
    db_post.close()


# ---------------------------------------------------------------------------
# 4. 重复业务键：TinyDB 只保证 doc_id 唯一，业务键不唯一
# ---------------------------------------------------------------------------

def test_duplicate_business_keys_on_resend(db_path):
    db, middleware = _make_db(db_path)

    # 第一批：业务键 k1 落盘（显式 flush，计数清零）
    db.insert({'biz_key': 'k1'})
    middleware.flush()
    assert middleware._cache_modified_count == 0
    first_id = 1

    # 没收到确认，重发同一业务键：doc_id 递增、键重复
    second_id = db.insert({'biz_key': 'k1'})
    middleware.flush()

    assert second_id == first_id + 1
    rows = _raw_file_docs(db_path)
    assert len(rows) == 2
    assert [row['biz_key'] for row in rows.values()] == [
        'k1', 'k1'
    ]

    # 业务键查询命中两条，“一批一次”语义被破坏
    assert db.count(where('biz_key') == 'k1') == 2

    db.close()


def test_idempotent_resend_with_business_key_guard(db_path):
    """安全重发：重发前用业务键做存在性检查。"""
    db, middleware = _make_db(db_path)

    def idempotent_insert(biz_key):
        if db.contains(where('biz_key') == biz_key):
            return None  # 已处理，跳过重发
        return db.insert({'biz_key': biz_key})

    assert idempotent_insert('k1') == 1
    middleware.flush()

    # 确认丢失后重发：闸门命中缓存已有记录，不产生重复
    assert idempotent_insert('k1') is None
    middleware.flush()
    assert db.count(where('biz_key') == 'k1') == 1

    db.close()


# ---------------------------------------------------------------------------
# 5. 计数轨迹：重复业务键 + 一次显式 flush，阈值前后观测
# ---------------------------------------------------------------------------

def test_counter_trajectory_with_duplicate_key_and_flush(db_path):
    db, middleware = _make_db(db_path)
    storage = middleware.storage
    trace = []

    def snapshot(tag):
        cache_ids = (
            sorted(_cache_docs(middleware).keys(), key=int)
            if middleware.cache else []
        )
        disk_ids = (
            sorted(_raw_file_docs(db_path).keys(), key=int)
            if os.path.getsize(db_path) else []
        )
        trace.append((
            tag,
            middleware._cache_modified_count,
            storage.write_calls,
            cache_ids,
            disk_ids,
        ))

    snapshot('open')

    db.insert({'biz_key': 'k1'})  # 计数 1
    snapshot('insert k1 #1')
    db.insert({'biz_key': 'k2'})  # 计数 2
    snapshot('insert k2')
    db.insert({'biz_key': 'k1'})  # 重复键；计数 3 自动 flush
    snapshot('insert duplicate k1 (auto-flush)')

    # 自动 flush 后：计数清零，底层写 1 次，磁盘 3 条
    assert middleware._cache_modified_count == 0
    assert storage.write_calls == 1
    assert len(_raw_file_docs(db_path)) == 3

    db.insert({'biz_key': 'k3'})  # 计数 1
    snapshot('insert k3')
    middleware.flush()            # 显式 flush：底层再写 1 次
    snapshot('manual flush')

    assert middleware._cache_modified_count == 0
    assert storage.write_calls == 2
    assert len(_raw_file_docs(db_path)) == 4

    db.close()
    snapshot('close')

    # 轨迹：(标签, 缓存计数, 写次数, 缓存doc_id, 磁盘doc_id)
    # close 会把 cache 置 None，故末行缓存列为 []
    assert trace == [
        ('open', 0, 0, [], []),
        ('insert k1 #1', 1, 0, ['1'], []),
        ('insert k2', 2, 0, ['1', '2'], []),
        (
            'insert duplicate k1 (auto-flush)', 0, 1,
            ['1', '2', '3'], ['1', '2', '3'],
        ),
        (
            'insert k3', 1, 1,
            ['1', '2', '3', '4'], ['1', '2', '3'],
        ),
        (
            'manual flush', 0, 2,
            ['1', '2', '3', '4'], ['1', '2', '3', '4'],
        ),
        ('close', 0, 2, [], ['1', '2', '3', '4']),
    ]


# ---------------------------------------------------------------------------
# 6. 小状态机：穷举阈值前后每个写入位置的失败与重试
#
# 状态：缓存计数 n ∈ {0,1,2,3}（阈值=3，n=3 即触发 flush）。
# 阈值前（第 1、2 次，不刷盘）与阈值处（第 3 次，刷盘）
# 分别注入 pre_write / post_write / 无故障，再各做一次重试，
# 枚举 3×3×2 = 18 条轨迹，断言磁盘行数与业务键重复数。
# ---------------------------------------------------------------------------

FAILURE_MODES = (None, 'pre_write', 'post_write')
RETRY_DECISIONS = ('resend', 'skip')


def _drive_machine(db_path, fail_at_write, failure_mode, retry):
    """跑一条轨迹，返回 (行数, 重复键数, 失败时计数)。"""
    db, middleware = _make_db(db_path)
    storage = middleware.storage
    failed_cache_count = None

    for write_index in range(1, CACHE_SIZE + 1):
        if write_index == fail_at_write:
            storage.fail_on_write = failure_mode
        try:
            db.insert({'biz_key': f'k{write_index}'})
        except CommitError:
            failed_cache_count = \
                middleware._cache_modified_count
            storage.fail_on_write = None
            if retry == 'resend':
                if failure_mode == 'pre_write':
                    # 缓存仍在：先 flush 再补插失败的业务键
                    middleware.flush()
                db.insert({'biz_key': f'k{write_index}'})
            else:
                # 跳过重发：缓存脏数据随后统一 flush
                middleware.flush()
            break

    middleware.flush()
    db.close()

    rows = list(_raw_file_docs(db_path).values())
    keys = [row['biz_key'] for row in rows]
    duplicates = len(keys) - len(set(keys))
    return len(rows), duplicates, failed_cache_count


def test_state_machine_exhausts_failure_before_and_at_threshold(
    tmp_path
):
    records = []
    # fail_at_write=1,2 为阈值前，=3 为阈值处
    for fail_at_write, failure_mode, retry in product(
        (1, 2, 3), FAILURE_MODES, RETRY_DECISIONS
    ):
        name = f'f{fail_at_write}_{failure_mode}_{retry}.db'
        path = str(tmp_path / name)
        row_count, duplicates, failed_count = _drive_machine(
            path, fail_at_write, failure_mode, retry
        )
        records.append({
            'fail_at_write': fail_at_write,
            'failure_mode': failure_mode,
            'retry': retry,
            'failed_cache_count': failed_count,
            'disk_rows': row_count,
            'duplicate_keys': duplicates,
        })

    # 无故障基线：3 条写入、0 重复
    clean = [r for r in records if r['failure_mode'] is None]
    assert len(clean) == 6
    assert all(
        r['disk_rows'] == 3 and r['duplicate_keys'] == 0
        for r in clean
    )

    # 规律一：失败都延迟到第 3 次写入触发 flush 时暴露。
    # 阈值前（fail_at_write=1,2）的 insert 只进缓存，不抛
    # 异常；故异常时计数恒为阈值 3，不会在 1、2 时失败。
    failures = [r for r in records if r['failure_mode'] is not None]
    assert len(failures) == 12
    assert all(
        r['failed_cache_count'] == CACHE_SIZE for r in failures
    )

    # 规律二：最终行数只取决于重发还是跳过，与失败模式
    # （pre/post）和位置无关。缓存把两种失败在同进程内
    # 收敛成同一种可恢复状态。
    resend = [r for r in failures if r['retry'] == 'resend']
    skip = [r for r in failures if r['retry'] == 'skip']
    assert len(resend) == 6 and len(skip) == 6
    assert all(r['disk_rows'] == 4 and r['duplicate_keys'] == 1
               for r in resend)
    assert all(r['disk_rows'] == 3 and r['duplicate_keys'] == 0
               for r in skip)

    # 最早漏记：阈值处 pre_write + 跳过。同进程缓存还在，
    # flush 可找回（故最终仍 3 行）；进程一旦死亡即漏记
    # （见上文独立测试），仅凭缓存无法跨进程恢复。
    pre_at_threshold_skip = next(
        r for r in records
        if r['failure_mode'] == 'pre_write'
        and r['fail_at_write'] == 3
        and r['retry'] == 'skip'
    )
    assert (
        pre_at_threshold_skip['failed_cache_count'] == CACHE_SIZE
    )

    # 最早重计：阈值处 post_write + 重发。第 3 条已 fsync，
    # 重发又插同业务键，磁盘 4 行、1 个重复键。pre_write
    # 重发在同进程内结果相同，二者只能靠失败瞬间的磁盘
    # 观测点区分，见 test_same_exception...。
    post_at_threshold_resend = next(
        r for r in records
        if r['failure_mode'] == 'post_write'
        and r['fail_at_write'] == 3
        and r['retry'] == 'resend'
    )
    assert post_at_threshold_resend['disk_rows'] == 4
    assert post_at_threshold_resend['duplicate_keys'] == 1
    assert (
        post_at_threshold_resend['failed_cache_count']
        == CACHE_SIZE
    )

    # 对照：阈值处 post_write + 跳过，磁盘保持 3 行、0 重复
    post_at_threshold_skip = next(
        r for r in records
        if r['failure_mode'] == 'post_write'
        and r['fail_at_write'] == 3
        and r['retry'] == 'skip'
    )
    assert post_at_threshold_skip['disk_rows'] == 3
    assert post_at_threshold_skip['duplicate_keys'] == 0


def test_earliest_double_count_and_miss_traces_are_distinct(tmp_path):
    """固定两条最早错误轨迹，防止重构改变边界行为。"""
    # 重计：post_write @ 阈值写入，重发 -> 重复业务键
    double_path = str(tmp_path / 'double.db')
    rows, duplicates, _ = _drive_machine(
        double_path, 3, 'post_write', 'resend'
    )
    assert (rows, duplicates) == (4, 1)

    # 漏记：pre_write @ 阈值写入，跳过，缓存随进程丢失
    miss_path = str(tmp_path / 'miss.db')
    db, middleware = _make_db(miss_path)
    middleware.storage.fail_on_write = 'pre_write'
    db.insert({'biz_key': 'k1'})
    db.insert({'biz_key': 'k2'})
    with pytest.raises(CommitError):
        db.insert({'biz_key': 'k3'})
    # 进程崩溃：缓存消失，未 flush 的不可恢复
    del db, middleware

    assert _read_in_subprocess(miss_path) == []


# ---------------------------------------------------------------------------
# 7. MemoryStorage 对照：无第二进程可核对的副本
# ---------------------------------------------------------------------------

def test_memory_storage_has_no_disk_tier_to_arbitrate():
    """纯内存 storage 没有观测点 4，无法靠外部状态仲裁。"""
    middleware = CachingMiddleware(MemoryStorage)
    middleware.WRITE_CACHE_SIZE = CACHE_SIZE
    middleware()
    underlying = middleware.storage
    underlying.memory = None

    for key in ('k1', 'k2'):
        middleware.write(
            {'_default': {'1': {'biz_key': key}}}
        )
    assert middleware._cache_modified_count == 2
    assert underlying.memory is None  # 阈值未到，底层无副本

    # 进程死亡等价于 middleware 被回收：无磁盘可重开核对，
    # 缓存里的两次写入彻底丢失，无法判断该重发哪一批。
    assert middleware.cache is not None
