"""定位撤销：撤销尚未继续使用的最近一次定位（操作 ID / 恢复起点与跳过）。"""

import math
import sqlite3

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """每个测试用独立 SQLite，并重新初始化模块级 store。"""
    db_path = tmp_path / "undo.db"
    monkeypatch.setenv("TEMPO_DB_PATH", str(db_path))

    import app.main as main
    from app.db import Store

    monkeypatch.setattr(main, "store", Store(str(db_path)))
    yield db_path


from app.main import app  # noqa: E402

client = TestClient(app)

# 120 BPM、4/4、PPQ=480：一拍 0.5s、一小节 2.0s
SCORE_BODY = {
    "ppq": 480,
    "time_signatures": [],
    "tempo_points": [{"position": "1:1:0", "bpm": 120, "curve": "step"}],
    "anchors": [{"position": "1:1:0", "locked": True}],
}
SLOW_POINTS = [{"position": "1:1:0", "bpm": 60, "curve": "step"}]


def create_score():
    resp = client.post("/scores", json=SCORE_BODY)
    assert resp.status_code == 201, resp.text
    return resp.json()


def create_rehearsal(score_id, cues):
    resp = client.post("/rehearsals", json={"score_id": score_id, "cues": cues})
    assert resp.status_code == 201, resp.text
    return resp.json()


def claim(rid, elapsed):
    return client.post(
        f"/rehearsals/{rid}/claims", json={"elapsed_seconds": elapsed}
    )


def locate(rid, elapsed, position):
    return client.post(
        f"/rehearsals/{rid}/locate",
        json={"elapsed_seconds": elapsed, "position": position},
    )


def undo(rid, operation_id):
    return client.post(
        f"/rehearsals/{rid}/locates/{operation_id}/undo", json={}
    )


def statuses(rid):
    got = client.get(f"/rehearsals/{rid}").json()
    return got, {c["text"]: c["status"] for c in got["cues"]}


def three_cue_rehearsal():
    """B=2:1:0(2.0s)、C=3:1:0(4.0s)、E=5:1:0(8.0s) 的排练。"""
    score = create_score()
    r = create_rehearsal(
        score["id"],
        [
            {"position": "2:1:0", "text": "B"},
            {"position": "3:1:0", "text": "C"},
            {"position": "5:1:0", "text": "E"},
        ],
    )
    return score, r


# ---------------------------------------------------------------------- #
# 定位响应新增操作 ID（原有字段保持不变）
# ---------------------------------------------------------------------- #
def test_locate_response_includes_operation_id_and_keeps_fields():
    _, r = three_cue_rehearsal()
    claim(r["id"], 2.0)  # B 已发，进度 2.0
    resp = locate(r["id"], 2.0, "4:1:0")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # 新增字段
    assert isinstance(data["operation_id"], str) and len(data["operation_id"]) == 32
    # 原有字段保持不变
    assert data["id"] == r["id"]
    assert data["elapsed_seconds"] == 2.0
    assert data["revision"] == 0
    assert data["origin"] == {
        "position": "4:1:0",
        "total_ticks": 5760,
        "elapsed_seconds": 2.0,
    }
    assert [s["text"] for s in data["skipped"]] == ["C"]
    assert len(data["cues"]) == 3


def test_consecutive_locates_get_distinct_operation_ids():
    _, r = three_cue_rehearsal()
    claim(r["id"], 2.0)
    op1 = locate(r["id"], 2.0, "4:1:0").json()["operation_id"]
    claim(r["id"], 4.0)  # E 发放，进度 4.0
    op2 = locate(r["id"], 4.0, "6:1:0").json()["operation_id"]
    assert op1 != op2


# ---------------------------------------------------------------------- #
# 撤销：恢复定位前起点与该次跳过的提示
# ---------------------------------------------------------------------- #
def test_undo_restores_origin_and_skipped_cues_become_pending():
    _, r = three_cue_rehearsal()
    rid = r["id"]
    claim(rid, 2.0)  # B 已发；C/E 待发
    op = locate(rid, 2.0, "4:1:0").json()["operation_id"]

    resp = undo(rid, op)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["operation_id"] == op
    assert data["revision"] == 0
    # 起点恢复为定位前的 1:1:0 / 0.0s
    assert data["origin"] == {
        "position": "1:1:0",
        "total_ticks": 0,
        "elapsed_seconds": 0.0,
    }
    # 该次跳过的 C 恢复待发；B 已发不动
    assert [c["text"] for c in data["restored"]] == ["C"]
    assert data["restored"][0]["total_ticks"] == 3840
    assert {c["text"]: c["status"] for c in data["cues"]} == {
        "B": "delivered",
        "C": "pending",
        "E": "pending",
    }

    got, st = statuses(rid)
    assert st == {"B": "delivered", "C": "pending", "E": "pending"}
    assert got["origin"]["position"] == "1:1:0"
    # 领取进度不变（撤销不回退进度）
    assert got["last_elapsed_seconds"] == 2.0


def test_restored_pending_cue_is_timed_by_restored_origin_and_current_map():
    # 撤销后待发提示按“恢复的起点 + 当前速度图”定时：
    # C(3:1:0) 恢复待发后，在原起点（1:1:0/0.0）下触发于 4.0s。
    _, r = three_cue_rehearsal()
    rid = r["id"]
    claim(rid, 2.0)
    op = locate(rid, 2.0, "4:1:0").json()["operation_id"]
    undo(rid, op)

    assert claim(rid, 2.0).json()["cues"] == []  # 2.0s 时 C 未到点
    due = claim(rid, 4.0).json()
    assert [c["text"] for c in due["cues"]] == ["C"]
    assert math.isclose(due["cues"][0]["trigger_seconds"], 4.0)
    assert due["cues"][0]["revision"] == 0


def test_undo_restores_to_origin_of_previous_locate_in_chain():
    # 连续两次定位都未继续使用时，先撤销第二次，恢复到第一次的起点。
    score = create_score()
    r = create_rehearsal(
        score["id"],
        [
            {"position": "2:1:0", "text": "B"},
            {"position": "3:1:0", "text": "C"},
            {"position": "5:1:0", "text": "E"},
            {"position": "7:1:0", "text": "G"},
        ],
    )
    rid = r["id"]
    op1 = locate(rid, 0.0, "2:1:0").json()["operation_id"]  # B 在目标处不跳过
    # 目标 2:1:0：B 位于目标处不跳过；C/E/G 在其后待发
    got, st = statuses(rid)
    assert st == {"B": "pending", "C": "pending", "E": "pending", "G": "pending"}
    assert got["origin"]["position"] == "2:1:0"

    op2 = locate(rid, 0.0, "6:1:0").json()["operation_id"]
    # 此时按坐标：B(2:1:0)、C(3:1:0)、E(5:1:0) 都在 6:1:0 之前 -> 跳过
    _, st = statuses(rid)
    assert st == {
        "B": "skipped",
        "C": "skipped",
        "E": "skipped",
        "G": "pending",
    }

    # 撤销第二次定位：回到第一次定位后的起点 2:1:0，第二次跳过的全部恢复
    resp = undo(rid, op2)
    assert resp.status_code == 200
    data = resp.json()
    assert data["origin"]["position"] == "2:1:0"
    assert sorted(c["text"] for c in data["restored"]) == ["B", "C", "E"]
    assert {c["text"]: c["status"] for c in data["cues"]} == {
        "B": "pending",
        "C": "pending",
        "E": "pending",
        "G": "pending",
    }

    # 再撤销第一次定位：回到未定位起点 1:1:0（第一次没有跳过任何提示）
    resp2 = undo(rid, op1)
    assert resp2.status_code == 200
    assert resp2.json()["origin"]["position"] == "1:1:0"
    assert resp2.json()["restored"] == []


def test_undo_backward_locate_restores_its_own_origin_only():
    # 普通回跳本身也是一次定位，撤销它只恢复回跳前的起点，
    # 不恢复更早定位跳过的提示。
    _, r = three_cue_rehearsal()
    rid = r["id"]
    claim(rid, 2.0)
    locate(rid, 2.0, "4:1:0")  # C 跳过，起点 4:1:0/2.0
    claim(rid, 4.0)  # E 发放，进度 4.0
    back = locate(rid, 4.0, "2:1:0").json()  # 回跳（不产生跳过）
    assert back["skipped"] == []

    resp = undo(rid, back["operation_id"])
    assert resp.status_code == 200
    data = resp.json()
    assert data["origin"]["position"] == "4:1:0"
    assert data["restored"] == []
    # C 仍是早先那次定位跳过的，保持原状
    assert {c["text"]: c["status"] for c in data["cues"]} == {
        "B": "delivered",
        "C": "skipped",
        "E": "delivered",
    }


# ---------------------------------------------------------------------- #
# 过期/已撤销：409 且不改状态
# ---------------------------------------------------------------------- #
def test_undo_twice_is_409_and_state_unchanged():
    _, r = three_cue_rehearsal()
    rid = r["id"]
    op = locate(rid, 0.0, "4:1:0").json()["operation_id"]
    assert undo(rid, op).status_code == 200

    resp = undo(rid, op)
    assert resp.status_code == 409
    detail = resp.json()["detail"][0]
    assert detail["reason"] == "already_undone"
    assert detail["operation_id"] == op

    got, st = statuses(rid)
    assert st == {"B": "pending", "C": "pending", "E": "pending"}
    assert got["origin"]["position"] == "1:1:0"


def test_undo_after_another_locate_is_409_superseded():
    # 连续定位只能撤销最后一次：撤销第一次 -> 409 superseded，状态不变。
    _, r = three_cue_rehearsal()
    rid = r["id"]
    op1 = locate(rid, 0.0, "3:1:0").json()["operation_id"]  # B 跳过
    locate(rid, 0.0, "5:1:0")  # 再定位（C 在 5:1:0 目标处不跳；E 目标处不跳）

    resp = undo(rid, op1)
    assert resp.status_code == 409
    assert resp.json()["detail"][0]["reason"] == "superseded"

    # 状态不变：起点仍是第二次定位的 5:1:0；B 仍跳过
    got, st = statuses(rid)
    assert got["origin"]["position"] == "5:1:0"
    assert st["B"] == "skipped"


def test_after_undo_new_locate_can_be_undone_and_old_skip_stays():
    # 撤销后再次定位：新操作可撤销；它只恢复自己的跳过，不碰更早状态。
    _, r = three_cue_rehearsal()
    rid = r["id"]
    op1 = locate(rid, 0.0, "3:1:0").json()["operation_id"]  # B 跳过
    undo(rid, op1)  # 恢复 B
    op2 = locate(rid, 0.0, "3:1:0").json()["operation_id"]  # B 再次跳过
    _, st = statuses(rid)
    assert st == {"B": "skipped", "C": "pending", "E": "pending"}

    resp = undo(rid, op2)
    assert resp.status_code == 200
    data = resp.json()
    assert [c["text"] for c in data["restored"]] == ["B"]
    _, st = statuses(rid)
    assert st == {"B": "pending", "C": "pending", "E": "pending"}


def test_undo_after_progress_advances_is_409():
    _, r = three_cue_rehearsal()
    rid = r["id"]
    claim(rid, 2.0)  # B 发放，进度 2.0
    op = locate(rid, 2.0, "4:1:0").json()["operation_id"]  # C 跳过
    # 进度前进到 3.0：E 在新起点下触发于 4.0，尚未发放（纯进度前进）
    assert claim(rid, 3.0).json()["cues"] == []

    resp = undo(rid, op)
    assert resp.status_code == 409
    detail = resp.json()["detail"][0]
    assert detail["reason"] == "progress_advanced"
    assert detail["elapsed_seconds"] == 3.0
    assert detail["locate_elapsed_seconds"] == 2.0

    # 状态不变：起点 4:1:0、C 仍跳过
    got, st = statuses(rid)
    assert got["origin"]["position"] == "4:1:0"
    assert got["last_elapsed_seconds"] == 3.0
    assert st == {"B": "delivered", "C": "skipped", "E": "pending"}


def test_undo_after_delivery_is_409_even_without_progress_change():
    # 定位后有提示发放（且发放发生在定位提交的同一秒、进度未变）也不可撤销。
    score = create_score()
    r = create_rehearsal(
        score["id"],
        [
            {"position": "2:1:0", "text": "B"},  # 已发
            {"position": "4:1:0", "text": "T"},  # 定位目标处，同秒可领
            {"position": "5:1:0", "text": "E"},
        ],
    )
    rid = r["id"]
    claim(rid, 2.0)  # B 发放，进度 2.0
    op = locate(rid, 2.0, "4:1:0").json()["operation_id"]
    due = claim(rid, 2.0).json()["cues"]  # 同秒领取目标处 T
    assert [c["text"] for c in due] == ["T"]

    resp = undo(rid, op)
    assert resp.status_code == 409
    detail = resp.json()["detail"][0]
    assert detail["reason"] == "cue_delivered"
    assert detail["delivered_count"] == 2
    assert detail["locate_delivered_count"] == 1
    got, st = statuses(rid)
    assert st == {"B": "delivered", "T": "delivered", "E": "pending"}


def test_undo_after_progress_advances_without_delivery_is_409():
    # 进度前进但没有任何新提示到点（前进到未达下一触发点）仍不可撤销。
    score = create_score()
    r = create_rehearsal(
        score["id"], [{"position": "5:1:0", "text": "E"}]  # 8.0s
    )
    rid = r["id"]
    op = locate(rid, 0.0, "2:1:0").json()["operation_id"]  # 无跳过（E 在后）
    claim(rid, 1.0)  # 进度 1.0，E 未到点
    resp = undo(rid, op)
    assert resp.status_code == 409
    detail = resp.json()["detail"][0]
    assert detail["reason"] == "progress_advanced"
    assert detail["elapsed_seconds"] == 1.0
    assert detail["locate_elapsed_seconds"] == 0.0

    got = client.get(f"/rehearsals/{rid}").json()
    assert got["origin"]["position"] == "2:1:0"
    assert got["last_elapsed_seconds"] == 1.0


def test_undo_after_annotation_is_409():
    _, r = three_cue_rehearsal()
    rid = r["id"]
    op = locate(rid, 0.0, "4:1:0").json()["operation_id"]  # B、C 跳过
    ann = client.post(
        f"/rehearsals/{rid}/annotations",
        json={"elapsed_seconds": 0.0, "text": "这里有问题"},
    )
    assert ann.status_code == 201, ann.text

    resp = undo(rid, op)
    assert resp.status_code == 409
    assert resp.json()["detail"][0]["reason"] == "annotation_registered"

    got, st = statuses(rid)
    assert got["origin"]["position"] == "4:1:0"
    assert st["B"] == "skipped" and st["C"] == "skipped"


def test_undo_after_publish_is_409():
    score, r = three_cue_rehearsal()
    rid = r["id"]
    op = locate(rid, 0.0, "4:1:0").json()["operation_id"]  # B、C 跳过
    pub = client.post(
        f"/scores/{score['id']}/revisions/0/publish",
        json={"tempo_points": SLOW_POINTS},
    )
    assert pub.status_code == 200

    resp = undo(rid, op)
    assert resp.status_code == 409
    detail = resp.json()["detail"][0]
    assert detail["reason"] == "revision_published"
    assert detail["revision"] == 1
    assert detail["locate_revision"] == 0

    got, st = statuses(rid)
    assert got["origin"]["position"] == "4:1:0"
    assert st["B"] == "skipped" and st["C"] == "skipped"
    assert st["E"] == "pending"


def test_same_elapsed_claim_with_no_delivery_does_not_block_undo():
    # 定位后同一秒重复领取但没有新提示到点（进度不前进）不算“继续使用”。
    _, r = three_cue_rehearsal()
    rid = r["id"]
    claim(rid, 2.0)
    op = locate(rid, 2.0, "4:1:0").json()["operation_id"]  # C 跳过
    # 同秒重复领取：没有目标处提示 -> 无发放，进度仍 2.0
    assert claim(rid, 2.0).json()["cues"] == []
    resp = undo(rid, op)
    assert resp.status_code == 200
    _, st = statuses(rid)
    assert st == {"B": "delivered", "C": "pending", "E": "pending"}


# ---------------------------------------------------------------------- #
# 404：操作 ID 不存在 / 不属于该排练 / 排练不存在
# ---------------------------------------------------------------------- #
def test_undo_unknown_operation_is_404():
    _, r = three_cue_rehearsal()
    rid = r["id"]
    locate(rid, 0.0, "4:1:0")
    resp = undo(rid, "0123456789abcdef0123456789abcdef")
    assert resp.status_code == 404
    detail = resp.json()["detail"][0]
    assert detail["loc"] == ["path", "operation_id"]
    assert detail["operation_id"] == "0123456789abcdef0123456789abcdef"


def test_undo_operation_of_other_rehearsal_is_404():
    _, r1 = three_cue_rehearsal()
    _, r2 = three_cue_rehearsal()
    op = locate(r1["id"], 0.0, "4:1:0").json()["operation_id"]
    # 拿 r1 的操作 ID 去撤销 r2：按“操作不存在”处理 404
    resp = undo(r2["id"], op)
    assert resp.status_code == 404
    # r1 状态不受影响
    _, st = statuses(r1["id"])
    assert st["B"] == "skipped"


def test_undo_on_unknown_rehearsal_is_404():
    resp = undo("deadbeef", "0123456789abcdef0123456789abcdef")
    assert resp.status_code == 404
    assert resp.json()["detail"][0]["loc"] == ["path", "rehearsal_id"]


# ---------------------------------------------------------------------- #
# 持久化：撤销依据与恢复结果在重启后一致
# ---------------------------------------------------------------------- #
def test_locate_undo_state_survives_store_restart(fresh_db, monkeypatch):
    from app.db import Store

    db_path = fresh_db
    _, r = three_cue_rehearsal()
    rid = r["id"]
    claim(rid, 2.0)
    op = locate(rid, 2.0, "4:1:0").json()["operation_id"]  # C 跳过

    # 重启（模拟 Compose 不删卷的 down/up）
    import app.main as main

    monkeypatch.setattr(main, "store", Store(str(db_path)))

    # 操作仍可撤销，起点/跳过正常恢复
    resp = undo(rid, op)
    assert resp.status_code == 200, resp.text
    assert resp.json()["origin"]["position"] == "1:1:0"
    _, st = statuses(rid)
    assert st == {"B": "delivered", "C": "pending", "E": "pending"}

    # 恢复后 C 按恢复的起点在当前速度图（120 BPM）下 4.0s 触发
    due = claim(rid, 4.0).json()
    assert [c["text"] for c in due["cues"]] == ["C"]
    assert math.isclose(due["cues"][0]["trigger_seconds"], 4.0)


def test_undo_after_restart_still_rejects_stale_operation(fresh_db, monkeypatch):
    from app.db import Store

    db_path = fresh_db
    _, r = three_cue_rehearsal()
    rid = r["id"]
    op = locate(rid, 0.0, "4:1:0").json()["operation_id"]
    claim(rid, 2.0)  # 定位后进度前进（同时 B 已发放）

    import app.main as main

    monkeypatch.setattr(main, "store", Store(str(db_path)))
    resp = undo(rid, op)
    assert resp.status_code == 409
    assert resp.json()["detail"][0]["reason"] == "progress_advanced"


# ---------------------------------------------------------------------- #
# 旧版库迁移：无定位操作表的旧库打开后自动建表，定位/撤销仍工作
# ---------------------------------------------------------------------- #
def test_old_schema_db_creates_locate_tables_and_undo_works(tmp_path):
    import json
    from app.db import Store
    from app.main import _skip_before_target

    db_path = tmp_path / "old-undo.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE scores (id TEXT PRIMARY KEY, ppq INTEGER NOT NULL,
            time_signatures TEXT NOT NULL, tempo_points TEXT NOT NULL,
            revision INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE anchors (score_id TEXT NOT NULL, position_index INTEGER NOT NULL,
            position TEXT NOT NULL, locked INTEGER NOT NULL, total_ticks INTEGER NOT NULL,
            original_seconds REAL NOT NULL, current_seconds REAL NOT NULL,
            tick_seconds REAL NOT NULL, PRIMARY KEY (score_id, position_index));
        CREATE TABLE rehearsals (id TEXT PRIMARY KEY, score_id TEXT NOT NULL,
            last_elapsed REAL NOT NULL DEFAULT 0.0,
            origin_ticks INTEGER NOT NULL DEFAULT 0,
            origin_elapsed REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE rehearsal_cues (rehearsal_id TEXT NOT NULL, cue_index INTEGER NOT NULL,
            position TEXT NOT NULL, text TEXT NOT NULL, total_ticks INTEGER NOT NULL,
            PRIMARY KEY (rehearsal_id, cue_index));
        CREATE TABLE cue_deliveries (rehearsal_id TEXT NOT NULL, cue_index INTEGER NOT NULL,
            trigger_seconds REAL NOT NULL, revision INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL,
            delivered_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (rehearsal_id, cue_index));
        CREATE TABLE cue_skips (rehearsal_id TEXT NOT NULL, cue_index INTEGER NOT NULL,
            revision INTEGER NOT NULL, elapsed_seconds REAL NOT NULL,
            skipped_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (rehearsal_id, cue_index));
        CREATE TABLE rehearsal_annotations (rehearsal_id TEXT NOT NULL,
            seq INTEGER NOT NULL, annotation_id TEXT NOT NULL UNIQUE,
            text TEXT NOT NULL, elapsed_seconds REAL NOT NULL,
            position TEXT NOT NULL, total_ticks INTEGER NOT NULL,
            revision INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (rehearsal_id, seq));
        """
    )
    points = json.dumps(
        [{"position": "1:1:0", "bpm": 120, "curve": "step"}], ensure_ascii=False
    )
    conn.execute(
        "INSERT INTO scores VALUES ('s1', 480, '[]', ?, 0, datetime('now'))",
        (points,),
    )
    conn.execute(
        "INSERT INTO anchors VALUES ('s1', 0, '1:1:0', 1, 0, 0.0, 0.0, 0.001)"
    )
    conn.execute(
        "INSERT INTO rehearsals (id, score_id, last_elapsed) VALUES ('r1','s1', 0.0)"
    )
    conn.execute("INSERT INTO rehearsal_cues VALUES ('r1', 0, '2:1:0', 'B', 1920)")
    conn.commit()
    conn.close()

    store = Store(str(db_path))
    located = store.locate("r1", 0.0, 5760, _skip_before_target)  # 跳到 4:1:0
    op_id = located["operation_id"]
    assert isinstance(op_id, str) and len(op_id) == 32
    got = store.get_rehearsal("r1")
    assert got["cues"][0]["status"] == "skipped"

    undone = store.undo_locate("r1", op_id)
    assert undone["restored_origin_ticks"] == 0
    assert undone["restored_origin_elapsed_seconds"] == 0.0
    got = store.get_rehearsal("r1")
    assert got["origin_ticks"] == 0
    assert got["cues"][0]["status"] == "pending"


# ---------------------------------------------------------------------- #
# 并发交错：撤销与领取共用写锁串行化——要么撤销成功（提示待发、起点恢复），
# 要么领取先到（进度前进、可能发放），撤销 409 且状态不变，绝不部分恢复
# ---------------------------------------------------------------------- #
def test_concurrent_undo_and_claim_are_serialized(fresh_db):
    import threading

    from app.db import (
        LocateUndoConflict,
        Store,
        UNDO_REASON_PROGRESS,
    )
    from app.main import _due_cues

    db_path = fresh_db
    _, r = three_cue_rehearsal()
    rid = r["id"]
    # 定位到 4:1:0：C 跳过；E(5:1:0) 在新起点下触发于 4.0s
    claim(rid, 2.0)  # B 发放，进度 2.0
    op_id = locate(rid, 2.0, "4:1:0").json()["operation_id"]

    n_workers = 8
    stores = [Store(str(db_path)) for _ in range(n_workers)]
    barrier = threading.Barrier(n_workers)
    outcomes: list[str] = []
    errors: list[Exception] = []

    def worker(i, store):
        try:
            barrier.wait(timeout=10)
            if i % 2 == 0:
                try:
                    store.undo_locate(rid, op_id)
                    outcomes.append("undone")
                except LocateUndoConflict as exc:
                    assert exc.reason == UNDO_REASON_PROGRESS
                    outcomes.append("rejected")
            else:
                # 领取到 4.0s：E 在定位后起点下到点；进度必然前进
                store.claim_cues(
                    rid, 4.0, lambda s, p: _due_cues(s, p, 4.0)
                )
        except Exception as exc:  # pragma: no cover - 失败时下面断言暴露
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(i, s))
        for i, s in enumerate(stores)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors

    # 二者必居其一，且撤销最多成功一次
    assert outcomes.count("undone") <= 1
    got = client.get(f"/rehearsals/{rid}").json()
    c = next(c for c in got["cues"] if c["text"] == "C")
    if outcomes.count("undone"):
        # 撤销先提交：C 恢复待发，起点回到 1:1:0；随后的领取在旧起点下
        # 把 C（4.0s）与 E（8.0s，未到点）处理为仅 C 发放
        assert c["status"] == "pending" or c["status"] == "delivered"
        # 起点必然是二者之一（撤销先到后，没有后续定位，故为 1:1:0）
        assert got["origin"]["position"] == "1:1:0"
    else:
        # 领取先提交：全部撤销 409，起点仍 4:1:0、C 仍跳过、E 已发
        assert got["origin"]["position"] == "4:1:0"
        assert c["status"] == "skipped"
        e = next(c for c in got["cues"] if c["text"] == "E")
        assert e["status"] == "delivered"
