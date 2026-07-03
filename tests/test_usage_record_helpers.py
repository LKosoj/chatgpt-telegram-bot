import json
import os
import importlib.machinery
import importlib.util
from datetime import date
from pathlib import Path
import sqlite3
import sys
import threading
import types

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.usage_tracker import UsageTracker
from bot.utils import (
    make_usage_tracker,
    record_chat_tokens,
    record_image_request,
    record_tts_request,
    record_transcription_seconds,
    record_vision_tokens,
)


def _make_config(allowed="42"):
    return {
        "allowed_user_ids": allowed,
        "token_price": 0.002,
        "image_prices": [0.016, 0.018, 0.02],
        "vision_token_price": 0.01,
        "tts_prices": [0.015, 0.030],
        "transcription_price": 0.006,
    }


def _make_tracker(tmp_path, user_id, name):
    return UsageTracker(user_id, name, logs_dir=str(tmp_path))


def _usage_db_rows(tmp_path, query, params=()):
    conn = sqlite3.connect(tmp_path / "usage.sqlite3")
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def test_record_chat_tokens_charges_guest_for_non_allowed(tmp_path):
    user = _make_tracker(tmp_path, 99, "stranger")
    guests = _make_tracker(tmp_path, "guests", "guests")
    usage = {99: user, "guests": guests}

    assert record_chat_tokens(usage, _make_config(allowed="42"), 99, 1000) is True

    assert sum(user.usage["usage_history"]["chat_tokens"].values()) == 1000
    assert sum(guests.usage["usage_history"]["chat_tokens"].values()) == 1000


def test_record_chat_tokens_skips_guest_for_allowed(tmp_path):
    user = _make_tracker(tmp_path, 42, "owner")
    guests = _make_tracker(tmp_path, "guests", "guests")
    usage = {42: user, "guests": guests}

    assert record_chat_tokens(usage, _make_config(allowed="42"), 42, 500) is True

    assert sum(user.usage["usage_history"]["chat_tokens"].values()) == 500
    assert guests.usage["usage_history"]["chat_tokens"] == {}


def test_record_chat_tokens_returns_false_on_zero(tmp_path):
    user = _make_tracker(tmp_path, 42, "owner")
    usage = {42: user}
    assert record_chat_tokens(usage, _make_config(), 42, 0) is False


def test_record_chat_tokens_rejects_negative_and_bool(tmp_path):
    user = _make_tracker(tmp_path, 42, "owner")
    usage = {42: user}

    assert record_chat_tokens(usage, _make_config(), 42, -1) is False
    assert record_chat_tokens(usage, _make_config(), 42, True) is False
    assert user.usage["usage_history"]["chat_tokens"] == {}


def test_record_chat_tokens_returns_false_for_unknown_user(tmp_path):
    usage = {}
    assert record_chat_tokens(usage, _make_config(), 7, 10) is False


def test_usage_tracker_uses_init_prices_when_add_called_without_price(tmp_path):
    tracker = UsageTracker(
        42, "user", logs_dir=str(tmp_path),
        token_price=0.5,
    )
    tracker.add_chat_tokens(1000)
    assert tracker.usage["current_cost"]["day"] == 0.5  # 1000 * 0.5 / 1000


def test_usage_tracker_records_chat_tokens_in_sqlite(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path), token_price=0.5)

    tracker.add_chat_tokens(1000)

    events = _usage_db_rows(
        tmp_path,
        "SELECT subject_id, subject_type, event_type, amount, cost FROM usage_events",
    )
    assert events == [{
        "subject_id": "42",
        "subject_type": "user",
        "event_type": "chat_tokens",
        "amount": 1000.0,
        "cost": 0.5,
    }]
    aggregates = _usage_db_rows(
        tmp_path,
        "SELECT chat_tokens, cost FROM usage_daily_aggregates WHERE subject_id = ?",
        ("42",),
    )
    assert aggregates == [{"chat_tokens": 1000, "cost": 0.5}]


def test_usage_tracker_writes_legacy_snapshot_after_sqlite_event(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path), token_price=0.5)

    tracker.add_chat_tokens(1000)

    snapshot = json.loads((tmp_path / "42.json").read_text(encoding="utf-8"))
    assert snapshot["usage_history"]["chat_tokens"] == tracker.usage["usage_history"]["chat_tokens"]
    assert snapshot["current_cost"]["all_time"] == 0.5


def test_usage_tracker_does_not_reimport_own_snapshot_on_restart(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path), token_price=0.5)
    tracker.add_chat_tokens(1000)

    restarted = UsageTracker(42, "owner", logs_dir=str(tmp_path), token_price=0.5)

    assert restarted.usage["current_cost"]["all_time"] == 0.5
    assert sum(restarted.usage["usage_history"]["chat_tokens"].values()) == 1000
    events = _usage_db_rows(tmp_path, "SELECT event_type, amount, cost FROM usage_events ORDER BY id")
    assert events == [{"event_type": "chat_tokens", "amount": 1000.0, "cost": 0.5}]
    imports = _usage_db_rows(tmp_path, "SELECT source_path FROM usage_imports")
    assert imports == [{"source_path": str(tmp_path / "42.json")}]


def test_usage_tracker_concurrent_instances_do_not_overwrite_newer_snapshot(tmp_path, monkeypatch):
    first = UsageTracker(42, "owner", logs_dir=str(tmp_path), token_price=0.5)
    second = UsageTracker(42, "owner", logs_dir=str(tmp_path), token_price=0.5)
    first_ready_to_write = threading.Event()
    second_finished = threading.Event()
    original_write = first._write_usage

    def delayed_first_write(*args, **kwargs):
        first_ready_to_write.set()
        second_finished.wait(timeout=0.2)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(first, "_write_usage", delayed_first_write)

    first_thread = threading.Thread(target=lambda: first.add_chat_tokens(1))
    first_thread.start()
    assert first_ready_to_write.wait(timeout=1)

    def record_second():
        second.add_chat_tokens(1)
        second_finished.set()

    second_thread = threading.Thread(target=record_second)
    second_thread.start()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    snapshot = json.loads((tmp_path / "42.json").read_text(encoding="utf-8"))
    assert sum(snapshot["usage_history"]["chat_tokens"].values()) == 2


def test_usage_tracker_imports_legacy_json_once(tmp_path):
    legacy = {
        "user_name": "owner",
        "current_cost": {"day": 0, "week": 0, "month": 0, "all_time": 11.25, "last_update": "2026-01-01"},
        "usage_history": {
            "chat_tokens": {"2026-01-01": 1000},
            "vision_tokens": {"2026-01-01": 2000},
            "number_images": {"2026-01-01": [1, 0, 2]},
            "tts_characters": {"tts-1-hd": {"2026-01-01": 3000}},
            "transcription_seconds": {"2026-01-01": 60},
        },
    }
    (tmp_path / "42.json").write_text(json.dumps(legacy), encoding="utf-8")

    tracker = UsageTracker(
        42,
        "owner",
        logs_dir=str(tmp_path),
        token_price=0.5,
        vision_token_price=0.5,
        image_prices=[1.0, 2.0, 3.0],
        tts_prices=[0.5, 0.75],
        transcription_price=0.5,
    )
    UsageTracker(
        42,
        "owner",
        logs_dir=str(tmp_path),
        token_price=0.5,
        vision_token_price=0.5,
        image_prices=[1.0, 2.0, 3.0],
        tts_prices=[0.5, 0.75],
        transcription_price=0.5,
    )

    assert tracker.usage["usage_history"]["chat_tokens"] == {"2026-01-01": 1000}
    assert tracker.usage["usage_history"]["number_images"] == {"2026-01-01": [1, 0, 2]}
    events = _usage_db_rows(tmp_path, "SELECT event_type FROM usage_events ORDER BY id")
    assert [row["event_type"] for row in events] == [
        "chat_tokens",
        "vision_tokens",
        "image_request",
        "image_request",
        "transcription_seconds",
        "tts_characters",
    ]
    imports = _usage_db_rows(tmp_path, "SELECT source_path FROM usage_imports")
    assert len(imports) == 1


def test_usage_tracker_import_preserves_legacy_all_time_cost(tmp_path):
    legacy = {
        "user_name": "owner",
        "current_cost": {"day": 0, "week": 0, "month": 0, "all_time": 99.0, "last_update": "2026-01-01"},
        "usage_history": {
            "chat_tokens": {"2026-01-01": 1000},
        },
    }
    (tmp_path / "42.json").write_text(json.dumps(legacy), encoding="utf-8")

    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path), token_price=0.5)

    assert tracker.get_current_cost()["cost_all_time"] == 99.0
    rows = _usage_db_rows(
        tmp_path,
        "SELECT event_type, event_date, cost FROM usage_events ORDER BY id",
    )
    assert rows == [
        {"event_type": "chat_tokens", "event_date": "2026-01-01", "cost": 0.5},
        {"event_type": "cost_adjustment", "event_date": "0001-01-01", "cost": 98.5},
    ]


def test_usage_tracker_recovers_corrupt_json_by_renaming_and_recreating(tmp_path):
    user_file = tmp_path / "42.json"
    user_file.write_text("{not-json", encoding="utf-8")

    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))

    assert tracker.usage["user_name"] == "owner"
    assert tracker.usage["usage_history"]["chat_tokens"] == {}
    assert json.loads(user_file.read_text(encoding="utf-8"))["user_name"] == "owner"
    corrupt_files = list(tmp_path.glob("42.json.corrupt*"))
    assert len(corrupt_files) == 1
    assert corrupt_files[0].read_text(encoding="utf-8") == "{not-json"


def test_usage_tracker_recovers_invalid_utf8_json_bytes(tmp_path):
    user_file = tmp_path / "42.json"
    user_file.write_bytes(b"\xff\xfe\x00")

    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))

    assert tracker.usage["user_name"] == "owner"
    assert json.loads(user_file.read_text(encoding="utf-8"))["user_name"] == "owner"
    corrupt_files = list(tmp_path.glob("42.json.corrupt*"))
    assert len(corrupt_files) == 1
    assert corrupt_files[0].read_bytes() == b"\xff\xfe\x00"


def test_usage_tracker_preserves_file_mode_when_recovering_corrupt_json(tmp_path):
    user_file = tmp_path / "42.json"
    user_file.write_text("{not-json", encoding="utf-8")
    user_file.chmod(0o664)

    UsageTracker(42, "owner", logs_dir=str(tmp_path))

    assert (user_file.stat().st_mode & 0o777) == 0o664


def test_usage_tracker_concurrent_corrupt_json_recovery_keeps_valid_file(tmp_path):
    user_file = tmp_path / "42.json"
    user_file.write_text("{not-json", encoding="utf-8")
    trackers = []

    def load_tracker():
        trackers.append(UsageTracker(42, "owner", logs_dir=str(tmp_path)))

    threads = [threading.Thread(target=load_tracker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(trackers) == 2
    assert json.loads(user_file.read_text(encoding="utf-8"))["user_name"] == "owner"
    assert len(list(tmp_path.glob("42.json.corrupt*"))) == 1
    assert all(tracker.usage["user_name"] == "owner" for tracker in trackers)


def test_usage_tracker_writes_atomically_in_same_directory(tmp_path, monkeypatch):
    logs_dir = tmp_path / "nested" / "usage"
    tracker = UsageTracker(42, "owner", logs_dir=str(logs_dir))
    real_replace = os.replace
    calls = []

    def capture_replace(src, dst):
        src_path = Path(src)
        dst_path = Path(dst)
        calls.append((src_path, dst_path))
        assert src_path.parent == logs_dir
        assert dst_path == Path(tracker.user_file)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", capture_replace)

    tracker.usage["current_cost"]["day"] = 1.0
    tracker._write_usage()

    assert calls
    assert Path(tracker.user_file).is_file()


def test_usage_tracker_preserves_existing_file_mode_on_atomic_replace(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    user_file = Path(tracker.user_file)
    tracker._write_usage()
    user_file.chmod(0o664)

    tracker.usage["current_cost"]["day"] = 1.0
    tracker._write_usage()

    assert (user_file.stat().st_mode & 0o777) == 0o664


def test_usage_tracker_failed_json_dump_leaves_existing_file_intact(tmp_path, monkeypatch):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    user_file = Path(tracker.user_file)
    old_content = '{"user_name": "owner", "current_cost": {"last_update": "2026-01-01"}, "usage_history": {}}'
    user_file.write_text(old_content, encoding="utf-8")

    def fail_dump(*args, **kwargs):
        raise RuntimeError("dump failed")

    monkeypatch.setattr(json, "dump", fail_dump)

    with pytest.raises(RuntimeError, match="dump failed"):
        tracker._write_usage()

    assert user_file.read_text(encoding="utf-8") == old_content


def test_get_current_cost_uses_sqlite_all_time_without_recalculating(tmp_path, monkeypatch):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    tracker.add_current_costs(12.34)

    def fail_initialize_all_time_cost(*args, **kwargs):
        raise AssertionError("sqlite all_time should not be recalculated")

    monkeypatch.setattr(tracker, "initialize_all_time_cost", fail_initialize_all_time_cost)

    assert tracker.get_current_cost()["cost_all_time"] == 12.34


def test_add_current_costs_updates_sqlite_all_time_without_recalculating(tmp_path, monkeypatch):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    tracker.add_current_costs(12.34)

    def fail_initialize_all_time_cost(*args, **kwargs):
        raise AssertionError("sqlite all_time should not be recalculated")

    monkeypatch.setattr(tracker, "initialize_all_time_cost", fail_initialize_all_time_cost)

    tracker.add_current_costs(0.01)

    assert tracker.usage["current_cost"]["all_time"] == 12.35


def test_usage_store_prunes_raw_events_without_touching_aggregates(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    tracker.store.record_event("42", "user", "chat_tokens", 100, 0.1, event_date="2026-05-01")
    tracker.store.record_event("42", "user", "chat_tokens", 200, 0.2, event_date="2026-07-01")

    deleted = tracker.store.prune_events(days=30, today=date(2026, 7, 2))

    assert deleted == 1
    events = _usage_db_rows(tmp_path, "SELECT event_date FROM usage_events ORDER BY event_date")
    assert events == [{"event_date": "2026-07-01"}]
    aggregates = _usage_db_rows(
        tmp_path,
        "SELECT event_date, chat_tokens, cost FROM usage_daily_aggregates ORDER BY event_date",
    )
    assert aggregates == [
        {"event_date": "2026-05-01", "chat_tokens": 100, "cost": 0.1},
        {"event_date": "2026-07-01", "chat_tokens": 200, "cost": 0.2},
    ]
    costs = tracker.store.load_costs("42", "user", today=date(2026, 7, 2))
    assert costs["all_time"] == pytest.approx(0.3)


def test_usage_history_can_be_loaded_with_retention_window(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    tracker.store.record_event("42", "user", "chat_tokens", 100, 0.1, event_date="2026-05-01")
    tracker.store.record_event("42", "user", "chat_tokens", 200, 0.2, event_date="2026-07-01")
    tracker.store.record_event(
        "42",
        "user",
        "tts_characters",
        1000,
        0.1,
        metadata={"model": "tts-1"},
        event_date="2026-05-01",
    )
    tracker.store.record_event(
        "42",
        "user",
        "tts_characters",
        2000,
        0.2,
        metadata={"model": "tts-1"},
        event_date="2026-07-01",
    )

    history = tracker.store.load_usage_history("42", "user", history_days=30, today=date(2026, 7, 2))

    assert history["chat_tokens"] == {"2026-07-01": 200}
    assert history["tts_characters"] == {"tts-1": {"2026-07-01": 2000}}
    full_history = tracker.store.load_usage_history("42", "user")
    assert full_history["chat_tokens"] == {"2026-05-01": 100, "2026-07-01": 200}


def test_usage_tracker_prune_store_writes_bounded_snapshot_preserving_all_time_cost(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    tracker.store.record_event("42", "user", "chat_tokens", 100, 1.0, event_date="2026-05-01")
    tracker.store.record_event("42", "user", "chat_tokens", 200, 2.0, event_date="2026-07-01")

    deleted = tracker.prune_store(event_days=30, history_days=30, today=date(2026, 7, 2))

    assert deleted == 1
    snapshot = json.loads(Path(tracker.user_file).read_text(encoding="utf-8"))
    assert snapshot["usage_history"]["chat_tokens"] == {"2026-07-01": 200}
    assert snapshot["current_cost"]["all_time"] == 3.0
    events = _usage_db_rows(tmp_path, "SELECT event_date FROM usage_events ORDER BY event_date")
    assert events == [{"event_date": "2026-07-01"}]


def test_usage_tracker_prune_store_keeps_bounded_snapshot_after_next_write(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    tracker.store.record_event("42", "user", "chat_tokens", 100, 1.0, event_date="2026-05-01")
    tracker.store.record_event("42", "user", "chat_tokens", 200, 2.0, event_date="2026-07-01")
    tracker.prune_store(event_days=30, history_days=30, today=date(2026, 7, 2))

    tracker.add_current_costs(0.5)

    snapshot = json.loads(Path(tracker.user_file).read_text(encoding="utf-8"))
    assert snapshot["usage_history"]["chat_tokens"] == {"2026-07-01": 200}
    assert snapshot["current_cost"]["all_time"] == 3.5


def test_usage_tracker_does_not_reimport_bounded_snapshot_on_restart(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    tracker.store.record_event("42", "user", "chat_tokens", 100, 1.0, event_date="2026-05-01")
    tracker.store.record_event("42", "user", "chat_tokens", 200, 2.0, event_date="2026-07-01")
    tracker.prune_store(event_days=30, history_days=30, today=date(2026, 7, 2))

    restarted = UsageTracker(42, "owner", logs_dir=str(tmp_path))

    assert restarted.usage["current_cost"]["all_time"] == 3.0
    assert restarted.usage["usage_history"]["chat_tokens"] == {"2026-05-01": 100, "2026-07-01": 200}
    events = _usage_db_rows(tmp_path, "SELECT event_date FROM usage_events ORDER BY event_date")
    assert events == [{"event_date": "2026-07-01"}]


def test_usage_tracker_prune_store_keeps_tts_breakdown_for_unbounded_history(tmp_path):
    tracker = UsageTracker(42, "owner", logs_dir=str(tmp_path))
    tracker.store.record_event("42", "user", "chat_tokens", 100, 1.0, event_date="2026-05-01")
    tracker.store.record_event(
        "42",
        "user",
        "tts_characters",
        1000,
        0.5,
        metadata={"model": "tts-1-hd"},
        event_date="2026-05-01",
    )

    deleted = tracker.prune_store(event_days=30, today=date(2026, 7, 2))

    assert deleted == 1
    snapshot = json.loads(Path(tracker.user_file).read_text(encoding="utf-8"))
    assert snapshot["usage_history"]["chat_tokens"] == {"2026-05-01": 100}
    assert snapshot["usage_history"]["tts_characters"] == {"tts-1-hd": {"2026-05-01": 1000}}
    events = _usage_db_rows(
        tmp_path,
        "SELECT event_type, event_date FROM usage_events ORDER BY event_type",
    )
    assert events == [{"event_type": "tts_characters", "event_date": "2026-05-01"}]


def test_record_image_request_uses_init_image_prices(tmp_path):
    tracker = UsageTracker(
        42, "user", logs_dir=str(tmp_path),
        image_prices=[1.0, 2.0, 3.0],
    )
    usage = {42: tracker}
    assert record_image_request(usage, _make_config(allowed="42"), 42, "256x256") is True
    assert tracker.usage["current_cost"]["day"] == 1.0


def test_record_vision_tokens_charges_user_and_guest(tmp_path):
    user = UsageTracker(99, "stranger", logs_dir=str(tmp_path), vision_token_price=0.5)
    guests = UsageTracker("guests", "guests", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {99: user, "guests": guests}
    assert record_vision_tokens(usage, _make_config(allowed="42"), 99, 2000) is True
    assert sum(user.usage["usage_history"]["vision_tokens"].values()) == 2000
    assert sum(guests.usage["usage_history"]["vision_tokens"].values()) == 2000
    assert user.usage["current_cost"]["day"] == 1.0
    assert guests.usage["current_cost"]["day"] == 1.0


def test_record_vision_tokens_accepts_numeric_string(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {42: user}

    assert record_vision_tokens(usage, _make_config(allowed="42"), 42, "2000") is True

    assert sum(user.usage["usage_history"]["vision_tokens"].values()) == 2000
    assert user.usage["current_cost"]["day"] == 1.0


def test_record_vision_tokens_skips_zero_string(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {42: user}

    assert record_vision_tokens(usage, _make_config(allowed="42"), 42, "0") is False

    assert user.usage["usage_history"]["vision_tokens"] == {}
    assert user.usage["current_cost"]["day"] == 0.0


def test_record_vision_tokens_rejects_negative_and_bool(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {42: user}

    assert record_vision_tokens(usage, _make_config(allowed="42"), 42, -1) is False
    assert record_vision_tokens(usage, _make_config(allowed="42"), 42, False) is False
    assert user.usage["usage_history"]["vision_tokens"] == {}
    assert user.usage["current_cost"]["day"] == 0.0


def test_record_tts_request_charges_user(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), tts_prices=[0.5, 0.75])
    usage = {42: user}
    assert record_tts_request(usage, _make_config(allowed="42"), 42, 1000, "tts-1-hd") is True
    model_history = user.usage["usage_history"]["tts_characters"]["tts-1-hd"]
    assert sum(model_history.values()) == 1000
    assert user.usage["current_cost"]["day"] == 0.75


def test_record_transcription_seconds_charges_user(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), transcription_price=0.5)
    usage = {42: user}
    assert record_transcription_seconds(usage, _make_config(allowed="42"), 42, 60) is True
    assert sum(user.usage["usage_history"]["transcription_seconds"].values()) == 60
    assert user.usage["current_cost"]["day"] == 0.5


def test_record_tts_and_transcription_reject_negative_and_bool(tmp_path):
    tts_user = UsageTracker(42, "owner", logs_dir=str(tmp_path / "tts"), tts_prices=[0.5, 0.75])
    transcription_user = UsageTracker(
        42,
        "owner",
        logs_dir=str(tmp_path / "transcription"),
        transcription_price=0.5,
    )

    assert record_tts_request({42: tts_user}, _make_config(allowed="42"), 42, -100, "tts-1") is False
    assert record_tts_request({42: tts_user}, _make_config(allowed="42"), 42, True, "tts-1") is False
    assert tts_user.usage["usage_history"]["tts_characters"] == {}

    assert record_transcription_seconds({42: transcription_user}, _make_config(allowed="42"), 42, -10) is False
    assert record_transcription_seconds({42: transcription_user}, _make_config(allowed="42"), 42, False) is False
    assert transcription_user.usage["usage_history"]["transcription_seconds"] == {}


def test_add_image_request_dalle3_wide_size(tmp_path):
    tracker = UsageTracker(42, "user", logs_dir=str(tmp_path), image_prices=[1.0, 2.0, 3.0])
    tracker.add_image_request("1792x1024")
    assert tracker.usage["current_cost"]["day"] == 3.0


def test_add_image_request_dalle3_tall_size(tmp_path):
    tracker = UsageTracker(42, "user", logs_dir=str(tmp_path), image_prices=[1.0, 2.0, 3.0])
    tracker.add_image_request("1024x1792")
    assert tracker.usage["current_cost"]["day"] == 3.0


def test_add_image_request_unknown_size_uses_max_price(tmp_path):
    tracker = UsageTracker(42, "user", logs_dir=str(tmp_path), image_prices=[1.0, 2.0, 3.0])
    tracker.add_image_request("9999x9999")
    assert tracker.usage["current_cost"]["day"] == 3.0


def test_add_image_request_empty_prices_does_not_raise(tmp_path):
    tracker = UsageTracker(42, "user", logs_dir=str(tmp_path), image_prices=[])
    tracker.add_image_request("1024x1024")
    assert tracker.usage["current_cost"]["day"] == 0.0


def test_make_usage_tracker_threads_config_prices(tmp_path):
    config = _make_config()
    config["token_price"] = 0.123
    config["image_prices"] = [9.0, 9.0, 9.0]
    config["vision_token_price"] = 0.456
    config["tts_prices"] = [0.7, 0.8]
    config["transcription_price"] = 0.321
    config["usage_retention_days"] = 14
    tracker = make_usage_tracker(config, 42, "owner", logs_dir=str(tmp_path))
    assert tracker.prices["token_price"] == 0.123
    assert tracker.prices["image_prices"] == [9.0, 9.0, 9.0]
    assert tracker.prices["vision_token_price"] == 0.456
    assert tracker.prices["tts_prices"] == [0.7, 0.8]
    assert tracker.prices["transcription_price"] == 0.321
    assert tracker._history_days == 14
