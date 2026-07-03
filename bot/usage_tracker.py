import os
import os.path
import pathlib
import json
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from datetime import date, timedelta


def year_month(date_str):
    # extract string of year-month from date, eg: '2023-03'
    return str(date_str)[:7]


def _coerce_today(today):
    if today is None:
        return date.today()
    if isinstance(today, date):
        return today
    return date.fromisoformat(str(today))


def _retention_cutoff(days, today=None):
    if isinstance(days, bool):
        raise ValueError("retention days must be a non-negative integer")
    days = int(days)
    if days < 0:
        raise ValueError("retention days must be a non-negative integer")
    return str(_coerce_today(today) - timedelta(days=days))


class _UsageSQLiteStore:
    _schema_lock = threading.RLock()

    def __init__(self, db_path):
        self.db_path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._schema_lock:
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS usage_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject_id TEXT NOT NULL,
                        subject_type TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        amount REAL NOT NULL DEFAULT 0,
                        cost REAL NOT NULL DEFAULT 0,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        event_date TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS usage_daily_aggregates (
                        subject_id TEXT NOT NULL,
                        subject_type TEXT NOT NULL,
                        event_date TEXT NOT NULL,
                        chat_tokens INTEGER NOT NULL DEFAULT 0,
                        vision_tokens INTEGER NOT NULL DEFAULT 0,
                        images_256 INTEGER NOT NULL DEFAULT 0,
                        images_512 INTEGER NOT NULL DEFAULT 0,
                        images_1024 INTEGER NOT NULL DEFAULT 0,
                        transcription_seconds REAL NOT NULL DEFAULT 0,
                        tts_characters INTEGER NOT NULL DEFAULT 0,
                        cost REAL NOT NULL DEFAULT 0,
                        PRIMARY KEY (subject_id, subject_type, event_date)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS usage_imports (
                        subject_id TEXT NOT NULL,
                        subject_type TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        source_mtime REAL,
                        source_size INTEGER,
                        imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (subject_id, subject_type, source_path)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_usage_events_subject_date
                    ON usage_events(subject_id, subject_type, event_date)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_usage_events_type_created
                    ON usage_events(event_type, created_at)
                """)

    def _has_import_marker(self, cursor, subject_id, subject_type, source_path):
        result = cursor.execute("""
            SELECT 1
            FROM usage_imports
            WHERE subject_id = ? AND subject_type = ? AND source_path = ?
        """, (subject_id, subject_type, source_path))
        return result.fetchone() is not None

    def has_import_marker(self, subject_id, subject_type, source_path):
        with self._lock:
            with self._connect() as conn:
                return self._has_import_marker(conn, subject_id, subject_type, source_path)

    def mark_imported(self, subject_id, subject_type, source_path, source_mtime, source_size):
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO usage_imports
                    (subject_id, subject_type, source_path, source_mtime, source_size)
                    VALUES (?, ?, ?, ?, ?)
                """, (subject_id, subject_type, source_path, source_mtime, source_size))

    def import_events(self, subject_id, subject_type, source_path, source_mtime, source_size, events):
        with self._lock:
            with self._connect() as conn:
                if self._has_import_marker(conn, subject_id, subject_type, source_path):
                    return False
                for event in events:
                    self._record_event(conn, subject_id, subject_type, **event)
                conn.execute("""
                    INSERT OR IGNORE INTO usage_imports
                    (subject_id, subject_type, source_path, source_mtime, source_size)
                    VALUES (?, ?, ?, ?, ?)
                """, (subject_id, subject_type, source_path, source_mtime, source_size))
                return True

    def record_event(self, subject_id, subject_type, event_type, amount, cost, metadata=None, event_date=None):
        if event_date is None:
            event_date = str(date.today())
        with self._lock:
            with self._connect() as conn:
                self._record_event(
                    conn,
                    subject_id,
                    subject_type,
                    event_type=event_type,
                    amount=amount,
                    cost=cost,
                    metadata=metadata,
                    event_date=event_date,
                )

    def _record_event(self, conn, subject_id, subject_type, event_type, amount, cost, metadata=None, event_date=None):
        metadata = metadata or {}
        event_date = str(event_date or date.today())
        amount = float(amount)
        cost = float(cost)
        conn.execute("""
            INSERT INTO usage_events
            (subject_id, subject_type, event_type, amount, cost, metadata, event_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            subject_id,
            subject_type,
            event_type,
            amount,
            cost,
            json.dumps(metadata, ensure_ascii=False),
            event_date,
        ))
        increments = {
            "chat_tokens": 0,
            "vision_tokens": 0,
            "images_256": 0,
            "images_512": 0,
            "images_1024": 0,
            "transcription_seconds": 0,
            "tts_characters": 0,
        }
        if event_type == "chat_tokens":
            increments["chat_tokens"] = int(amount)
        elif event_type == "vision_tokens":
            increments["vision_tokens"] = int(amount)
        elif event_type == "image_request":
            tier = str(metadata.get("tier", "1024"))
            increments[f"images_{tier}" if tier in {"256", "512", "1024"} else "images_1024"] = int(amount)
        elif event_type == "transcription_seconds":
            increments["transcription_seconds"] = amount
        elif event_type == "tts_characters":
            increments["tts_characters"] = int(amount)

        conn.execute("""
            INSERT OR IGNORE INTO usage_daily_aggregates
            (subject_id, subject_type, event_date)
            VALUES (?, ?, ?)
        """, (subject_id, subject_type, event_date))
        conn.execute("""
            UPDATE usage_daily_aggregates
            SET chat_tokens = chat_tokens + ?,
                vision_tokens = vision_tokens + ?,
                images_256 = images_256 + ?,
                images_512 = images_512 + ?,
                images_1024 = images_1024 + ?,
                transcription_seconds = transcription_seconds + ?,
                tts_characters = tts_characters + ?,
                cost = cost + ?
            WHERE subject_id = ? AND subject_type = ? AND event_date = ?
        """, (
            increments["chat_tokens"],
            increments["vision_tokens"],
            increments["images_256"],
            increments["images_512"],
            increments["images_1024"],
            increments["transcription_seconds"],
            increments["tts_characters"],
            cost,
            subject_id,
            subject_type,
            event_date,
        ))

    def prune_events(self, days=30, today=None):
        cutoff_date = _retention_cutoff(days, today=today)
        with self._lock:
            with self._connect() as conn:
                # TTS model breakdown has no separate aggregate table; keep those rows
                # so unbounded compatibility snapshots can still reconstruct it.
                cursor = conn.execute(
                    """
                    DELETE FROM usage_events
                    WHERE event_date < ? AND event_type != 'tts_characters'
                    """,
                    (cutoff_date,),
                )
                return cursor.rowcount

    def load_usage_history(self, subject_id, subject_type, history_days=None, today=None):
        history = {
            "chat_tokens": {},
            "transcription_seconds": {},
            "number_images": {},
            "tts_characters": {},
            "vision_tokens": {},
        }
        aggregate_filter = ""
        tts_filter = ""
        params = [subject_id, subject_type]
        tts_params = [subject_id, subject_type]
        if history_days is not None:
            cutoff_date = _retention_cutoff(history_days, today=today)
            aggregate_filter = " AND event_date >= ?"
            tts_filter = " AND event_date >= ?"
            params.append(cutoff_date)
            tts_params.append(cutoff_date)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(f"""
                    SELECT event_date, chat_tokens, vision_tokens, images_256, images_512,
                           images_1024, transcription_seconds
                    FROM usage_daily_aggregates
                    WHERE subject_id = ? AND subject_type = ?{aggregate_filter}
                    ORDER BY event_date
                """, tuple(params)).fetchall()
                for row in rows:
                    event_date = row["event_date"]
                    if row["chat_tokens"]:
                        history["chat_tokens"][event_date] = int(row["chat_tokens"])
                    if row["vision_tokens"]:
                        history["vision_tokens"][event_date] = int(row["vision_tokens"])
                    images = [int(row["images_256"]), int(row["images_512"]), int(row["images_1024"])]
                    if any(images):
                        history["number_images"][event_date] = images
                    if row["transcription_seconds"]:
                        history["transcription_seconds"][event_date] = row["transcription_seconds"]

                tts_rows = conn.execute(f"""
                    SELECT event_date, amount, metadata
                    FROM usage_events
                    WHERE subject_id = ? AND subject_type = ? AND event_type = 'tts_characters'{tts_filter}
                    ORDER BY event_date, id
                """, tuple(tts_params)).fetchall()
                for row in tts_rows:
                    try:
                        metadata = json.loads(row["metadata"] or "{}")
                    except json.JSONDecodeError:
                        metadata = {}
                    model = metadata.get("model") or "tts-1"
                    history["tts_characters"].setdefault(model, {})
                    history["tts_characters"][model][row["event_date"]] = (
                        history["tts_characters"][model].get(row["event_date"], 0) + int(row["amount"])
                    )
        return history

    def load_costs(self, subject_id, subject_type, today=None):
        today = _coerce_today(today)
        today_str = str(today)
        week = today.isocalendar()[:2]
        month = year_month(today_str)
        costs = {"day": 0.0, "week": 0.0, "month": 0.0, "all_time": 0.0}
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT event_date, cost
                    FROM usage_daily_aggregates
                    WHERE subject_id = ? AND subject_type = ?
                """, (subject_id, subject_type)).fetchall()
        for row in rows:
            event_date = str(row["event_date"])
            cost = float(row["cost"] or 0)
            costs["all_time"] += cost
            if event_date == today_str:
                costs["day"] += cost
            try:
                parsed_date = date.fromisoformat(event_date)
            except ValueError:
                continue
            if parsed_date.isocalendar()[:2] == week:
                costs["week"] += cost
            if year_month(event_date) == month:
                costs["month"] += cost
        return costs


class UsageTracker:
    """
    UsageTracker class
    Enables tracking of daily/weekly/monthly usage per user.
    Usage events are stored in usage_logs/usage.sqlite3. Legacy per-user JSON
    files are imported once and can still be written as compatibility snapshots.
    Snapshot example:
    {
        "user_name": "@user_name",
        "current_cost": {
            "day": 0.45,
            "week": 1.20,
            "month": 3.23,
            "all_time": 3.23,
            "last_update": "2023-03-14"},
        "usage_history": {
            "chat_tokens": {
                "2023-03-13": 520,
                "2023-03-14": 1532
            },
            "transcription_seconds": {
                "2023-03-13": 125,
                "2023-03-14": 64
            },
            "number_images": {
                "2023-03-12": [0, 2, 3],
                "2023-03-13": [1, 2, 3],
                "2023-03-14": [0, 1, 2]
            }
        }
    }
    """

    _file_lock = threading.RLock()

    def __init__(
        self,
        user_id,
        user_name,
        logs_dir="usage_logs",
        *,
        token_price=None,
        image_prices=None,
        vision_token_price=None,
        tts_prices=None,
        transcription_price=None,
        history_days=None,
    ):
        """
        Initializes UsageTracker for a user with current date.
        Loads usage from SQLite and imports the legacy usage log file if needed.
        :param user_id: Telegram ID of the user
        :param user_name: Telegram user name
        :param logs_dir: path to directory of usage logs, defaults to "usage_logs"
        :param token_price: chat tokens price per 1k tokens (falls back to historical default)
        :param image_prices: image prices, one per size ["256x256", "512x512", "1024x1024"]
        :param vision_token_price: vision tokens price per 1k tokens
        :param tts_prices: TTS prices per 1k characters, one per [standard, hd]
        :param transcription_price: transcription price per minute
        :param history_days: optional compatibility snapshot history window
        """
        self.user_id = user_id
        self.logs_dir = logs_dir
        self.subject_id = str(user_id)
        self.subject_type = "guest_pool" if str(user_id) == "guests" else "user"
        # Defaults match production config shape (see bot/__main__.py): lists for
        # image_prices/tts_prices, floats for scalar prices.
        self.prices = {
            'token_price': 0.002 if token_price is None else token_price,
            'image_prices': [0.016, 0.018, 0.02] if image_prices is None else image_prices,
            'vision_token_price': 0.01 if vision_token_price is None else vision_token_price,
            'tts_prices': [0.015, 0.030] if tts_prices is None else tts_prices,
            'transcription_price': 0.006 if transcription_price is None else transcription_price,
        }
        #self.logs_dir = os.path.join("../../",os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usage_logs'))
        # path to usage file of given user
        pathlib.Path(logs_dir).mkdir(parents=True, exist_ok=True)
        self.user_file = str(pathlib.Path(logs_dir) / f"{user_id}.json")
        self.store = _UsageSQLiteStore(pathlib.Path(logs_dir) / "usage.sqlite3")
        self._history_days = history_days

        with self._file_lock:
            self._import_legacy_json_if_needed(user_name)
            self.usage = self._snapshot_usage(user_name, history_days=self._history_days)

    def _new_usage(self, user_name):
        return {
            "user_name": user_name,
            "current_cost": {"day": 0.0, "week": 0.0, "month": 0.0, "all_time": 0.0, "last_update": str(date.today())},
            "usage_history": {"chat_tokens": {}, "transcription_seconds": {}, "number_images": {}, "tts_characters": {}, "vision_tokens":{}}
        }

    def _next_corrupt_file_path(self):
        corrupt_path = pathlib.Path(f"{self.user_file}.corrupt")
        if not corrupt_path.exists():
            return corrupt_path
        suffix = 1
        while True:
            candidate = pathlib.Path(f"{self.user_file}.corrupt.{suffix}")
            if not candidate.exists():
                return candidate
            suffix += 1

    def _import_legacy_json_if_needed(self, user_name):
        user_file = pathlib.Path(self.user_file)
        if not user_file.exists() or self.store.has_import_marker(
            self.subject_id,
            self.subject_type,
            str(user_file),
        ):
            return
        try:
            with open(user_file, "r", encoding="utf-8") as file:
                legacy_usage = json.load(file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            target_stat = user_file.stat()
            os.replace(self.user_file, self._next_corrupt_file_path())
            self.usage = self._new_usage(user_name)
            self._write_usage(target_stat=target_stat)
            return

        usage_history = legacy_usage.setdefault("usage_history", {})
        usage_history.setdefault("chat_tokens", {})
        usage_history.setdefault("transcription_seconds", {})
        usage_history.setdefault("number_images", {})
        usage_history.setdefault("tts_characters", {})
        usage_history.setdefault("vision_tokens", {})
        stat = user_file.stat()
        events = self._legacy_usage_events(legacy_usage)
        self.store.import_events(
            self.subject_id,
            self.subject_type,
            str(user_file),
            stat.st_mtime,
            stat.st_size,
            events,
        )

    def _legacy_usage_events(self, legacy_usage):
        usage_history = legacy_usage.get("usage_history", {})
        events = []
        for event_date, tokens in usage_history.get("chat_tokens", {}).items():
            tokens = int(tokens)
            events.append({
                "event_type": "chat_tokens",
                "amount": tokens,
                "cost": round(tokens * self.prices["token_price"] / 1000, 6),
                "metadata": {},
                "event_date": str(event_date),
            })
        for event_date, tokens in usage_history.get("vision_tokens", {}).items():
            tokens = int(tokens)
            events.append({
                "event_type": "vision_tokens",
                "amount": tokens,
                "cost": round(tokens * self.prices["vision_token_price"] / 1000, 2),
                "metadata": {},
                "event_date": str(event_date),
            })
        image_prices = list(self.prices["image_prices"])
        for event_date, counts in usage_history.get("number_images", {}).items():
            for tier, count, price in zip(("256", "512", "1024"), list(counts)[:3], image_prices):
                count = int(count)
                if count <= 0:
                    continue
                events.append({
                    "event_type": "image_request",
                    "amount": count,
                    "cost": count * price,
                    "metadata": {"tier": tier},
                    "event_date": str(event_date),
                })
        for event_date, seconds in usage_history.get("transcription_seconds", {}).items():
            seconds = int(seconds)
            events.append({
                "event_type": "transcription_seconds",
                "amount": seconds,
                "cost": round(seconds * self.prices["transcription_price"] / 60, 2),
                "metadata": {},
                "event_date": str(event_date),
            })
        for tts_model, model_usage in usage_history.get("tts_characters", {}).items():
            price = self._tts_price_for_model(tts_model, self.prices["tts_prices"])
            for event_date, characters in model_usage.items():
                characters = int(characters)
                events.append({
                    "event_type": "tts_characters",
                    "amount": characters,
                    "cost": round(characters * price / 1000, 2),
                    "metadata": {"model": tts_model},
                    "event_date": str(event_date),
                })
        current_cost = legacy_usage.get("current_cost", {})
        try:
            legacy_all_time = float(current_cost["all_time"])
        except (KeyError, TypeError, ValueError):
            legacy_all_time = None
        if legacy_all_time is not None:
            imported_cost = sum(float(event.get("cost") or 0) for event in events)
            baseline_delta = round(legacy_all_time - imported_cost, 6)
            if baseline_delta:
                events.append({
                    "event_type": "cost_adjustment",
                    "amount": 0,
                    "cost": baseline_delta,
                    "metadata": {"source": "legacy_current_cost_all_time"},
                    "event_date": "0001-01-01",
                })
        return events

    def _snapshot_usage(self, user_name, history_days=None, today=None):
        snapshot_date = _coerce_today(today)
        costs = self.store.load_costs(self.subject_id, self.subject_type, today=snapshot_date)
        return {
            "user_name": user_name,
            "current_cost": {
                "day": costs["day"],
                "week": costs["week"],
                "month": costs["month"],
                "all_time": costs["all_time"],
                "last_update": str(snapshot_date),
            },
            "usage_history": self.store.load_usage_history(
                self.subject_id,
                self.subject_type,
                history_days=history_days,
                today=today,
            ),
        }

    def _refresh_usage(self, history_days=None, today=None):
        self.usage = self._snapshot_usage(
            self.usage.get("user_name", ""),
            history_days=history_days,
            today=today,
        )

    def prune_store(self, event_days=30, history_days=None, today=None):
        with self._file_lock:
            deleted = self.store.prune_events(event_days, today=today)
            self._history_days = history_days
            self._refresh_usage(history_days=history_days, today=today)
            self._write_usage()
            return deleted

    def _record_usage_event(self, event_type, amount, cost, metadata=None, event_date=None):
        with self._file_lock:
            self.store.record_event(
                self.subject_id,
                self.subject_type,
                event_type,
                amount,
                cost,
                metadata=metadata,
                event_date=event_date,
            )
            self._refresh_usage(history_days=self._history_days)
            self._write_usage()

    def _write_usage(self, target_stat=None):
        user_file = pathlib.Path(self.user_file)
        user_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = None
        with self._file_lock:
            if target_stat is None and user_file.exists():
                target_stat = user_file.stat()
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=str(user_file.parent),
                    delete=False,
                ) as outfile:
                    temp_file = outfile.name
                    json.dump(self.usage, outfile, ensure_ascii=False)
                    outfile.flush()
                    os.fsync(outfile.fileno())
                if target_stat is not None:
                    os.chmod(temp_file, target_stat.st_mode & 0o7777)
                    try:
                        os.chown(temp_file, target_stat.st_uid, target_stat.st_gid)
                    except (AttributeError, PermissionError, OSError):
                        pass
                os.replace(temp_file, self.user_file)
                temp_file = None
                updated_stat = user_file.stat()
                self.store.mark_imported(
                    self.subject_id,
                    self.subject_type,
                    str(user_file),
                    updated_stat.st_mtime,
                    updated_stat.st_size,
                )
            finally:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)

    # token usage functions:

    def add_chat_tokens(self, tokens, tokens_price=None):
        """Adds used tokens from a request to a users usage history and updates current cost
        :param tokens: total tokens used in last request
        :param tokens_price: price per 1000 tokens; falls back to self.prices['token_price']
        """
        if tokens_price is None:
            tokens_price = self.prices['token_price']
        today = date.today()
        token_cost = round(float(tokens) * tokens_price / 1000, 6)
        self._record_usage_event("chat_tokens", int(tokens), token_cost, event_date=str(today))

    def get_current_token_usage(self):
        """Get token amounts used for today and this month

        :return: total number of tokens used per day and per month
        """
        self._refresh_usage()
        today = date.today()
        if str(today) in self.usage["usage_history"]["chat_tokens"]:
            usage_day = self.usage["usage_history"]["chat_tokens"][str(today)]
        else:
            usage_day = 0
        month = str(today)[:7]  # year-month as string
        usage_month = 0
        for today, tokens in self.usage["usage_history"]["chat_tokens"].items():
            if today.startswith(month):
                usage_month += tokens
        return usage_day, usage_month

    # image usage functions:

    def add_image_request(self, image_size, image_prices=None):
        """Add image request to users usage history and update current costs.

        :param image_size: requested image size (DALL·E 3 sizes 1792x1024 and 1024x1792
                           are billed at the 1024x1024 tier)
        :param image_prices: prices for images of sizes ["256x256", "512x512", "1024x1024"];
                             falls back to self.prices['image_prices']
        """
        if image_prices is None:
            image_prices = self.prices['image_prices']
        size_to_price_idx = {"256x256": 0, "512x512": 1, "1024x1024": 2, "1792x1024": 2, "1024x1792": 2}
        prices = list(image_prices)
        if not prices:
            return
        max_idx = len(prices) - 1
        requested_size = min(size_to_price_idx.get(image_size, max_idx), max_idx)
        image_cost = prices[requested_size]
        today = date.today()
        tier = ("256", "512", "1024")[min(requested_size, 2)]
        self._record_usage_event(
            "image_request",
            1,
            image_cost,
            metadata={"size": image_size, "tier": tier},
            event_date=str(today),
        )

    def get_current_image_count(self):
        """Get number of images requested for today and this month.

        :return: total number of images requested per day and per month
        """
        self._refresh_usage()
        today = date.today()
        if str(today) in self.usage["usage_history"]["number_images"]:
            usage_day = sum(self.usage["usage_history"]["number_images"][str(today)])
        else:
            usage_day = 0
        month = str(today)[:7]  # year-month as string
        usage_month = 0
        for today, images in self.usage["usage_history"]["number_images"].items():
            if today.startswith(month):
                usage_month += sum(images)
        return usage_day, usage_month


    # vision usage functions
    def add_vision_tokens(self, tokens, vision_token_price=None):
        """
         Adds requested vision tokens to a users usage history and updates current cost.
        :param tokens: total tokens used in last request
        :param vision_token_price: price per 1K vision tokens;
                                   falls back to self.prices['vision_token_price']
        """
        if vision_token_price is None:
            vision_token_price = self.prices['vision_token_price']
        today = date.today()
        token_price = round(tokens * vision_token_price / 1000, 2)
        self._record_usage_event("vision_tokens", int(tokens), token_price, event_date=str(today))

    def get_current_vision_tokens(self):
        """Get vision tokens for today and this month.

        :return: total amount of vision tokens per day and per month
        """
        self._refresh_usage()
        today = date.today()
        if str(today) in self.usage["usage_history"]["vision_tokens"]:
            tokens_day = self.usage["usage_history"]["vision_tokens"][str(today)]
        else:
            tokens_day = 0
        month = str(today)[:7]  # year-month as string
        tokens_month = 0
        for today, tokens in self.usage["usage_history"]["vision_tokens"].items():
            if today.startswith(month):
                tokens_month += tokens
        return tokens_day, tokens_month

    # tts usage functions:

    def _tts_price_for_model(self, tts_model, tts_prices):
        tts_models = ['tts-1', 'tts-1-hd', 'llmgateway/silero-tts']
        prices = list(tts_prices)
        if not prices:
            return 0
        if tts_model in tts_models and tts_models.index(tts_model) < len(prices):
            return prices[tts_models.index(tts_model)]
        return prices[0]

    def add_tts_request(self, text_length, tts_model, tts_prices=None):
        if tts_prices is None:
            tts_prices = self.prices['tts_prices']
        price = self._tts_price_for_model(tts_model, tts_prices)
        today = date.today()
        tts_price = round(text_length * price / 1000, 2)
        self._record_usage_event(
            "tts_characters",
            int(text_length),
            tts_price,
            metadata={"model": tts_model},
            event_date=str(today),
        )

    def get_current_tts_usage(self):
        """Get length of speech generated for today and this month.

        :return: total amount of characters converted to speech per day and per month
        """

        self._refresh_usage()
        today = date.today()
        characters_day = 0
        for tts_model in self.usage["usage_history"]["tts_characters"]:
            if tts_model in self.usage["usage_history"]["tts_characters"] and \
                str(today) in self.usage["usage_history"]["tts_characters"][tts_model]:
                characters_day += self.usage["usage_history"]["tts_characters"][tts_model][str(today)]

        month = str(today)[:7]  # year-month as string
        characters_month = 0
        for tts_model in self.usage["usage_history"]["tts_characters"]:
            if tts_model in self.usage["usage_history"]["tts_characters"]: 
                for today, characters in self.usage["usage_history"]["tts_characters"][tts_model].items():
                    if today.startswith(month):
                        characters_month += characters
        return int(characters_day), int(characters_month)


    # transcription usage functions:

    def add_transcription_seconds(self, seconds, minute_price=None):
        """Adds requested transcription seconds to a users usage history and updates current cost.
        :param seconds: total seconds used in last request
        :param minute_price: price per minute transcription;
                             falls back to self.prices['transcription_price']
        """
        if minute_price is None:
            minute_price = self.prices['transcription_price']
        today = date.today()
        transcription_price = round(seconds * minute_price / 60, 2)
        self._record_usage_event(
            "transcription_seconds",
            int(seconds),
            transcription_price,
            event_date=str(today),
        )

    def add_current_costs(self, request_cost):
        """
        Add current cost to all_time, day, week and month cost and update last_update date.
        """
        self._record_usage_event("cost_adjustment", 0, float(request_cost), event_date=str(date.today()))

    def get_current_transcription_duration(self):
        """Get minutes and seconds of audio transcribed for today and this month.

        :return: total amount of time transcribed per day and per month (4 values)
        """
        self._refresh_usage()
        today = date.today()
        if str(today) in self.usage["usage_history"]["transcription_seconds"]:
            seconds_day = self.usage["usage_history"]["transcription_seconds"][str(today)]
        else:
            seconds_day = 0
        month = str(today)[:7]  # year-month as string
        seconds_month = 0
        for today, seconds in self.usage["usage_history"]["transcription_seconds"].items():
            if today.startswith(month):
                seconds_month += seconds
        minutes_day, seconds_day = divmod(seconds_day, 60)
        minutes_month, seconds_month = divmod(seconds_month, 60)
        return int(minutes_day), round(seconds_day, 2), int(minutes_month), round(seconds_month, 2)

    # general functions
    def get_current_cost(self):
        """Get total USD amount of all requests of the current day, week and month

        :return: cost of current day, week and month
        """
        self._refresh_usage()
        cost_day = self.usage["current_cost"]["day"]
        cost_week = self.usage["current_cost"].get("week", cost_day)
        cost_month = self.usage["current_cost"]["month"]
        if "all_time" in self.usage["current_cost"]:
            cost_all_time = self.usage["current_cost"]["all_time"]
        else:
            cost_all_time = self.initialize_all_time_cost()
        return {"cost_today": cost_day, "cost_week": cost_week, "cost_month": cost_month, "cost_all_time": cost_all_time}

    def initialize_all_time_cost(self, tokens_price=0.002, image_prices="0.016,0.018,0.02", minute_price=0.006, vision_token_price=0.01, tts_prices='0.015,0.030'):
        """Get total USD amount of all requests in history
        
        :param tokens_price: price per 1000 tokens, defaults to 0.002
        :param image_prices: prices for images of sizes ["256x256", "512x512", "1024x1024"],
            defaults to [0.016, 0.018, 0.02]
        :param minute_price: price per minute transcription, defaults to 0.006
        :param vision_token_price: price per 1K vision token interpretation, defaults to 0.01
        :param tts_prices: price per 1K characters tts per model ['tts-1', 'tts-1-hd'], defaults to [0.015, 0.030]
        :return: total cost of all requests
        """
        total_tokens = sum(self.usage['usage_history']['chat_tokens'].values())
        token_cost = round(total_tokens * tokens_price / 1000, 6)

        total_images = [sum(values) for values in zip(*self.usage['usage_history']['number_images'].values())]
        image_prices_list = [float(x) for x in image_prices.split(',')]
        image_cost = sum([count * price for count, price in zip(total_images, image_prices_list)])

        total_transcription_seconds = sum(self.usage['usage_history']['transcription_seconds'].values())
        transcription_cost = round(total_transcription_seconds * minute_price / 60, 2)

        total_vision_tokens = sum(self.usage['usage_history']['vision_tokens'].values())
        vision_cost = round(total_vision_tokens * vision_token_price / 1000, 2)

        tts_prices_list = [float(x) for x in tts_prices.split(',')]
        tts_cost = round(sum(
            sum(model_usage.values()) * self._tts_price_for_model(tts_model, tts_prices_list) / 1000
            for tts_model, model_usage in self.usage['usage_history']['tts_characters'].items()
        ), 2)

        all_time_cost = token_cost + transcription_cost + image_cost + vision_cost + tts_cost
        return all_time_cost
