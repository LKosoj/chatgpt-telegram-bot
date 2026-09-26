import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from .plugin import Plugin
from .background import BackgroundTask
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import ContextTypes

# Lease window for a claimed-but-not-yet-delivered reminder. Short on purpose:
# sending is a single Telegram HTTP call, not an agent turn, so it only needs
# enough headroom over the 60s check tick (reminders.py `check`) to survive a
# slow send without another worker racing to reclaim the same row.
REMINDER_LEASE_SECONDS = 120


class RemindersPlugin(Plugin):
    """
    Плагин для управления напоминаниями и интеграциями
    """
    _MAX_SEND_ATTEMPTS = 3

    def __init__(self):
        self.reminders_file = os.path.join(os.path.dirname(__file__), "reminders.json")
        self.db_handle = None

    def get_source_name(self) -> str:
        return "Reminders"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "set_reminder",
            "description": (
                "Schedule a delayed Telegram notification for the current user at an absolute date/time. "
                "Call when the user asks to be reminded about something later — resolve any relative "
                "phrasing ('tomorrow', 'in 2 hours') into the absolute YYYY-MM-DD HH:MM 'time' value "
                "yourself before calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Target reminder date/time in 'YYYY-MM-DD HH:MM' 24-hour local format."
                    },
                    "message": {
                        "type": "string",
                        "description": "Text shown to the user when the reminder fires."
                    },
                    "current_time": {
                        "type": "string",
                        "description": (
                            "Caller's best estimate of the current local date/time as 'YYYY-MM-DD HH:MM', "
                            "used as the reference point for resolving relative phrasings into 'time'."
                        )
                    },
                    "integration": {
                        "type": "string",
                        "description": "Delivery channel for the reminder; only 'telegram' is supported.",
                        "enum": ["telegram",]
                    }
                },
                "required": ["time", "message", "integration","current_time"]
            }
        },
        {
            "name": "list_reminders",
            "description": (
                "List the active (not yet fired) reminders for the current Telegram user with their ids "
                "and scheduled times. Call when the user asks what is on their reminder list or to look "
                "up an id before deleting one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_time": {
                        "type": "string",
                        "description": (
                            "Caller's best estimate of the current local date/time as 'YYYY-MM-DD HH:MM'; "
                            "used only for formatting relative due times in the response."
                        )
                    }
                },
            }
        },
        {
            "name": "delete_reminder",
            "description": (
                "Remove one scheduled reminder belonging to the current Telegram user by its id. Call "
                "when the user explicitly asks to cancel a reminder — use list_reminders first if the "
                "id is not already known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "string",
                        "description": "Reminder id as returned by list_reminders."
                    }
                },
                "required": ["reminder_id"]
            }
        }]

    def get_commands(self) -> List[Dict]:
        """Возвращает список команд, которые поддерживает плагин"""
        return [
            {
                "command": "set_reminder",
                "description": self.t("reminders_command_set_description"),
                "handler": self.execute,
                "handler_kwargs": {"function_name": "set_reminder"},
                "args": self.t("reminders_args_set"),
                "plugin_name": "reminders",
            },
            {
                "command": "list_reminders",
                "description": self.t("reminders_command_list_description"),
                "handler": self.handle_prompt_constructor,
                "handler_kwargs": {},
                "plugin_name": "reminders",
                "add_to_menu": True,
            },
            {
                "command": "delete_reminder",
                "description": self.t("reminders_command_delete_description"),
                "args": self.t("reminders_args_delete"),
                "handler": self.execute,
                "handler_kwargs": {"function_name": "delete_reminder"},
                "plugin_name": "reminders"
            },
            {
                # Обработчик для всех callback_query плагина
                "callback_query_handler": self.handle_reminder_callback,
                "callback_pattern": "^reminder:",
                "plugin_name": "reminders",
                "handler_kwargs": {}
            }
        ]

    def _build_reminders_keyboard(self, reminders: list) -> InlineKeyboardMarkup:
        """Строит клавиатуру списка напоминаний: ряд просмотр/удаление на каждое + кнопка закрытия"""
        keyboard = []

        for r in reminders:
            reminder_time = datetime.fromisoformat(r['time'])
            formatted_time = reminder_time.strftime('%d.%m.%Y %H:%M')

            # Создаем ряд из двух кнопок для каждого напоминания
            keyboard.append([
                InlineKeyboardButton(
                    text=self.t(
                        "reminders_button_label",
                        time=formatted_time,
                        message=r['message']
                    ),
                    callback_data=f"reminder:view:{r['id']}"
                ),
                InlineKeyboardButton(
                    text=self.t("reminders_delete_button"),
                    callback_data=f"reminder:delete:{r['id']}"
                )
            ])

        # Добавляем кнопку закрытия
        keyboard.append([
            InlineKeyboardButton(self.t("reminders_close_menu"), callback_data="reminder:close_menu:")
        ])

        return InlineKeyboardMarkup(keyboard)

    async def handle_prompt_constructor(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик для конструктора промптов"""
        message = update.message
        if message is None:
            return
        assert message.from_user is not None
        user_id = str(message.from_user.id)
        rows = await self.db_handle.fetch_all(
            "SELECT * FROM reminders WHERE owner_id = ? AND status != 'sent' ORDER BY created_at ASC", (user_id,)
        )

        if not rows:
            await message.reply_text(
                self.t("reminders_none"),
                parse_mode='Markdown'
            )
            return

        # Создаем разметку с кнопками
        reply_markup = self._build_reminders_keyboard(rows)

        await message.reply_text(
            self.t("reminders_title"),
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    def initialize(self, openai=None, bot=None, storage_root: str | None = None,
                    db=None, plugin_config=None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        self.db_handle = db
        if storage_root:
            self.reminders_file = os.path.join(storage_root, "reminders.json")
        if self.db_handle is not None:
            self.db_handle.run_sync_blocking(self._import_json_reminders_sync)

    def register_schema(self) -> List[str]:
        return [
            '''
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                target_chat_id TEXT NOT NULL,
                time TEXT NOT NULL,
                fire_at_utc TEXT,
                message TEXT NOT NULL,
                integration TEXT NOT NULL,
                reply_to_message_id INTEGER,
                send_attempts INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                locked_at TEXT,
                locked_by TEXT,
                created_at TEXT NOT NULL
            )
            ''',
            '''
            CREATE INDEX IF NOT EXISTS idx_reminders_owner ON reminders(owner_id)
            ''',
            '''
            CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, fire_at_utc, time)
            ''',
        ]

    def _import_json_reminders_sync(self, db) -> None:
        with db.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
        if count > 0:
            # Table already has data: either import already ran, or the
            # process crashed after the INSERT commit below but before the
            # rename. Finish the rename so the legacy file doesn't linger
            # forever, but never re-import (idempotent).
            if os.path.exists(self.reminders_file):
                os.replace(self.reminders_file, self.reminders_file + ".migrated")
            return
        if not os.path.exists(self.reminders_file):
            return
        try:
            with open(self.reminders_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            logging.exception("Failed to read reminders.json for import")
            return
        if not isinstance(data, dict):
            return
        rows = []
        for owner_id, user_reminders in data.items():
            if not isinstance(user_reminders, dict):
                continue
            for reminder_id, reminder in user_reminders.items():
                if not isinstance(reminder, dict):
                    continue
                fire_at_utc = reminder.get("fire_at_utc")
                if fire_at_utc is not None:
                    try:
                        datetime.fromisoformat(str(fire_at_utc))
                    except ValueError:
                        # Corrupt fire_at_utc would otherwise sort unpredictably against
                        # the UTC claim cutoff (plain string comparison, see
                        # _claim_due_reminders_sync) — drop it so the row falls back to
                        # the legacy naive-'time' comparison instead.
                        fire_at_utc = None
                reminder_time = reminder.get("time")
                if fire_at_utc is None:
                    # No usable fire_at_utc: due-detection for this row falls back
                    # entirely to the legacy naive 'time' column (lexicographic compare
                    # in _claim_due_reminders_sync). HEAD's check_reminders caught
                    # ValueError/KeyError on a missing/invalid 'time' and skipped the
                    # record forever (logged, never sent) instead of firing it — mirror
                    # that here by not importing it, rather than inserting an empty
                    # string that would sort as immediately due.
                    try:
                        datetime.fromisoformat(str(reminder_time))
                    except ValueError:
                        logging.warning(
                            "Reminders: skipping legacy import of reminder %s for owner %s: missing/invalid time",
                            reminder_id, owner_id,
                        )
                        continue
                rows.append((
                    reminder_id, owner_id,
                    str(reminder.get("target_chat_id") or reminder.get("user_id") or owner_id),
                    reminder_time or "", fire_at_utc,
                    reminder.get("message", ""), reminder.get("integration", "telegram"),
                    reminder.get("reply_to_message_id"),
                    int(reminder.get("send_attempts") or 0), "pending",
                    reminder.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                ))
        with db.get_connection() as conn:
            conn.executemany(
                '''INSERT OR IGNORE INTO reminders (
                    id, owner_id, target_chat_id, time, fire_at_utc, message, integration,
                    reply_to_message_id, send_attempts, status, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                rows,
            )
        os.replace(self.reminders_file, self.reminders_file + ".migrated")

    def get_background_tasks(self) -> List[BackgroundTask]:
        return [
            BackgroundTask(
                name="check",
                interval_seconds=60.0,
                coroutine_factory=self._check_reminders_tick,
            )
        ]

    async def _check_reminders_tick(self, *, application) -> None:
        await self.check_reminders(application.bot)

    def _claim_due_reminders_sync(
        self, db, now_local_iso: str, now_utc_iso: str, lease_cutoff_iso: str, worker_id: str
    ) -> List[Dict[str, Any]]:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                '''
                SELECT id FROM reminders
                WHERE status != 'sent'
                  AND (status != 'processing' OR locked_at IS NULL OR locked_at <= ?)
                  AND (
                        (fire_at_utc IS NOT NULL AND fire_at_utc <= ?)
                     OR (fire_at_utc IS NULL AND time <= ?)
                  )
                ''',
                (lease_cutoff_iso, now_utc_iso, now_local_iso),
            )
            ids = [r[0] for r in cursor.fetchall()]
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            cursor.execute(
                f"UPDATE reminders SET status='processing', locked_at=?, locked_by=? WHERE id IN ({placeholders})",
                (now_local_iso, worker_id, *ids),
            )
            cursor.execute(f"SELECT * FROM reminders WHERE id IN ({placeholders})", ids)
            return [dict(r) for r in cursor.fetchall()]

    async def check_reminders(self, helper: Any) -> None:
        """
        Проверка и отправка напоминаний
        """
        if self.db_handle is None:
            return

        # Rows left in status='sent' had a successful send whose follow-up
        # DELETE failed on a prior tick; retry only the delete here, never
        # the send (see the send/delete split in the loop below).
        sent_rows = await self.db_handle.fetch_all("SELECT id FROM reminders WHERE status = 'sent'")
        for row in sent_rows:
            try:
                await self.db_handle.execute("DELETE FROM reminders WHERE id = ?", (row["id"],))
            except Exception:
                logging.exception(
                    "Reminders: retry-delete failed for reminder %s marked as sent", row["id"],
                )

        now_local_iso = datetime.now().isoformat(timespec="seconds")
        # No timespec="seconds" here: fire_at_utc is stored with full microsecond
        # precision (reminders.py set_reminder), and truncating this comparison
        # value to seconds could delay firing by up to ~1s at a second boundary.
        now_utc_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        lease_cutoff_iso = (datetime.now() - timedelta(seconds=REMINDER_LEASE_SECONDS)).isoformat(timespec="seconds")
        worker_id = str(os.getpid())

        reminders = await self.db_handle.run_sync(
            self._claim_due_reminders_sync, now_local_iso, now_utc_iso, lease_cutoff_iso, worker_id,
        )

        for reminder in reminders:
            reminder_id = reminder["id"]
            try:
                reminder_for_send = dict(reminder)
                reminder_for_send["user_id"] = reminder["owner_id"]
                await self.send_reminder(reminder_for_send, helper)
            except Exception as exc:
                attempts = int(reminder.get("send_attempts") or 0) + 1
                if attempts >= self._MAX_SEND_ATTEMPTS:
                    logging.error(
                        "Reminders: giving up on reminder %s for user %s after %d attempts: %s",
                        reminder_id, reminder.get("owner_id"), attempts, exc,
                    )
                    await self.db_handle.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
                else:
                    logging.warning(
                        "Reminders: send failed for reminder %s (attempt %d/%d): %s",
                        reminder_id, attempts, self._MAX_SEND_ATTEMPTS, exc,
                    )
                    await self.db_handle.execute(
                        '''UPDATE reminders
                           SET send_attempts = ?, status = 'pending', locked_at = NULL, locked_by = NULL
                           WHERE id = ?''',
                        (attempts, reminder_id),
                    )
                continue

            # Send succeeded outside the except above: a failure past this
            # point must never be treated as a send failure (that would
            # resend on the next tick). Fall back to a 'sent' marker so a
            # later tick retries only the deletion, never the send.
            try:
                await self.db_handle.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            except Exception:
                logging.exception(
                    "Reminders: sent reminder %s but failed to delete the row; marking as "
                    "sent so a later tick retries only the deletion", reminder_id,
                )
                await self.db_handle.execute("UPDATE reminders SET status = 'sent' WHERE id = ?", (reminder_id,))

    async def send_reminder(self, reminder: Dict, helper) -> None:
        """
        Отправка напоминания через выбранную интеграцию
        """
        if reminder["integration"] == "telegram":
            # target_chat_id — куда слать (группа или личка);
            # fallback на user_id для старых записей без target_chat_id
            chat_id = reminder.get("target_chat_id") or reminder.get("user_id")
            await helper.send_message(
                chat_id=chat_id,
                text=self.t("reminders_notification", message=reminder['message']),
                reply_to_message_id=reminder.get("reply_to_message_id")
            )
        # Здесь можно добавить другие интеграции (email, slack)

    def _get_reply_to_message_id(self, kwargs):
        request_context = kwargs.get("request_context")
        if request_context is not None:
            return request_context.message_id
        return kwargs.get("message_id")

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """
        Выполнение функций плагина
        """
        # owner_id — кто создал (from_user.id), используется как ключ верхнего уровня.
        # В группах chat_id != user_id; берём user_id если он есть, иначе chat_id.
        owner_id = str(kwargs.get('user_id') or kwargs.get('chat_id'))
        if function_name == "set_reminder":
            reminder_id = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{owner_id}'
            # Convert datetime to ISO format string
            reminder_time = datetime.strptime(kwargs["time"], "%Y-%m-%d %H:%M")

            # Вычислить fire_at_utc из current_time если передан (исправление TZ)
            fire_at_utc = None
            current_time_str = kwargs.get("current_time")
            if current_time_str:
                try:
                    parsed_local_now = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M")
                    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
                    offset = parsed_local_now - utc_now
                    fire_at_utc = (reminder_time - offset).isoformat()
                except (ValueError, TypeError):
                    pass  # деградируем к legacy-сравнению по серверному времени

            target_chat_id = str(kwargs.get('chat_id') or owner_id)
            reply_to_message_id = self._get_reply_to_message_id(kwargs)
            created_at = datetime.now().isoformat(timespec="seconds")

            await self.db_handle.execute(
                '''INSERT INTO reminders (
                    id, owner_id, target_chat_id, time, fire_at_utc, message, integration,
                    reply_to_message_id, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?)''',
                (
                    reminder_id, owner_id, target_chat_id, reminder_time.isoformat(), fire_at_utc,
                    kwargs["message"], kwargs["integration"], reply_to_message_id, created_at,
                ),
            )

            return {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": self.t("reminders_set_at", time=kwargs['time'])
                }
            }

        elif function_name == "list_reminders":
            rows = await self.db_handle.fetch_all(
                "SELECT * FROM reminders WHERE owner_id = ? AND status != 'sent' ORDER BY created_at ASC", (owner_id,)
            )
            if not rows:
                value = self.t("reminders_none")
            else:
                lines = [self.t("reminders_title")]
                for reminder in rows:
                    reminder_time = datetime.fromisoformat(reminder['time'])
                    formatted_time = reminder_time.strftime('%d.%m.%Y %H:%M')
                    lines.append(f"- {formatted_time} - {reminder['message']} (id: `{reminder['id']}`)")
                value = "\n".join(lines)

            return {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": value
                }
            }

        elif function_name == "delete_reminder":
            reminder_id_to_delete = kwargs.get("reminder_id") or kwargs.get("query")
            row = await self.db_handle.fetch_one(
                "SELECT id FROM reminders WHERE owner_id = ? AND id = ?", (owner_id, reminder_id_to_delete)
            )

            if row is not None:
                await self.db_handle.execute(
                    "DELETE FROM reminders WHERE owner_id = ? AND id = ?", (owner_id, reminder_id_to_delete)
                )
                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": self.t("reminders_deleted", reminder_id=reminder_id_to_delete)
                    }
                }

            return {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": self.t("reminders_not_found")
                }
            }

        return {"error": self.t("reminders_unknown_function")}

    async def handle_reminder_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик callback-запросов от кнопок напоминаний"""
        query = update.callback_query
        if query is None:
            return

        try:
            assert query.data is not None
            action, command, reminder_id = query.data.split(":")

            # Обработка закрытия меню
            if action == "reminder" and command == "close_menu":
                await query.answer(self.t("reminders_menu_closed"))
                assert isinstance(query.message, Message)
                await query.message.delete()
                return

            if action == "reminder" and command == "view":
                user_id = str(query.from_user.id)

                # Проверяем существование напоминания
                row = await self.db_handle.fetch_one(
                    "SELECT * FROM reminders WHERE owner_id = ? AND id = ?", (user_id, reminder_id)
                )
                if row is not None:
                    reminder_time = datetime.fromisoformat(row['time'])
                    formatted_time = reminder_time.strftime('%d.%m.%Y %H:%M')

                    # Показываем детали напоминания во всплывающем окне
                    await query.answer(
                        text=self.t(
                            "reminders_popup_details",
                            time=formatted_time,
                            message=row['message']
                        ),
                        show_alert=True,
                        cache_time=0
                    )
                    return
                else:
                    await query.answer(self.t("reminders_not_found"), show_alert=True)
                    return

            if action == "reminder" and command == "delete":
                user_id = str(query.from_user.id)
                await query.answer()  # Отвечаем на callback запрос для удаления

                # Проверяем существование напоминания
                existing = await self.db_handle.fetch_one(
                    "SELECT id FROM reminders WHERE owner_id = ? AND id = ?", (user_id, reminder_id)
                )
                if existing is not None:
                    # Удаляем напоминание
                    await self.db_handle.execute(
                        "DELETE FROM reminders WHERE owner_id = ? AND id = ?", (user_id, reminder_id)
                    )

                    # Обновляем сообщение со списком напоминаний
                    rows = await self.db_handle.fetch_all(
                        "SELECT * FROM reminders WHERE owner_id = ? AND status != 'sent' ORDER BY created_at ASC",
                        (user_id,)
                    )
                    if rows:
                        await query.edit_message_text(
                            text=self.t("reminders_title"),
                            reply_markup=self._build_reminders_keyboard(rows),
                            parse_mode='markdown'
                        )
                    else:
                        await query.edit_message_text(
                            text=self.t("reminders_none"),
                            parse_mode='markdown'
                        )

                    return

        except Exception as e:
            logging.error(f"Ошибка при обработке callback запроса: {e}")
            await query.edit_message_text(
                text=self.t("reminders_delete_error", error=str(e)),
                parse_mode='markdown'
            )
