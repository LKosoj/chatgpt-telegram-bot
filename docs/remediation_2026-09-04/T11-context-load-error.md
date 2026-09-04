# T11. `get_conversation_context`: ошибка чтения ≠ «контекста нет»

Источники задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел T11) и
`docs/architecture_code_review_2026-09-04.md` §4.2 (первый пункт, «`get_conversation_context`
глотает ошибки чтения»). Термины: *сентинел (sentinel)* — заранее условленное «пустое»
значение, которым функция обозначает «данных нет», в отличие от исключения (exception),
которым Python обозначает «что-то пошло не так». Проблема T11 в том, что сейчас оба случая
возвращают один и тот же сентинел, и вызывающий код не может их различить.

## Цель

`Database.get_conversation_context` (`bot/database.py:934-982`) должна:

1. Пробрасывать наверх реальную ошибку чтения (временную — «database is locked»; или
   постоянную — битый JSON в колонке `context`) как исключение, а не подменять её тем же
   значением, что и легитимное «контекста ещё нет».
2. Возвращать факт-соответствующий тип (сейчас аннотация `Optional[Dict[str, Any]]`, реально
   — 5-элементный кортеж).
3. Использовать один дефолт `max_tokens_percent` во всех «нет данных» ветках (сейчас три места
   отдают `80`, а успешный путь — `100`, притом что `100` — это и есть дефолт колонки в схеме:
   `bot/database.py`, `CREATE TABLE ... max_tokens_percent INTEGER DEFAULT 100`).

Ниже — полный обход всех потребителей (`bot/openai_helper.py`, `bot/telegram_bot.py`,
`bot/openai_tool_handler.py`, тесты), который показывает, что почти везде **код уже готов**
к исключению: он либо уже ловит `except Exception` и превращает ошибку в `chat_fail`/текст для
пользователя, либо это заведомо best-effort путь, где так и было задумано «залогировать и
продолжить». Правки по факту нужны только в `bot/database.py`; остальное — подтверждение,
что менять не требуется, с указанием, где именно и почему.

## Анализ потребителей

Легенда столбца «Реакция на исключение»: цитата/пересказ существующего кода, не предположение.

| # | Место (file:line) | Функция | Своя обработка ошибок? | Реакция на исключение | Правка нужна? |
|---|---|---|---|---|---|
| 1 | `bot/database.py:1380-1391` | `get_conversation_context_async` | Нет (тонкая обёртка `_run_db_method` → `_run_in_db_thread` → `loop.run_in_executor`) | `run_in_executor` пробрасывает исключение из sync-функции нативно через `await` — уже работает как надо | Только тип-аннотация |
| 2 | `bot/openai_helper.py:428-442` | `get_conversation_stats` | Нет своего `try` | Пробросится к вызывающему. **Не вызывается нигде в продакшн-коде** (`grep` по всему дереву — только тесты `tests/test_reset_chat_history_async.py`) | Нет |
| 3 | `bot/openai_helper.py:805` (внутри `ask`, 788-846) | `ask` | Да: `except Exception as e: logger.error(...); raise` (`:844-846`) | Пробрасывается дальше уже залогированным | Нет |
| 4 | `bot/openai_helper.py:1178` (внутри `_get_chat_response_stream_locked`, `try` `:1173`/`except` `:1289`) | стриминговый путь | Да, внешний `except Exception as e` (`:1289-1293`) | `yield f"Error generating response: {str(e)}", '0'` — пользователь видит текст ошибки как «ответ», новая сессия не создаётся | Нет (текст не локализован через `chat_fail`, но это существующее поведение для *любого* исключения в этой функции — не регрессия T11, см. «Риски») |
| 5 | `bot/openai_helper.py:1314, 1337` | `resolve_allowed_plugins` | Нет своего `try` | Вызывается из (а) `_get_chat_response_stream_locked:1213` — внутри `try:1212/except:1227-1230`, даёт `yield f"Error in function call: {str(e)}", '0'`; (б) `__common_get_chat_response:1596` — внутри общего `try` функции (см. строку 7) | Нет |
| 6 | `bot/openai_helper.py:1420` | `_maybe_apply_auto_chat_mode` | Нет своего `try` | Единственный вызыватель — `__common_get_chat_response:1471`, внутри её общего `try` (см. строку 7) | Нет |
| 7 | `bot/openai_helper.py:1457` (внутри `__common_get_chat_response`, `try:1450`) | основной non-stream путь | Да: цепочка `except openai.RateLimitError` / `BadRequestError` / `ValueError` / `except Exception as e` (`:1690-1696`) оборачивает в `Exception(f"⚠️ _{localized_text('error', ...)}._ ⚠️\n{error_message}")` и **перевызывает** | Дальше это либо напрямую всплывает в `get_chat_response` (нет своего `try`, только `finally` на `:907`), либо через `ChatRun.run_non_stream` (`bot/chat_run.py:58/253-257`, тоже `except: log; raise`) — в обоих случаях долетает до `bot/telegram_bot.py:4608-4617`, где `except Exception as e: ... text=f"{localized_text('chat_fail', ...)} {error_message}"` — ровно то, что просит задача | Нет |
| 8 | `bot/openai_helper.py:2461` (внутри `__common_get_chat_response_vision`, `try:2458`) | vision-путь | Да: аналогичная цепочка (`:2544-2555`) | Тот же паттерн, что и строка 7 | Нет |
| 9 | `bot/openai_helper.py:3316` (внутри `_dispatch_before_create_session_prune`, per-session `try:3314-3328`) | prune-хук перед `create_session` | Да, явно best-effort: «Failures here must not block session creation» (докстринг `:3295-3296`) — `except Exception: logger.error(...); continue` | Пропускает *эту* сессию, остальные обрабатывает — уже есть тест `tests/test_summarise_overflow_dispatch.py::test_dispatch_before_create_session_prune_continues_on_per_session_failure`, который явно проверяет это на `RuntimeError` | Нет |
| 10 | `bot/openai_helper.py:3385` (внутри `reset_chat_history`, `try:3364-3423`) | сброс истории/создание сессии | Да: `except Exception as e: logger.error(...); raise` (`:3421-3423`) | Уже ровно та цель, к которой стремится T11 — раскрыть, а не проглотить | Нет |
| 11 | `bot/openai_helper.py:3505` (внутри `__add_to_history`) | добавление сообщения в историю | Нет своего `try` | Единственные вызыватели — внутри уже проверенных `try` (строки 4, 7, 8) плюс `chat_run.py:207,218` внутри `ChatRun.run_non_stream`'s `try` (строка 7) | Нет |
| 12 | `bot/openai_helper.py:3537` (внутри `record_plugin_exchange`) | зеркалирование ответа плагина в историю | Нет своего `try` | Единственный вызыватель — `bot/telegram_bot.py:_mirror_plugin_exchange` (`:4735/4767-4773`): `except Exception as exc: logger.warning(...)` — best-effort, потому что пользователь уже получил прямой ответ плагина | Нет |
| 13 | `bot/telegram_bot.py:865` (внутри `_dispatch_session_before_delete`, `try:864-875`) | снимок сессии перед PII-удалением | Да: `except Exception as exc: logger.error(...); raise` | Пробрасывается в `_dispatch_and_delete_oldest_sessions_for_limit:990` (без своего `try`) → к её вызывателям: `:2422` внутри `_handle_prompt_selection_locked`'s `try:2361/except:2504` (`error_with_details`); `:6116` внутри `_handle_session_callback_locked`'s `try:6107/except:6258` (`error_with_details`) | Нет |
| 14 | `bot/telegram_bot.py:2440` | `_handle_prompt_selection_locked`, «холодный» кэш при выборе режима | Косвенно — внутри `try:2361/except:2504` | `error_with_details` пользователю | Нет |
| 15 | `bot/telegram_bot.py:6057` | `_handle_session_callback_locked`, ветка `preview` | **Нет** — этот код (`:6044-6105`) идёт *до* `try:6107` этой же функции | Пробросится необработанным до глобального `error_handler` PTB (`bot/utils.py:647-657`) — тот только логирует, пользователю ничего не покажет | Не обязательна для T11, но есть готовое лёгкое улучшение — см. «Рекомендация (опционально)» |
| 16 | `bot/telegram_bot.py:6150` | `_handle_session_callback_locked`, ветка `switch` | Да, внутри `try:6107/except:6258` | `error_with_details` | Нет |
| 17 | `bot/telegram_bot.py:6167` | `_handle_session_callback_locked`, ветка `delete` | Да, внутри `try:6107/except:6258` | `error_with_details` | Нет |
| 18 | `bot/openai_tool_handler.py:262-278` (`_reentry_session`) | обновление `max_tokens_percent` при повторном заходе после tool-call | Да: `except Exception as exc: logger.warning(...); return session_id, 80` | Не создаёт сессию, деградирует к уже известному `session_id` | Нет (см. заметку о `80` в «Риски») |

Вывод: единственная точка, где сегодня реальная ошибка неотличима от «данных нет» — это сама
`get_conversation_context` (её собственный `except Exception` на `:980-982` и неявный сентинел
при неудавшемся `create_session` на `:951-953`). Всё, что выше по стеку, уже устроено так, что
раскрытое исключение доходит либо до `chat_fail`/`error_with_details` для пользователя, либо до
уже спроектированного best-effort пропуска — без создания лишней сессии.

## Дизайн

### Решение: (а) пробрасывать исключение, без общего сентинел-маркера

Выбран вариант (а) из формулировки задачи. Обоснование:

- Все 18 потребителей выше уже построены вокруг `except Exception` — где нужно, ошибка
  превращается в `chat_fail`/`error_with_details`; где не нужно (хуки best-effort), уже стоит
  `except Exception: log; continue/return`. Вариант (б) — явный маркер (например,
  `ContextLoadError` как *возвращаемое* значение, не исключение) — потребовал бы добавить
  `isinstance(result, ContextLoadError)` в 10+ мест и продублировал бы работу, которую сейчас
  бесплатно делает `except Exception`. Это лишняя абстракция ради проблемы, которой при
  ближайшем рассмотрении в 17 из 18 мест не существует.
- В `bot/openai_helper.py` уже есть точный прецедент нужного паттерна: `reset_chat_history`
  (`bot/openai_helper.py:3364-3423`) — `except Exception as e: logger.error(...); raise`.
  T11 приводит `get_conversation_context` (в `bot/database.py`) к тому же паттерну.

Как исключения используются: не «голый» `except Exception: raise` без узла — вводятся два
целевых типа, оба — подкласс `RuntimeError`, оба определены в `bot/database.py` рядом с
классом `Database` (модуль пока не содержит собственных исключений):

```python
class ConversationContextError(RuntimeError):
    """Не удалось загрузить контекст разговора из-за отказа хранилища
    (заблокированная БД, неожиданная ошибка драйвера, либо сама сессия не
    смогла создаться). Не путать с легитимным «загружать ещё нечего»
    (свежий пользователь, либо session_id, которому не соответствует ни одной
    строки) — это по-прежнему обычный ConversationContextResult с context=None.
    Вызывающий код не должен в ответ на это исключение создавать новую сессию.
    """


class ConversationContextCorruptError(ConversationContextError):
    """Колонка conversation_context.context не парсится как JSON. В отличие
    от заблокированной БД это проблема данных, а не временная — повтор не
    поможет. Отдельный подкласс (а не переиспользование ConversationContextError)
    даёт возможность найти именно эти случаи по логам/типу исключения для
    ручного разбора одной строки, при этом весь существующий код, ловящий
    Exception/ConversationContextError, продолжает реагировать одинаково.
    """
```

Почему не «лог + карантин строки» для битого JSON (что задача предлагала как альтернативу):
карантин означал бы, что *функция чтения* тихо пишет в БД (перемещает/обнуляет повреждённую
строку) — неожиданный побочный эффект для read-пути, новая таблица/миграция ради ещё
не существующего потребителя, и риск потерять данные, если «повреждение» было не повреждением,
а гонкой при чтении на границе транзакции. Вместо этого: явный отдельный тип исключения +
подробный `logger.error` с `user_id`/`session_id` (без содержимого битой строки, чтобы не
раздувать логи) — этого достаточно, чтобы оператор нашёл строку и поправил её вручную
(`UPDATE conversation_context SET context='{"messages": []}' WHERE user_id=... AND
session_id=...`), а функция чтения остаётся чистой (без записи).

### Что классифицируется как ошибка, а что — как легитимное «данных нет»

Через `get_conversation_context` сегодня проходят четыре варианта возврата «нет данных»
(`bot/database.py:934-982`); только один из них — реальная ошибка:

1. `:948-953` — активной сессии нет, `session_id` не передан → вызывается `create_session(...)`.
   Если она вернула `None` — это **не** «данных нет», а замаскированный отказ: `create_session`
   сама ловит все исключения и возвращает `None` (`bot/database.py:1305-1307`,
   `except Exception as e: logger.error(...); return None`) — то есть `None` здесь *всегда*
   означает, что внутри было реальное исключение. **Меняем на `raise
   ConversationContextError(...)`** — иначе `get_conversation_context` замаскирует эту ошибку
   второй раз тем же сентинелом, а вызывающий код (например, `_get_chat_response_stream_locked`)
   пойдёт в `reset_chat_history(session_id=None)`, который попробует создать сессию ещё раз и
   в итоге тоже упадёт (`ValueError` на `bot/openai_helper.py:3381`) — то есть результат тот же
   `chat_fail`, но через двойную попытку записи и двойной лог. Раскрытие сразу даёт тот же
   `chat_fail` без дублирования.
2. `:957-960` — защитная ветка (`session_id` всё ещё не определён после блока выше).
   На практике недостижима: к этой точке `session_id` гарантированно взят из `create_session`
   (успешно — иначе см. пункт 1), из активной строки (`result[0]`, где `result` был найден и
   потому не может быть `NULL`), либо это исходный `session_id`, переданный вызывающим кодом
   (тогда он не мог быть falsy, иначе сработала бы ветка 1). Оставляем как есть — не ошибка,
   не трогаем логику, только унифицируем дефолт (80→100).
3. `:977-978` — `session_id` определён, но `SELECT` по нему не вернул строку (сессию успели
   удалить/спрунить, либо вызывающий код передал устаревший `session_id`). Это результат
   штатного SQL-запроса без исключений — не ошибка чтения, а «для этого id данных нет».
   Существующие потребители (`resolve_allowed_plugins`, `reset_chat_history`) уже рассчитаны на
   этот исход как на нормальный (например, `resolve_allowed_plugins` при отсутствии системного
   сообщения сама вызывает `reset_chat_history`). Не меняем — только дефолт.
4. `:980-982` — блок `except Exception`, который сегодня ловит вообще всё: реальные ошибки
   соединения/курсора (`sqlite3.OperationalError` и т.п.) и `json.JSONDecodeError` при разборе
   колонки `context`. **Это единственное место, которое действительно скрывает ошибку.**
   Убираем перехват; `json.JSONDecodeError`/`TypeError` из `json.loads` перехватываются точечно
   и заворачиваются в `ConversationContextCorruptError` (см. код ниже), всё остальное — просто
   не перехватываем на этом уровне (не оборачиваем зря: `sqlite3.OperationalError` и так
   осмысленное имя, и ни один из 18 потребителей не различает типы исключений — всем достаточно
   `except Exception`, так что оборачивание не добавило бы им ничего, а по конвенции файла
   ошибки соединения и так пробрасываются нативно — см. `tests/test_database.py::
   test_outer_commit_failure_rolls_back`, которая явно ожидает `sqlite3.OperationalError`
   наружу без обёртки).

Единственное добавление сверх «просто убрать `except`» — обёртка *вокруг всей функции*, которая
не глотает, а только диагностически логирует user_id/session_id перед повторным `raise` (тот же
приём, что уже применён в `reset_chat_history`), чтобы в логе на месте отказа сразу было видно,
для кого он произошёл — иначе эта информация теряется к моменту, когда исключение долетает до
общих `except Exception` в `openai_helper.py`.

### Тип возврата: `NamedTuple` вместо кортежа без структуры

```python
class ConversationContextResult(NamedTuple):
    context: Optional[Dict[str, Any]]
    parse_mode: str
    temperature: float
    max_tokens_percent: int
    session_id: Optional[str]
```

`NamedTuple` — это подкласс обычного `tuple`, поэтому весь существующий код вида
`context, parse_mode, temperature, max_tokens_percent, session_id = await
self._db_call("get_conversation_context", ...)` (все места из таблицы выше) продолжает
работать без изменений — распаковка по позиции не отличает `NamedTuple` от «голого» кортежа.
Так же не ломаются тесты, которые задают моки через `MagicMock(return_value=(...))` или
`side_effect=[(...), (...)]` — они возвращают обычные кортежи, а не `ConversationContextResult`,
и это нормально: реальный тип нужен только там, где вызывается настоящая `Database`.

### `create_session` (`bot/database.py:1214-1307`) — не трогаем, но фиксируем ограничение

`create_session` продолжает сама ловить исключения и возвращать `None`
(`bot/database.py:1305-1307`) — вне scope T11 (task называет только `get_conversation_context`).
Следствие: после исправления `get_conversation_context` при неудаче `create_session` бросит
`ConversationContextError("Не удалось создать сессию для пользователя …")`, но **исходное**
исключение (то, что случилось внутри `create_session`) в `__cause__` не попадёт — оно видно
только в соседней строке лога `create_session`'а («Ошибка при создании сессии: …»,
`exc_info=True`). Обе строки лога окажутся рядом по времени, так что для диагностики этого
достаточно; если понадобится точнее — отдельная задача (не T11) сделать так, чтобы
`create_session` сама пробрасывала исключение, а не глотала его.

### `get_conversation_context_async` и `DbHandle` — проверено, изменений не требуется

- `get_conversation_context_async` (`bot/database.py:1380-1391`) — `_run_db_method` вызывает
  `_run_in_db_thread`, которая делает `await loop.run_in_executor(...)`; исключение из
  sync-функции, выполненной в executor'е, `run_in_executor` поднимает наружу автоматически при
  `await` — дополнительный код не нужен, меняется только type hint (см. правки).
- `DbHandle` (`bot/plugins/db_handle.py`) не вызывает `Database.get_conversation_context` по
  имени — у него только «сырые» SQL-методы (`execute`/`fetch_one`/`fetch_all`/…). Единственные
  плагины, читающие таблицу `conversation_context`, — `bot/plugins/hindsight_memory.py:909,
  1461` — делают это прямым SQL через `db_handle`, минуя `get_conversation_context` целиком.
  Т.е. `DbHandle` этим изменением не затрагивается.

## Правки

### 1. `bot/database.py:5` — добавить `NamedTuple` в импорт

```python
# было
from typing import Dict, Any, Optional, List, Generator
# стало
from typing import Dict, Any, Optional, List, Generator, NamedTuple
```

### 2. `bot/database.py:62` (после `_numeric_env`, перед `class Database:`) — новые типы

```python
class ConversationContextResult(NamedTuple):
    """Структурированный результат ``Database.get_conversation_context``.

    Оставлен как tuple (не dataclass), чтобы позиционная распаковка на местах
    вызова (``context, parse_mode, temperature, max_tokens_percent, session_id
    = ...``) продолжала работать без изменений.
    """

    context: Optional[Dict[str, Any]]
    parse_mode: str
    temperature: float
    max_tokens_percent: int
    session_id: Optional[str]


class ConversationContextError(RuntimeError):
    """Не удалось загрузить контекст разговора из-за отказа хранилища
    (заблокированная БД, неожиданная ошибка драйвера, либо сама сессия не
    смогла создаться). Не путать с легитимным «загружать ещё нечего» —
    это по-прежнему обычный ConversationContextResult с context=None.
    Вызывающий код не должен в ответ на это исключение создавать новую сессию.
    """


class ConversationContextCorruptError(ConversationContextError):
    """conversation_context.context не парсится как JSON. В отличие от
    заблокированной БД это проблема данных, а не временная — повтор не
    поможет. Отдельный подкласс — чтобы находить именно эти случаи по типу
    исключения для ручного разбора одной строки.
    """
```

### 3. `bot/database.py:934-982` — тело `get_conversation_context`

```python
# было (bot/database.py:934-982)
    def get_conversation_context(self, user_id: int, session_id: str = None, openai_helper = None) -> Optional[Dict[str, Any]]:
        """Получение контекста разговора с поддержкой сессий"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Проверяем наличие активной сессии
                cursor.execute('''
                    SELECT session_id FROM conversation_context
                    WHERE user_id = ? AND is_active = 1
                ''', (user_id,))
                result = cursor.fetchone()

                # Если нет активной сессии и не указан session_id, создаем новую
                if not result and not session_id:
                    logger.info(f"Создаем новую сессию для пользователя {user_id}")
                    session_id = self.create_session(user_id, openai_helper=openai_helper)
                    if not session_id:
                        logger.warning(f"Не удалось создать сессию для пользователя {user_id}")
                        return None, 'HTML', 0.8, 80, None
                elif not session_id and result:
                    session_id = result[0]

                # Если сессия всё ещё не определена, используем значения по умолчанию
                if not session_id:
                    logger.warning(f"Не удалось определить сессию для пользователя {user_id}")
                    return None, 'HTML', 0.8, 80, None

                cursor.execute('''
                    SELECT context, parse_mode, temperature, max_tokens_percent
                    FROM conversation_context
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))

                result = cursor.fetchone()
                if result:
                    context = json.loads(result[0]) if result[0] is not None else {'messages': []}
                    parse_mode = result[1] if result[1] is not None else 'HTML'
                    temperature = round(result[2], 2) if result[2] is not None else 0.8
                    max_tokens_percent = result[3] if result[3] is not None else 100

                    return context, parse_mode, temperature, max_tokens_percent, session_id

                logger.info(f"Контекст не найден для сессии {session_id}, возвращаем значения по умолчанию")
                return None, 'HTML', 0.8, 80, None

        except Exception as e:
            logger.error(f'Ошибка получения контекста сессии: {e}', exc_info=True)
            return None, 'HTML', 0.8, 80, None
```

```python
# стало
    def get_conversation_context(
        self, user_id: int, session_id: str = None, openai_helper = None
    ) -> ConversationContextResult:
        """Получение контекста разговора с поддержкой сессий.

        Бросает ConversationContextError (или её подкласс
        ConversationContextCorruptError для битого JSON) вместо того, чтобы
        подменять отказ чтения тем же результатом, что и «контекста ещё нет».
        Вызывающий код не должен создавать новую сессию в ответ на это
        исключение. Легитимное «загружать нечего» (свежий пользователь,
        либо session_id без единой строки) по-прежнему возвращается как
        обычный ConversationContextResult с context=None — это не ошибка.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Проверяем наличие активной сессии
                cursor.execute('''
                    SELECT session_id FROM conversation_context
                    WHERE user_id = ? AND is_active = 1
                ''', (user_id,))
                result = cursor.fetchone()

                # Если нет активной сессии и не указан session_id, создаем новую
                if not result and not session_id:
                    logger.info(f"Создаем новую сессию для пользователя {user_id}")
                    session_id = self.create_session(user_id, openai_helper=openai_helper)
                    if not session_id:
                        # create_session сама ловит все свои исключения и в этом
                        # случае уже залогировала причину — здесь это не «данных
                        # нет», а замаскированный отказ записи.
                        raise ConversationContextError(
                            f"Не удалось создать сессию для пользователя {user_id}"
                        )
                elif not session_id and result:
                    session_id = result[0]

                # Защитная ветка: практически недостижима — к этой точке
                # session_id уже гарантированно взят из create_session, из
                # активной строки, либо это исходный аргумент вызывающего кода.
                if not session_id:
                    logger.warning(f"Не удалось определить сессию для пользователя {user_id}")
                    return ConversationContextResult(None, 'HTML', 0.8, 100, None)

                cursor.execute('''
                    SELECT context, parse_mode, temperature, max_tokens_percent
                    FROM conversation_context
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))

                result = cursor.fetchone()
                if result:
                    try:
                        context = json.loads(result[0]) if result[0] is not None else {'messages': []}
                    except (TypeError, json.JSONDecodeError) as e:
                        # Ошибка данных, не временная — отдельный подкласс, чтобы
                        # её можно было найти по типу исключения и поправить
                        # строку вручную, а не тихо подменять контекст пустым.
                        raise ConversationContextCorruptError(
                            f"Повреждён JSON контекста user_id={user_id}, session_id={session_id}: {e}"
                        ) from e
                    parse_mode = result[1] if result[1] is not None else 'HTML'
                    temperature = round(result[2], 2) if result[2] is not None else 0.8
                    max_tokens_percent = result[3] if result[3] is not None else 100

                    return ConversationContextResult(
                        context, parse_mode, temperature, max_tokens_percent, session_id
                    )

                logger.info(f"Контекст не найден для сессии {session_id}, возвращаем значения по умолчанию")
                return ConversationContextResult(None, 'HTML', 0.8, 100, None)

        except ConversationContextError:
            raise
        except Exception as e:
            logger.error(
                f'Ошибка получения контекста сессии user_id={user_id} session_id={session_id}: {e}',
                exc_info=True,
            )
            raise
```

Изменения по сути: (1) убран сентинел при неудаче `create_session` → `raise
ConversationContextError`; (2) точечный перехват `json.loads` → `raise
ConversationContextCorruptError ... from e`; (3) внешний `except Exception` больше не глотает —
логирует с `user_id`/`session_id` и пробрасывает; (4) все дефолты `80` → `100`; (5) возвращаемый
тип — `ConversationContextResult` вместо «голого» кортежа.

### 4. `bot/database.py:1380-1391` — тип-аннотация `get_conversation_context_async`

```python
# было
    async def get_conversation_context_async(
        self,
        user_id: int,
        session_id: str = None,
        openai_helper = None,
    ) -> Optional[Dict[str, Any]]:
        return await self._run_db_method(
            "get_conversation_context",
            user_id,
            session_id,
            openai_helper,
        )
```

```python
# стало
    async def get_conversation_context_async(
        self,
        user_id: int,
        session_id: str = None,
        openai_helper = None,
    ) -> ConversationContextResult:
        """См. ``get_conversation_context`` — исключения (включая
        ConversationContextError/ConversationContextCorruptError) пробрасываются
        через ``_run_in_db_thread``/``run_in_executor`` без изменений."""
        return await self._run_db_method(
            "get_conversation_context",
            user_id,
            session_id,
            openai_helper,
        )
```

Тело не меняется — только тип-аннотация и уточняющий докстринг.

### `bot/openai_helper.py`, `bot/telegram_bot.py`, `bot/openai_tool_handler.py` — без изменений

Как показано в таблице «Анализ потребителей», все 17 мест за пределами `bot/database.py` уже
реагируют на исключение так, как того требует задача (не создают сессию, показывают
`chat_fail`/`error_with_details`, либо это осознанный best-effort пропуск с существующим
тестом). Изменять их не нужно — правка сентинела в `get_conversation_context` меняет то, что
они *получают*, а не то, как они на это реагируют.

### Рекомендация (опционально, отдельно от обязательного объёма T11)

`bot/telegram_bot.py:6044-6105` (`_handle_session_callback_locked`, ветка `preview`) не имеет
своего `try`/`except` — при отказе чтения исключение сегодня (через `AttributeError` на
`None.get(...)`) и после правки (через `ConversationContextError`) одинаково долетает до
глобального `error_handler` PTB и не показывается пользователю. Регрессии нет (поведение то
же самое, до и после), но раз уж потребители обходились, стоит для симметрии с соседними
ветками `switch`/`delete` (`:6150, 6167`, которые уже внутри `try:6107/except:6258`) либо
расширить этот `try` на ветку `preview`, либо обернуть `preview` в такой же локальный
`try/except`, показывающий `error_with_details`. Альтернатива — оставить как есть: ветка
редко используется и деградирует не хуже, чем сегодня. Решение — на усмотрение разработчика/
ревьюера T11, в обязательный список правок не включено.

## Тесты

Новые/обновляемые тесты — в `tests/test_database.py`, рядом с существующими тестами
`get_conversation_context` (`test_malformed_max_sessions_env_falls_back_for_real_session_paths`,
`test_create_session_copies_active_mode_before_pruning_oldest`, миграционный тест на
`:447`) — эти три существующих теста упражняют только happy-path и должны остаться зелёными
без изменений (регрессионная проверка «happy-path не меняется»).

```python
import sqlite3
import pytest
from bot.database import (
    ConversationContextError,
    ConversationContextCorruptError,
)


def test_get_conversation_context_propagates_read_error_without_creating_session(db):
    """Временная ошибка чтения не должна создавать новую сессию."""
    helper = DummyOpenAI()
    session_id = db.create_session(1, openai_helper=helper)
    sessions_before = db.list_user_sessions(1)

    real_execute = sqlite3.Cursor.execute

    def failing_execute(self, sql, *args, **kwargs):
        if "SELECT context, parse_mode" in sql:
            raise sqlite3.OperationalError("database is locked")
        return real_execute(self, sql, *args, **kwargs)

    import unittest.mock as mock
    with mock.patch.object(sqlite3.Cursor, "execute", failing_execute):
        with pytest.raises(sqlite3.OperationalError):
            db.get_conversation_context(1, session_id)

    # Ни одна новая сессия не появилась и не пропала.
    assert db.list_user_sessions(1) == sessions_before


def test_get_conversation_context_raises_on_create_session_failure(db, monkeypatch):
    """create_session вернула None (замаскированное исключение) -> явная ошибка,
    а не сентинел «данных нет»."""
    monkeypatch.setattr(db, "create_session", lambda *a, **k: None)

    with pytest.raises(ConversationContextError):
        db.get_conversation_context(999)  # нет активной сессии, session_id не передан


def test_get_conversation_context_raises_on_corrupt_json(db):
    """Битый JSON в context — ошибка данных, не тихий сентинел."""
    helper = DummyOpenAI()
    session_id = db.create_session(1, openai_helper=helper)
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE conversation_context SET context = ? WHERE user_id = ? AND session_id = ?",
            ("{not valid json", 1, session_id),
        )

    with pytest.raises(ConversationContextCorruptError):
        db.get_conversation_context(1, session_id)


def test_get_conversation_context_missing_session_defaults_to_100_not_80(db):
    """Унификация дефолта max_tokens_percent: legit «нет данных» -> 100, не 80."""
    context, parse_mode, temperature, max_tokens_percent, session_id = (
        db.get_conversation_context(1, "definitely-missing-session-id")
    )
    assert context is None
    assert max_tokens_percent == 100


@pytest.mark.asyncio
async def test_get_conversation_context_async_propagates_corrupt_json_error(db):
    """Асинхронная обёртка не глотает исключение из sync-метода."""
    helper = DummyOpenAI()
    session_id = await db.create_session_async(1, openai_helper=helper)
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE conversation_context SET context = ? WHERE user_id = ? AND session_id = ?",
            ("{not valid json", 1, session_id),
        )

    with pytest.raises(ConversationContextCorruptError):
        await db.get_conversation_context_async(1, session_id)
```

Плюс один тест на уровне `OpenAIHelper` (в `tests/test_reset_chat_history_async.py`, по образцу
уже существующей фикстуры `_make_helper_stats`) — подтверждает требование задачи «ошибка чтения
не создаёт сессию» именно там, где это проверяется пользователем задачи:

```python
@pytest.mark.asyncio
async def test_get_conversation_stats_propagates_error_without_creating_session():
    helper = object.__new__(OpenAIHelper)
    helper.conversations = {}
    helper.loaded_conversation_sessions = {}
    helper.config = {'max_sessions': 5}
    helper.db = SimpleNamespace(
        create_session_async=AsyncMock(side_effect=AssertionError("must not be called")),
        get_conversation_context_async=AsyncMock(
            side_effect=ConversationContextError("db locked")
        ),
    )

    with pytest.raises(ConversationContextError):
        await helper.get_conversation_stats(42)

    helper.db.create_session_async.assert_not_called()
```

Существующее покрытие, которое подтверждает, что best-effort ветка не нужно менять и не
регрессирует: `tests/test_summarise_overflow_dispatch.py::
test_dispatch_before_create_session_prune_continues_on_per_session_failure` (уже проверяет, что
исключение из `get_conversation_context` в одной из сессий не блокирует обработку остальных).

## Команды проверки

```bash
# baseline — подтвердить, что оба файла из задания зелёные до правок
python3 -m pytest tests/test_database.py tests/test_openai_helper_db_offload.py -q -p no:cacheprovider

# после правок в bot/database.py — те же файлы плюс новые тесты выше
python3 -m pytest tests/test_database.py tests/test_openai_helper_db_offload.py \
  tests/test_reset_chat_history_async.py tests/test_summarise_overflow_dispatch.py \
  -q -p no:cacheprovider

# более широкий прогон — потребители из таблицы выше, ни один не должен был измениться
python3 -m pytest tests/test_openai_helper_tool_calls.py tests/test_group_session_flow.py \
  tests/test_telegram_streaming.py tests/test_callback_authorization.py \
  tests/test_hindsight_memory.py tests/test_per_conversation_serialization.py \
  tests/test_skills_agent_gate.py -q -p no:cacheprovider

# полный прогон, без evals/ (см. AGENTS.md Testing And Verification)
python3 -m pytest -q -p no:cacheprovider
```

## Риски

- **`create_session`'а собственное глотание исключений (`bot/database.py:1305-1307`) остаётся
  вне scope.** После правки `get_conversation_context` корректно поднимет
  `ConversationContextError`, но без оригинальной причины в `__cause__` — она видна только в
  соседней строке лога `create_session`. Приемлемо (обе строки рядом по времени), но стоит
  явно проговорить с ревьюером, что это осознанная граница, а не недосмотр.
- **Стриминговый путь показывает нелокализованный текст ошибки** (`bot/openai_helper.py:1293`,
  `f"Error generating response: {str(e)}"`) вместо `chat_fail`. Это существующее поведение для
  *любого* исключения в `_get_chat_response_stream_locked`, не специфичное для T11 и не
  регрессия — но если ревьюер сочтёт нужным унифицировать текст с `chat_fail`, это отдельная,
  более широкая правка (задевает форматирование всех ошибок стрима, не только контекста).
- **`bot/telegram_bot.py:6044-6105` (`preview`) без своего `try`** — подробно разобрано в
  «Рекомендация (опционально)»: не регрессия, но кандидат на отдельный маленький фикс.
- **`bot/openai_tool_handler.py:277,278,285` использует свой независимый дефолт `max_tokens_percent
  = 80`** при деградации в `_reentry_session` — это не то же самое значение, что дефолт колонки
  БД, но и не связано напрямую с сентинелом `get_conversation_context` (там уже стоит
  `except Exception`, до которого добираются раньше, чем до дефолта колонки). Можно
  синхронизировать с `100` для единообразия, но это отдельная, не обязательная для T11 правка
  (файл не входит в список потребителей, явно указанный в задаче).
- **Тест на «database is locked»** в разделе «Тесты» патчит `sqlite3.Cursor.execute` глобально
  на время `with` — если параллельно с этим тестом в процессе выполняется другой тест с реальным
  доступом к SQLite (не должно происходить при обычном последовательном запуске pytest, но
  стоит проверить при `-n auto`/`pytest-xdist`, если он используется в CI).
- **Меняется публичный тип исключения** при отказе чтения: было `None`-подобный кортеж, стало
  `ConversationContextError`/`ConversationContextCorruptError`/нативные `sqlite3.*` ошибки.
  Любой внешний код (вне этого репозитория), который наблюдает `Database.get_conversation_context`
  напрямую и не проверил задачу T11, увидит новое исключение вместо старого сентинела —
  ожидаемо и является целью задачи, но стоит явно отметить в описании PR/коммита как breaking
  change контракта функции (не публичного API бота — `Database` не экспортируется наружу
  процесса).

## Критерии готовности

- `bot/database.py`: `get_conversation_context` возвращает `ConversationContextResult` на
  легитимном «данных нет» и бросает `ConversationContextError`/`ConversationContextCorruptError`
  на реальном отказе чтения/создания сессии; все дефолты `max_tokens_percent` в этой функции —
  `100`; аннотация типа у `get_conversation_context`/`get_conversation_context_async` соответствует
  факту.
- Новые тесты из раздела «Тесты» зелёные; существующие `tests/test_database.py`,
  `tests/test_openai_helper_db_offload.py`, `tests/test_reset_chat_history_async.py`,
  `tests/test_summarise_overflow_dispatch.py` и весь список из «Команды проверки» — зелёные без
  изменений в самих тестах (кроме добавленных новых).
- Полный `python3 -m pytest -q -p no:cacheprovider` (без `evals/`) зелёный.
- В `bot/openai_helper.py`, `bot/telegram_bot.py`, `bot/openai_tool_handler.py` изменений нет
  (кроме опциональной правки `preview`-ветки, если ревьюер её одобрит) — таблица «Анализ
  потребителей» приложена как обоснование, почему это не требуется.

## Постскриптум после ревью

Ревью (Sonnet, персона reviewer) нашло одну регрессию, пропущенную таблицей потребителей:
`ChatGPTTelegramBot._dispatch_session_before_delete` перебрасывала ошибку чтения снимка, и
после T11 отказ чтения (в т.ч. `ConversationContextCorruptError`) блокировал удаление сессии
и prune-перед-созданием новой. Исправлено: снимок для хука best-effort — при ошибке
логируется и возвращается `0`, удаление продолжается (Policy A из AGENTS.md). Тест
`test_dispatch_session_before_delete_swallows_snapshot_read_error` в
`tests/test_group_session_flow.py`.

Отклонение от плана в тесте: `mock.patch.object(sqlite3.Cursor, "execute", ...)` невозможен
на Python 3.12 (immutable C-тип), использован паттерн подмены `db._local.connection`.

Замечание без действия (не в рамках T11): `Database.list_user_sessions` при битом JSON в
одной сессии возвращает пустой список всех сессий (`except Exception: return []`).
