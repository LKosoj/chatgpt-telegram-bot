# T02 — план: попутные баги + доступ в группах 1.4

Ссылки на master-plan: `docs/improvement_2026-09-25/00-master-plan.md`, раздел
«T02. Попутные баги + доступ в группах 1.4 (волна 1)» (строки 89-121 на момент
написания этого плана).

Владение файлами (не выходить за пределы списка): `bot/utils.py` (только `is_allowed`,
`_charge_user_and_guest`/`_charge_user_and_guest_async`, при необходимости мелкий
помощник рядом), `bot/__main__.py`, `bot/plugins/chief.py`, `bot/telegram_stream.py`,
`bot/telegram_bot.py` (только строки с `exc.retry_after`), `README.md`, `README.ru.md`,
`.env.example`, тесты: `tests/test_callback_authorization.py`, `tests/test_usage_budget.py`,
`tests/test_chief_model_choice.py`, `tests/test_telegram_stream_core.py`.

Окружение: `~/.venvs/ctb/bin/python`. Тесты:
`~/.venvs/ctb/bin/python -m pytest <paths> -q --no-header -p no:cacheprovider`.
`rg`/`grep` в этой среде искажают вывод — искать через Read/Grep или инлайн `python3`.

## Проверка расхождений с мастер-планом (номера строк на HEAD 08bc457)

Все place-ссылки мастер-плана проверены чтением кода; расхождения:

1. `.strip()` в бюджете: мастер-план называет `bot/utils.py:866-870` и `:881-884`.
   Фактически:
   - `_charge_user_and_guest` — определение начинается на `utils.py:862`, баг-строка
     (без `.strip()`) на **`utils.py:868`**.
   - `_charge_user_and_guest_async` — определение на `utils.py:876`, баг-строка на
     **`utils.py:882`**.
   Дрейф на 1-2 строки, не критично, но план ниже ссылается на фактические номера.
2. `chief.py:_parse_menu_preferences` — сигнатура подтверждена на **`chief.py:428`**
   (совпадает с мастер-планом). Баг: если `re.search(r'\{.*\}', ...)` не находит JSON,
   `if json_match:` пропускается, `try` завершается без `return`/`raise` →
   функция неявно возвращает `None` → вызывающий код
   `preferences, tokens_used = await self._parse_menu_preferences(...)` (`chief.py:548`)
   падает с `TypeError: cannot unpack non-iterable NoneType object` (воспроизведено
   вручную). Это НЕ `TypeError` из самой функции, а падение вызывающей стороны —
   значит фикс должен сделать так, чтобы `_parse_menu_preferences` сама подняла
   `ValueError` до момента `return`/распаковки.
3. `retry_after`: мастер-план — `telegram_stream.py:199`, `telegram_bot.py:4314`, `:4457`.
   Все три номера подтверждены точным совпадением (`grep` по `retry_after`/`RetryAfter`
   в обоих файлах даёт ровно эти три сайта плюс объявление `StreamOutcome`/импорт —
   других использований `exc.retry_after`/`e.retry_after` в дереве нет).
4. `is_allowed` цикл проверки членства: мастер-план — `utils.py:684-690`. Фактически
   комментарий на `:684`, `if not is_inline and is_group_chat(update):` на **`:685`**,
   `for user in itertools.chain(...)` на **`:687-690`**, `logging.info` (при отказе) на
   `:691-692`. Дрейф на 1 строку, план ниже правит фактическую строку `:685`.
5. Установленная версия `python-telegram-bot` — **22.8** (проверено
   `python -c "import telegram; print(telegram.__version__)"`). В этой версии
   `RetryAfter.retry_after` — property: по умолчанию (без `PTB_TIMEDELTA=true`)
   возвращает `int`/`float` и печатает `PTBDeprecationWarning` ("в будущей мажорной
   версии тип будет `datetime.timedelta`"); при `PTB_TIMEDELTA=true` уже сейчас
   возвращает `datetime.timedelta`. Проверено вручную (`RetryAfter(5).retry_after`
   → `5` с warning; `PTB_TIMEDELTA=true` → `datetime.timedelta(seconds=5)`).
   `asyncio.sleep(timedelta(...))` реально падает с `TypeError` (проверено). Отсюда
   и нужен `retry_after_seconds(exc) -> float`, принимающий `int | float | timedelta`.

Остальные факты мастер-плана (файл владения, структура шагов) подтверждены без
расхождений.

---

## Шаг 1. `.strip()` в учёте гостевого бюджета

**Файл:** `bot/utils.py`.

Баг: `is_allowed` парсит `allowed_user_ids` с `.strip()`
(`[x.strip() for x in config['allowed_user_ids'].split(',') if x.strip()]`,
`utils.py:680`), а `_charge_user_and_guest`/`_charge_user_and_guest_async` — без
`.strip()`. Если админ пишет `ALLOWED_TELEGRAM_USER_IDS="1, 2"` (с пробелом после
запятой — частый стиль), `is_allowed` для `user_id=2` вернёт `True` по явному
совпадению (`684-690` уже не нужен), но учёт стоимости решит, что `user_id=2` —
"гость", и спишет с общего гостевого бюджета вместо личного.

### 1.1. `_charge_user_and_guest` (`utils.py:862-874`)

Текущая строка `utils.py:868`:
```python
        allowed_user_ids = config['allowed_user_ids'].split(',')
```
→
```python
        allowed_user_ids = [x.strip() for x in config['allowed_user_ids'].split(',') if x.strip()]
```

### 1.2. `_charge_user_and_guest_async` (`utils.py:876-888`)

Текущая строка `utils.py:882`:
```python
        allowed_user_ids = config['allowed_user_ids'].split(',')
```
→
```python
        allowed_user_ids = [x.strip() for x in config['allowed_user_ids'].split(',') if x.strip()]
```

Не выносить в общий хелпер: та же однострочная конструкция уже продублирована 5 раз
в файле (`is_admin`, `is_allowed`, `get_user_budget` ×2) без общего хелпера — де-дуп
запланирован отдельно в волне T12 («Копипаста»), в T02 — только паритет с
существующим паттерном.

### Тест (падает до фикса) — `tests/test_usage_budget.py`

Добавить рядом с существующими тестами (после `test_remaining_budget_initializes_...`,
файл уже импортирует `from types import SimpleNamespace` и `bot.utils as utils`).
Добавить импорт:
```python
from bot.utils import _charge_user_and_guest, _charge_user_and_guest_async
```
(рядом с существующим `from bot.utils import get_remaining_budget`).

```python
def test_charge_user_and_guest_strips_whitespace_in_allowed_ids():
    """"1, 2" должно парситься так же, как в is_allowed: до фикса ' 2' (с пробелом)
    не совпадает с str(user_id), и явно разрешённый user_id=2 всё равно списывается
    с общего гостевого бюджета."""
    calls = []
    usage = {2: SimpleNamespace(name="user"), "guests": SimpleNamespace(name="guests")}
    config = {"allowed_user_ids": "1, 2"}

    charged = _charge_user_and_guest(usage, config, 2, lambda tracker: calls.append(tracker))

    assert charged is True
    assert calls == [usage[2]]  # guests-трекер не должен быть тронут


async def test_charge_user_and_guest_async_strips_whitespace_in_allowed_ids():
    calls = []

    async def charge_fn(tracker):
        calls.append(tracker)

    usage = {2: SimpleNamespace(name="user"), "guests": SimpleNamespace(name="guests")}
    config = {"allowed_user_ids": "1, 2"}

    charged = await _charge_user_and_guest_async(usage, config, 2, charge_fn)

    assert charged is True
    assert calls == [usage[2]]
```
`asyncio_mode = auto` (`pytest.ini`) — декоратор `@pytest.mark.asyncio` не нужен
(файл `tests/test_chief_model_choice.py` уже следует этому же стилю без декоратора;
в `test_usage_budget.py` асинхронных тестов пока нет — вводим этот стиль как более
свежий локальный прецедент).

Проверено вручную интерпретатором: до фикса `calls` содержит оба трекера
(`guests` тоже вызывается) — тест падает на `assert calls == [usage[2]]`.

---

## Шаг 2. `chief.py:_parse_menu_preferences` — ValueError вместо неявного None

**Файл:** `bot/plugins/chief.py`, метод `_parse_menu_preferences` (`chief.py:428-469`).

Текущий код (`chief.py:449-469`):
```python
        try:
            # Извлекаем JSON из ответа
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                preferences = json.loads(json_match.group(0))

                # Добавляем days к preferences перед валидацией
                preferences['days'] = days

                # Убираем дубликаты из фильтров
                if 'health_filters' in preferences:
                    preferences['health_filters'] = list(set(preferences['health_filters']))
                if 'dietary_preferences' in preferences:
                    preferences['dietary_preferences'] = list(set(preferences['dietary_preferences']))

                validate(instance=preferences, schema=self.menu_plan_schema)
                return preferences, tokens_used
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(f"Error parsing menu preferences: {str(e)}")
            raise ValueError("Не удалось разобрать предпочтения. Пожалуйста, уточните ваши пожелания.")
```

Правка — добавить `else:` к `if json_match:` (тот же текст ошибки, что в `except`):
```python
        try:
            # Извлекаем JSON из ответа
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                preferences = json.loads(json_match.group(0))

                # Добавляем days к preferences перед валидацией
                preferences['days'] = days

                # Убираем дубликаты из фильтров
                if 'health_filters' in preferences:
                    preferences['health_filters'] = list(set(preferences['health_filters']))
                if 'dietary_preferences' in preferences:
                    preferences['dietary_preferences'] = list(set(preferences['dietary_preferences']))

                validate(instance=preferences, schema=self.menu_plan_schema)
                return preferences, tokens_used
            else:
                logging.error("Error parsing menu preferences: no JSON object found in response")
                raise ValueError("Не удалось разобрать предпочтения. Пожалуйста, уточните ваши пожелания.")
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(f"Error parsing menu preferences: {str(e)}")
            raise ValueError("Не удалось разобрать предпочтения. Пожалуйста, уточните ваши пожелания.")
```
`ValueError` не перехватывается соседним `except (json.JSONDecodeError, ValidationError)`
(оба — не `ValueError` напрямую использующиеся классы except; `JSONDecodeError` —
подкласс `ValueError`, но `except` матчит по фактическому типу исключения, а не по
MRO включения в кортеж — поднятый обычный `ValueError` в этот `except` не попадёт),
поэтому распространяется наружу как есть и попадает во внешний
`except Exception as e: return {"error": self.t("chief_request_error", error=str(e))}`
(`chief.py:569-570`) с понятным текстом вместо текста про `NoneType`.

### Тест (падает до фикса) — `tests/test_chief_model_choice.py`

Использовать существующий `plugin`-fixture (`monkeypatch.setenv(EDAMAM_APP_ID/KEY)` +
`ChiefPlugin()`) и `FakeHelper` из этого же файла — они уже присутствуют
(`tests/test_chief_model_choice.py:15-32`), новых фикстур не создавать.

```python
async def test_parse_menu_preferences_raises_value_error_when_no_json_found(plugin):
    """До фикса функция неявно возвращает None, и эта же строка падает с
    TypeError: cannot unpack non-iterable NoneType object — тем же способом,
    каким падает вызывающий код в execute() (chief.py:548)."""
    helper = FakeHelper("Извините, не могу разобрать ваши пожелания.")

    with pytest.raises(ValueError, match="Не удалось разобрать предпочтения"):
        preferences, tokens_used = await plugin._parse_menu_preferences("что-то", helper, user_id=1)
```
Проверено вручную: до фикса — `TypeError: cannot unpack non-iterable NoneType object`
(не перехватывается `pytest.raises(ValueError)`, тест падает/ошибается); после
фикса — чистый `ValueError` с нужным текстом, тест проходит.

---

## Шаг 3. `retry_after_seconds(exc) -> float`

**Файл:** `bot/telegram_stream.py` — новая функция; **файл:** `bot/telegram_bot.py` —
только замена двух строк, использующих `exc.retry_after`/`e.retry_after`.

### 3.1. Новая функция в `bot/telegram_stream.py`

Добавить импорт `timedelta` (сейчас в файле нет `import datetime`/`from datetime
import ...`):
```python
from datetime import timedelta
```
(рядом с `import asyncio` / `import logging`, `telegram_stream.py:18-19`).

Добавить функцию сразу после `logger = logging.getLogger(__name__)`
(`telegram_stream.py:27`) и перед `@dataclass class StreamOutcome`:
```python
def retry_after_seconds(exc: RetryAfter) -> float:
    """Normalize ``RetryAfter.retry_after`` to a plain float of seconds.

    In python-telegram-bot 22.x ``retry_after`` is ``int`` by default and
    ``datetime.timedelta`` when ``PTB_TIMEDELTA=true`` (opt-in early for a future
    major version where ``timedelta`` becomes the only type); ``asyncio.sleep``
    does not accept ``timedelta`` directly.
    """
    value = exc.retry_after
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(value)
```

### 3.2. Использование в `telegram_stream.py:199`

Текущее:
```python
        except RetryAfter as exc:
            backoff += 5
            await asyncio.sleep(exc.retry_after)
```
→
```python
        except RetryAfter as exc:
            backoff += 5
            await asyncio.sleep(retry_after_seconds(exc))
```

### 3.3. Использование в `bot/telegram_bot.py:4314` и `:4457`

Импорт: добавить `retry_after_seconds` к уже существующему импорту
`from .telegram_stream import stream_to_telegram` (`telegram_bot.py:48`) →
```python
from .telegram_stream import retry_after_seconds, stream_to_telegram
```

`telegram_bot.py:4310-4314` — текущее:
```python
                            except RetryAfter as e:
                                if rich_stream_required:
                                    raise
                                backoff += 5
                                await asyncio.sleep(e.retry_after)
```
→ строка `4314`:
```python
                                await asyncio.sleep(retry_after_seconds(e))
```
(остальные строки блока `4310-4313` не трогать).

`telegram_bot.py:4455-4457` — текущее:
```python
                            except RetryAfter as e:
                                backoff += 5
                                await asyncio.sleep(e.retry_after)
```
→ строка `4457`:
```python
                                await asyncio.sleep(retry_after_seconds(e))
```

### Тесты — `tests/test_telegram_stream_core.py`

Файл уже содержит `from bot import telegram_stream` (`test_telegram_stream_core.py:12`)
и autouse-фикстуру `_no_real_sleep`, которая подменяет `telegram_stream.asyncio.sleep`
на `AsyncMock()` (`:25-27`) — существующий параметризованный тест
`test_intermediate_edit_failure_backs_off_and_final_tokens_stay_correct`
(`:82-105`) уже гоняет `RetryAfter(retry_after=1)` через `stream_to_telegram` и не
трогает голый `asyncio.sleep`, поэтому он и так не отличит `int` от `float` —
эти существующие тесты не меняются.

Добавить отдельные unit-тесты на саму функцию (без мока sleep, интеграции со
стримом не нужно — тестируем конвертацию значений). `retry_after_seconds` обращается
только к атрибуту `.retry_after`, поэтому для теста достаточно duck-typed
`SimpleNamespace`, а не реального `RetryAfter` — это не зависит от версии PTB и от
`PTB_TIMEDELTA`/deprecation-warning:

**Важно (проверено вручную, ошибка первой версии этого плана):** прямое присваивание
`exc._retry_after = timedelta(...)` на реальном `RetryAfter` НЕ заставляет `.retry_after`
вернуть `timedelta` — свойство `get_timedelta_value` всё равно конвертирует в
`int`/`float`, если `PTB_TIMEDELTA` не `true` в `os.environ` (проверено:
`exc._retry_after = timedelta(seconds=2.5)` без `PTB_TIMEDELTA=true` →
`exc.retry_after` возвращает `2.5` типа `float`, не `timedelta`). Значит тест на
реальном `RetryAfter` без `monkeypatch.setenv("PTB_TIMEDELTA", "true")` не дойдёт до
`isinstance(value, timedelta)` в `retry_after_seconds` и ничего не проверит. Поэтому
использовать `SimpleNamespace`, а не реальный `RetryAfter`, для проверки timedelta-ветки:

```python
from types import SimpleNamespace
from datetime import timedelta


def test_retry_after_seconds_passes_through_int():
    assert telegram_stream.retry_after_seconds(SimpleNamespace(retry_after=3)) == 3.0


def test_retry_after_seconds_converts_timedelta():
    exc = SimpleNamespace(retry_after=timedelta(seconds=2.5))
    assert telegram_stream.retry_after_seconds(exc) == 2.5
```
Второй тест проверяет ветку `isinstance(value, timedelta)`; без фикса (голый
`float(value)`) он падает с `TypeError: float() argument must be a string or a real
number, not 'datetime.timedelta'` (проверено вручную интерпретатором).

Опционально, для полноты (не обязательно, чтобы не плодить тесты сверх
необходимого): тест на реальном `RetryAfter` с
`monkeypatch.setenv("PTB_TIMEDELTA", "true")`, подтверждающий поведение актуальной
установленной PTB 22.8 в этом режиме:
```python
def test_retry_after_seconds_converts_real_retry_after_timedelta(monkeypatch):
    monkeypatch.setenv("PTB_TIMEDELTA", "true")
    exc = RetryAfter(retry_after=2)
    assert telegram_stream.retry_after_seconds(exc) == 2.0
```

---

## Шаг 4. Флаг `ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER`

### 4.1. `bot/__main__.py` — новый ключ конфига

В `telegram_config` (`__main__.py:321-359`), сразу после
`'allowed_user_ids': os.environ.get('ALLOWED_TELEGRAM_USER_IDS', '*'),`
(`__main__.py:327`) добавить:
```python
        'allow_group_members_via_authorized_user': env_bool(
            'ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER', True,
        ),
```
Выбор `env_bool` (а не строгий `parse_bool_env`): все соседние булевы ключи
`telegram_config` (`enable_quoting`, `enable_image_generation`, `enable_transcription`,
`enable_vision`, `enable_tts_generation`, `ignore_group_transcriptions`,
`ignore_group_vision`, `voice_reply_transcript`) используют `env_bool`; `parse_bool_env`
в файле применяется только к `TELEGRAM_LOCAL_MODE`/`SUMMARY_ENABLED`, где есть
отдельный тест на строгий отказ при опечатке. Дополнительный довод: опечатка в
значении `env_bool` схлопывается в `False` (a не `True`) — то есть отказоустойчиво
в безопасную сторону (более строгий режим, а не более открытый). Если ревьюер
предпочтёт `parse_bool_env` для консистентности с `telegram_local_mode` — это
не противоречит мастер-плану (там сказано только `config.get(..., True)` — про сам
парсинг env явно не сказано), альтернатива тоже приемлема.

### 4.2. `bot/utils.py` — `is_allowed`

`utils.py:685` (комментарий на `:684`, сам код был процитирован в разделе
«Проверка расхождений»):
```python
    if not is_inline and is_group_chat(update):
```
→
```python
    if (not is_inline and is_group_chat(update)
            and config.get('allow_group_members_via_authorized_user', True)):
```
Ничего внутри блока (`utils.py:686-692`) не трогать — при `False` весь блок
(включая цикл `is_user_in_group` и финальный `logging.info` об отказе) просто не
выполняется, функция сразу переходит к `return False` (`utils.py:693`). Это даёт
ровно то, что просит мастер-план: "чужой в группе отклонён и `get_chat_member` не
вызывается" — `is_user_in_group` (и, соответственно, `context.bot.get_chat_member`)
физически не вызывается при `False`.

### 4.3. `bot/__main__.py` — лог режима при старте

После завершения сборки `telegram_config` (`__main__.py:359`, закрывающая `}`) и
перед `plugin_config = {...}` (`__main__.py:361`) добавить:
```python
    if telegram_config['allow_group_members_via_authorized_user']:
        logging.info(
            'Group access mode: ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER is enabled -- '
            'any member of a group chat is treated as allowed if the group also '
            'contains an allowed/admin user.'
        )
    else:
        logging.info(
            'Group access mode: ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER is disabled -- '
            'group chat messages are allowed only from the sender\'s own allowed/admin id.'
        )
```
(английский текст — единственный язык логов в `__main__.py`, см. существующие
`logging.error`/`logging.warning` в этом файле; в файле пока нет ни одного
`logging.info` — это будет первый).

### Тесты — `tests/test_callback_authorization.py`

Добавить после существующих трёх групповых тестов
(`test_restricted_group_membership_unexpected_bad_request_is_not_cached`,
заканчивается на `:479`), перед
`test_unauthorized_plugin_callback_query_does_not_call_plugin_handler` (`:483`).
Использовать существующие `FakeCallbackUpdate`, `_make_context`, `_make_bot`,
`is_allowed` (уже импортирован, `:51`), `ChatMember` (уже импортирован, `:8`).

```python
@pytest.mark.asyncio
async def test_group_membership_grant_disabled_rejects_non_member_without_api_call():
    update = FakeCallbackUpdate("session:back", user_id=999, chat_id=-100321)
    update.effective_chat.type = "supergroup"
    context = _make_context()
    config = {
        "allowed_user_ids": "111",
        "admin_user_ids": "-",
        "bot_language": "en",
        "allow_group_members_via_authorized_user": False,
    }

    allowed = await is_allowed(config, update, context)

    assert allowed is False
    context.bot.get_chat_member.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_callback_rejected_when_membership_grant_disabled():
    bot = _make_bot(allowed_user_ids="111")
    bot.config["allow_group_members_via_authorized_user"] = False
    update = FakeCallbackUpdate("session:back", user_id=999, chat_id=-100654)
    update.effective_chat.type = "supergroup"
    context = _make_context()
    # Даже если бы код спросил Telegram, ответ был бы "участник" -- проверяем,
    # что при выключенном флаге код вообще не долетает до этого вызова.
    context.bot.get_chat_member.return_value = SimpleNamespace(status=ChatMember.MEMBER)

    await bot.reset(update, context)

    bot.db.list_user_sessions.assert_not_called()
    context.bot.get_chat_member.assert_not_awaited()
    update.callback_query.edit_message_text.assert_awaited_once_with(
        text=localized_text("access_denied_command", "en")
    )


@pytest.mark.asyncio
async def test_group_callback_allowed_member_passes_without_api_call_when_membership_grant_disabled():
    bot = _make_bot(allowed_user_ids="111")
    bot.config["allow_group_members_via_authorized_user"] = False
    update = FakeCallbackUpdate("session:back", user_id=111, chat_id=-100987)
    update.effective_chat.type = "supergroup"
    context = _make_context()

    await bot.reset(update, context)

    bot.db.list_user_sessions.assert_called_once_with(111)
    context.bot.get_chat_member.assert_not_awaited()
```
`SimpleNamespace` уже импортирован в файле (`test_callback_authorization.py:4`).

Существующие три групповых теста (`:421-479`) не меняются — их `config`/`bot.config`
не содержат `allow_group_members_via_authorized_user`, поэтому `config.get(...,
True)` даёт прежнее поведение.

---

## Шаг 5. Документация

### 5.1. `README.md`

Новая строка в таблице «Telegram Core» (`README.md:263-276`), сразу после
`ALLOWED_TELEGRAM_USER_IDS` (`README.md:268`):
```
| `ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER` | `true` | bool | In group chats, treat any member as allowed when the group also contains an allowed/admin user. Set `false` to require the sender's own ID in `ALLOWED_TELEGRAM_USER_IDS`/`ADMIN_USER_IDS`. |
```

Уточнить строку `GUEST_BUDGET` в таблице «Budgets And Pricing» (`README.md:357`).
Текущее:
```
| `GUEST_BUDGET` | `100.0` | float | Budget for guests when group chats are addressed by an allowed user. |
```
→
```
| `GUEST_BUDGET` | `100.0` | float | Shared budget for group members who are allowed only via `ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER` (not themselves listed in `ALLOWED_TELEGRAM_USER_IDS`). Unused when that flag is `false`. |
```

### 5.2. `README.ru.md`

Новая строка в таблице «Telegram core» (`README.ru.md:271-283`), сразу после
`ALLOWED_TELEGRAM_USER_IDS` (`README.ru.md:276`):
```
| `ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER` | `true` | bool | В групповых чатах считать разрешённым любого участника, если в группе также состоит allowed/admin пользователь. `false` — решает только сам отправитель (нужен его ID в `ALLOWED_TELEGRAM_USER_IDS`/`ADMIN_USER_IDS`). |
```

Уточнить `GUEST_BUDGET` (`README.ru.md:365`). Текущее:
```
| `GUEST_BUDGET` | `100.0` | float | Бюджет для гостей в групповых чатах. |
```
→
```
| `GUEST_BUDGET` | `100.0` | float | Общий бюджет для участников группы, допущенных только через `ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER` (сами не входят в `ALLOWED_TELEGRAM_USER_IDS`). Не используется при `false`. |
```

### 5.3. `.env.example`

После блока `ALLOWED_TELEGRAM_USER_IDS` (`.env.example:10-11`) добавить:
```
# In group chats, allow any member when the group also contains an allowed/admin user.
# Set to false to require the sender's own ID in ALLOWED_TELEGRAM_USER_IDS/ADMIN_USER_IDS.
# ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER=true
```

---

## Порядок реализации (рекомендация разработчику)

1. Шаг 1 (`.strip()`) и тесты — независим от остальных, можно первым.
2. Шаг 2 (`chief.py`) и тест — независим.
3. Шаг 3 (`retry_after_seconds`) и тесты — независим.
4. Шаг 4 (флаг групп) и тесты — зависит только от `bot/utils.py`/`bot/__main__.py`,
   не пересекается с шагами 1-3 по коду, но лучше делать после шага 1 (тот же файл
   `utils.py`), чтобы не гонять diff дважды.
5. Шаг 5 (документация) — после шага 4, когда точное имя флага и формулировки
   зафиксированы кодом.

## Команды приёмки

```bash
~/.venvs/ctb/bin/python -m pytest \
  tests/test_callback_authorization.py \
  tests/test_usage_budget.py \
  tests/test_chief_model_choice.py \
  tests/test_chief_close_async.py \
  tests/test_telegram_stream_core.py \
  -q --no-header -p no:cacheprovider

python3 -m mypy bot/utils.py bot/__main__.py bot/plugins/chief.py bot/telegram_stream.py bot/telegram_bot.py \
  --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports

~/.venvs/ctb/bin/python -m ruff check \
  bot/utils.py bot/__main__.py bot/plugins/chief.py bot/telegram_stream.py bot/telegram_bot.py \
  tests/test_callback_authorization.py tests/test_usage_budget.py tests/test_chief_model_choice.py \
  tests/test_telegram_stream_core.py
```
Дополнительно (защита от порчи чужих тестов, эти файлы НЕ редактируются, но
пересекаются по коду): прогнать полный набор, затрагивающий `bot/__main__.py`
и `bot/telegram_bot.py`, чтобы убедить ревьюера, что T02 не сломал соседние
задачи:
```bash
~/.venvs/ctb/bin/python -m pytest \
  tests/test_telegram_builder_config.py tests/test_telegram_streaming.py \
  -q --no-header -p no:cacheprovider
```

## Риски

- `retry_after_seconds`: `float(value)` для нестандартного типа (не `int`/`float`/
  `timedelta`) поднимет `TypeError`/`ValueError` внутри `except RetryAfter` — но
  `exc.retry_after` по контракту PTB 22.x всегда один из этих типов, риск
  теоретический.
- Флаг группы: если конфиг где-то в кодовой базе строится не как `dict`, а как
  объект без `.get` — `config.get(...)` упадёт. Проверено: везде выше по коду
  `config['allowed_user_ids']` уже используется как `dict` (bracket-доступ), риска
  нет.
- Изменение `env_bool` vs `parse_bool_env` для нового флага — управленческое
  решение, а не баг; при разногласии с ревьюером изменить одну строку, тесты не
  зависят от конкретной функции парсинга (только от итогового `bool` в
  `bot.config`).
- `tests/test_telegram_builder_config.py` (1007 строк, вне владения T02) проверен
  агентом `large-file-explorer`: ни одного `assert` на точное равенство всего
  `telegram_config`/`bot.config`, ни одного `caplog`-теста на точное количество
  записей — новый ключ конфига и новые `logging.info` не должны его сломать; тем не
  менее команда приёмки выше включает его прогон явно.
- `chief.py`: `else`-ветка меняет поведение только для ответов модели без единого
  `{...}` в тексте — редкий, но реальный кейс (модель ответила отказом без JSON).
