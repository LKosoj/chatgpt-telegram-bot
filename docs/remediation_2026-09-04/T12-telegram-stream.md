# T12. Единый стриминговый рендерер и повтор финального чанка

Источники задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел T12, «Волна 3») и
`docs/architecture_code_review_2026-09-04.md` §3.6 (плюс предложение П7 в §5.3). Номера строк в
обоих документах устарели — код ниже проверен по актуальному дереву (HEAD `af382fb` + правки
после него) поиском по именам функций, а не по старым `file:line`.

Термины: **чанк (chunk)** — здесь два разных смысла одновременно, и это важно не путать: (1)
кусок текста ≤4096 символов, на которые режется длинный ответ, потому что Telegram не принимает
сообщение длиннее; (2) один элемент из стрима модели `(content, tokens)` — по факту это не
дельта, а **весь ответ целиком на данный момент**, растущий с каждой итерацией. **backoff** —
временная пауза перед повторной попыткой после ошибки, которая растёт при повторных неудачах.
**sentinel** (см. также T11) — здесь `tokens == 'not_finished'`: строковый маркер «это ещё не
последний чанк», в отличие от финального чанка, где `tokens` — это строка с числом токенов.
**draft (черновик)** — новый механизм, описанный ниже: промежуточное сообщение, которое можно
многократно перезаписывать без отдельного вызова `editMessageText`, через нестандартный метод
Bot API `sendRichMessageDraft`.

## Цель

1. Новый модуль `bot/telegram_stream.py` с функцией `stream_to_telegram(...)`, которая
   инкапсулирует общую часть трёх копий стримингового цикла: обработку `(content, tokens)`,
   backoff, чанкование по 4096 символов (`split_into_chunks`), обработку `RetryAfter`/`TimedOut`/
   `BadRequest`, и (это главное) **гарантированную доставку последнего чанка**.
2. Устранить CERTAIN-баг §3.6: сейчас при ошибке `edit_message_text` на последней итерации
   (`RetryAfter`/`TimedOut`/`Exception`) код делает `continue`, но `async for` больше не отдаёт
   значений — итоговый текст теряется безвозвратно, а `total_tokens` остаётся 0, потому что
   присвоение `total_tokens = int(tokens)` стоит **после** блока `try/except` и тоже
   пропускается.
3. Перевести три места на модуль поэтапно (T12 просит оценить риск полной унификации — ниже
   объясняется, почему это не один шаг, а минимум четыре), с тестами на каждом шаге.

## Находка, меняющая рамки задачи: rich-режим — это **дефолт** для приватных чатов

Ни `docs/audit_remediation_plan_2026-09-04.md`, ни §3.6 в `architecture_code_review_2026-09-04.md`
не упоминают этот механизм — видимо, он попал в код после того ревью. Он полностью меняет оценку
того, какой именно код сегодня отвечает за стриминг в чате один-на-один с ботом.

- `bot/__main__.py:37-45` (`parse_telegram_rich_mode_env`) — дефолт `TELEGRAM_RICH_MESSAGES` =
  `"auto"`; `bot/__main__.py:198` — `TELEGRAM_RICH_DRAFTS` дефолт `True`. Оба идут в конфиг без
  доп. настройки оператора (`bot/__main__.py:212-213, 306-307`).
- `bot/telegram_rich.py:166-172` (`rich_messages_enabled`) — режим `"auto"` **включён**
  (`{"auto", "required"}`), выключен только явным `TELEGRAM_RICH_MESSAGES=off`.
- `bot/telegram_bot.py:674-681` (`_should_stream_rich_drafts`) — `True`, если rich включён,
  драфты включены (дефолт), и чат приватный (`ChatType.PRIVATE`).
- Значит: **в личной переписке с ботом (самый частый случай использования) по умолчанию
  работает не тот код, который правит T12**, а параллельная ветка `rich_stream_active`
  (`bot/telegram_bot.py:4317-4404`), которая шлёт не `editMessageText`, а нестандартные методы
  Bot API `sendRichMessage`/`sendRichMessageDraft` (`bot/telegram_rich.py:102-153`) — это
  расширение локального Bot API сервера (`base_url=http://localhost:8081/bot`,
  `bot/telegram_bot.py:6288-6303` по AGENTS.md), а не часть python-telegram-bot. У этой ветки
  **лимит не 4096 символов, а 32768 байт** (`MAX_RICH_MARKDOWN_BYTES`,
  `bot/telegram_rich.py:15`), и `split_into_chunks` там вообще не используется.
- «Легаси»-ветка (`split_into_chunks`/`edit_message_with_retry`/`reply_text`, ровно то, что
  описывает §3.6) сегодня выполняется по умолчанию только в **групповых чатах** — в
  `_should_stream_rich_drafts` чат должен быть `PRIVATE`, group туда не проходит — либо когда
  оператор явно поставил `TELEGRAM_RICH_MESSAGES=off`. Тесты это неявно подтверждают:
  `_make_bot()` в `tests/test_telegram_streaming.py:463-471` (словарь `bot.config`) не кладёт
  `telegram_rich_messages`, поэтому `rich_messages_mode` возвращает `"off"`
  (`bot/telegram_rich.py:166-172`, дефолт при отсутствующем ключе) — тесты по умолчанию гоняют
  именно легаси-путь, а rich-тесты (`test_streaming_uses_rich_draft_and_final_message` и
  соседние, `tests/test_telegram_streaming.py:1136-1459`) явно ставят
  `bot.config["telegram_rich_messages"] = "auto"`.
- У rich-ветки **тот же класс бага**, но с другим исходом. При `rich_stream_required` (режим
  `"required"`) ошибка на `send_rich_markdown`/`send_rich_markdown_draft` не поглощается, а
  `raise`-ится (`bot/telegram_bot.py:4351-4352, 4371-4372`) и долетает до общего
  `except Exception as e` в конце `_process_message_locked` (`:4608-4617`), который показывает
  пользователю `chat_fail: <текст ошибки>` — т.е. **текст ответа всё равно теряется**, но хотя бы
  пользователь видит явную ошибку, а не тишину/устаревший черновик. При `rich_stream_active` без
  `required` (дефолтный `"auto"`) код красивее деградирует: ловит исключение, выставляет
  `rich_stream_active = False` (`bot/telegram_bot.py:4355, 4373`) и с этого момента следующие
  чанки идут через легаси-путь (если черновик ещё не отправлялся — текущий чанк тоже публикуется
  сразу через `_publish_legacy_stream_snapshot`, `:4364-4369, 4382-4387`) — но если сбой
  случился именно на **последнем** чанке и черновик уже был отправлен, эта ветка выходит из
  `try/except` без повторной попытки — тот же по сути баг §3.6, просто в другой обёртке.

**Вывод для рамок T12.** Задача, как она сформулирована в аудите и плане, — про буквальное
дублирование одного и того же паттерна (`split_into_chunks` + `edit_message_with_retry` +
`RetryAfter`/`TimedOut`) в трёх местах. Rich-режим — структурно другой транспорт (другой лимит,
другой протокол, отдельная сущность «драфт»), и заворачивать его в тот же `stream_to_telegram` в
этом тикете означало бы почти удвоить поверхность модуля ради части, которая формально не
входит в описание задачи. Ниже T12 реализован строго по заданным рамкам (легаси-паттерн во всех
трёх местах), но с явной рекомендацией — см. «Риски» — завести отдельный тикет на тот же класс
бага в rich-ветке, потому что именно она сегодня работает в личных чатах по умолчанию.

## Репро текущего бага

В `/tmp/hunt/test_repro_final_chunk.py` уже лежит рабочее репро (импортирует
`tests/test_telegram_streaming.py` как модуль и переиспользует его `FakeMessage`/`_make_bot`).
Прогнано сейчас (без изменений в коде):

```
$ ~/.venvs/ctb/bin/python -m pytest -q /tmp/hunt/test_repro_final_chunk.py
.                                                                        [100%]
1 passed in 1.84s
```

Тест патчит `edit_message_with_retry` так, чтобы бросать `TimedOut` именно на финальном тексте
`"Hello final answer"`, и проверяет три вещи одновременно: (1) финальный текст ни разу не ушёл в
`reply_text` как новое сообщение, (2) `record_chat_tokens` вызван с `0` вместо реальных токенов,
(3) повторной попытки доставить именно этот текст не было. Это и есть формальная спецификация
того, что должно измениться после T12 (см. «Критерии готовности»).

## Анализ трёх мест

| # | Место | `async for` | Источник стрима | Доставка | Финал direct_result | Что теряется при ошибке |
|---|---|---|---|---|---|---|
| 1 | `vision()` → вложенная `_run_vision_model_request` | `bot/telegram_bot.py:3249-3348` | `self.openai.interpret_image_stream(...)` (`:3233`) | `chat_id` + `sent_message.message_id`, `update.effective_message.reply_text` для первого/overflow-сообщений | `self._handle_direct_result(update, content)` + `record_vision_tokens` (`:3250-3255`) | Текст и токены — `except Exception: backoff += 5; continue` (`:3340-3342`), то же для `RetryAfter`/`TimedOut` (`:3330-3338`) |
| 2 | `_process_message_locked`, легаси-ветка (когда `rich_stream_active` и `rich_stream_final_only` оба `False`) | `bot/telegram_bot.py:4295-4513` | `self.openai.get_chat_response_stream(...)` (`:4233-4239`) | `chat_id` + `sent_message.message_id`, `reply_text` для первого/overflow-сообщений | `self._handle_direct_result` + `_dispatch_assistant_response_observer` + `_record_chat_usage` (`:4296-4310`) | То же самое: `except Exception: backoff += 5; continue` (`:4505-4507`) |
| 3 | `handle_callback_inline_query` | `bot/telegram_bot.py:4918-4976` | `self.openai.get_chat_response_stream(chat_id=user_id, ...)` (`:4909-4914`) | **только** `edit_message_text` по `inline_message_id`; **новое сообщение отправить нельзя** — это плейсхолдер под чужим сообщением в другом чате | `direct_result_inline_fallback_text(...)` (текстовый суррогат, не настоящая доставка файлов/фото) + `cleanup_intermediate_files` + `_record_chat_usage` (`:4919-4929`) | То же самое: `except Exception: backoff += 5; continue` (`:4968-4970`) |

### Чем три места реально отличаются (не совпадения ради унификации)

- **Транспорт доставки.** (1) и (2) — обычные сообщения в чате, могут открыть новое сообщение
  через `reply_text`. (3) — инлайн-плейсхолдер: единственная операция — `edit_message_text` с
  `inline_message_id`; `send_message` невозможен в принципе (у бота нет права/смысла слать новое
  сообщение в чужой чат, куда был вставлен инлайн-результат). Значит «гарантия доставки через
  fallback новым сообщением» физически применима только к (1) и (2); для (3) гарантия — это
  «повторить `edit` агрессивнее, потом честно залогировать неудачу», не более.
- **Что происходит при direct_result.** (1)/(2) реально доставляют результат пользователю
  (фото/файл/HTML и т.д.) через `_handle_direct_result` → `handle_direct_result`
  (`bot/utils.py`, вызывает нужный `bot.send_*`). (3) не может — инлайн-режим не поддерживает
  большинство direct_result'ов, поэтому строит текстовый fallback
  (`direct_result_inline_fallback_text`) и выполняет `cleanup_intermediate_files`, которого нет
  в (1)/(2). Это не то, что можно спрятать за одним колбэком «доставь результат» — вызывающая
  сторона должна сама решать, что значит «доставлено», поэтому `on_direct_result` в дизайне ниже
  — чистый колбэк без возвращаемого значения, вся логика остаётся у вызывающего места, как
  сегодня.
- **Учёт токенов/расходов.** (1) — `record_vision_tokens`; (2)/(3) — `_record_chat_usage`,
  причём у (2) есть побочный `_dispatch_assistant_response_observer`, которого нет ни у (1), ни
  у (3). Это тоже не про сам модуль стриминга — вызывается **после** возврата из
  `stream_to_telegram`, не колбэком внутри цикла.
  Пример: `text=f'{query}\n\n_{answer_tr}:_\n{content}'`.
- **Поведение при неудаче самого первого сообщения (i==0).** Три места ведут себя по-разному
  **уже сегодня**, и это не баг, а разные компромиссы:
  - vision (`:3300-3320`) — `except Exception: logger.debug(...); continue` — тихо пробует
    заново на следующем чанке модели;
  - легаси-текст (`:4477-4486`) — `except Exception: logger.error(...)`, пытается один раз
    отправить `chat_fail`, затем **`break`** — полностью прекращает стриминг;
  - инлайн (`:4943-4948`) — `except Exception: logger.debug(...); continue` — как vision.

  Дизайн ниже сознательно **меняет** это несовпадение (см. «Дизайн», пункт про i==0) — единое
  поведение «повторить на следующем чанке, а если поток так и закончится — сработает гарантия
  финальной доставки» строго не хуже нынешнего для (1)/(3) и лучше для (2) (сегодняшний `break`
  в (2) не гарантирует даже доставку `chat_fail`: если и он не пройдёт, ошибка проглатывается
  тем же `except Exception: continue` на уровне `edit`, но это уже другой `try` — сообщение
  `chat_fail` не ретраится вовсе). Это осознанное расширение поведения, не молчаливая правка —
  выносится в «Риски» как пункт, требующий подтверждения ревьюера.

## Дизайн модуля

### Что подтверждено, а что в тексте задачи было неточным

- В задаче упомянуты `helper.stream_chunk_delay`/`stream_flush` — таких атрибутов/констант в
  коде нет (`python3 -c` grep по `bot/` не нашёл ни одного вхождения). Реальная пауза между
  правками — жёстко зашитая `await asyncio.sleep(0.01)` после каждого успешного `edit`
  (`bot/telegram_bot.py:3344, 4509, 4972`), throttling-порог — `get_stream_cutoff_values`
  (`bot/utils.py:261-270`, зависит от длины контента и типа чата). Модуль принимает оба как
  параметры с теми же дефолтами, а не читает несуществующие поля `helper`.
- `TELEGRAM_MAX_LEN` как именованная константа тоже не существует — есть
  `split_into_chunks(text, chunk_size=4096)` (`bot/utils.py:288`), 4096 — просто дефолт
  параметра. Модуль так же принимает `chunk_size=4096` параметром.
- `edit_message_with_retry` (`bot/utils.py:585-645`) уже делает Markdown-fallback: при
  `markdown=True` рендерит через `render_markdown_message_entities`
  (telegramify-markdown, `bot/utils.py:433-441`) с `parse_mode=None` + `entities=...`; если
  Telegram вернёт `BadRequest` (кроме «message is not modified», который тихо проглатывается,
  `:629-631`), делает вторую попытку с сырым текстом без сущностей (`:632-638`). Это ровно
  «Markdown-fallback (parse_mode → plain)» из формулировки задачи — **уже реализовано и не
  требует изменений**, модуль просто продолжает вызывать эту функцию из колбэка `edit`, не
  дублируя её логику.

### API

```python
# bot/telegram_stream.py
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from telegram.error import RetryAfter, TimedOut

from .utils import is_direct_result, log_exception_shape, split_into_chunks

logger = logging.getLogger(__name__)

# Колбэки — единственное, что знает о конкретном транспорте (обычный чат
# по chat_id+message_id, инлайн по inline_message_id, vision-реплай и т.п.).
# Сам модуль ни разу не вызывает python-telegram-bot напрямую.
EditFn = Callable[[Any, str, bool], Awaitable[None]]   # (message_id, text, markdown) -> None
SendFn = Callable[[str], Awaitable[Any]]               # (text) -> объект с .message_id (.chat_id опц.)
DirectResultFn = Callable[[str, int], Awaitable[None]] # (raw_content, total_tokens) -> None
CutoffFn = Callable[[str], int]                        # (content) -> порог для throttling


@dataclass
class StreamOutcome:
    final_text: str = ""
    total_tokens: int = 0
    message_ids: list = field(default_factory=list)
    direct_result: Any = None   # сырой content, если стрим завершился direct_result'ом
    delivered: bool = True      # False, только если финальный чанк не доставлен даже фолбэком


async def stream_to_telegram(
    stream: AsyncIterator[tuple[str, str]],
    *,
    edit: EditFn,
    send: Optional[SendFn],
    cutoff_for: CutoffFn,
    on_direct_result: Optional[DirectResultFn] = None,
    chunk_size: int = 4096,
    max_final_attempts: int = 3,
    retry_backoff_seconds: float = 5.0,
    inter_edit_delay: float = 0.01,
) -> StreamOutcome:
    """
    Общий рендер (content, tokens)-стрима OpenAIHelper в Telegram. Контракт
    ``stream`` — как у ``get_chat_response_stream``/``interpret_image_stream``:
    ``content`` — это ПОЛНЫЙ накопленный ответ на текущий момент (не дельта),
    ``tokens`` — строка ``'not_finished'`` на всех чанках, кроме последнего,
    где это строка с числом токенов. ``content`` может быть JSON-строкой с
    ``direct_result`` — тогда обрабатывает её ``on_direct_result`` и цикл
    завершается немедленно.

    ``send is None`` означает, что у вызывающей стороны нет способа отправить
    НОВОЕ сообщение (инлайн-режим) — гарантия финальной доставки в этом случае
    сводится к повтору ``edit`` без фолбэка новым сообщением.

    Побочных эффектов сверх переданных колбэков нет: запись usage, диспетч
    хуков, `_record_chat_usage`/`record_vision_tokens` остаются на вызывающей
    стороне — три места делают это по-разному (см. «Анализ трёх мест»).
    """
    i = 0
    prev = ""
    backoff = 0
    last_published_chunk = 0
    sent_message: Any = None
    final_text = ""
    total_tokens = 0
    final_delivered = False
    message_ids: list = []

    async for content, tokens in stream:
        if on_direct_result is not None and is_direct_result(content):
            if tokens != "not_finished":
                total_tokens = int(tokens)
            await on_direct_result(content, total_tokens)
            return StreamOutcome(
                total_tokens=total_tokens, message_ids=message_ids,
                direct_result=content, delivered=True,
            )

        if len(content.strip()) == 0:
            continue

        final_text = content
        is_final = tokens != "not_finished"
        if is_final:
            # Токены модели фиксируются независимо от успеха доставки —
            # закрывает "usage=0 при потерянном чанке" из репро-теста.
            total_tokens = int(tokens)
        final_delivered = False

        # --- переполнение chunk_size: закрыть предыдущий чанк отдельным
        # сообщением, продолжить расти в новом ("хвостовой" чанк) ---
        chunks = split_into_chunks(content, chunk_size)
        if len(chunks) > 1:
            content = chunks[-1]
            if last_published_chunk != len(chunks) - 1:
                last_published_chunk += 1
                previous_chunk = chunks[-2]
                if sent_message is not None:
                    await _swallow(edit(sent_message.message_id, previous_chunk, True))
                elif send is not None:
                    sent_message = await _swallow_send(send, previous_chunk or "...", message_ids)
                if send is not None:
                    sent_message = await _swallow_send(send, content or "...", message_ids)
                # инлайн (send is None): второе сообщение открыть нечем,
                # следующая итерация продолжит расти в том же edit.
                continue

        cutoff = cutoff_for(content) + backoff

        if sent_message is None:
            if send is not None:
                try:
                    sent_message = await send(content)
                    message_ids.append(sent_message.message_id)
                    prev = content
                    final_delivered = is_final
                except Exception as exc:
                    logger.warning(
                        "stream_to_telegram: initial send failed error=%s",
                        log_exception_shape(exc),
                    )
                    # Не поднимаем: следующий чанк модели попробует снова.
                    # Если поток так и закончится без успешной отправки,
                    # сработает гарантия финальной доставки ниже.
            else:
                # Инлайн: "первое сообщение" — это edit уже существующего
                # плейсхолдера (inline_message_id создан заранее).
                try:
                    await edit(None, content, is_final)
                    prev = content
                    final_delivered = is_final
                except Exception as exc:
                    logger.debug(
                        "stream_to_telegram: initial inline edit failed error=%s",
                        log_exception_shape(exc),
                    )
            i += 1
            continue

        if not (abs(len(content) - len(prev)) > cutoff or is_final):
            i += 1
            continue

        prev = content
        try:
            await edit(sent_message.message_id, content, is_final)
            final_delivered = is_final
            await asyncio.sleep(inter_edit_delay)
        except RetryAfter as exc:
            backoff += 5
            await asyncio.sleep(exc.retry_after)
        except TimedOut:
            backoff += 5
            await asyncio.sleep(0.5)
        except Exception as exc:
            backoff += 5
            logger.debug(
                "stream_to_telegram: throttled edit failed error=%s",
                log_exception_shape(exc),
            )
        i += 1

    if final_text and not final_delivered:
        final_delivered = await _guarantee_final_delivery(
            sent_message, final_text,
            edit=edit, send=send, message_ids=message_ids,
            max_attempts=max_final_attempts,
            backoff_seconds=retry_backoff_seconds,
        )

    return StreamOutcome(
        final_text=final_text, total_tokens=total_tokens,
        message_ids=message_ids, direct_result=None, delivered=final_delivered,
    )


async def _guarantee_final_delivery(
    sent_message, final_text, *, edit, send, message_ids, max_attempts, backoff_seconds,
) -> bool:
    """Повторяет доставку финального текста до ``max_attempts`` раз с растущим
    backoff; если сообщение уже существовало и все попытки `edit` провалились,
    один раз пробует ``send`` как новое сообщение (когда он доступен)."""
    for attempt in range(max_attempts):
        try:
            if sent_message is not None:
                await edit(sent_message.message_id, final_text, True)
            elif send is not None:
                sent_message = await send(final_text)
                message_ids.append(sent_message.message_id)
            else:
                await edit(None, final_text, True)  # инлайн без открытого сообщения
            return True
        except Exception as exc:
            logger.warning(
                "stream_to_telegram: final delivery attempt %d/%d failed error=%s",
                attempt + 1, max_attempts, log_exception_shape(exc),
            )
            await asyncio.sleep(backoff_seconds * (attempt + 1))

    if send is not None:
        try:
            fallback = await send(final_text)
            message_ids.append(fallback.message_id)
            return True
        except Exception as exc:
            logger.error(
                "stream_to_telegram: fallback send_message also failed error=%s",
                log_exception_shape(exc),
            )
    else:
        logger.error(
            "stream_to_telegram: final text could not be delivered "
            "(inline mode has no send fallback)"
        )
    return False
```

(`_swallow`/`_swallow_send` — тривиальные обёртки «выполнить корутину, залогировать и
проглотить исключение», по образцу уже существующих `try/except logger.debug(...)` вокруг
overflow-чанков в текущем коде, например `bot/telegram_bot.py:3266-3277`. Не расписаны отдельно
— однострочные хелперы, разработчик впишет их по месту.)

### Ключевые решения дизайна

1. **Два колбэка (`edit`/`send`), не флаги.** Всё, что отличает три места — конкретный способ
   что-то отправить/отредактировать — инкапсулировано в замыкания, которые строит вызывающий
   код (он уже держит `context`, `chat_id`/`inline_message_id`, `update`). Модуль не содержит
   `is_inline`/`is_vision`: если `send is None`, это уже само по себе означает «инлайн-режим»,
   отдельный флаг не нужен.
2. **`on_direct_result` — колбэк без возврата, не встроенная доставка.** Как показано в
   «Анализ трёх мест», доставка direct_result у (1)/(2) и у (3) — принципиально разные операции
   (реальная отправка файла vs текстовый суррогат). Модуль просто передаёт управление и
   останавливается; вся политика — на вызывающей стороне, как и сегодня. Это соответствует
   правилу проекта «код решает, что дальше» (раздел AGENTS.md «Deterministic Routing In Agent
   Plugins») — модуль не пытается угадывать политику доставки за три разных сценария.
3. **Отсчёт токенов отделён от успеха доставки.** `total_tokens = int(tokens)` выполняется, как
   только получен финальный `(content, tokens)`-элемент, до попытки `edit`/`send`. Это прямая
   правка §3.6-подпункта «`_record_chat_usage` пишет 0» — модель уже потратила токены независимо
   от того, дошло ли обновлённое сообщение до Telegram.
4. **Гарантия финальной доставки отделена от throttling-цикла.** Внутри `async for` поведение
   почти не меняется (тот же cutoff/backoff, те же исключения) — риск регрессии в
   «горячем» пути минимален. Отдельная функция `_guarantee_final_delivery` вызывается один раз,
   после выхода из цикла, только если последний обработанный элемент был финальным и его
   доставка не подтвердилась. Именно тут теперь реализованы «до N повторов, затем `send` как
   fallback», которых не было вовсе.
5. **Поведение i==0 объединено на «тихий повтор на следующем чанке» (как сегодня у vision/
   инлайн), а не на «break» (как сегодня у легаси-текста).** Явное изменение поведения (2) —
   см. «Риски». Мотивация: если так и не появится ни одного успешного `edit`/`send` за весь
   стрим, `_guarantee_final_delivery` всё равно отработает в конце и попробует достучаться до
   пользователя; сегодняшний `break` в (2) не даёт такой гарантии даже для одной попытки
   `chat_fail`.
6. **Rich-режим полностью вне модуля** (см. «Находка, меняющая рамки задачи») — `stream_to_telegram`
   вызывается только из легаси-ветки; условие `if rich_stream_active / elif rich_stream_final_only`
   в `_process_message_locked` остаётся как есть, `else:` (легаси) заменяется на вызов модуля.

## Правки по `file:line`

Поэтапно, как просит задача (риск полной синхронной унификации — три разных вызывающих
контекста, из которых один вообще без тестового покрытия retry-ветки внутри `handle_callback_
inline_query`, и один частично мёртв по умолчанию из-за rich-режима):

### Этап 1 — модуль + основной текстовый путь (легаси-ветка)

- Новый файл `bot/telegram_stream.py` (код выше).
- `bot/telegram_bot.py:4295-4514` (тело `async for` внутри легаси-ветки, между
  `else:`/`elif rich_stream_final_only:` и `assistant_response_text = last_stream_content`) —
  заменить на построение колбэков и вызов `stream_to_telegram`:

  ```python
  async def _edit(message_id, text, markdown):
      await edit_message_with_retry(context, chat_id, str(message_id), text=text, markdown=markdown)

  async def _send(text):
      return await update.effective_message.reply_text(
          message_thread_id=get_thread_id(update),
          reply_to_message_id=get_reply_to_message_id(self.config, update),
          text=text,
      )

  async def _on_direct_result(content, tokens):
      nonlocal assistant_response_text
      assistant_response_text = self._direct_result_observer_text(content)
      await self._handle_direct_result(update, content)

  outcome = await stream_to_telegram(
      stream_response,
      edit=_edit, send=_send,
      cutoff_for=lambda content: get_stream_cutoff_values(update, content),
      on_direct_result=_on_direct_result,
  )
  if outcome.direct_result is not None:
      await self._dispatch_assistant_response_observer(..., text=assistant_response_text, tokens=outcome.total_tokens, ...)
      self._record_chat_usage(chat_id, user_id, outcome.total_tokens)
      return
  assistant_response_text = outcome.final_text
  total_tokens = outcome.total_tokens
  ```

  Важный нюанс (i==0 → полноценная markdown-разметка через `render_markdown_message_entities`,
  `bot/telegram_bot.py:4457-4468`) внутри `_send` теряется буквально — сегодня самое первое
  сообщение, если оно сразу финальное, шлётся с готовыми entities, а не голым текстом. Решение:
  `_send` сам проверяет `is_final` через замыкание над последним увиденным `tokens` **или**
  модуль передаёт готовый `markdown`-флаг и в `send`, а не только в `edit` (расширение сигнатуры
  `SendFn` до `Callable[[str, bool], Awaitable[Any]]` — минимальная правка дизайна, вносится на
  этом этапе, т.к. без неё первое короткое сообщение потеряет форматирование).
- Импорт `from .telegram_stream import stream_to_telegram` в `bot/telegram_bot.py` рядом с
  остальными импортами модуля (`:30-44`).

### Этап 2 — инлайн-запрос

- `bot/telegram_bot.py:4918-4976` — та же замена, `send=None`, `_edit` замыкается на
  `inline_message_id`, `cutoff_for` не меняется. `on_direct_result` строит
  `direct_result_inline_fallback_text` + `cleanup_intermediate_files` + свой `edit`, как сегодня
  (`:4919-4929`), не через реальную доставку.
- Особое внимание: сегодня `text = f'{query}\n\n{divider}{answer_tr}:{divider}\n{content}'`
  (`:4955`) — префикс `query` добавляется к каждому чанку. `_edit`/`_send`-замыкания должны
  оборачивать переданный модулем `content` в этот префикс *внутри* колбэка, не в модуле (иначе
  `cutoff`/`chunk_size` считались бы по чужой длине).

### Этап 3 — vision

- `bot/telegram_bot.py:3229-3349` — аналогично этапу 1, `on_direct_result` вызывает
  `self._handle_direct_result` + `record_vision_tokens` (не `_record_chat_usage`).
- `_send` для vision учитывает `reply_to_message_id` только на первом сообщении, как сегодня
  (`:3300-3308` использует его лишь при `i==0`) — раз `send` в новом дизайне используется и для
  overflow, и для i==0, и для fallback, реализация `_send` должна сама решать, добавлять ли
  `reply_to_message_id` (например, только если ещё не было ни одного отправленного сообщения в
  этом вызове — состояние, которое `_send`-замыкание держит через `nonlocal`).

Rich-ветка (`bot/telegram_bot.py:4317-4404`, `rich_stream_active`/`rich_stream_final_only`) не
трогается ни на одном из трёх этапов.

## Тесты

`tests/test_telegram_streaming.py` уже существует (1816 строк, 44 теста) и уже содержит нужные
фейки — новых базовых заглушек создавать не нужно:

- `FakeMessage`/`FakeUpdate`/`FakeContextBot`/`_make_context`/`_make_bot`
  (`tests/test_telegram_streaming.py:374-485`) — переиспользуются как есть для этапа 1.
  `FakeMessage.reply_text` уже поддерживает `reply_side_effects` (список исключений/результатов
  по очереди, конструктор `:380`, логика в `reply_text` `:414-419`) — то, что нужно для теста
  «N неудачных `edit`, потом успешный `send`».
- `FakeOpenAI.get_chat_response_stream` (`:177-184`) уже строит произвольный
  `(content, tokens)`-стрим из `chunks=[...]` — не нужно менять для инлайн/vision, только завести
  параллельные `_make_inline_bot`/`_make_vision_bot`-хелперы по образцу `_make_bot`, если в
  файле их ещё нет для этих путей (сейчас `_make_bot` покрывает только `process_message`).
- Для инлайн-теста нужен `update.callback_query` (`inline_message_id`, `data`,
  `from_user`) — в `tests/test_telegram_streaming.py` такого фейка сегодня нет, придётся
  добавить `FakeCallbackQuery`/`FakeInlineUpdate` по аналогии с `FakeUpdate`.
- Для vision — `FakeVisionOpenAI` уже есть (`:207-263`), но только для нестримингового
  `interpret_image`; понадобится добавить `interpret_image_stream` аналогично
  `FakeOpenAI.get_chat_response_stream`.

### Новые тесты

1. **Юнит-тесты модуля в изоляции** — новый файл `tests/test_telegram_stream_core.py`:
   стримит вручную собранные `(content, tokens)`-последовательности через `stream_to_telegram`
   с игрушечными `edit`/`send` (обычные `AsyncMock`/списки вызовов, без `ChatGPTTelegramBot`
   вообще). Кейсы:
   - обычный растущий ответ, единственное сообщение (i==0 → финал);
   - overflow на два+ чанка (`chunk_size` намеренно маленький, например 20, чтобы не тянуть
     4096-символьные фикстуры);
   - `RetryAfter`/`TimedOut`/`BadRequest`/произвольный `Exception` на промежуточном чанке —
     backoff растёт, `total_tokens` не спутан с промежуточным;
   - **основной кейс задачи**: ошибка `edit` ровно на финальном чанке — `max_final_attempts - 1`
     неудач подряд, затем успех (гарантия отработала через retry, `send` не понадобился);
   - тот же кейс, но все `max_final_attempts` попыток `edit` проваливаются — проверить, что
     `send` вызван один раз с полным `final_text`, `delivered=True`;
   - тот же кейс с `send=None` (инлайн) — после исчерпания попыток `delivered=False`, ошибка
     залогирована, `send` не звался (его и нет);
   - `direct_result`: `on_direct_result` вызван один раз с правильным `total_tokens`, цикл
     останавливается немедленно, `StreamOutcome.direct_result` — это исходный `content`.
2. **Формализовать репро** — перенести `/tmp/hunt/test_repro_final_chunk.py` в
   `tests/test_telegram_streaming.py` как постоянный тест (например
   `test_final_chunk_delivered_after_edit_failure`), **изменив ожидания на противоположные**:
   после фикса `TimedOut` на `"Hello final answer"` не должен приводить к потере — либо повтор
   `edit` успевает пройти (замокать так, чтобы второй вызов не бросал), либо срабатывает
   `reply_text`-fallback. Явно проверить `recorded.call_args.args[3] == 3` (было `0`).
3. **Обновить существующие тесты, если они закрепляют старое поведение.** Целевой прогон перед
   правкой — все 44 теста зелёные (см. «Команды проверки»); нужно перепроверить вручную
   `test_streaming_reply_text_failure_is_logged_and_does_not_retry_each_chunk`
   (`tests/test_telegram_streaming.py:1790-1816`) — она сегодня проверяет **сегодняшнее**
   поведение отказа на i==0 (не более одной попытки `reply_text`, `edit_message_with_retry` не
   вызывается вовсе). Дизайн этапа 1 сохраняет «не более одной попытки на конкретный чанк», но
   меняет исход при исчерпании: раньше стрим просто останавливался (`break`), теперь дойдёт до
   `_guarantee_final_delivery` в конце. Нужно решить с ревьюером, остаётся ли это утверждение
   теста нормативным (см. «Риски», пункт про i==0) — если да, тест не трогать; если поведение
   меняется намеренно, переписать assertion на «после исчерпания попыток по ходу стрима сработал
   финальный fallback».

## Команды проверки

```bash
# baseline — сегодняшнее поведение перед правками
~/.venvs/ctb/bin/python -m pytest -q tests/test_telegram_streaming.py
~/.venvs/ctb/bin/python -m pytest -q /tmp/hunt/test_repro_final_chunk.py

# этап 1 — модуль + основной текстовый путь
~/.venvs/ctb/bin/python -m pytest -q tests/test_telegram_stream_core.py tests/test_telegram_streaming.py

# этап 2 — инлайн (файл тестов инлайн-колбэка ещё не выделен — либо новые тесты
# в tests/test_telegram_streaming.py, либо отдельный tests/test_inline_streaming.py)
~/.venvs/ctb/bin/python -m pytest -q -k "inline" tests/test_telegram_streaming.py

# этап 3 — vision
~/.venvs/ctb/bin/python -m pytest -q -k "vision" tests/test_telegram_streaming.py

# соседние наборы, которые могут зависеть от edit_message_with_retry/split_into_chunks/utils
~/.venvs/ctb/bin/python -m pytest -q tests/test_utils_send_long_response_file.py \
  tests/test_callback_authorization.py tests/test_telegram_transcribe.py

# полный прогон (без evals/, см. AGENTS.md Testing And Verification)
~/.venvs/ctb/bin/python -m pytest -q
```

## Риски

- **Rich-режим — дефолт для приватных чатов, но остаётся вне T12.** Самый частый в проде путь
  (`TELEGRAM_RICH_MESSAGES=auto`, приватный чат) не получает фикс §3.6 в этом тикете. Такой же
  класс бага там уже есть (`raise` при `rich_stream_required`, потеря текста при `rich_stream_
  active` без required и ошибке на последнем чанке) — рекомендуется отдельный тикет
  (`bot/telegram_bot.py:4317-4404`, `bot/telegram_rich.py:102-153`), не блокирующий T12.
- **Изменение поведения на i==0 у легаси-текста** (`break` → тихий повтор + финальная гарантия)
  — расширяет число ретраев в худшем случае (раньше стрим быстро сдавался, теперь может
  пытаться до конца потока модели). Не приводит к зависанию (нет `while True`, ограничено
  длиной стрима модели), но меняет наблюдаемое поведение в тесте
  `test_streaming_reply_text_failure_is_logged_and_does_not_retry_each_chunk` — нужно явное
  решение ревьюера, разбирается в «Тесты», пункт 3.
- **`SendFn` меняет сигнатуру между «Дизайном» и этапом 1** (добавление `markdown: bool`
  параметра, см. «Правки», этап 1) — сделано намеренно, чтобы не потерять
  entity-based-разметку самого первого короткого ответа; отражено в обеих секциях, но
  разработчику стоит явно выбрать сигнатуру `SendFn` **до** написания юнит-тестов модуля,
  иначе тесты этапа 1 придётся переписывать.
- **`FakeOpenAI`/`FakeMessage` в `tests/test_telegram_streaming.py` разошлись с реальным
  дефолтным конфигом бота** (`telegram_rich_messages` отсутствует в `_make_bot`, хотя в проде
  дефолт `"auto"`) — при написании новых тестов легко случайно протестировать не тот путь,
  который реально исполняется в продакшене по умолчанию. Стоит явно комментировать в новых
  тестах, что `bot.config` намеренно эмулирует легаси-режим (`TELEGRAM_RICH_MESSAGES=off` или
  групповой чат), а не дефолт `bot/__main__.py`.
- **Ретраи с `asyncio.sleep(backoff_seconds * attempt)` в юнит-тестах модуля** должны
  мокаться (`monkeypatch.setattr(telegram_stream.asyncio, "sleep", AsyncMock())` — по образцу
  уже сделанного в `/tmp/hunt/test_repro_final_chunk.py`), иначе `max_final_attempts=3` и
  `retry_backoff_seconds=5.0` дадут тесту реальные секунды ожидания.
- **`_process_message_locked` — 524-строчная функция** (по аудиту, §2 пункт 1); правка этапа 1
  затрагивает только один `else`-блок внутри неё, но замыкания `_edit`/`_send`/`_on_direct_result`
  захватывают много локальных переменных функции (`chat_id`, `context`, `update`,
  `assistant_response_text`, `total_tokens` через `nonlocal`) — при переносе кода нужно сверить
  каждое замыкание построчно с оригиналом, чтобы не потерять какую-то деталь вроде
  `reply_to_message_id` только на первом сообщении.

## Критерии готовности

1. `bot/telegram_stream.py` существует, покрыт `tests/test_telegram_stream_core.py` без
   обращения к `ChatGPTTelegramBot`/Telegram API.
2. Репро-тест (см. «Тесты», п.2) зелёный с обратными по смыслу assertion'ами: ошибка `edit` на
   финальном чанке больше не приводит к потере текста и к `total_tokens == 0`.
3. Этапы 1–3 переведены на `stream_to_telegram`; rich-ветка (`bot/telegram_bot.py:4317-4404`)
   не изменена ни строкой.
4. `~/.venvs/ctb/bin/python -m pytest -q` — весь набор `tests/` и `bot/tests/` зелёный (базовая
   численность на HEAD ревью — 1449 passed, 1 skipped; после T12 число тестов должно вырасти за
   счёт новых, не уменьшиться).
5. Решение по «Риски» (i==0 поведение, вынесенное расширение `SendFn`) явно согласовано с
   ревьюером до слияния, а не решено implicit-но по ходу правки.
6. Найденный rich-режим зафиксирован как отдельный follow-up (тикет/пункт волны) с ссылкой на
   этот документ, а не потерян как побочное наблюдение планировщика.

## Постскриптум после ревью

Ревью (Sonnet, персона reviewer) подтвердило паритет с HEAD для backoff/`RetryAfter`/`TimedOut`,
однократный вызов `direct_result`-колбэков, байт-в-байт совпадение rich-ветки `else:` и
фиксацию `total_tokens` до попытки доставки. Найдены и исправлены две ошибки в
`bot/telegram_stream.py`:

1. **Гарантия финальной доставки переписывала хвостовое сообщение полным ответом.** `final_text`
   (полный текст для `StreamOutcome`/хуков) передавался в `_guarantee_final_delivery`, тогда как
   после нарезки по `chunk_size` редактируемое сообщение содержит только хвост. Введена
   отдельная переменная `pending_content` (текст текущего сообщения); `StreamOutcome.final_text`
   по-прежнему полный ответ. При неудачном открытии хвостового сообщения `sent_message`
   сбрасывается в `None`, чтобы следующий чанк или гарантия открыли его заново, а не
   перезаписали уже закрытый чанк.
2. **Инлайн-режим потерял throttling.** Признак «сообщение уже открыто» был завязан на
   `sent_message is not None`, который для `send=None` никогда не наступает, поэтому каждый чанк
   уходил в `edit`. Добавлен флаг `inline_opened` (выставляется после первого успешного edit,
   как `i == 0` в HEAD); промежуточные чанки инлайна снова отсекаются по `cutoff`.

Попутно восстановлен паритет с HEAD для инлайна: там нарезки по 4096 не было (весь текст идёт
одним edit, `edit_message_with_retry` сам усекает), а новый код усекал инлайн-ответ до хвоста.
Нарезка теперь применяется только при `send is not None`.

Добавлены тесты в `tests/test_telegram_stream_core.py`: переполнение на финальном чанке
(гарантия доставляет хвост, а не полный текст; повторное открытие хвоста после неудачного
send), throttling инлайна, повтор первого инлайн-edit без throttling после неудачи, отсутствие
нарезки в инлайне. В `tests/test_telegram_streaming.py` в тесты `records_once` добавлен
`recorded.assert_called_once()` (замечание ревью).

Принятые отклонения (не блокеры по оценке ревью): при первом же чанке > 4096 хвост не получает
`reply_to_message_id` (HEAD делал это только в этом узком случае через delete+resend; теперь
поведение единообразно). Rich-drafts ветка — вне периметра, отдельный тикет.

Проверка после правок: `ruff check bot tests bot/tests` чист, полный прогон `1535 passed`.
