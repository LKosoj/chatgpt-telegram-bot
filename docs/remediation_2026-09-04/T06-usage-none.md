# T06. Провайдер превращает отсутствующие prompt/completion_tokens в 0

Источники задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел T06) и
`docs/architecture_code_review_2026-09-04.md` §4.1 (пункт «Провайдер превращает отсутствующие
prompt/completion_tokens в 0»). Термины: *prompt_tokens* — сколько токенов (кусочков текста,
на которые модель режет ввод) ушло на запрос пользователя; *completion_tokens* — сколько ушло
на ответ модели; *total_tokens* — их сумма, которую некоторые шлюзы (gateway — прокси перед
реальным API модели) присылают одним числом, не раскладывая на составляющие.

## Цель

Когда шлюз (gateway) не прислал разбивку `prompt_tokens`/`completion_tokens` (прислал только
`total_tokens` или вообще ничего), код не должен подменять отсутствующее значение на `0`.
`0` — это «известно, что токенов не было», а не «неизвестно, сколько было». Из-за подмены
`bot.pricing.resolve_chat_cost()` (файл `bot/pricing.py`, решает, как посчитать цену запроса)
выбирает ветку `model_split` (цена считается отдельно за prompt и completion) с обеими частями
по нулю → стоимость запроса всегда получается `0.0`, хотя реально потрачены токены и должна
была сработать соседняя ветка `model_blended` (цена считается по `total_tokens` через среднюю
ставку prompt/completion). Это нарушает правило из AGENTS.md (раздел Chat Token Pricing):
«report unknown rather than a split that silently omits» (лучше явно сказать «неизвестно», чем
тихо занизить сумму).

## Репро

Баг воспроизведён в `/tmp/t06_repro.py` (не часть репозитория, чистый скрипт для проверки):

```python
import asyncio
from types import SimpleNamespace

from bot.ai_providers.openai_compatible import OpenAICompatibleProvider
from bot.ai_provider import AIProviderRequest, collect_ai_response
from bot.pricing import resolve_chat_cost


async def main():
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi", tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(total_tokens=5, prompt_tokens=None, completion_tokens=None),
    )

    async def create_chat_completion(**kwargs):
        return fake_response

    provider = OpenAICompatibleProvider(create_chat_completion)
    response = await collect_ai_response(provider.stream_response(
        AIProviderRequest(model="m", messages=()),
    ))
    print("AIUsage:", response.usage)

    cost, price_source = resolve_chat_cost(
        model="m", total_tokens=5,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        fallback_price_per_1k=0.002, table={"m": (1.0, 2.0)},
    )
    print("resolve_chat_cost ->", cost, price_source)


asyncio.run(main())
```

Запуск (`PYTHONPATH=. python3 /tmp/t06_repro.py`) на текущем коде даёт:

```
AIUsage: AIUsage(prompt_tokens=0, completion_tokens=0, total_tokens=5)
resolve_chat_cost -> 0.0 model_split
```

Ожидалось: `AIUsage(prompt_tokens=None, completion_tokens=None, total_tokens=5)` и
`(cost, 'model_blended')` с `cost = 5 * ((1.0 + 2.0) / 2) / 1000`.

Второй репро-скрипт (`/tmp/t06_pricing_check.py`) гоняет тот же случай через реальный путь
бота — `OpenAIHelper.get_chat_response()` с фейковым HTTP-клиентом
(`tests/test_openai_helper_tool_calls.py::FakeResponse(..., prompt_tokens=None,
completion_tokens=None, total_tokens=5)`) — и показывает тот же эффект «на два уровня выше»:
`helper.get_last_chat_usage_split(1)` возвращает `(0, 0)` вместо `None`, а показанная
пользователю строка использования получается `"💰 5 tokens (0 prompt, 0 completion)"` вместо
скрытия разбивки. Третий скрипт (`/tmp/t06_vision_check.py`) повторяет то же самое для
vision-пути (`OpenAIHelper._interpret_image_text_response`) и получает
`"💰 5 tokens (None prompt, None completion)"` — то есть после точечного фикса только
`AIUsage`/`_usage()` (без доп. правки в местах вывода) в ответ пользователю попадёт буквальное
слово `None`.

## Путь данных (проверено по коду, что не так и что уже работает правильно)

`OpenAIHelper.chat_completion()` (`bot/openai_helper.py:444`) — единственная точка входа
нестримингового запроса — всегда идёт через `_create_chat_response_completion()`
(`bot/openai_helper.py:687-690`), которая при `chat_run_variant_b_enabled=True` (дефолт,
`bot/openai_helper.py:342`, `bot/__main__.py:214`) вызывает `_timed_create_via_ai_provider()`
(`bot/openai_helper.py:613-676`). Тот собирает `AIProviderResponse` через
`collect_ai_response()` (`bot/ai_provider.py:57-104`), а `.usage` в нём — это `AIUsage`,
построенный `_usage()` в `bot/ai_providers/openai_compatible.py:254-262`.

Это значит, что «легаси»-тело `bot/openai_helper.py:934-1091` и `ChatRun.run_non_stream()`
(`bot/chat_run.py`) — не два независимых пути с разными объектами `response`: они читают
**один и тот же** `AIProviderResponse`, когда флаг включён (по умолчанию). При выключенном
флаге `chat_completion()` вместо этого возвращает сырой объект SDK (`_timed_create()`,
`bot/openai_helper.py:490-518`), который никогда не проходит через `_usage()` — там баг не
воспроизводится (его собственное потенциальное отсутствие полей — другая, вне-scope T06,
история, см. «Риски»).

Виновник — `_usage()`/`_int_or_zero()`:

```python
# bot/ai_providers/openai_compatible.py:254-269 (текущий код)
def _usage(response: Any) -> AIUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return AIUsage(
        prompt_tokens=_int_or_zero(getattr(usage, "prompt_tokens", 0)),
        completion_tokens=_int_or_zero(getattr(usage, "completion_tokens", 0)),
        total_tokens=_int_or_zero(getattr(usage, "total_tokens", 0)),
    )


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
```

`_int_or_zero(None)` → `int(None or 0)` → `0`. Отсутствие поля и «реально было 0 токенов»
становятся неразличимы уже на этом шаге, до того как значение попадёт куда-либо ещё.

Ниже по потоку логика уже написана правильно и рассчитана на `None`, но с рождения получает
`0` и потому никогда не срабатывает для этого случая:

- `bot/chat_response_utils.py:53-68` (`response_prompt_completion_tokens`) — «legacy»-путь
  сравнения: `if prompt_tokens is None or completion_tokens is None: return None`. Именно так
  и должно быть — это образец поведения, к которому нужно привести `AIUsage`.
- `bot/chat_response_utils.py:72-89` (`aggregate_usage_split`) — считает разбивку по всем
  «прогонам» (round trips, отдельным обращениям к модели за один ход диалога) только если для
  каждого из них есть запись в `usage_accumulator`; неполный набор → `None`.
- `bot/pricing.py:82-120` (`resolve_chat_cost`) — `model_split`, только если
  `prompt_tokens is not None and completion_tokens is not None`; иначе `model_blended` (если
  модель есть в таблице цен) или `legacy_fallback`.

Прямые потребители `AIUsage.prompt_tokens`/`.completion_tokens` (не через
`response_prompt_completion_tokens`, а напрямую как атрибут) — их два, и оба живые в проде при
дефолтном флаге:

- `bot/chat_run.py:240-245` — строка с деталями использования токенов в ответе пользователю
  (`ChatRun.run_non_stream`, обычный текстовый путь).
- `bot/openai_helper.py:2778-2781` — тот же паттерн в `_interpret_image_text_response`
  (vision-путь, работает независимо от флага, т.к. у vision нет отдельного ChatRun-варианта).

Оба сейчас читают `response.usage.prompt_tokens`/`.completion_tokens` без проверки на `None`
— если после фикса `_usage()` поле станет `None`, здесь в ответ пользователю уйдёт буквальный
текст `"None"` (см. репро выше). Их нужно поправить в этом же PR — иначе фикс `_usage()`
превратит тихий баг с ценой в видимый баг с текстом ответа.

Стриминг (`OpenAIHelper.get_chat_response_stream`, `bot/openai_helper.py:1240` и
`bot/openai_tool_handler.py:1114`) использует `_response_prompt_completion_tokens` (алиас той
же `response_prompt_completion_tokens`) прямо на сыром чанке SDK — эта ветка вообще не ходит
через `AIUsage`/`_usage()`, значит T06 её не касается и `tests/test_stream_usage.py` останется
как есть (см. «Команды проверки» — гоняем его только чтобы убедиться, что фикс ничего не
задел).

`telegram_bot.py:805-811` (`_record_chat_usage`, вызывает `get_last_chat_usage_split`) уже
написан безопасно: `prompt_tokens=last_split[0] if last_split else None`.
`bot/openai_helper.py:551-562` (INFO-лог отладки, читает сырой `response.usage` до провайдера)
тоже уже безопасен (`getattr(usage, 'prompt_tokens', None)`). Их трогать не нужно.

## Правки

### 1. `bot/ai_events.py:19-22` — `AIUsage`: разрешить `None` для split-полей

Сейчас:

```python
@dataclass(frozen=True, slots=True)
class AIUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

Правка (меняются только две строки, `total_tokens` не трогаем — его семантика «0, если
неизвестно» и так уже совпадает с `response_total_tokens()` из
`bot/chat_response_utils.py:46-50`, `getattr(..., "total_tokens", 0) or 0`, это не часть
бага):

```python
@dataclass(frozen=True, slots=True)
class AIUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int = 0
```

Проверено: `AIUsage(...)` конструируется только в `bot/ai_providers/openai_compatible.py:258`
и в тестах с явными значениями (`grep -rn "AIUsage(" bot tests` — единственные места); голых
`AIUsage()` без аргументов в дереве нет, так что смена дефолта не меняет поведение нигде, кроме
того единственного места, которое и чиним.

### 2. `bot/ai_providers/openai_compatible.py:254-269` — новая `_int_or_none`, `_usage` использует её для split-полей

Сейчас (см. «Путь данных» выше). Правка — добавить `_int_or_none` рядом с `_int_or_zero` и
переключить на неё только prompt/completion (total_tokens и `_int_or_zero` в
`stream_chunk_tool_call_deltas`, `bot/ai_providers/openai_compatible.py:149`, не трогаем):

```python
def _usage(response: Any) -> AIUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return AIUsage(
        prompt_tokens=_int_or_none(getattr(usage, "prompt_tokens", None)),
        completion_tokens=_int_or_none(getattr(usage, "completion_tokens", None)),
        total_tokens=_int_or_zero(getattr(usage, "total_tokens", 0)),
    )


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _int_or_none(value: Any) -> int | None:
    """Как _int_or_zero, но не подменяет отсутствующее значение нулём.

    bot.pricing.resolve_chat_cost() читает None как "разбивка неизвестна" и
    считает по model_blended; 0 читается как "известно, что токенов не
    было" и уводит в model_split с нулевой ценой. См.
    docs/remediation_2026-09-04/T06-usage-none.md.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
```

Явный `0` (реальный, не пропущенный) по-прежнему остаётся `0`, а не `None` — `_int_or_none(0)`
возвращает `0` (проверка `value is None` идёт раньше `int(value)`), что и нужно: настоящий
ноль токенов — это не то же самое, что «неизвестно».

### 3. `bot/chat_run.py:240-245` — не печатать `None` пользователю, если split неизвестен

Сейчас:

```python
                if total_tokens == usage_tokens:
                    answer += (
                        f" ({str(response.usage.prompt_tokens)} {localized_text('prompt', bot_language)},"
                        f" {str(response.usage.completion_tokens)} "
                        f"{localized_text('completion', bot_language)})"
                    )
```

Правка:

```python
                usage = response.usage
                if (
                    total_tokens == usage_tokens
                    and usage is not None
                    and usage.prompt_tokens is not None
                    and usage.completion_tokens is not None
                ):
                    answer += (
                        f" ({str(usage.prompt_tokens)} {localized_text('prompt', bot_language)},"
                        f" {str(usage.completion_tokens)} "
                        f"{localized_text('completion', bot_language)})"
                    )
```

Побочный эффект (в плюс, бесплатно вместе с той же правкой): раньше если `response.usage` в
принципе `None` (сырой ответ вообще без `usage`) и при этом `total_tokens == usage_tokens == 0`
совпадали, код падал бы на `AttributeError: 'NoneType' object has no attribute
'prompt_tokens'`. Это была отдельная, более редкая заготовка бага, не введённая T06, но
`usage is not None` в том же условии закрывает и её без лишних строк.

Имя локальной переменной `usage` в этой функции свободно — рядом есть `usage_accumulator` и
`usage_tokens`, но не `usage` (проверено по всей функции `run_non_stream`,
`bot/chat_run.py:47-248`).

### 4. `bot/openai_helper.py:2778-2781` — тот же guard в vision-пути

Сейчас:

```python
            if total_tokens == usage_tokens:
                answer += \
                      f" ({str(response.usage.prompt_tokens)} {localized_text('prompt', bot_language)}," \
                      f" {str(response.usage.completion_tokens)} {localized_text('completion', bot_language)})"
```

Правка:

```python
            usage = response.usage
            if (
                total_tokens == usage_tokens
                and usage is not None
                and usage.prompt_tokens is not None
                and usage.completion_tokens is not None
            ):
                answer += \
                      f" ({str(usage.prompt_tokens)} {localized_text('prompt', bot_language)}," \
                      f" {str(usage.completion_tokens)} {localized_text('completion', bot_language)})"
```

Это внутри `_interpret_image_text_response` (`bot/openai_helper.py:2749-2787`); имя `usage` там
тоже свободно (в этой функции есть только `usage_tokens`, не `usage`).

### Не трогаем (вне scope T06)

`bot/openai_helper.py:1074-1090` — тот же паттерн внутри «легаси»-тела `__common_get_chat_response`
(`bot/openai_helper.py:934-1091`). Он выполняется только при `chat_run_variant_b_enabled=False`
(`bot/openai_helper.py:922-926` отдаёт управление в `ChatRun` при дефолтном `True`), и в этой
ветке `response` — сырой объект SDK из `_timed_create()`, а не `AIProviderResponse`: он никогда
не проходит через `_usage()`/`AIUsage`, значит правка T06 его не касается. Если у сырого SDK-ответа
`prompt_tokens`/`completion_tokens` тоже могут быть `None` — это уже существовало и не зависит от
данного фикса; трогать не будем, чтобы не выходить за рамки задачи (`§5.1 П1` архитектурного
обзора и так предлагает целиком удалить это тело как дубликат `ChatRun`).

## Новые тесты

### `tests/test_ai_provider.py` — юнит-тест на `_usage()`/`AIUsage` через публичный API провайдера

```python
from types import SimpleNamespace

from bot.ai_providers.openai_compatible import OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_usage_keeps_missing_prompt_completion_as_none():
    """Шлюз прислал total_tokens, но не прислал prompt/completion_tokens.

    Regression test for T06: раньше _usage()/_int_or_zero превращали
    отсутствующие поля в 0, из-за чего resolve_chat_cost решал, что
    разбивка известна (model_split, цена 0.0), вместо честного
    model_blended.
    """
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi", tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(total_tokens=5, prompt_tokens=None, completion_tokens=None),
    )

    async def create_chat_completion(**kwargs):
        return fake_response

    provider = OpenAICompatibleProvider(create_chat_completion)
    response = await collect_ai_response(provider.stream_response(
        AIProviderRequest(model="m", messages=()),
    ))

    assert response.usage == AIUsage(prompt_tokens=None, completion_tokens=None, total_tokens=5)
```

`SimpleNamespace`/`OpenAICompatibleProvider` уже импортируются в файле (кроме
`OpenAICompatibleProvider` и `SimpleNamespace` — их нужно добавить в импорты сверху файла);
`AIUsage`, `AIProviderRequest`, `collect_ai_response` уже импортированы.

### `tests/test_pricing.py` — сквозной тест через реальный путь бота (аналог уже существующего `test_usage_split_reaches_record_chat_tokens_through_tool_call_round_trip`)

```python
@pytest.mark.asyncio
async def test_usage_split_stays_unknown_when_gateway_omits_split():
    """Regression test for T06: шлюз прислал total_tokens=5, но не прислал
    prompt/completion_tokens. get_last_chat_usage_split должен вернуть
    None (а не (0, 0)), resolve_chat_cost -- посчитать по model_blended
    (а не по model_split с нулевой ценой), и в ответе пользователю не
    должно быть строки "None".
    """
    pm = DummyPluginManager({}, specs=[])
    client = DummyClient([
        FakeResponse(content="final answer", total_tokens=5, prompt_tokens=None, completion_tokens=None),
    ])
    helper = _make_helper(pm, client=client)
    helper.config["chat_run_variant_b_enabled"] = True
    helper.config["enable_functions"] = False
    helper.config["show_usage"] = True

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1, query="hi", user_id=1,
    )

    assert total_tokens == 5
    assert "None" not in answer
    assert helper.get_last_chat_usage_split(1) is None

    model = helper.get_last_chat_model(1)
    cost, price_source = resolve_chat_cost(
        model=model,
        total_tokens=total_tokens,
        prompt_tokens=None,
        completion_tokens=None,
        fallback_price_per_1k=0.002,
        table={model: (1.0, 2.0)},
    )

    assert price_source == "model_blended"
    assert cost == pytest.approx(5 * ((1.0 + 2.0) / 2) / 1000)
```

Технически проверено черновиком `/tmp/t06_pricing_check.py` (см. «Репро») — на текущем коде
даёт `split (0, 0)` и `(0.0, 'model_split')`, то есть без фикса этот тест красный, как и
положено regression-тесту.

### `tests/test_pricing.py` — тот же случай для vision-пути

```python
@pytest.mark.asyncio
async def test_interpret_image_text_response_omits_split_detail_when_unknown():
    """Regression test for T06 (vision-путь): то же самое, что предыдущий
    тест, но для OpenAIHelper._interpret_image_text_response, которая
    задействована независимо от chat_run_variant_b_enabled.
    """
    pm = DummyPluginManager({}, specs=[])
    helper = _make_helper(pm, client=DummyClient([]))
    helper.config["show_usage"] = True
    response = FakeResponse(content="a photo", total_tokens=5, prompt_tokens=None, completion_tokens=None)

    answer, total_tokens = await helper._interpret_image_text_response(
        chat_id=1, response=response, token_accumulator=[],
    )

    assert total_tokens == 5
    assert "None" not in answer
    assert "💰 5" in answer
```

Технически проверено черновиком `/tmp/t06_vision_check.py` — на текущем коде даёт
`'a photo\n\n---\n💰 5 tokens (None prompt, None completion)'`, тест красный без фикса.

Три новых теста написаны так, чтобы упасть на нынешнем коде и пройти после правок 1-4 — это
и есть критерий «регрессионный тест», а не просто новый тест.

## Команды проверки

```bash
# baseline (уже зелёный до правок, подтверждает, что имена файлов/тестов верны)
python3 -m pytest tests/test_ai_provider.py tests/test_pricing.py tests/test_stream_usage.py \
  tests/test_usage_record_helpers.py -q -p no:cacheprovider

# после правок 1-4 и добавления трёх новых тестов -- те же файлы плюс сопутствующие
python3 -m pytest tests/test_ai_provider.py tests/test_pricing.py tests/test_stream_usage.py \
  tests/test_usage_record_helpers.py tests/test_ai_events.py \
  tests/test_openai_helper_tool_calls.py -q -p no:cacheprovider

# полный прогон -- убедиться, что смена дефолта AIUsage.prompt_tokens/completion_tokens
# ни на что больше не повлияла
python3 -m pytest -q -p no:cacheprovider
```

`tests/test_openai_helper_tool_calls.py` — обязательно в список, т.к. там же лежат `FakeResponse`,
`DummyClient`, `_make_helper`, используемые новыми тестами, и там же
`test_timed_create_via_provider_returns_provider_response_not_raw_sdk` (строка 765) уже сравнивает
`AIUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)` с реальными числами — эта строка
не должна была измениться в поведении (числа не `None`), но стоит перепроверить явно.

## Риски

- **Молчаливая правка дефолта `AIUsage`.** Смена `int = 0` на `int | None = None` — потенциальный
  breaking change для любого стороннего кода, который сравнивает `AIUsage.prompt_tokens == 0` или
  делает над ним арифметику без проверки на `None`. В дереве таких мест не найдено (проверено
  grep'ом по `AIUsage(` и по `.prompt_tokens`/`.completion_tokens`/`.total_tokens` на `response.usage`
  и `usage.` во всём `bot/`), но если T01 (параллельная задача, поднимает `openai` до 3.x) добавит
  новых потребителей `AIUsage` в те же дни — нужно перепроверить после мержа T01.
- **`_int_or_zero` используется и для `total_tokens`, и для индекса tool-call дельты
  (`bot/ai_providers/openai_compatible.py:149`).** Правка не трогает `_int_or_zero`, только
  добавляет отдельную `_int_or_none` — риск случайно поменять поведение индекса tool-call
  отсутствует, но при реализации проверить, что импорт/место вставки `_int_or_none` не спутано
  с `_int_or_zero` по имени при копировании.
- **`bot/openai_helper.py:1074-1090` (легаси-тело) намеренно не трогаем** — при будущем удалении
  этого тела (§5.1 П1 архитектурного обзора) не забыть, что T06 покрывает только
  `AIProviderResponse`-путь; если легаси-тело оживят или скопируют, тот же guard понадобится и там.
- **Пустая строка вместо скобок.** После правок 3-4 при неизвестном split пользователь просто не
  увидит `(N prompt, M completion)` — это осознанное поведение (лучше промолчать, чем соврать), но
  стоит явно принять как ожидаемое, а не как "недостающий UI".
- Тесты идут через `FakeResponse`/`DummyClient` (не реальный `openai` SDK), поэтому апгрейд `openai`
  до 3.x в параллельной задаче T01 не должен требовать правок в этом PR — но команда проверки
  «полный прогон» в конце стоит повторить после мержа T01, на новом venv.

## Критерии готовности

1. `bot/ai_events.py`, `bot/ai_providers/openai_compatible.py`, `bot/chat_run.py`,
   `bot/openai_helper.py` изменены точно по правкам 1-4 выше, без побочных правок в
   несвязанных местах.
2. Три новых теста добавлены (`tests/test_ai_provider.py` — 1 тест;
   `tests/test_pricing.py` — 2 теста) и до правок 1-4 (на текущем коде) красные, после — зелёные.
   Проверить вручную `git stash` до/после, либо прогнать тесты на копии без правок 1-4.
3. `python3 /tmp/t06_repro.py` после правок печатает
   `AIUsage(prompt_tokens=None, completion_tokens=None, total_tokens=5)` и
   `resolve_chat_cost -> <ненулевая цена> model_blended`.
4. Полный прогон из «Команды проверки» зелёный, включая `tests/test_stream_usage.py` (стриминг
   не должен был измениться) и `tests/test_openai_helper_tool_calls.py::test_timed_create_via_provider_returns_provider_response_not_raw_sdk`.
5. В новых тестах нет буквального слова `"None"` в проверяемом пользовательском тексте (`answer`)
   — то есть regression покрывает не только цену, но и то, что видит пользователь.

## Постскриптум после ревью (2026-09-04)

Реализовано 1:1 по плану: `AIUsage.prompt_tokens`/`completion_tokens` → `int | None`
(`bot/ai_events.py`), `_int_or_none` в `bot/ai_providers/openai_compatible.py`, guard перед выводом
строки usage пользователю в `bot/chat_run.py` и `bot/openai_helper.py`. Ревью (Sonnet, persona
reviewer): ошибок и предупреждений нет; все потребители `prompt_tokens`/`completion_tokens`
(`chat_response_utils`, `pricing`, `utils.record_chat_tokens`, `session_logger`, `session_otel`)
проверены — арифметики над `None` нет.
