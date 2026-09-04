# T20. Общие тестовые заглушки — план (код не менялся)

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md:182-185` («Волна 5 — тесты и
документация»): «`tests/fakes.py` (`FakeMessage`, `FakeDB`, `FakeHelper`, `FakeEncoding`,
`FakePluginManager`) + фикстуры в `tests/conftest.py`; мигрировать самые частые дубли; `assert` в
`test_concurrent_access_smoke`». Находка — `docs/architecture_code_review_2026-09-04.md:498`:
«213 классов `Fake*/Dummy*/Stub*` в 53 файлах (`FakeMessage` — 15 копий, `FakeDB` — 12,
`FakeHelper` — 11). Нужен `tests/fakes.py`».

Роль этого документа — план для следующего агента (разработчика). Режим этой задачи —
только чтение: код и тесты не менялись, создан один файл — этот.

**Главный вывод инвентаризации, который меняет объём задачи:** одинаковое имя класса
(`FakeDB`, `FakeHelper`, `FakePluginManager`, отчасти `FakeMessage`) в этом проекте почти
всегда означает «минимальная заглушка, написанная заново под конкретный тест», а не
скопированный код. Реальных байт-в-байт (или почти байт-в-байт) дублей с ≥3 копиями нашлось
всего два кластера классов и один кластер фикстур — они и есть содержимое `tests/fakes.py` и
новой фикстуры в `tests/conftest.py`. Остальные три из пяти названных в задаче имён
(`FakeDB`, `FakeHelper`, `FakePluginManager`) **не выносятся** — ниже показано, чем именно
отличаются их копии и почему объединение их сломало бы либо испортило тесты.

## Термины простым языком

- **Заглушка / фейк (fake, test double)** — упрощённый объект, который тесты подсовывают вместо
  настоящего (базы данных, Telegram-сообщения, ответа OpenAI), чтобы не ходить в сеть и не
  поднимать реальную инфраструктуру.
- **Дубликат (почти одинаковая копия)** — здесь считается «почти одинаковым», только если тело
  класса/функции совпадает дословно или отличается на 1 неважную деталь (комментарий,
  порядок ключей). Разные значения по умолчанию, разный набор методов или разный порядок
  аргументов — это **не** дубликат, а разные заглушки, которые случайно называются одинаково.
- **Фикстура (fixture)** — функция pytest с декоратором `@pytest.fixture`, которую тест получает
  как параметр; pytest сам находит её по имени параметра — файл, где она объявлена, не нужно
  импортировать явно. Если фикстура лежит в `conftest.py`, она доступна всем тестам в этой папке
  и её подпапках без импорта.
- **autouse-фикстура** — фикстура, которая выполняется для каждого теста автоматически, даже
  если тест её не запрашивал как параметр. Задача явно запрещает делать новые фикстуры
  autouse — только «по запросу».
- **namespace package (пакет без `__init__.py`)** — начиная с Python 3.3 папка без файла
  `__init__.py` всё равно может импортироваться как пакет (`import tests`), если её путь есть в
  `sys.path`. Это отличается от обычного пакета только тем, что Python ищет её «неявно».
- **`sys.path`** — список папок, где Python ищет модули при `import`. `tests/conftest.py`
  вставляет туда корень репозитория при старте, поэтому `from bot... import ...` работает из
  любого теста.

## Цель

Выяснить, какие из «одноимённых» тестовых заглушек в `tests/*.py` (и `bot/tests/*.py`)
действительно дублируют друг друга, спроектировать минимальный `tests/fakes.py` и одну фикстуру
в `tests/conftest.py` только для подтверждённых дублей, расписать точечные правки по
`file:line`, и предложить `assert`-ы для `test_concurrent_access_smoke`, который сейчас ничего
не проверяет. Реализация — в компетенции следующего агента.

## Инвентаризация

### 1. Методология

Поиск через `ast`-обход всех `.py` в `tests/` и `bot/tests/` (не `rg`/`grep` — по правилам
окружения они иногда искажают вывод на этой машине; `python3` с модулем `ast` даёт точные
`file:line` и точные тела классов/функций для посимвольного сравнения). Проверено:

- все классы с именем, начинающимся на `Fake`/`Dummy`/`Stub`/`Mock` (с учётом варианта с
  ведущим подчёркиванием, например `_FakeEncoding`, и регистра);
- все функции-фабрики `_make_*`/`make_*` (`_make_bot`, `_make_helper`, `_make_db` и т. п.);
  все именованные pytest-фикстуры (`@pytest.fixture`), сгруппированные по имени функции;
- для каждой группы с ≥2 совпадениями — построчное сравнение тел через `ast.get_source_segment`
  и, где нужно, реальные вызовы (`grep`-по-питону) на месте использования, чтобы отличить
  «то же имя, тот же контракт» от «то же имя, другой контракт».

`bot/tests/` (MCP-тесты) не содержит ни одного класса `Fake*/Dummy*/Stub*/Mock*` — вся находка
целиком лежит в `tests/`.

### 2. Общая картина

- 199 классов с именем `Fake…`/`Dummy…`/`Stub…`/`Mock…` без ведущего подчёркивания в 50 файлах
  `tests/*.py` (223, если считать варианты с ведущим `_`, например `_FakeEncoding`,
  `_FakePluginManager`, `_FakeHindsightPlugin`). Это ниже, чем «213» в
  `docs/architecture_code_review_2026-09-04.md:498`, — на новом коде с 2026-09-04 могло
  чуть измениться; порядок величины подтверждён, точное число из старого обзора **не
  бралось на веру** (проверяй `file:line`, не число из прошлого документа — это ровно та
  ошибка, от которой предостерегает `AGENTS.md`: не доверять `file:line`/утверждениям другого
  прохода без переоткрытия).
- Самые частые имена (по счётчику, а не по факту дублирования — см. §3):
  `FakeMessage` 15, `FakeDB` 13, `FakeHelper` 12, `FakeResponse` 10, `_FakeEncoding` 10,
  `FakePluginManager` 8, `FakeToolCall` 6, `FakeChoice` 6, `FakeUpdate` 6.
- Фабрики с одинаковым именем: `_make_bot` — 8 файлов, `_make_helper`-семейство (`_make_helper`,
  `_make_helper_ask`, `_make_helper_for_close`, `_make_helper_for_save`, `_make_helper_resolve`,
  `_make_helper_stats`) — 13 функций, `make_helper` — 1 (в `tests/test_hindsight_memory.py`).
  `_install_module_if_missing` (не класс, вспомогательная функция для подмены отсутствующего
  модуля через `sys.modules`) — 15 копий.
- Именованные pytest-фикстуры, встречающиеся под одним именем ≥2 раза: `plugin` (7), `agent_db`
  (5), `db` (3), `_reset_to_thread_calls` (3).

### 3. Разбор по группам — что реально дублируется, а что только называется одинаково

| Группа | Копий (имя) | Реально дублируется? | Куда идёт |
|---|---|---|---|
| `_FakeEncoding` (tiktoken-заглушка) | 10 | Да — все 10 тел байт-в-байт идентичны | `tests/fakes.py::FakeEncoding` |
| `_install_module_if_missing` (функция, не класс) | 15 | Да — все 15 тел байт-в-байт идентичны | бонус, вне исходного списка задачи, см. §«Риски» |
| `FakeMessage` + `FakeChoice` (форма ответа OpenAI: `tool_calls`/`content`) | 4 файла из 15 «FakeMessage» и 6 «FakeChoice» | Да, для этих 4 файлов — оба класса байт-в-байт идентичны | `tests/fakes.py::FakeMessage`, `FakeChoice` |
| `agent_db` (pytest-фикстура: чистая SQLite + DDL `AgentToolsPlugin`) | 5 | Да — 4 тела дословно идентичны, 1 отличается одним комментарием | новая фикстура в `tests/conftest.py` |
| `FakeToolCall` (в тех же 4 файлах, что и `FakeMessage`/`FakeChoice`) | 6 | **Нет** — 4 файла с разным контрактом (см. ниже) | остаётся локальным |
| `FakeResponse` | 10 | **Нет** — разный набор атрибутов (`usage` есть не везде, разные конструкторы) | остаётся локальным |
| `FakeDB` | 13 | **Нет** (кроме одной пары из 2) — 9 из 13 это вложенные в конкретный тест `test_plugin_manager.py` одноразовые классы с разными возвращаемыми данными | остаётся локальным |
| `FakeHelper` | 12 | **Нет** (кроме одной пары из 2) — под одним именем то и дело живёт совершенно другой набор методов | остаётся локальным |
| `FakePluginManager` | 8 | **Нет** — 8 разных наборов методов, общего подмножества, нужного ≥3 файлам одинаково, не нашлось | остаётся локальным |
| `_make_bot` | 8 | **Нет** — каждый вручную выставляет свой набор атрибутов на `object.__new__(ChatGPTTelegramBot)` | остаётся локальным |
| `_make_helper`/`make_helper`-семейство | 13 (+1) | Частично — 2 из проверенных строят реальный `OpenAIHelper` из ~39-ключевого конфига, совпадающего на 90%; остальные — «ручная» частичная сборка через `object.__new__`, разная везде | не входит в этот план, см. §«Риски» («вторая волна») |

Ниже — доказательства по самым важным строчкам таблицы.

#### 3.1. `_FakeEncoding` → `FakeEncoding` (безопасно, 10 копий)

Во всех 10 файлах тело идентично:

```python
class _FakeEncoding:
    def encode(self, value):
        return list(value)
```

Файлы и точные диапазоны класса (`file:lineno-end_lineno`):

| Файл | Класс | Следующие строки (использование) |
|---|---|---|
| `tests/test_openai_helper_tool_calls.py:22-24` | `_FakeEncoding` | `:27-30` — `_tiktoken.encoding_for_model/get_encoding = lambda …: _FakeEncoding()`, `_install_module_if_missing("tiktoken", _tiktoken)` |
| `tests/test_skills_agent_gate.py:30-32` | `_FakeEncoding` | `:35-38` |
| `tests/test_telegram_streaming.py:27-29` | `_FakeEncoding` | `:32-35` |
| `tests/test_telegram_builder_config.py:21-23` | `_FakeEncoding` | `:26-29` |
| `tests/test_per_conversation_serialization.py:21-23` | `_FakeEncoding` | `:26-29` |
| `tests/test_group_session_flow.py:21-23` | `_FakeEncoding` | `:26-29` |
| `tests/test_plugin_handlers_registration.py:20-22` | `_FakeEncoding` | `:25-28` |
| `tests/test_plugin_menu_force_reply.py:20-22` | `_FakeEncoding` | `:25-28` |
| `tests/test_callback_authorization.py:21-23` | `_FakeEncoding` | `:26-29` |
| `tests/test_telegram_transcribe.py:22-24` | `_FakeEncoding` | `:27-30` |

Смысл конструкции: `tiktoken` — библиотека OpenAI для подсчёта токенов; если она не
установлена (`importlib.util.find_spec("tiktoken") is None`), файл подкладывает в
`sys.modules["tiktoken"]` фиктивный модуль с `encoding_for_model`/`get_encoding`, которые
возвращают `_FakeEncoding()` (`encode()` считает не токены, а символы — этого достаточно для
тестов, которым важна не точность подсчёта, а сам факт, что подсчёт произошёл).

**Важно для окружения, где эта задача выполняется:** в `~/.venvs/ctb` `tiktoken` реально
установлен (`pip show tiktoken` → 0.14.0), поэтому во всех 10 файлах guard
`if importlib.util.find_spec(name) is None` сейчас **не срабатывает** — фиктивный модуль не
подставляется вообще, и `FakeEncoding` в этой среде не используется ни разу. Это не повод не
выносить класс: подмена нужна для сред без `tiktoken` (например, минимальный CI-образ без
сети), и вынос не меняет условие срабатывания — просто у всех 10 файлов сработает (или не
сработает) один и тот же класс вместо десяти копий.

#### 3.2. `FakeMessage` + `FakeChoice` — безопасный кластер из 4 файлов

Ровно в 4 файлах оба класса дословно совпадают:

```python
class FakeMessage:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls
        self.content = content


class FakeChoice:
    def __init__(self, tool_calls=None, content=""):
        self.message = FakeMessage(tool_calls=tool_calls, content=content)
        self.delta = None
        self.finish_reason = None
```

| Файл | `FakeMessage` | `FakeChoice` |
|---|---|---|
| `tests/test_openai_helper_tool_calls.py` | `:359-362` | `:365-369` |
| `tests/test_plugin_chat_id_contract.py` | `:39-42` | `:45-49` |
| `tests/test_reflection_on_tool_error.py` | `:48-51` | `:54-58` |
| `tests/test_skills_agent_gate.py` | `:401-404` | `:407-411` |

Оба класса эмулируют форму ответа OpenAI SDK (`response.choices[i].message.{tool_calls,
content}`), которую читает `bot/openai_tool_handler.py`. Форма стабильна и специфична для этих
4 файлов, значит вынос безопасен.

**Проверка обратных импортов (чтобы не сломать транзитивных потребителей):** в проекте уже
есть кросс-файловые импорты из `tests/test_openai_helper_tool_calls.py` —
`tests/test_pricing.py:6` и `tests/test_stream_usage.py:11` берут оттуда `DummyPluginManager`,
`_make_helper`, `FakeResponse`, `FakeToolCall`, `DummyClient`, `FakeStreamItem`,
`FakeClosableAsyncStream`; `tests/test_skills_prompt_fragment.py` берёт `DummyPluginManager`,
`_make_helper`. Ни один из них не импортирует `FakeMessage`/`FakeChoice`/`_FakeEncoding`
напрямую — значит перенос этих трёх имён в `tests/fakes.py` их не задевает.
`FakeResponse` в `test_openai_helper_tool_calls.py` продолжает работать: он ссылается на имя
`FakeChoice` в области видимости модуля, а после миграции это имя останется определено —
просто через `from tests.fakes import FakeChoice`, а не через локальный `class FakeChoice`.

#### 3.3. Почему `FakeToolCall` в тех же 4 файлах — НЕ дубликат (важно не унифицировать)

Внешне похожи (все хранят `self.function = SimpleNamespace(name=..., arguments=...)` и
опциональный `id`), но реальные контракты вызова несовместимы:

| Файл | Сигнатура `FakeToolCall.__init__` | Что делает с `arguments` | Как вызывается в тестах |
|---|---|---|---|
| `tests/test_openai_helper_tool_calls.py:353` | `(self, name, arguments, id=None)` | сохраняет как есть (без кодирования) | всегда передают **уже готовую JSON-строку**: `FakeToolCall("p.do", "{}")`, `FakeToolCall(..., json.dumps({...}))` |
| `tests/test_plugin_chat_id_contract.py:34` | `(self, name, arguments)` (без `id`!) | `arguments=json.dumps(arguments)` — кодирует сам | вызывается только изнутри локального `FakeResponse`: `FakeToolCall(tool_name, arguments or {})` — передают **словарь** |
| `tests/test_reflection_on_tool_error.py:42` | `(self, name, arguments, call_id=None)` — параметр называется `call_id`, не `id` | `arguments=json.dumps(arguments)` — кодирует сам | всегда передают **словарь**: `FakeToolCall("alpha.fail", {})`, `FakeToolCall(..., {"query": "same"}, call_id="call_1")` |
| `tests/test_skills_agent_gate.py:395` | `(self, name, arguments="{}", id=None)` | сохраняет как есть | вызывают часто **без `arguments` вообще**: `FakeToolCall("terminal.terminal")` — полагаются на дефолт `"{}"` |

Если подставить сюда одну общую реализацию, один из двух исходов гарантирован: словарь,
переданный в версию «без автокодирования», уедет в `SimpleNamespace(arguments=<dict>)` вместо
JSON-строки и сломает код, который делает `json.loads(tool_call.function.arguments)`; либо
JSON-строка, переданная в версию «с автокодированием», получит двойное экранирование
(`json.dumps("{}")` → `'"{}"'`). Плюс параметр `call_id` у `test_reflection_on_tool_error.py`
без изменений вызывающего кода не совпадёт с `id` у остальных. Вывод: `FakeToolCall` **не**
идёт в `tests/fakes.py`, несмотря на то что живёт бок о бок с уже вынесенными `FakeMessage`/
`FakeChoice` в тех же 4 файлах.

По той же причине не унифицируется `FakeResponse` в этом кластере: у
`test_openai_helper_tool_calls.py:372` и `test_skills_agent_gate.py:414` есть `.usage`
(`SimpleNamespace` с токенами), у `test_plugin_chat_id_contract.py:52` и
`test_reflection_on_tool_error.py:61` атрибута `.usage` нет вовсе — код, который читает
`response.usage.total_tokens`, упал бы с `AttributeError` на двух из четырёх, если бы им
подставили «богатую» версию, и наоборот выглядел бы неполным, если бы всем раздали «бедную».

#### 3.4. Почему `FakeDB`, `FakeHelper`, `FakePluginManager` не идут в `tests/fakes.py`

**`FakeDB` (13 копий).** 9 из 13 — классы, объявленные *внутри тела теста* в
`tests/test_plugin_manager.py` (`:652, :698, :715, :731, :753, :766, :796, :847, :888, :926` —
10 мест, некоторые с одинаковым `record_tool_call_event`, но большинство с разными,
специально подобранными под конкретный сценарий значениями `get_user_settings`, например
`:698` возвращает `{"disabled_plugins": ["weather", "time"]}`, а `:715` — `{"disabled_plugins":
["weather"]}`, `:731` — список с дублями и пробелами специально для проверки нормализации).
Собирать их в одну заглушку означало бы либо потерять эту вариативность, либо сделать
универсальный конструктор параметрами — то есть новую абстракцию ради тестов, которые и так
работают, что прямо запрещено правилом «не расширять поведение заглушек».
Единственная настоящая пара-дубликат — `tests/test_plugin_chat_id_contract.py:109` и
`tests/test_reflection_on_tool_error.py:88` (`list_user_sessions`/`_async`, возвращают `[]`),
но это 2 копии, не 3 — ниже порога задачи «≥3 почти одинаковых копии»; помечено как
кандидат на будущее, если появится третья.

**`FakeHelper` (12 копий).** Каждая — минимальный дублёр `OpenAIHelper`, реализующий ровно
тот один-два метода, которые вызывает код под тестом (`ask()` в
`tests/test_movie_info_plugin.py:14` и `tests/test_chief_model_choice.py:15` — разные сигнатуры
`ask()`; `resolve_allowed_plugins()` в `tests/test_agent_tools_plan_rule_mutator.py:17`,
`tests/test_agent_tools_replan.py:57`, `tests/test_agent_tools_verify.py:56`; `chat_completion()`
в `tests/test_llm_gateway_routing.py:36`, `tests/test_model_utilities.py:22`; полноценный набор
из 7 методов в `tests/test_plugin_chat_id_contract.py:125` и
`tests/test_reflection_on_tool_error.py:117`). Единственная точно совпадающая пара —
`test_agent_tools_replan.py:57` / `test_agent_tools_verify.py:56` (байт-в-байт), опять 2 копии,
не 3. `test_plugin_chat_id_contract.py:125` и `test_reflection_on_tool_error.py:117` почти
совпадают, но не дословно: у первого есть метод `_localized_text`, которого нет у второго —
не тождественные заглушки, объединение изменило бы набор методов у одной из них.

**`FakePluginManager` (8 копий).** Полный разброс: от двухметодного
(`tests/test_plugin_menu_force_reply.py:73` — только `is_plugin_disabled_for_user`/
`disabled_plugins_for_user`, обе с захардкоженным ответом) до многометодного
(`tests/test_telegram_streaming.py:97` — `set_db`, `user_settings_scope`, счётчики вызовов) —
ни одного метода, который был бы нужен в одинаковой реализации ≥3 файлам одновременно, не
нашлось.

**`_make_bot` (8 копий, дополнительная находка, не входила в исходный список задачи).** Тот же
паттерн: `object.__new__(ChatGPTTelegramBot)` + вручную выставленный набор атрибутов, который в
каждом файле свой — потому что настоящий `ChatGPTTelegramBot.__init__` не вызывается (слишком
тяжёлый), и каждый тест выставляет только то подмножество атрибутов, которое трогает его код.

### 4. `agent_db` — фикстура-дубликат (5 копий, безопасно)

```python
@pytest.fixture()
def agent_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "agent.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        for stmt in AgentToolsPlugin().register_schema():
            conn.execute(stmt)
    yield db
    Database._reset_singleton()
```

| Файл | Строки | Отличие от эталона |
|---|---|---|
| `tests/test_agent_tools_session_reset.py:30-39` | эталон | — |
| `tests/test_agent_tools_plugin.py:375-385` | — | одна лишняя строка-комментарий `# Stage 3: agent_plan_* DDLs live in the plugin now.` |
| `tests/test_agent_tools_plan_rule_mutator.py:35-44` | идентично | — |
| `tests/test_agent_tools_verify.py:44-53` | идентично | — |
| `tests/test_agent_tools_replan.py:45-54` | идентично | — |

Все 5 файлов уже импортируют `from bot.database import Database` и
`from bot.plugins.agent_tools import AgentToolsPlugin` (нужны им и для остального тела теста,
не только для фикстуры) — при удалении локальной фикстуры эти импорты, скорее всего, останутся
использоваться в других местах файла; проверять индивидуально при миграции (см. §«Риски»).

Это единственная в проекте фикстура-дубликат с ≥3 практически идентичными копиями — то, что
задача просит вынести «в `tests/conftest.py`».

Также найдена `_reset_to_thread_calls` (3 копии: `tests/test_movie_info_plugin.py:56`,
`tests/test_webshot_plugin.py:38`, `tests/test_show_me_diagrams_plantuml.py:25`, тело везде
`to_thread_calls.clear(); yield; to_thread_calls.clear()`), но `to_thread_calls` — список,
объявленный на уровне модуля в каждом из этих трёх файлов отдельно; общая фикстура в
`conftest.py` не сможет сослаться на имя, которого в её собственной области видимости нет —
потребовался бы отдельный механизм (реестр, параметризация), то есть новая абстракция ради
трёх строк. Не выносится.

### 5. `test_concurrent_access_smoke` — почему без `assert` тест не проверяет ничего

`tests/test_database.py:712-720`, текущий код:

```python
def test_concurrent_access_smoke(db):
    def worker(idx):
        db.save_user_settings(idx, {"x": idx})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
```

Причина, по которой это не тест, а имитация теста: исключение, брошенное **внутри**
`threading.Thread.run()`, не пробрасывается в поток, который вызвал `.join()` — Python по
умолчанию просто печатает traceback через `threading.excepthook` и поток тихо завершается.
Значит, даже если `db.save_user_settings` кидает исключение на каждом из 10 вызовов, тест
всё равно долетит до конца функции без единого `assert` и pytest зачтёт его как `PASSED`.
Единственное, что тест сейчас гарантирует — что процесс не зависает навсегда (потому что
`.join()` без таймаута рано или поздно вернётся, если только не случится настоящий deadlock).

В этом же файле есть готовый локальный паттерн для перехвата исключений из фонового потока —
`tests/test_database.py:673-706`
(`test_get_connection_context_manager_recovers_after_failed_commit`, метод `write_after_failure`
использует `worker_errors = []` + `except BaseException as exc: worker_errors.append(exc)`,
затем `assert worker_errors == []`). Предлагаемое исправление копирует этот же приём (то есть
не вводит новый стиль в файл), плюс проверяет, что все 10 записей реально попали в базу:

```python
def test_concurrent_access_smoke(db):
    worker_errors = []

    def worker(idx):
        try:
            db.save_user_settings(idx, {"x": idx})
        except BaseException as exc:
            worker_errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert worker_errors == []
    for idx in range(10):
        assert db.get_user_settings(idx) == {"x": idx}
```

Сигнатуры подтверждены: `Database.save_user_settings(self, user_id, settings)`
(`bot/database.py:797`) пишет `json.dumps(settings)` через `INSERT … ON CONFLICT DO UPDATE`;
`Database.get_user_settings(self, user_id)` (`bot/database.py:815`) читает и делает
`json.loads`, возвращая тот же словарь — значит `{"x": idx}` сравнивается корректно.
Таймаут на `.join()` сознательно не добавляется — это расширение поведения теста (сейчас при
дедлоке тест зависает, что тоже сигнал о проблеме, только через таймаут pytest/CI, а не через
`assert`); можно предложить отдельным пунктом, не обязательным для T20.

### 6. Текущее состояние `conftest.py`

`tests/conftest.py` (21 строка, целиком):

```python
import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _close_pytest_asyncio_baseline_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    if not loop.is_closed():
        loop.close()
    asyncio.set_event_loop(None)
```

Единственная существующая фикстура — служебная, `autouse`, не про заглушки. `bot/tests/conftest.py`
(6 строк) — только вставка корня в `sys.path`, фикстур нет. Ни `tests/__init__.py`, ни
`bot/tests/__init__.py` не существует — обе папки резолвятся как **namespace-пакеты** (PEP 420):
подтверждено запуском `python -c "import tests; print(tests.__spec__...)"` из корня после
вставки `ROOT` в `sys.path`, ровно как это делает `tests/conftest.py` при сборе тестов.

**Импорт `from tests.fakes import ...` уже используется в проекте** — не нужно ничего менять
в `pytest.ini` (там нет и не станет `pythonpath =`), потому что `tests/conftest.py` уже вставляет
корень репозитория в `sys.path` до сбора любого теста в этой папке. Прецедент:
`tests/test_stream_usage.py:11` и `tests/test_pricing.py:6` делают
`from tests.test_openai_helper_tool_calls import (...)`, `tests/test_skills_prompt_fragment.py`
— то же самое. Значит `from tests.fakes import FakeEncoding, FakeMessage, FakeChoice` будет
работать так же, без дополнительной настройки.

## Дизайн `tests/fakes.py`

```python
"""Общие тестовые заглушки, доказанно дублирующиеся в ≥3 файлах.

Смотри docs/remediation_2026-09-04/T20-test-fakes.md — там разбор, почему
именно эти классы, а не «все Fake*/Dummy*/Stub*». Одинаковое имя класса в
разных тестовых файлах в этом проекте почти всегда означает разные, специально
подогнанные под конкретный тест заглушки — не копируй сюда что-то новое, пока
не найдёшь минимум 3 файла с побайтово (или почти побайтово) одинаковым телом.
"""


class FakeEncoding:
    """Заменитель tiktoken.Encoding: encode() считает "токеном" каждый символ.

    Подключается через sys.modules["tiktoken"] только если настоящий пакет не
    установлен (см. _install_module_if_missing в каждом файле, где это
    используется). В .venv этого репозитория tiktoken установлен, поэтому
    здесь класс сейчас не активируется ни в одном тесте — он нужен для сред
    без tiktoken (например, CI-образ без сети).
    """

    def encode(self, value):
        return list(value)


class FakeMessage:
    """Заменитель response.choices[i].message из OpenAI SDK: только
    tool_calls и content — ровно то, что читает bot/openai_tool_handler.py."""

    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls
        self.content = content


class FakeChoice:
    """Заменитель response.choices[i], оборачивает FakeMessage."""

    def __init__(self, tool_calls=None, content=""):
        self.message = FakeMessage(tool_calls=tool_calls, content=content)
        self.delta = None
        self.finish_reason = None
```

Сознательно **не включены**: `FakeDB`, `FakeHelper`, `FakePluginManager`, `FakeToolCall`,
`FakeResponse`, `_make_bot`, `_make_helper` — обоснование по каждому в §3.3–3.4. Если в будущем
у `FakeDB`/`FakeHelper` (пары из §3.4) появится третья байт-в-байт идентичная копия — тогда их
можно будет добавить сюда по тому же принципу.

## Дизайн фикстуры в `tests/conftest.py`

Добавить импорты и одну не-autouse фикстуру:

```python
from bot.database import Database
from bot.plugins.agent_tools import AgentToolsPlugin


@pytest.fixture()
def agent_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "agent.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        for stmt in AgentToolsPlugin().register_schema():
            conn.execute(stmt)
    yield db
    Database._reset_singleton()
```

Фикстура не `autouse` (по правилу задачи) — доступна только тем тестам, которые явно берут
`agent_db` как параметр, ровно как сегодня 5 локальных копий. Импорт `bot.database`/
`bot.plugins.agent_tools` на верхнем уровне `conftest.py` безопасен: `tests/conftest.py` уже
вставляет корень репозитория в `sys.path` до этого места (строки 7-9 текущего файла), и
`bot.database`/`bot.plugins.agent_tools` не тянут за собой ничего специфичного для конкретного
теста (в отличие, например, от `bot.openai_helper`, который на импорте не должен требовать
`tiktoken`-заглушку — но `Database`/`AgentToolsPlugin` от `tiktoken` не зависят, проверено
чтением обоих модулей: ни один не импортирует `tiktoken`).

## Правки по `file:line` (план миграции, не выполнено)

### A. `FakeEncoding` — 10 файлов, механическая замена

Для каждого файла: удалить класс `_FakeEncoding` (диапазон строк ниже), добавить
`from tests.fakes import FakeEncoding` в блок импортов **после** `bot.*`-импортов (там же, где
в файле уже стоят импорты с `# noqa: E402` — модульный код подмены `sys.modules` выполняется до
них, поэтому первые «настоящие» импорты и так помечены `# noqa: E402`; новый импорт стоит той
же пометкой), заменить `_FakeEncoding()` → `FakeEncoding()` в двух строках лямбд.

| Файл | Класс (удалить) | Лямбды (заменить `_FakeEncoding()`→`FakeEncoding()`) |
|---|---|---|
| `tests/test_openai_helper_tool_calls.py` | `:22-24` | `:28-29` |
| `tests/test_skills_agent_gate.py` | `:30-32` | `:36-37` |
| `tests/test_telegram_streaming.py` | `:27-29` | `:33-34` |
| `tests/test_telegram_builder_config.py` | `:21-23` | `:27-28` |
| `tests/test_per_conversation_serialization.py` | `:21-23` | `:27-28` |
| `tests/test_group_session_flow.py` | `:21-23` | `:27-28` |
| `tests/test_plugin_handlers_registration.py` | `:20-22` | `:26-27` |
| `tests/test_plugin_menu_force_reply.py` | `:20-22` | `:26-27` |
| `tests/test_callback_authorization.py` | `:21-23` | `:27-28` |
| `tests/test_telegram_transcribe.py` | `:22-24` | `:28-29` |

### B. `FakeMessage` + `FakeChoice` — 4 файла кластера

Удалить оба локальных класса, добавить `from tests.fakes import FakeMessage, FakeChoice`.
Локальные `FakeToolCall`/`FakeResponse`/`FakeDB`/`FakeHelper` в этих же файлах **не трогать**
(см. §3.3).

| Файл | `FakeMessage` (удалить) | `FakeChoice` (удалить) |
|---|---|---|
| `tests/test_openai_helper_tool_calls.py` | `:359-362` | `:365-369` |
| `tests/test_plugin_chat_id_contract.py` | `:39-42` | `:45-49` |
| `tests/test_reflection_on_tool_error.py` | `:48-51` | `:54-58` |
| `tests/test_skills_agent_gate.py` | `:401-404` | `:407-411` |

Для `tests/test_openai_helper_tool_calls.py` и `tests/test_skills_agent_gate.py` пункты A и B
объединяются в один PR-диф на файл (оба класса убираются, один импорт добавляется).

### C. `agent_db` — 5 файлов

Удалить локальную фикстуру, ничего не импортировать взамен (pytest найдёт `agent_db` в
`tests/conftest.py` по имени параметра автоматически). После удаления проверить, остаются ли
`from bot.database import Database` и `from bot.plugins.agent_tools import AgentToolsPlugin`
нужны файлу для чего-то ещё (в реальном коде тестов эти классы почти наверняка
используются и для прямых вызовов, не только внутри фикстуры, — смотреть индивидуально, не
удалять импорт вслепую).

| Файл | Фикстура (удалить) |
|---|---|
| `tests/test_agent_tools_session_reset.py` | `:30-39` |
| `tests/test_agent_tools_plugin.py` | `:375-385` |
| `tests/test_agent_tools_plan_rule_mutator.py` | `:35-44` |
| `tests/test_agent_tools_verify.py` | `:44-53` |
| `tests/test_agent_tools_replan.py` | `:45-54` |

### D. `test_concurrent_access_smoke` — 1 файл

`tests/test_database.py:712-720` — заменить тело на вариант из §5 (добавить перехват
исключений и 11 `assert`).

## Тесты / критерии готовности

- Тесты для самого `tests/fakes.py` не нужны (по заданию) — сами вынесенные классы уже
  покрыты всеми тестами файлов, которые их используют; поведение не меняется, значит
  регрессия проявится как падение существующих тестов, а не как отсутствие новых.
- После правок A–D: полный прогон обязан остаться зелёным, и число тестов не должно
  уменьшиться (см. базовые числа ниже — 1530).
- `ruff check bot tests bot/tests` обязан остаться чистым (сейчас чист — см. ниже).
- Точечно после правок A/B — файлы из таблиц A и B по отдельности и все вместе:
  `tests/test_openai_helper_tool_calls.py`, `tests/test_skills_agent_gate.py`,
  `tests/test_plugin_chat_id_contract.py`, `tests/test_reflection_on_tool_error.py`,
  `tests/test_telegram_streaming.py`, `tests/test_telegram_builder_config.py`,
  `tests/test_per_conversation_serialization.py`, `tests/test_group_session_flow.py`,
  `tests/test_plugin_handlers_registration.py`, `tests/test_plugin_menu_force_reply.py`,
  `tests/test_callback_authorization.py`, `tests/test_telegram_transcribe.py`.
- Точечно после правки C: `tests/test_agent_tools_session_reset.py`,
  `tests/test_agent_tools_plugin.py`, `tests/test_agent_tools_plan_rule_mutator.py`,
  `tests/test_agent_tools_verify.py`, `tests/test_agent_tools_replan.py` — плюс любой другой
  файл, использующий имя `agent_db` как параметр (проверить, что не появилась коллизия имени
  фикстуры с ещё каким-то локальным `agent_db`, кроме этих пяти, — на дату написания плана
  других нет).
- Точечно после правки D: `tests/test_database.py -k test_concurrent_access_smoke`, затем
  весь `tests/test_database.py`.
- Убедиться, что кросс-файловые импорты не сломались:
  `tests/test_pricing.py`, `tests/test_stream_usage.py`, `tests/test_skills_prompt_fragment.py`.

## Команды проверки

Выполнять из корня репозитория `/srv/git_projects/chatgpt-telegram-bot`, интерпретатор
`~/.venvs/ctb/bin/python`.

```bash
# Базовый прогон до правок (зафиксировано в этом плане, 2026-09-04, HEAD af382fb):
~/.venvs/ctb/bin/python -m pytest -q -p no:cacheprovider tests bot/tests
# -> 1530 passed, 3 warnings (PTBDeprecationWarning, не связано с этой задачей)

~/.venvs/ctb/bin/python -m ruff check bot tests bot/tests
# -> All checks passed!

# После правок — та же команда, число passed не меньше 1530, 0 failed.

# Точечно по кластеру FakeMessage/FakeChoice:
~/.venvs/ctb/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_openai_helper_tool_calls.py tests/test_plugin_chat_id_contract.py \
  tests/test_reflection_on_tool_error.py tests/test_skills_agent_gate.py \
  tests/test_pricing.py tests/test_stream_usage.py tests/test_skills_prompt_fragment.py

# Точечно по agent_db:
~/.venvs/ctb/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_agent_tools_session_reset.py tests/test_agent_tools_plugin.py \
  tests/test_agent_tools_plan_rule_mutator.py tests/test_agent_tools_verify.py \
  tests/test_agent_tools_replan.py

# Точечно по FakeEncoding (10 файлов из таблицы A):
~/.venvs/ctb/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_telegram_streaming.py tests/test_telegram_builder_config.py \
  tests/test_per_conversation_serialization.py tests/test_group_session_flow.py \
  tests/test_plugin_handlers_registration.py tests/test_plugin_menu_force_reply.py \
  tests/test_callback_authorization.py tests/test_telegram_transcribe.py

# test_concurrent_access_smoke отдельно:
~/.venvs/ctb/bin/python -m pytest -q -p no:cacheprovider \
  "tests/test_database.py::test_concurrent_access_smoke"
```

## Риски

- **Не перепутать «то же имя» с «тот же код».** Основной риск этой задачи — соблазн
  унифицировать по названию класса, а не по факту дублирования. В §3.3–3.4 показано минимум
  два места (`FakeToolCall`, `FakeResponse` в общем с `FakeMessage`/`FakeChoice` кластере), где
  это привело бы к скрытой порче теста (двойное JSON-кодирование, `AttributeError` на `.usage`).
  Следующий агент должен перед каждой миграцией сверять тело класса и вызовы на месте, а не
  ориентироваться на этот документ как на готовую истину без повторной проверки — как и было
  сказано в самой задаче про верификацию прошлых утверждений.
- **`_install_module_if_missing` (15 идентичных копий) — не входит в исходный список задачи.**
  Технически подходит под правило «≥3 идентичных копий», и его можно было бы тоже вынести в
  `tests/fakes.py`. Он не включён в дизайн выше, чтобы не расширять объём задачи сверх
  названных пяти имён без явного запроса; следующий агент/владелец задачи может решить вынести
  его отдельным маленьким PR при желании — поведение при этом не меняется (функция и так везде
  одинаковая).
- **Порядок кода вокруг `_FakeEncoding`/`_install_module_if_missing` — не фикстура, а
  модуль-код, выполняющийся до импорта `bot.*`.** Его нельзя механически перенести в
  `tests/conftest.py` целиком (не только сам класс, а весь «подставить фейковый tiktoken до
  импорта bot.openai_helper, затем убрать из sys.modules» танец): `conftest.py` грузится один
  раз на сессию для всей папки, а разным файлам нужно снимать подмену в разные моменты (после
  разных `bot.*`-импортов) — общий conftest не знает, когда именно каждый файл закончил
  импортировать нужные ему модули bot. Поэтому в этот план идёт вынос только самого класса
  `FakeEncoding`, а обвязка (`_install_module_if_missing`, `_tiktoken = types.ModuleType(...)`,
  `for _module_name in _INSERTED_MODULES: sys.modules.pop(...)`) остаётся в каждом файле как
  есть.
- **Удаление импортов `Database`/`AgentToolsPlugin` при миграции `agent_db` (правка C) — делать
  только после проверки, что файл их не использует больше нигде.** На дату написания плана не
  проверялось построчно для всех 5 файлов — это тривиальная, но обязательная проверка перед
  удалением строки импорта (иначе `ruff` поймает неиспользуемый импорт, что и так входит в
  критерий готовности).
- **Конфликт имён фикстуры `agent_db`.** Если после миграции у какого-то шестого файла
  появится собственный `agent_db` с другим контрактом (например, без DDL `AgentToolsPlugin`),
  он будет молча иметь приоритет над версией из `conftest.py` (pytest даёт локальному
  переопределению приоритет) — не поломка, но источник путаницы; стоит грепать по имени
  `agent_db` при следующих правках, чтобы не плодить вторую, другую версию под тем же именем.
- **`_make_helper`/`make_helper`-семейство (13+1 функций) сознательно оставлено вне этого
  плана.** Частичная проверка (`tests/test_openai_helper_tool_calls.py:441` и
  `tests/test_hindsight_memory.py:159`) показала, что 2 из них строят настоящий `OpenAIHelper`
  из ~39-ключевого конфига, совпадающего процентов на 90 (разные `enable_functions`,
  `vision_model`, плюс лишний ключ `hindsight_auto_save` у одного), а
  `tests/test_summarise_overflow_dispatch.py:32` вместо этого использует
  `object.__new__(OpenAIHelper)` с 4 вручную выставленными атрибутами — совершенно другой
  паттерн под тем же префиксом имени. Остальные ~10 функций семейства не проверялись построчно
  в рамках этой задачи (это увеличило бы объём исследования за пределы «дубли для 5 названных
  заглушек» из T20) — если браться за них, это отдельная задача с отдельной инвентаризацией по
  той же методологии §1, а не хвост к T20.
- **Тест-провайдер `db`-фикстур не унифицируется.** `tests/test_database.py:26` и
  `tests/test_db_handle.py:10` называются одинаково (`db`), но одна не делает `yield`/сброс
  после теста, другая делает — то есть у них разное поведение очистки, а не только разный код;
  объединение изменило бы порядок сброса `Database`-синглтона для одного из двух файлов.

---

## Постскриптум после реализации и ревью

### Что сделано (разработчик, Sonnet)

- **A. `FakeEncoding`.** Создан `tests/fakes.py`; локальные `_FakeEncoding` удалены и заменены
  импортом из `tests.fakes` в 10 файлах: `test_openai_helper_tool_calls.py`,
  `test_skills_agent_gate.py`, `test_telegram_streaming.py`, `test_telegram_builder_config.py`,
  `test_per_conversation_serialization.py`, `test_group_session_flow.py`,
  `test_plugin_handlers_registration.py`, `test_plugin_menu_force_reply.py`,
  `test_callback_authorization.py`, `test_telegram_transcribe.py`.
- **B. Кластер `FakeMessage` + `FakeChoice`.** Вынесен в `tests/fakes.py`, локальные копии
  удалены в 4 файлах (`test_openai_helper_tool_calls.py`, `test_plugin_chat_id_contract.py`,
  `test_reflection_on_tool_error.py`, `test_skills_agent_gate.py`). `FakeToolCall`/`FakeResponse`
  в тех же файлах не тронуты — план верно объяснил, что их контракты различаются.
- **C. Фикстура `agent_db`** вынесена в `tests/conftest.py` (не autouse), локальные копии удалены
  в 5 файлах `test_agent_tools_*.py`.
- **D. `test_concurrent_access_smoke`** (`tests/test_database.py`) больше не пустой: добавлены
  `assert worker_errors == []` (иначе исключения из потоков молча съедаются `excepthook`,
  а `.join()` их не пробрасывает) и проверка, что все 10 конкурентных записей реально попали
  в БД.
- `FakeDB`/`FakeHelper`/`FakePluginManager` **не выносились** — пункт D плана подтверждён.

### Расхождения план ↔ факт (исправлено по коду, а не по документу)

1. Номера строк в таблицах плана местами устарели (`_FakeEncoding` в
   `test_telegram_streaming.py` был на строке 40, а не 27; в
   `test_per_conversation_serialization.py` — на 24, а не 21). Позиции перепроверены `ast`-ом.
2. План предлагал импортировать `FakeMessage, FakeChoice` во всех 4 файлах кластера, но
   `FakeMessage` нигде не упоминается напрямую (создаётся внутри `FakeChoice.__init__`) —
   лишний импорт падал бы на `ruff` F401. Импортируется только `FakeChoice`.
   Ревьюер подтвердил `ast`-подсчётом: 0 узлов `Name(id="FakeMessage")` во всех 4 файлах.
3. План советовал «скорее всего, импорты `Database`/`AgentToolsPlugin` останутся нужны». По
   факту после удаления фикстуры `Database` стал мёртвым во всех 5 файлах, а `pytest` — в
   `test_agent_tools_plan_rule_mutator.py` (там `@pytest.fixture` был единственным
   применением). Ревьюер перепроверил `ast`-подсчётом узлов `Name` и подтвердил: `Database` —
   0 обращений в пяти файлах, `pytest` — 0 только в том одном файле, в остальных четырёх
   5–64 обращения, и там импорт корректно оставлен.
4. Названный в плане образец `test_get_connection_context_manager_recovers_after_failed_commit`
   в файле отсутствует; паттерн `worker_errors=[]` / `except BaseException` взят из реально
   существующего соседнего теста.

### Ревью (Sonnet, persona reviewer, read-only)

**Вердикт: `## Ошибки` — нет.** Ревьюер сверил удалённые тела классов с общими через
`git diff` (побайтовое совпадение, включая `content=""`, `delta=None`, `finish_reason=None` —
не `content=None`), проверил, что похожие по имени `FakeMessage`/`FakeChoice` в
`test_concurrent_tool_state.py` и `test_openai_compatible_provider.py` намеренно и правильно
не тронуты (другие контракты), а восемь оставшихся `FakeMessage` — это заглушки Telegram
`Message` совсем другого домена. Отдельно проверил, что фикстура `agent_db` — единственное
определение в дереве и не autouse.

Проверка «assert-ы не косметические»: ревьюер скопировал `tests/test_database.py` в `/tmp`,
сделал так, чтобы одна из десяти записей тихо не сохранялась (без исключения) — тест покраснел
(`assert None == {'x': 5}`). Прогнал `test_concurrent_access_smoke` 15 раз подряд — стабильно
зелёный, флаки не обнаружены.

### Предупреждения ревьюера — разбор

1. **Расхождение план/факт по `FakeMessage`** — см. пункт 2 выше, действий не требует.
2. **`worker_errors.append` из 10 потоков без блокировки** безопасен только благодаря GIL.
   Ревьюер сам отметил, что это ровно тот стиль, который уже принят в соседнем тесте того же
   файла, и вопрос не блокирующий. Оставлено как есть — иначе один тест стал бы отличаться от
   соседнего по стилю без выигрыша в надёжности.
3. **Хрупкое место с порядком импортов.** `from tests.fakes import FakeEncoding` стоит после
   лямбд `_tiktoken.encoding_for_model = lambda _model: FakeEncoding()`. Это работает только
   потому, что `tiktoken.encoding_for_model` вызывается внутри функции
   (`bot/openai_helper.py:3796`), а не на уровне модуля. Если когда-нибудь вызов переедет на
   уровень модуля, получится `NameError: FakeEncoding is not defined`. Сейчас не баг; кроме
   того, в этом окружении `tiktoken` установлен (0.14.0), поэтому guard
   `if importlib.util.find_spec(name) is None` не срабатывает и подмена вообще не активируется —
   `FakeEncoding` нужен только для сред без `tiktoken` (например, CI-образ без сети). Это же
   записано в докстринге `tests/fakes.py`.

### Проверка

- `pytest -q tests bot/tests` — **1636 passed**, 0 failed, 3 warnings (`PTBDeprecationWarning`,
  к задаче не относятся). Число тестов не изменилось: T20 — рефакторинг дублей плюс
  ассерты в уже существующий тест, новых тестовых функций не добавлялось.
- `ruff check bot tests bot/tests` — All checks passed.
- Ревьюер дополнительно прогнал все 18 затронутых файлов **по отдельности** (проверка на
  зависимость от порядка сборки, из-за подмены `sys.modules`) — все зелёные и в изоляции, и в
  общем прогоне; кросс-файловые импортёры (`test_pricing.py`, `test_stream_usage.py`,
  `test_skills_prompt_fragment.py`) не сломаны.
