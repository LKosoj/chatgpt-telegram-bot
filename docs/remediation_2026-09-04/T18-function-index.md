# T18. Индекс «имя функции → (плагин, spec)»

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md`, раздел «Волна 4» (P5 идёт
параллельно с T15/П1, друг от друга не зависят); находка —
`docs/architecture_code_review_2026-09-04.md` §5.1 (буллет «Резолв «имя функции → плагин» без
индекса…») и §5.3 (П5). Роль этого документа — план для разработчика; код не менялся, только
прочитан.

Владелец задачи явно решил (см. `docs/audit_remediation_plan_2026-09-04.md:9`): `_guard_tool_call`
**не удаляется** и не переносится в базовый `Plugin`. П5 в обзоре одним пунктом упоминала и
индекс, и `_guard_tool_call` — T18 в плане исправлений сузил задачу до индекса. Этот план
`_guard_tool_call` не трогает.

Термины:
- **линейный перебор (linear scan)** — цикл, который проверяет все N элементов по очереди;
  время растёт пропорционально N. Индекс (словарь `dict`) вместо этого находит элемент по
  ключу за примерно постоянное время, независимо от N.
- **spec (спецификация функции)** — JSON-описание одного инструмента (имя, описание, параметры)
  в формате, который понимает модель. Каждый плагин отдаёт список таких описаний через
  `get_spec()`.
- **каноническое имя функции** — имя вида `<plugin>.<function>` (с точкой), как его видит код
  бота. **model-safe имя** — то же имя, но без точки (модели иногда не принимают спецсимволы в
  именах инструментов), например `alpha.do` → `alpha_do`. Взаимное преобразование уже есть:
  `to_model_function_name`/`to_canonical_function_name` (`bot/plugin_manager.py:368-403`).
- **инвалидация (invalidation)** — явный сброс кэша/индекса, когда исходные данные изменились и
  старое значение больше не годится.

## Цель

Убрать O(N) линейный перебор всех плагинов (и вызов дорогого `get_spec()` каждого из них) на
каждый вызов `get_spec_by_function_name`, `get_plugin_name_by_function_name` и
`is_function_allowed` — эти методы вызываются на каждый tool-call модели (`call_function`,
`bot/plugin_manager.py:405-507`) и при построении списка разрешённых функций для суб-агентов.
Заменить на словарь `_function_index: dict[str, tuple[plugin_name, spec]]`, который строится один
раз и переиспользуется, с явной инвалидацией (`invalidate_function_index()`) и ленивой
пересборкой при промахе — чтобы MCP-плагин, который добавляет инструменты уже после запуска
(регистрация сервера админом, фоновое обновление списка тулов), не оставался с устаревшим
индексом.

Код не меняется — правки по этому плану выполняет следующий агент (разработчик).

## Анализ

### 1. Кто и как сегодня резолвит «имя функции → плагин/spec»

| Метод | `file:line` | Что делает | Стоимость |
|---|---|---|---|
| `get_spec_by_function_name` | `bot/plugin_manager.py:587-597` | Находит плагин через `__get_plugin_by_function_name` (см. ниже), затем **ещё раз** вызывает `plugin.get_spec()` и линейно ищет нужный spec в его списке | 2 прохода: полный скан всех плагинов ради имени плагина + повторный `get_spec()` найденного плагина |
| `get_plugin_name_by_function_name` | `bot/plugin_manager.py:599-624` | Линейно перебирает `self.plugins.keys()`, для каждого — `get_plugin()` (лениво инстанцирует) + `plugin.get_spec()`, ищет совпадение по имени (см. «Сопоставление имён» ниже) | O(N) вызовов `get_spec()`, N = число загруженных плагинов (сейчас 39, см. §2) |
| `is_function_allowed` | `bot/plugin_manager.py:626-633` | `if allowed_plugins == ['All']: return True` (без обращения к индексу); иначе вызывает `get_plugin_name_by_function_name` | Наследует стоимость предыдущего пункта, кроме случая `['All']` |
| `__get_plugin_by_function_name` (приватный, name-mangled) | `bot/plugin_manager.py:674-681` | `get_plugin_name_by_function_name` + `get_plugin(plugin_name)` | То же самое + инстанцирование |
| `call_function` | `bot/plugin_manager.py:405-507` | Вызывает `__get_plugin_by_function_name` (:415) и `get_spec_by_function_name` (:448) — то есть **на каждый вызов одного тула сегодня выполняется 3 независимых линейных скана** всех плагинов | 3×O(N) на один tool-call |
| `get_plugin_source_name` | `bot/plugin_manager.py:578-585` | `__get_plugin_by_function_name` | O(N), не в горячем пути tool-call, не трогается |
| `is_subagent_function_allowed` | `bot/plugin_manager.py:663-672` | Вызывает `is_function_allowed` | Наследует ускорение автоматически, отдельно не трогается |

`reload_plugins` и `_plugin_for_function`, упомянутые в тексте задачи как гипотетические имена,
в коде **не существуют** (проверено `grep`/AST-обходом всего `bot/plugin_manager.py`). Реальный
эквивалент «перезагрузки» — `reinitialize()` (`bot/plugin_manager.py:298-303`): чистит
`self.plugins`/`self.plugin_instances` и вызывает `load_plugins()` заново.

Другие места, перебирающие `self.plugins.keys()`/`.items()` (`bot/plugin_manager.py:322` —
`get_functions_specs`, `:510` — `_guard_tool_call`, `:826,872,951,964,977,1003,1172,1258` —
списки команд/хуков/фоновых задач), в задачу не входят: это не резолв «одно имя функции → один
плагин», а сознательный проход по всем плагинам ради их описания/хуков/задач — индекс по имени
функции им не поможет и не нужен (`get_functions_specs` намеренно сканирует только
`allowed_plugins`, чтобы НЕ считать `get_spec()` отключённых модой плагинов — см. §3).

### 2. Стоимость `get_spec()` по плагинам

`get_spec()` не делает сеть/диск синхронно ни у одного плагина (проверено чтением каждого файла
из списка ниже) — «дорого» означает объём Python-литерала, который метод пересобирает с нуля на
каждый вызов (аллокация вложенных `dict`/`list`), а не I/O:

| Плагин | `get_spec()` | Строк |
|---|---|---|
| `agent_tools.py` | `:481-884` | 403 |
| `skills.py` | `:375-752` | 377 (см. ниже — не растёт от числа установленных skills) |
| `mcp_server.py` | `:201-320` | 119 (из них 3 статичных описания + цикл по `self.servers` — растёт с числом MCP-серверов) |
| `text_document_qa.py` | `:144-265` | 121 |
| `task_management.py` | `:27-109` | 82 |
| `spotify.py` | `:34-113` | 79 |
| `hindsight_memory.py` | `:2635-2710` | 75 |
| `reminders.py` | `:25-99` | 74 |
| остальные 32 плагина | — | 2–68 строк каждый |

Всего по дереву 39 плагинов (см. полный список — `python3 -c` обход
`bot/plugins/*.py` кроме `NON_PLUGIN_MODULES`). Сумма строк всех `get_spec()` — around 1700+;
именно это пересобирается **трижды на каждый tool-call** сегодня (см. таблицу §1).

Уточнение по двум плагинам, которые в тексте задачи предположительно называются «дорогими»:
- **`skills.py`** (`AGENTS.md`, раздел «Tool And Context Footprint», пункт 2): `get_spec()`
  возвращает **фиксированный** набор из 6 тул-спеков (`list_skills`, `get_skill`,
  `get_skill_reference`, `get_skill_resource`, `find_installable_skills`, `install_skill`),
  не зависящий от числа установленных skills — сам список skills отдаётся моделью через вызов
  `list_skills`, не через `get_spec()`. Диск не читается внутри `get_spec()`.
- **`mcp_server.py`**: `get_spec()` синхронно читает **только** `self.servers` — словарь в
  памяти, загруженный из `mcp_servers.json` при `initialize()` (`:82-88`, `load_servers_config`).
  Сети внутри `get_spec()` нет. Но если у сервера ещё нет закэшированных `tools` (`server_config
  ["tools"]` пусто), `get_spec()` вызывает `_schedule_tools_refresh(server_name)` (`:321-345`) —
  это **побочный эффект**: если есть работающий event loop, планируется фоновая
  `asyncio.create_task(self._refresh_server_tools(server_name))`, которая позже (асинхронно,
  вне текущего вызова) сходит в сеть/stdio-процесс и допишет `server_config["tools"]`. Это и
  есть источник «динамических спеков», которые должны инвалидировать индекс (см. §4).

### 3. Как per-user disabled-plugins и mode allow-list сочетаются с индексом

`allowed_plugins` (список имён плагинов, разрешённых текущим chat-mode) и per-user
disabled-plugins объединяются **до** любого вызова в `plugin_manager` —
`OpenAIHelper._apply_user_disabled_plugins` (`bot/openai_helper.py:1324-1334`): если
`allowed_plugins == ['All']`, разворачивает его в список `self.plugin_manager.plugins.keys()`
минус отключённые пользователем; иначе просто вычитает отключённые из явного списка. Результат —
обычный `List[str]`, который затем передаётся в `get_functions_specs`/`is_function_allowed`/
`is_subagent_function_allowed`.

Внутри `is_function_allowed` (`:626-633`) это применяется **после** резолва плагина по имени
функции — `plugin_name = get_plugin_name_by_function_name(...)`, затем `plugin_name in
allowed_plugins`. Замена резолва на индекс не трогает эту проверку: индекс отвечает только на
вопрос «какому плагину принадлежит эта функция», а allow-list/disabled-set — отдельный фильтр
поверх результата, применяемый тем же кодом, что и сегодня. Семантика не меняется.

Отдельно: `get_functions_specs` (`:305-348`) **намеренно** не должен получить общий
полный индекс вместо своего текущего цикла — он специально пропускает `get_spec()` для
плагинов, не входящих в `allowed_plugins`, чтобы не тратить время на узких chat-mode (см.
`AGENTS.md`, «Tool And Context Footprint», пункт 3: «Narrowing `tools:` — главный рычаг
сокращения payload на вызов»). Полный индекс, наоборот, специально считает **все** загруженные
плагины один раз, чтобы резолв **уже вызванной** моделью функции был мгновенным — это два разных
компромисса для двух разных операций (построить список для промпта vs найти владельца уже
пришедшего имени), их не следует объединять.

### 4. Сопоставление имён — почему индекс должен хранить оба ключа

`get_plugin_name_by_function_name` сегодня (`:601, 609-612`) сначала приводит входное имя к
канонической форме (`to_canonical_function_name`), а затем для каждого spec проверяет **два**
условия: точное совпадение канонического имени ИЛИ `to_model_function_name(spec_name) ==
requested_name` (сравнение с *исходным*, ещё не канонизированным именем). Второе условие —
подстраховка на случай, если `to_canonical_function_name` не смогла развернуть имя (кэш
`_model_tool_name_to_canonical`, `:72-73`, ещё не заполнен для этой конкретной функции — так
бывает только до первого построения спеков для модели в рамках жизни процесса, но код это не
гарантирует и не проверяет). Тест `test_model_safe_function_name_collision_round_trips_to_
correct_plugin` (`tests/test_plugin_manager.py:311-330`) специально создаёт коллизию
model-safe имён (`RawNamePlugin` с пустым `function_prefix`, чьё каноническое имя `.alpha_do`
после `to_model_function_name` превращается в тот же кандидат `alpha_do`, что и у
`AlphaPlugin`, и получает суффикс-хэш) — чтобы не потерять эту гарантию, индекс должен
регистрировать запись под **обоими** ключами: каноническим именем и его model-safe формой.

## Дизайн

### Место в `__init__`

Объявить рядом с `_model_tool_name_to_canonical`/`_canonical_tool_name_to_model`
(`bot/plugin_manager.py:72-73`):

```python
        self._model_tool_name_to_canonical: dict[str, str] = {}
        self._canonical_tool_name_to_model: dict[str, str] = {}
        self._function_index: dict[str, tuple[str, dict]] | None = None
```

`None` означает «индекс не построен/инвалидирован» — отличается от пустого `dict` (валидный
случай «плагинов нет ни одного»).

### Построение, кэш, инвалидация, ленивая пересборка при промахе

Новый блок методов — сразу после `to_canonical_function_name` (`bot/plugin_manager.py:397-403`),
перед `call_function` (`:405`):

```python
    def invalidate_function_index(self) -> None:
        """Явно сбрасывает индекс «имя функции → (плагин, spec)».

        Вызывается после load_plugins()/reinitialize() (набор плагинов мог
        измениться) и любым плагином с динамическими спеками после изменения
        набора инструментов (см. MCPServerPlugin.register_server/remove_server/
        _refresh_server_tools).
        """
        self._function_index = None

    def _get_function_index(self) -> dict[str, tuple[str, dict]]:
        if self._function_index is None:
            self._function_index = self._build_function_index()
        return self._function_index

    def _build_function_index(self) -> dict[str, tuple[str, dict]]:
        """Полный проход по всем загруженным плагинам — ровно тот же цикл,
        что раньше выполнялся отдельно в get_plugin_name_by_function_name на
        каждый вызов. Один сломанный плагин не должен ронять сборку индекса —
        поведение один в один с текущим (см. test_call_function_lookup_skips_
        unrelated_broken_plugin), strict_validation здесь сознательно не
        учитывается (этот метод никогда не raise'ит — как и оба публичных
        метода, которые он заменяет)."""
        index: dict[str, tuple[str, dict]] = {}
        for plugin_name in self.plugins.keys():
            try:
                plugin_instance = self.get_plugin(plugin_name)
                if not plugin_instance:
                    continue
                specs = self._normalize_specs(plugin_instance.get_spec(), plugin_instance)
            except Exception as exc:  # noqa: BLE001 — см. docstring
                logger.error(
                    "Error building function index for plugin %s: %s",
                    plugin_name, exc, exc_info=True,
                )
                continue
            for spec in specs:
                name = spec.get("name")
                if not name:
                    continue
                # setdefault: первый встреченный плагин побеждает при коллизии
                # имён — так же, как сегодняшний линейный проход возвращает
                # первое совпадение по self.plugins.keys().
                index.setdefault(name, (plugin_name, spec))
                model_name = self.to_model_function_name(name)
                if model_name != name:
                    index.setdefault(model_name, (plugin_name, spec))
        return index

    def _lookup_function(self, function_name: str) -> tuple[str, dict] | None:
        canonical = self.to_canonical_function_name(function_name)
        index = self._get_function_index()
        entry = index.get(canonical) or index.get(function_name)
        if entry is not None:
            return entry
        # Промах: спека могла появиться после последней сборки индекса без
        # явного invalidate_function_index() (например, плагин не вызвал
        # колбэк). Пересобираем один раз и пробуем снова — не бесконечный
        # цикл, т.к. второй промах просто возвращает None, как и сегодня.
        self.invalidate_function_index()
        index = self._get_function_index()
        return index.get(canonical) or index.get(function_name)
```

### Изменение `get_spec_by_function_name`/`get_plugin_name_by_function_name`

Было (`bot/plugin_manager.py:587-624`) — два независимых линейных скана. Станет:

```python
    def get_spec_by_function_name(self, function_name):
        entry = self._lookup_function(function_name)
        return entry[1] if entry else None

    def get_plugin_name_by_function_name(self, function_name):
        entry = self._lookup_function(function_name)
        return entry[0] if entry else None
```

`is_function_allowed` (`:626-633`), `__get_plugin_by_function_name` (`:674-681`),
`is_subagent_function_allowed` (`:663-672`) — **код не меняется**, они уже вызывают
`get_plugin_name_by_function_name`/наследуют его стоимость, ускорение получают бесплатно.

Метод должен остаться вызываемым по имени именно так (`self.get_spec_by_function_name(...)`
внутри `call_function:448`, не инлайнить в прямой доступ к индексу) — два существующих теста
подменяют его целиком через `monkeypatch.setattr(pm, "get_spec_by_function_name", ...)`
(`tests/test_plugin_manager.py:506, 522`).

### Инвалидация при перезагрузке плагинов

`load_plugins()` (`bot/plugin_manager.py:223-240`), в конец метода, после
`self._validate_enabled_plugins()`:

```python
        self._validate_enabled_plugins()
        self.invalidate_function_index()
```

Покрывает и первый вызов из `__init__` (:78, безопасно — `_function_index` уже `None`), и
`reinitialize()` (:298-303, которая сама вызывает `load_plugins()`), так что отдельно
трогать `reinitialize()` не нужно.

### Инвалидация из MCP-плагина (динамические спеки)

Плагины не хранят ссылку на `PluginManager` — они получают только `openai`/`bot`/
`storage_root`/`db`/`plugin_config` через `_call_initialize` (`bot/plugin_manager.py:110-135`),
который **фильтрует** kwargs по сигнатуре `plugin.initialize(...)` (плагин получает только те
параметры, которые сам объявил — сегодня `db`/`plugin_config` тоже так работают: большинство
плагинов их не объявляют и просто не получают). Тот же механизм используется, чтобы дать
плагину колбэк на инвалидацию, не заводя нового понятия в фреймворке и не хардкодя имя MCP
плагина в `plugin_manager.py` (что запрещено для core-модулей `AGENTS.md`, «Plugin And Tool
Rules», и в целом идёт вразрез со стилем этого файла).

**`bot/plugin_manager.py`** — добавить `invalidate_function_index=self.invalidate_function_index`
в kwargs всех трёх существующих вызовов `_call_initialize` (они уже передают одинаковый набор
`openai=/bot=/storage_root=/db=/plugin_config=`):
- `set_openai` (`:90-97`)
- `get_plugin`, ветка «инстанс уже в кэше, но `openai` не установлен» (`:694-701`)
- `get_plugin`, ветка «создаём новый инстанс» (`:717-724`)

Любой плагин, которому это нужно, объявляет параметр в своей сигнатуре `initialize(...)` — так
же, как `mcp_server.py` уже объявляет `storage_root` вместо принятия `**kwargs`. Плагины, не
объявившие параметр (все остальные 38), просто его не получат — `_call_initialize` фильтрует
kwargs по `inspect.signature(method).parameters` (`:126-135`).

**`bot/plugins/mcp_server.py`**:

1. `__init__` (`:66-79`) — добавить безопасный дефолт (симметрично `self.openai = None`,
   `self.bot = None` на тех же строках), чтобы методы ниже не падали, если вызваны до
   `initialize()` (в тестах такое не происходит, но это тот же защитный стиль, что уже есть в
   классе):
   ```python
           self._invalidate_function_index = lambda: None
   ```

2. `initialize()` (`:82-88`) — новый опциональный параметр:
   ```python
       def initialize(
           self, openai=None, bot=None, storage_root: str | None = None,
           invalidate_function_index=None,
       ) -> None:
           super().initialize(openai=openai, bot=bot, storage_root=storage_root)
           if invalidate_function_index is not None:
               self._invalidate_function_index = invalidate_function_index
           if storage_root:
               ...
   ```

3. `register_server()` — после `self.save_servers_config()` (`:621`), перед `tool_names = ...`
   (`:623`):
   ```python
               self.save_servers_config()
               self._invalidate_function_index()
   ```
   Покрывает оба транспорта (stdio и http) — это общий хвост метода после обеих веток.

4. `remove_server()` — после `self.save_servers_config()` (`:684`):
   ```python
           self.save_servers_config()
           self._invalidate_function_index()
   ```

5. `_refresh_server_tools()` — фоновая асинхронная задача (`_schedule_tools_refresh`,
   `:321-345`), которая мутирует `server_config["tools"]` **вне** какого-либо tool-call — это
   и есть случай, который `register_mcp_server`/`remove_mcp_server` не покрывают, а «промах →
   пересборка» в `_lookup_function` подстраховывает даже без этого явного вызова. Добавить в
   обе ветки, сразу после существующих `self.save_servers_config()`:
   ```python
   # stdio-ветка, было :377-378
                       server_config["tools"] = tools_data
                       self.save_servers_config()
                       self._invalidate_function_index()
   # http-ветка, было :388-389
                   server_config["tools"] = tools_data
                   self.save_servers_config()
                   self._invalidate_function_index()
   ```

## Почему это не меняет наблюдаемое поведение

- **Порядок совпадений при коллизии имён** — `index.setdefault` в порядке `self.plugins.keys()`
  (тот же порядок вставки, что и сегодняшний `for plugin_name in self.plugins.keys()`) — первый
  плагин по алфавиту побеждает, как и сейчас.
- **Сломанный плагин не ломает резолв других** — try/except на уровне одного плагина внутри
  `_build_function_index`, без учёта `strict_validation` (текущие `get_plugin_name_by_function_
  name`/`get_spec_by_function_name` тоже никогда не поднимают исключение из-за чужого плагина;
  `strict_validation` сегодня влияет только на `get_functions_specs`, отдельный метод, который
  этим планом не трогается).
- **Конкурентность**: `_build_function_index()` не содержит `await` (все `get_spec()` в дереве
  синхронные, `get_plugin()` тоже синхронный) — весь проход выполняется одним непрерывным
  куском на event loop без переключения на другую задачу, значит параллельные `call_function`
  из одного `asyncio.gather` (`bot/openai_tool_handler.py:181`) не могут увидеть индекс в
  «наполовину построенном» состоянии.
- **disabled-plugins/mode allow-list** — применяются после резолва плагина, тем же кодом, что и
  сегодня (§3) — индекс их не видит и не должен.

## Тесты

### Обязаны остаться зелёными без изменений (регрессионный барьер)

Прогнать `tests/test_plugin_manager.py` и `bot/tests/test_mcp_server.py` целиком — базовый
прогон на 2026-09-04 (до правок): **29 passed** и **21 passed** соответственно. Отдельного
внимания при ревью правки заслуживают:

- `test_call_function_lookup_skips_unrelated_broken_plugin` (`:334-344`) — сломанный
  `get_spec()` одного плагина не должен мешать резолву другого. Это главный регрессионный
  барьер для try/except внутри `_build_function_index`.
- `test_model_safe_function_name_collision_round_trips_to_correct_plugin` (`:311-330`) —
  проверяет двойное индексирование по каноническому и model-safe имени (§4 «Дизайна»).
- `test_function_allowlist_uses_plugin_ownership` (`:373-386`) — прямые ассерты на
  `get_plugin_name_by_function_name`/`is_function_allowed`.
- `test_call_function_returns_error_when_spec_missing_and_does_not_execute` (`:499-511`) и
  `test_call_function_records_missing_spec_as_error_telemetry` (`:515-541`) — монkeypatch'ат
  `get_spec_by_function_name` целиком; метод обязан остаться настоящим переопределяемым
  атрибутом, а не быть инлайнен в `call_function`.
- `test_namespacing_and_collision` (`:294-307`), `test_call_function_accepts_model_safe_
  function_name` (`:348-359`), `test_call_function_respects_plugin_guard` (`:545-554`).
- `bot/tests/test_mcp_server.py::test_register_server`/`test_register_server_unauthorized`/
  `test_remove_server`/`test_remove_server_unauthorized`/`test_get_spec` — используют фикстуру
  `mcp_plugin` (`:27-31`), которая вызывает `initialize()` **без** нового параметра — новый
  параметр обязан иметь дефолт `None` → no-op, иначе все 21 тест файла упадут разом.

### Новые тесты — `tests/test_plugin_manager.py`

1. **Индекс строится один раз и переиспользуется.** Плагин со счётчиком вызовов `get_spec()`:
   ```python
   def _write_counting_plugin(path: Path):
       code = """
   from bot.plugins.plugin import Plugin

   class CountingPlugin(Plugin):
       call_count = 0

       def get_source_name(self) -> str:
           return "Counting"

       def get_spec(self):
           CountingPlugin.call_count += 1
           return [{"name": "do", "description": "x",
                     "parameters": {"type": "object", "properties": {}, "required": []}}]

       async def execute(self, function_name, helper, **kwargs):
           return {"result": "ok"}
   """
       path.write_text(textwrap.dedent(code), encoding="utf-8")
   ```
   ```python
   def test_function_index_builds_spec_once_across_repeated_lookups(tmp_path):
       plugin_dir = tmp_path / "plugins"
       plugin_dir.mkdir()
       _write_counting_plugin(plugin_dir / "counting.py")

       pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
       assert pm._function_index is None  # ещё не построен

       assert pm.get_plugin_name_by_function_name("counting.do") == "counting"
       assert pm.get_spec_by_function_name("counting.do") is not None
       assert pm.is_function_allowed("counting.do", ["counting"]) is True

       # get_spec() вызван ровно один раз на все три обращения к резолву —
       # считаем через сам зарегистрированный класс, не через отдельный импорт.
       plugin_class = pm.plugins["counting"]
       assert plugin_class.call_count == 1
   ```

2. **Явная инвалидация подхватывает изменённый (не новый) spec.** Промах здесь не сработает —
   ключ `"alpha.do"` уже есть в старом индексе, значит без явного `invalidate_function_index()`
   вернётся старая закэшированная spec-запись:
   ```python
   def test_invalidate_function_index_picks_up_changed_spec(tmp_path):
       plugin_dir = tmp_path / "plugins"
       plugin_dir.mkdir()
       _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")
       pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
       pm.get_plugin_name_by_function_name("alpha.do")  # строит и кэширует индекс

       plugin = pm.get_plugin("alpha")
       plugin.get_spec = lambda: [{
           "name": "do", "description": "новое описание",
           "parameters": {"type": "object", "properties": {}, "required": []},
       }]

       # Без инвалидации индекс всё ещё отдаёт старую (закэшированную) spec-запись.
       stale_spec = pm.get_spec_by_function_name("alpha.do")
       assert stale_spec["description"] == "x"  # "x" — из _write_plugin

       pm.invalidate_function_index()
       fresh_spec = pm.get_spec_by_function_name("alpha.do")
       assert fresh_spec["description"] == "новое описание"
   ```

3. **Промах вызывает ленивую пересборку (новая функция без явной инвалидации).**
   ```python
   def test_function_index_rebuilds_lazily_on_miss(tmp_path):
       plugin_dir = tmp_path / "plugins"
       plugin_dir.mkdir()
       _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")
       pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
       pm.get_plugin_name_by_function_name("alpha.do")  # строит индекс без "alpha.extra"

       plugin = pm.get_plugin("alpha")
       plugin.get_spec = lambda: [
           {"name": "do", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}},
           {"name": "extra", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}},
       ]

       # Без вызова invalidate_function_index() — промах должен сам пересобрать индекс.
       assert pm.get_plugin_name_by_function_name("alpha.extra") == "alpha"
   ```

4. **`reinitialize()` сбрасывает индекс.**
   ```python
   def test_reinitialize_resets_function_index(tmp_path):
       plugin_dir = tmp_path / "plugins"
       plugin_dir.mkdir()
       _write_plugin(plugin_dir / "alpha.py", "AlphaPlugin", "do")
       pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
       pm.get_plugin_name_by_function_name("alpha.do")
       assert pm._function_index is not None

       _write_plugin(plugin_dir / "beta.py", "BetaPlugin", "run")
       pm.reinitialize()

       assert pm._function_index is None
       assert pm.get_plugin_name_by_function_name("beta.run") == "beta"
   ```

5. **`PluginManager` прокидывает `invalidate_function_index` в `initialize()` плагина** (без
   привязки к MCP — общий контракт фреймворка):
   ```python
   def _write_dynamic_spec_plugin(path: Path):
       code = """
   from bot.plugins.plugin import Plugin

   class DynamicPlugin(Plugin):
       def initialize(self, openai=None, bot=None, storage_root=None, invalidate_function_index=None):
           super().initialize(openai=openai, bot=bot, storage_root=storage_root)
           self.invalidate = invalidate_function_index

       def get_source_name(self) -> str:
           return "Dynamic"

       def get_spec(self):
           return [{"name": "do", "description": "x",
                     "parameters": {"type": "object", "properties": {}, "required": []}}]

       async def execute(self, function_name, helper, **kwargs):
           if self.invalidate:
               self.invalidate()
           return {"result": "ok"}
   """
       path.write_text(textwrap.dedent(code), encoding="utf-8")


   def test_plugin_receives_invalidate_function_index_callback(tmp_path):
       plugin_dir = tmp_path / "plugins"
       plugin_dir.mkdir()
       _write_dynamic_spec_plugin(plugin_dir / "dynamic.py")
       pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))

       plugin = pm.get_plugin("dynamic")
       assert plugin.invalidate == pm.invalidate_function_index
   ```

### Новые тесты — `bot/tests/test_mcp_server.py`

```python
async def test_register_server_invalidates_function_index(mcp_plugin, mock_env_vars):
    called = []
    mcp_plugin._invalidate_function_index = lambda: called.append(True)
    with patch.object(MCPServerPlugin, "_fetch_server_tools", new=AsyncMock(
        return_value=[{"name": "t", "description": "d", "parameters": {}}]
    )):
        result = await mcp_plugin.register_server(
            "srv", 123, base_url="http://example.com",
        )
    assert result.get("success") is True
    assert called


async def test_remove_server_invalidates_function_index(mcp_plugin, mock_env_vars):
    mcp_plugin.servers["srv"] = {"transport": "http", "base_url": "http://x", "tools": []}
    called = []
    mcp_plugin._invalidate_function_index = lambda: called.append(True)

    result = await mcp_plugin.remove_server("srv", 123)

    assert result.get("success") is True
    assert called


async def test_refresh_server_tools_invalidates_function_index(mcp_plugin, mock_env_vars):
    mcp_plugin.servers["srv"] = {"transport": "http", "base_url": "http://x", "tools": []}
    called = []
    mcp_plugin._invalidate_function_index = lambda: called.append(True)
    with patch.object(MCPServerPlugin, "_fetch_server_tools", new=AsyncMock(
        return_value=[{"name": "t", "description": "d", "parameters": {}}]
    )):
        await mcp_plugin._refresh_server_tools("srv")

    assert called
```
(Конкретные аргументы `register_server`/сигнатуру `_fetch_server_tools` уточнить по актуальному
коду на момент реализации — см. `bot/plugins/mcp_server.py:546-624, 662-687, 359-393`.)

Точные параметры `register_server(...)` в примерах выше и admin-права (`user_id=123` — уже
входит в `ADMIN_USER_IDS` из `mock_env_vars`, `bot/tests/test_mcp_server.py:14-23`) взять из
существующего `test_register_server` (`:82-102`), чтобы не дублировать константы вручную.

## Команды проверки

Выполнять из корня репозитория `/srv/git_projects/chatgpt-telegram-bot`, интерпретатор
`~/.venvs/ctb/bin/python` (`.venv` проекта недоступен по правам).

```bash
# Базовый прогон до правок (зафиксировано в этом плане, 2026-09-04):
~/.venvs/ctb/bin/python -m pytest tests/test_plugin_manager.py bot/tests/test_mcp_server.py \
  -q -p no:cacheprovider
# -> 29 passed, 21 passed (раздельно) / 50 passed (вместе)

# После правок — те же файлы плюс новые тесты:
~/.venvs/ctb/bin/python -m pytest tests/test_plugin_manager.py bot/tests/test_mcp_server.py \
  -q -p no:cacheprovider -x

# Смежное: маршрутизация tool-call (не должна измениться по наблюдаемому поведению):
~/.venvs/ctb/bin/python -m pytest tests/test_openai_helper_tool_calls.py -q -p no:cacheprovider -x
# -> baseline 134 passed

# Полный прогон (testpaths = tests bot/tests, evals/ не собираются):
~/.venvs/ctb/bin/python -m pytest -q -p no:cacheprovider
# -> baseline на 2026-09-04 (до этой задачи, HEAD af382fb): 1503 passed, 1 skipped, 5 failed —
#    все 5 падений в tests/test_database.py (test_get_conversation_context_*), не связаны с
#    plugin_manager.py и не должны становиться ни больше, ни меньше от этой задачи.
```

## Риски

- **`_function_index` — новый публичный по сути атрибут** (тесты обращаются к нему напрямую,
  `pm._function_index is None`/`is not None`, как в §«Тесты»). Имя с ведущим underscore, но раз
  тесты на него полагаются — при рефакторинге в будущем нужно синхронно править и тесты.
- **Порядок обхода `self.plugins.keys()` определяет, какой плагин получает «читаемое» имя при
  коллизии model-safe имён**, если коллизия вообще возникает (см. `to_model_function_name`,
  `:368-395`, hash-суффикс). При полной сборке индекса сразу (весь `self.plugins`, а не только
  плагины конкретного chat-mode) порядок теперь не зависит от того, какой mode/request первым
  вызвал `get_functions_specs` — это **более** детерминированно, чем сегодня (сегодня
  `_model_tool_name_to_canonical` заполняется первым запросом, который до него дотянется, и
  это НЕ обязательно `['All']`). Не регрессия по корректности (обе стороны биекции всё равно
  различимы), но если где-то есть тест/документация, жёстко фиксирующие «плагин X получает
  хэш-суффикс» вне уже разобранного `test_model_safe_function_name_collision_round_trips_to_
  correct_plugin` — проверить его отдельно при реализации.
- **Эффект от промаха на действительно несуществующее имя** (модель дала галлюцинированное имя
  функции) — вызывает одну лишнюю полную пересборку индекса (`invalidate_function_index()` +
  `_get_function_index()`), т.е. ту же стоимость, что и сегодняшний единственный линейный скан
  для такого случая. Не регрессия, но если модель начнёт систематически слать несуществующие
  имена (например, баг в промпте) — каждый такой вызов будет платить полную пересборку заново
  (кэш «этого имени точно нет» не заводится). Не считаю нужным чинить сейчас: сегодняшнее
  поведение для такого случая — тоже O(N) на каждый промах, значит хуже не станет; заводить
  negative-cache — уже за рамками задачи (over-engineering под гипотетический сценарий).
- **MCP: `initialize()` без `**kwargs`.** Новый параметр `invalidate_function_index` добавляется
  как явный именованный параметр (симметрично уже существующему `storage_root`), а не через
  `**kwargs` — сохраняет читаемую сигнатуру, но требует не забыть его в сигнатуре при следующей
  правке `_call_initialize`, если появится ещё один новый kwarg общего назначения.
- **Фоновая `_refresh_server_tools`** запускается `asyncio.create_task` без обработки исключений
  сверх уже существующего `_done`-колбэка (`:337-345`, логирует и не пробрасывает) — если сам
  колбэк `self._invalidate_function_index()` бросит (не должен, это простой `def ...: self.
  _function_index = None`), исключение уйдёт туда же, в лог фоновой задачи, а не наружу. Не
  меняет существующий контракт устойчивости этого фонового пути.
- **Плагины, которые сами держат spec-мутирующее состояние, но не MCP** (гипотетически, в
  будущем) — не получат авто-инвалидацию, если не объявят `invalidate_function_index` в своей
  `initialize()`. Это осознанный выбор (нет способа для core-кода узнать «какие плагины
  динамические» без хардкода конкретных plugin-id, что запрещено `AGENTS.md`) — подстраховка
  для них — общий механизм «промах → пересборка» (§«Дизайн»), который сработает при первом
  обращении к НОВОМУ имени, но не при замене существующего под тем же именем без вызова
  `invalidate_function_index()` явно.

## Критерии готовности

- `bot/plugin_manager.py`: `_function_index` объявлен в `__init__` рядом с
  `_model_tool_name_to_canonical`; `invalidate_function_index()`, `_get_function_index()`,
  `_build_function_index()`, `_lookup_function()` реализованы; `get_spec_by_function_name`/
  `get_plugin_name_by_function_name` переписаны на `_lookup_function` без изменения сигнатур и
  возвращаемых значений (`spec | None`, `plugin_name | None`).
- `load_plugins()` вызывает `invalidate_function_index()` в конце (покрывает и `__init__`, и
  `reinitialize()`).
- Все три вызова `_call_initialize` (`set_openai`, обе ветки `get_plugin`) передают
  `invalidate_function_index=self.invalidate_function_index`.
- `bot/plugins/mcp_server.py`: `initialize()` принимает и сохраняет
  `invalidate_function_index` (дефолт `None` → no-op); `register_server`, `remove_server` и обе
  ветки `_refresh_server_tools` вызывают его после успешной мутации `self.servers`.
- `is_function_allowed`, `__get_plugin_by_function_name`, `is_subagent_function_allowed`,
  `call_function`, `get_plugin_source_name` — код не менялся (ускорение унаследовано).
- `tests/test_plugin_manager.py` (29 существующих + новые из §«Тесты») и
  `bot/tests/test_mcp_server.py` (21 существующий + новые) — все зелёные.
- `tests/test_openai_helper_tool_calls.py` (134) без изменений в поведении.
- Полный прогон `pytest -q` не хуже базового (1503 passed / 1 skipped / 5 pre-existing failed
  в `tests/test_database.py`, не связанных с этой задачей).
- `_guard_tool_call` (`bot/plugin_manager.py:509-532`) не тронут — вне рамок T18.

## Постскриптум после ревью

Ревью (Sonnet, персона reviewer): ошибок и предупреждений нет. Подтверждено: построение
индекса не содержит `await` и не может быть увидено «наполовину» параллельными
`call_function`; победитель при коллизии имён совпадает с прежним линейным сканом; промах
на несуществующее имя стоит один ребилд (не хуже прежнего скана, negative-cache не нужен);
`_call_initialize` фильтрует новый параметр по сигнатуре, `TypeError` невозможен.
Отмечен побочный эффект: кэш model-safe имён теперь наполняется по всем плагинам сразу
(при реальной коллизии хэш-суффикс может достаться другому плагину) — на корректность
резолва не влияет.
