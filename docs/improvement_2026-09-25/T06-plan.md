# T06 — Промпты-данные: chat_modes.yml тексты, миграция mode_key, shared_blocks, skills metadata

**Владение файлами:** `bot/chat_modes.yml`, `bot/chat_modes_registry.py`, `bot/database.py`
(только миграция mode_key), `bot/skills/**`, `tests/test_chat_modes_registry.py`,
`tests/test_database.py` (миграция), новые тесты skills-метаданных.
**НЕ трогать:** списки `tools:` в режимах, любые `get_spec()`, любые имена/описания/параметры
функций плагинов.

Все числа и цитаты ниже проверены заново в этой сессии (`bot/chat_modes.yml`,
`bot/chat_modes_registry.py`, `bot/database.py`, `bot/skill_script_routing.py`,
`bot/validation.py`, `bot/chat_response_utils.py`, `bot/plugin_manager.py`,
`bot/plugins/skills.py`, `bot/skills/**`, реальный `PluginManager` в `~/.venvs/ctb`,
мастер-план `docs/improvement_2026-09-25/00-master-plan.md:206-253`), не переносились из
памяти без перечитывания.

---

## 0. Важное отклонение, которое нужно решить до реализации

Мастер-план (шаг 1) просит «миграцию БД» без уточнения деталей. При проектировании нашёл
скрытую проблему: `_reconcile_schema_version_with_shape()` (`bot/database.py:627-643`)
угадывает «настоящую» версию схемы ИСКЛЮЧИТЕЛЬНО по форме таблицы (наличию колонок
`session_id`, `version`) и откатывает `schema_version`, если записанная версия оказалась
«впереди» формы. Это защита от порчи БД, рассчитанная на миграции 1 и 2 — обе добавляют
колонку.

Миграция 3 (mode_key) — **чисто дата-миграция**: `mode_key` живёт внутри JSON-поля
`context`, не в колонке (`AGENTS.md`: «mode_key… lives inside `conversation_context.context`
JSON blob»). Форма таблицы после миграции 3 ничем не отличается от формы после миграции 2.
Если не поправить `_reconcile_schema_version_with_shape`, то при каждом следующем старте
бота эта функция увидит `recorded_version=3`, посчитает по форме `actual_version=2`,
решит, что 3 «неправдоподобно далеко», и удалит запись о версии 3
(`DELETE FROM schema_version WHERE version > 2`, `bot/database.py:643`). Миграция 3 будет
запускаться заново на каждом старте (полный скан `conversation_context`) — не ломает данные
(миграция идемпотентна), но тратит время старта впустую и на каждом старте пишет
warning-лог «schema_version=3 is ahead of… resetting to 2».

**Решение:** ввести константу-потолок «до какой версии форма таблицы вообще может служить
доказательством» и не трогать версии выше него. Правка — одно новое условие, без изменения
существующей защиты для версий 1 и 2 (детали в §3.3).

---

## 1. Инвентаризация bot/chat_modes.yml (25 режимов, границы верифицированы)

Границы режимов (номер первой строки top-level ключа):
`assistant`5, `text_improver`39, `travel_guide`113, `content_creator`180,
`technical_writer`270, `summary_assistant`361, `primitives`445, `chief_assistant`524,
`medical_assistant`579, `legal_assistant_ru`640, `code_assistant`713, `sql_assistant`801,
`artist`885, `english_tutor`901, `psychologist`983, `movie_expert`1065, `school_tutor`1147,
`code_interpreter`1220, `startup_idea_generator`1253, `money_maker`1337, `accountant`1421,
`project_manager`1505, `meta_writer`1589, `personal_finance_planner`1672, `skills_agent`1744.

Проверенные числа:
- **17×** опечатка «исползуй» (строки 157, 242, 301, 417, 501, 567, 789, 871, 971, 1053,
  1135, 1239, 1323, 1407, 1491, 1575, 1659) — все 17 являются третьей строкой одного и того
  же 3-строчного блока «ВАЖНО! ЕСЛИ НЕОБХОДИМО ИСКАТЬ…» (точечно проверено смещение
  start+2 для блоков на 155→157, 240→242, 299→301 — совпадает).
- **23×** копии блока «ВАЖНО! ЕСЛИ НЕОБХОДИМО ИСКАТЬ В ИНТЕРНЕТЕ, ПРАВИЛА ВЫБОРА МЕТОДА
  ПОИСКА» (строки 30, 97, 155, 240, 299, 415, 499, 565, 618, 690, 787, 869, 969, 1051, 1133,
  1197, 1237, 1321, 1405, 1489, 1573, 1657, 1721) — 17 с опечаткой, 6 уже с «используй».
  Отсутствует только в `artist` и `skills_agent`.
- **18×** байт-идентичная строка «research_articles ОДИН РАЗ для поиска актуальных данных,
  если необходимо» — почти всегда прямо перед блоком ВАЖНО (пример: `text_improver:95`
  перед блоком `97-99`).
- **4×** строка с некорректным именем инструмента `optimize_prompt` в прозе (строки 26, 153,
  238, 297) — правильный `tools:`-идентификатор плагина `prompt_perfect`. Остальные 4 из 7
  вхождений «НЕ ПЫТАЙТЕСЬ…» (строки 616, 687, 1193, 1718) уже используют корректное имя
  `prompt_perfect`.
- **1×** «В конце всегда интересуйтесь, нужна ли дополнительная помощь…» (строка 29,
  только `assistant`).
- **1×** `prompt_markers` во всём файле (строки 1799-1801, только `skills_agent`:
  `["локальные skills", "skills."]`).
- `ОБЯЗАТЕЛЬНО` встречается только в `skills_agent` (строки 1758, 1759).
  `ЗАПРЕЩЕНО` встречается только в `skills_agent` (строки 1791, 1792).

**Проверка реальных имён инструментов, которые видит модель** (загрузка настоящего
`PluginManager` в `~/.venvs/ctb`, `to_model_function_name`, `bot/plugin_manager.py:395-424`):

| Бытовое имя в прозе YAML | Плагин | Реальное имя функции | Имя, которое видит модель |
|---|---|---|---|
| `research_articles` | `web_research` | `research_articles` | `web_research_research_articles` |
| `web_search` (общее) | один из `ddg_web_search` / `google_web_search` / `jina_web_search` | `web_search` (у всех трёх) | `ddg_web_search_web_search` / `google_web_search_web_search` / `jina_web_search_web_search` |
| `website_content` | `website_content` | `website_content` | `website_content_website_content` |
| `optimize_prompt` | `prompt_perfect` | `optimize_prompt` | `prompt_perfect_optimize_prompt` |

Вывод: ни одно бытовое имя из прозы не совпадает с тем, что модель реально может вызвать.
Раньше это «работало», потому что модель видит и прозу, и настоящие JSON-специ (там верные
имена) одновременно и как-то сопоставляет их сама — но это лишний риск (модель может
попытаться вызвать несуществующее имя `research_articles`). Решение по неймингу — §4.2.

---

## 2. Порядок выполнения (важно)

Миграция БД (§3) должна попасть в код **до** правки текстов режимов (§4-6), т.к.
`LEGACY_PROMPT_FINGERPRINTS` — это хэши текущего (HEAD) `prompt_start`, снятые ДО правки.
Если породок перепутать, слепки будут сняты уже с нового текста и не смогут сматчить старые
сессии в БД со старым текстом. Хэши ниже (§3.2) уже вычислены с текущего HEAD и не зависят
от того, когда физически применяется патч — но **тексты в §4-6 нельзя коммитить раньше**,
чем структура `LEGACY_PROMPT_FINGERPRINTS` из этого документа попадёт в `database.py`,
иначе при повторной генерации кем-то ещё хэши будут сняты с уже отредактированного файла.

---

## 3. Миграция БД: backfill mode_key (`bot/database.py`)

### 3.1 TARGET_SCHEMA_VERSION и список миграций

```python
# bot/database.py:180-183, было:
    # Текущая целевая версия схемы. Миграция 1 — переход с legacy-таблицы
    # `conversation_context` без session_id на новую схему с сессиями.
    # Миграция 2 — добавление колонки version (монотонный счётчик ревизий).
    TARGET_SCHEMA_VERSION = 2

# стало:
    # Текущая целевая версия схемы. Миграция 1 — переход с legacy-таблицы
    # `conversation_context` без session_id на новую схему с сессиями.
    # Миграция 2 — добавление колонки version (монотонный счётчик ревизий).
    # Миграция 3 — чисто дата-миграция (без изменения колонок): backfill
    # mode_key внутрь messages[0] системного сообщения в JSON `context` для
    # сессий, сохранённых до появления mode_key. См. LEGACY_PROMPT_FINGERPRINTS.
    TARGET_SCHEMA_VERSION = 3
```

```python
# bot/database.py:608-612, было:
    def _schema_migrations(self):
        return (
            (1, self._migrate_conversation_context_to_sessions),
            (2, self._migrate_conversation_context_version_column),
        )

# стало:
    def _schema_migrations(self):
        return (
            (1, self._migrate_conversation_context_to_sessions),
            (2, self._migrate_conversation_context_version_column),
            (3, self._migrate_conversation_context_backfill_mode_key),
        )
```

### 3.2 LEGACY_PROMPT_FINGERPRINTS

Модульная константа (место — рядом с другими модульными константами в начале
`bot/database.py`, до класса `Database`). Ключ — `sha256(prompt_start.strip())` текущего
(HEAD, до правок T06) `bot/chat_modes.yml`, значение — `mode_key`. Вычислено сейчас через
`yaml.safe_load` (та же логика чтения, что использует `ChatModesRegistry`), проверено:

```python
# Слепки prompt_start каждого режима из bot/chat_modes.yml ДО правок T06 (2026-09-25,
# HEAD 08bc457). Нужны, чтобы миграция 3 могла определить mode_key старых сессий, чей
# system-контент совпадает с ТЕКСТОМ ДО правки промптов, а не с текущим YAML.
# sha256(prompt_start.strip()) -> mode_key
LEGACY_PROMPT_FINGERPRINTS = {
    "b2c1c8abd73b2c73e540f0373f48303f1c09370ea0f73bc120fec97ab1eaef08": "assistant",
    "9be4c9b324b6a539d3ff61a4b081ba5e4110c3fedf1d4867e181b5abb6062e98": "text_improver",
    "7933d561247b29bb3010001a6bba9901b0c7b701c163797d666eee00ebf0ee87": "travel_guide",
    "b5d714aed40a91e8d6764d640c95e6ed5ae9ef912d42aaf46b177e2b66a5391a": "content_creator",
    "6d7202e82a2ea741d7f4be75afa5eb23a24bf85f443251a534bdd99ed36c1bd6": "technical_writer",
    "7ee9d354b4d76eb1169705ec95eb6b2d489d1d24923bc5caf8d9449bdd9ba312": "summary_assistant",
    "b6ca32103b07a5eca6b3319387679f3d831fb0f82b49bd16e4b33ed3f38208ab": "primitives",
    "f025f6641ea8c5a20238971c27e0ce66da6cf9fda8100f27d7bf5d7d4573775d": "chief_assistant",
    "065d4af6dba74620b9793a2eaacd0af408600db3f9f653d8638e43821a2594af": "medical_assistant",
    "cbe60fa1973fa9447eb97acf1a0ef269736aa31b50cb0f56d14add122556d0c4": "legal_assistant_ru",
    "3fb6127506423b9afe41c530021d8ce7c2ea263132ab8b9fec9271a1f41d0b47": "code_assistant",
    "c8656f9a7e3552eabfdb76569a657d08fa536943968f70562eb4b9da37267074": "sql_assistant",
    "64f4189c68d50ea0230beb85535a3521c1e880114e53792f0079c0f093d601e4": "artist",
    "f405f6e049ce7228ab0cc2a55e956d63703b5d93d7d1b9e073e836276c097ba9": "english_tutor",
    "9e124562a5fb93c1d01fae5f877f417df0991f6bffc2d2065fbb3c730a732dc7": "psychologist",
    "3c232872069309f3726f10a64a14b76fe88c231b510558f299363fe8013557c4": "movie_expert",
    "db5685b99655ca1eab5f9527f75aa62f37ce81a513a69c22610f57daeb613c0a": "school_tutor",
    "210dd566b31af11655af47fe77a7d30a5e637b5ef70ec759f3570a749b3b553d": "code_interpreter",
    "a7d0a0741577c59037d23411a3e7e51ef15327226ec6428f5b1e693ee2d51d9e": "startup_idea_generator",
    "ac8baf5213845478928347370f322a342b2edd7499996281e8c6ca3eefbd869b": "money_maker",
    "79f72e51feb8707d25d01919084c59d75656d57e28f2d9990261a31f83bd1921": "accountant",
    "8f61cbd329dd4429c00457424e991084a2bc972b0fcbb5caccaad43f1e8276a9": "project_manager",
    "c776c0d7598ed7a35bc43f80f1b3f3f2e8d12e5c5608d393ef132ea6a210ddc3": "meta_writer",
    "7697f97a7fbafaa604efafe8d8a9316b2b920eaa9fd2b0e40a1ced84563096e2": "personal_finance_planner",
    "9367c6a4cccb4cc6ab3b6f8d1f55fca35f03a1fa400eb4ca34edf99cae0ffaca": "skills_agent",
}
```

Воспроизвести (для проверки при ревью):
```bash
~/.venvs/ctb/bin/python -c "
import yaml, hashlib
data = yaml.safe_load(open('bot/chat_modes.yml', encoding='utf-8'))
for k, m in data.items():
    print(hashlib.sha256(m['prompt_start'].strip().encode()).hexdigest(), k)
"
```
(запускать на копии файла ДО правок §4-6, иначе хэши не совпадут).

### 3.3 Миграция и фикс reconcile (код)

```python
def _migrate_conversation_context_backfill_mode_key(self, cursor: sqlite3.Cursor) -> None:
    """Миграция 3: чисто данные, без изменения колонок. Для сессий без mode_key
    в system-сообщении находит режим по sha256 старого (до правок T06) prompt_start
    и проставляет mode_key. Идемпотентна: пропускает строки, где mode_key уже есть
    или контент не совпал ни с одним слепком."""
    cursor.execute("SELECT user_id, session_id, context FROM conversation_context")
    rows = cursor.fetchall()
    updated = 0
    for user_id, session_id, context_json in rows:
        try:
            context = json.loads(context_json)
        except (TypeError, ValueError):
            continue
        messages = context.get("messages") if isinstance(context, dict) else None
        if not isinstance(messages, list) or not messages:
            continue
        first = messages[0]
        if not isinstance(first, dict) or first.get("role") != "system":
            continue
        if first.get("mode_key"):
            continue
        content = first.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        fingerprint = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        mode_key = LEGACY_PROMPT_FINGERPRINTS.get(fingerprint)
        if not mode_key:
            continue
        first["mode_key"] = mode_key
        cursor.execute(
            "UPDATE conversation_context SET context = ?, version = version + 1 "
            "WHERE user_id = ? AND session_id = ?",
            (json.dumps(context, ensure_ascii=False), user_id, session_id),
        )
        updated += 1
    logger.info("Migration 3: backfilled mode_key for %d session(s)", updated)
```

`json`/`hashlib` уже импортированы в `bot/database.py` (строки 7, 13) — новых импортов не
требуется. Стиль (raw `cursor.execute`, без похода через async-обёртки) — как в
`_migrate_conversation_context_to_sessions` и `_migrate_conversation_context_version_column`.

Фикс reconcile (`bot/database.py:627-643`) — одно новое условие плюс одна константа класса:

```python
    # Миграции 1 и 2 меняют колонки conversation_context — их можно проверить по форме
    # таблицы. Миграция 3+ — чисто дата-миграции (mode_key живёт в JSON, не в колонке),
    # форма таблицы их не отражает. Поэтому reconcile не пытается судить о версиях выше
    # этого потолка — иначе завершённая миграция 3 выглядела бы «убежавшей вперёд формы»
    # и откатывалась бы на каждом старте.
    SHAPE_VERIFIABLE_SCHEMA_VERSION = 2

    def _reconcile_schema_version_with_shape(self, cursor: sqlite3.Cursor) -> None:
        columns = set(self._conversation_context_columns(cursor))
        actual_version = 0
        if 'session_id' in columns:
            actual_version = 1
        if 'version' in columns:
            actual_version = 2
        recorded_version = self._schema_version(cursor)
        if recorded_version <= actual_version:
            return
        if recorded_version > self.SHAPE_VERIFIABLE_SCHEMA_VERSION:
            return
        logger.warning(
            "schema_version=%s is ahead of conversation_context shape=%s; resetting to %s",
            recorded_version,
            sorted(columns),
            actual_version,
        )
        cursor.execute('DELETE FROM schema_version WHERE version > ?', (actual_version,))
```

Поведение для версий 1/2 не меняется (защита от порчи по-прежнему работает). Для версии 3
reconcile теперь просто ничего не делает — форма таблицы физически не может ни подтвердить,
ни опровергнуть, что миграция 3 прошла.

---

## 4. shared_blocks: механизм (`bot/chat_modes_registry.py`)

### 4.1 Код подстановки

```python
# bot/chat_modes_registry.py:26-46, после строки `self._data = data` (было 45),
# перед `self._mtime = mtime` (было 46):
            self._data = data
            self._substitute_shared_blocks()
            self._mtime = mtime

    def _substitute_shared_blocks(self) -> None:
        shared_blocks = self._data.pop("shared_blocks", None)
        if not isinstance(shared_blocks, dict):
            return
        for mode_data in self._data.values():
            if not isinstance(mode_data, dict):
                continue
            prompt = mode_data.get("prompt_start")
            if not isinstance(prompt, str):
                continue
            for key, block in shared_blocks.items():
                marker = "{{shared:%s}}" % key
                if marker in prompt:
                    prompt = prompt.replace(marker, str(block).strip())
            mode_data["prompt_start"] = prompt
```

`shared_blocks` выталкивается из `self._data` до того, как любой метод класса его увидит —
`all_modes()`, `get_all_modes_list()`, `validate_tools()`, `get_mode_by_system_prompt()`
все читают только `self._data` после `_load_if_needed()`, поэтому одна правка закрывает все
четыре требования мастер-плана («не видят служебный ключ») без четырёх отдельных патчей.

### 4.2 Контент shared_blocks.web_search_rules

Маркер — ровно `{{shared:web_search_rules}}` (имя зафиксировано мастер-планом, шаг 6).
Объединяет старую строку «research_articles ОДИН РАЗ…» и 3-строчный блок ВАЖНО в один блок
(они всегда шли рядом и об одном и том же — какой инструмент искать/читать использовать).
Опечатка «исползуй» пропадает автоматически (текст пишется один раз и правильно).

Имена инструментов — **идентификаторы плагинов из `tools:`** (`web_research`,
`website_content`, `ddg_web_search`/`google_web_search`/`jina_web_search`), а не итоговые
underscore-имена вида `web_research_research_articles` — обоснование: `tools:`-списки и так
используют именно словарь идентификаторов плагинов, а не функций; сохранение того же
словаря в прозе не даёт двум местам разъехаться, и настоящее JSON-имя функции модель и так
видит в своей function-schema. Для `web_search` называю все три возможных плагина через
слэш, т.к. в разных режимах в `tools:` встречается разный поднабор (общий блок не может
жёстко называть один — какого нет в `tools:` этого режима, того не будет и в function-array
модели, упоминание лишнего имени безвредно, т.к. модель просто не увидит его spec).

```yaml
shared_blocks:
  web_search_rules: |
    Если нужны данные из интернета, выбирайте инструмент по типу запроса и вызывайте
    нужный только один раз за подзадачу:
    - Для общих вопросов (например, "как работает блокчейн") используйте web_research.
    - Для точных фактов (курс валют на дату, конкретные цифры, адреса) используйте
      доступный по списку tools этого режима веб-поиск (ddg_web_search / google_web_search
      / jina_web_search) и затем website_content, чтобы прочитать страницу.
```

(Верхнеуровневый ключ `shared_blocks:` добавляется в самый низ `chat_modes.yml`, после
последнего режима `skills_agent`, либо в начало файла после комментария о списке tools —
не принципиально, `ChatModesRegistry` читает весь YAML как один dict.)

### 4.3 Правило замены по режимам + representative before/after

Общее правило: 3-строчный блок ВАЖНО заменяется на `{{shared:web_search_rules}}`; если ему
непосредственно предшествует байт-идентичная строка «research_articles ОДИН РАЗ для поиска
актуальных данных, если необходимо», эта строка удаляется тоже (её смысл вошёл в общий
блок). Другие соседние, режимо-специфичные строки (например, «website_content для проверки
фактов», «optimize_prompt/prompt_perfect для улучшения запроса…») не трогаются — они
остаются как есть рядом с маркером.

**Пример 1 — `assistant` (блок встроен как пункт нумерованного списка, строки 30-32):**
```
# было (строка 30-32, пункт "15."):
15. ВАЖНО! ЕСЛИ НЕОБХОДИМО ИСКАТЬ В ИНТЕРНЕТЕ, ПРАВИЛА ВЫБОРА МЕТОДА ПОИСКА:
 - Если запрос общий (например, "как работает блокчейн"), используй web_research
 - Если запрос конкретный (например, "курс доллара на дату", "расстояние от Москвы до Санкт-Петербурга"), используй web_search и website_content

# стало:
15. {{shared:web_search_rules}}
```

**Пример 2 — `text_improver` (строки 95, 97-99, простой случай):**
```
# было:
    Использование инструментов:
    - research_articles ОДИН РАЗ для поиска актуальных данных, если необходимо

    ВАЖНО! ЕСЛИ НЕОБХОДИМО ИСКАТЬ В ИНТЕРНЕТЕ, ПРАВИЛА ВЫБОРА МЕТОДА ПОИСКА:
    - Если запрос общий (например, "как работает блокчейн"), используй web_research
    - Если запрос конкретный (например, "курс доллара на дату", "расстояние от Москвы до Санкт-Петербурга"), используй web_search и website_content

# стало:
    Использование инструментов:
    {{shared:web_search_rules}}
```

**Пример 3 — `technical_writer` (строки 295-301, есть соседние режимо-специфичные строки,
которые остаются нетронутыми):**
```
# было:
    - research_articles ОДИН РАЗ для поиска актуальных данных, если необходимо
    - website_content для проверки фактов
    - optimize_prompt для улучшения запроса. НЕ ПЫТАЙТЕСЬ оптимизировать уже оптимизированный промпт

    ВАЖНО! ЕСЛИ НЕОБХОДИМО ИСКАТЬ В ИНТЕРНЕТЕ, ПРАВИЛА ВЫБОРА МЕТОДА ПОИСКА:
    - Если запрос общий (например, "как работает блокчейн"), исползуй web_research
    - Если запрос конкретный (например, "курс доллара на дату", "расстояние от Москвы до Санкт-Петербурга"), исползуй web_search и website_content

# стало:
    - website_content для проверки фактов
    - prompt_perfect для улучшения запроса. Не оптимизируйте уже оптимизированный промпт

    {{shared:web_search_rules}}
```
(здесь же заодно правится `optimize_prompt`→`prompt_perfect`, см. §5).

Итого по всем 23 копиям: implementer применяет то же правило (удалить 1 строку
research_articles, если есть; заменить 3-строчный блок на маркер) — точный список всех 23
позиций для замены: строки 30, 97, 155, 240, 299, 415, 499, 565, 618, 690, 787, 869, 969,
1051, 1133, 1197, 1237, 1321, 1405, 1489, 1573, 1657, 1721 (начало каждого блока).

---

## 5. Точечные правки текста режимов (вне shared-блока)

- `assistant`, строка 24, правило 9 «Указывайте, какой инструмент используется и для чего.»
  — **удалить целиком** (мастер-план шаг 4: «убрать требование объявлять инструмент перед
  вызовом»; так же требует `skills_agent` правило 9 в разделе безопасности — «Не пишите
  пользователю, что вы "сейчас вызовете"…» — оставление старого текста было бы внутренне
  противоречиво между режимами).
- `assistant`, строка 27, правило 12 «При вопросах, связанных с программированием,
  переходите в режим Code Assistant (/reset code_assistant).» — команда `/reset` не
  принимает аргумент-имя режима (проверено: `ChatGPTTelegramBot.reset()`,
  `bot/telegram_bot.py:2148-2181`, читает только `update`/`context`, `context.args` не
  используется). **Удалить упоминание `/reset code_assistant`**; предлагаемая замена:
  «При вопросах, связанных с программированием, переключитесь на режим Code Assistant через
  меню /mode.» (точную формулировку меню-команды сверить с реальной командой переключения
  режима — вне зоны владения T06, только не оставлять несуществующий синтаксис).
- `assistant`, строка 29, правило 14 «В конце всегда интересуйтесь, нужна ли дополнительная
  помощь или подробное пояснение.» — **удалить** (мастер-план шаг 7, единственная копия в
  файле).
- Нумерация правил 10-15 в `assistant` сдвигается после удаления 9/12/14 — implementer
  перенумеровывает оставшиеся пункты по порядку (1..N), это чисто механическая правка.
- 4 места с некорректным именем `optimize_prompt` в прозе (строки 26, 153, 238, 297) →
  заменить на `prompt_perfect` (реальный `tools:`-идентификатор плагина).
- Лишние двойные пустые строки перед `parse_mode:`/`tools:` после схлопывания
  3-строчного блока ВАЖНО в один маркер — оставлять одну пустую строку, не более (косметика,
  не отдельная задача — просто не тащить за собой лишний перенос после замены).

---

## 6. skills_agent: построчный разбор правил (`bot/chat_modes.yml:1744-1803`)

### 6.1 Что удаляется (код-дубли, подтверждено чтением кода в этой сессии)

| # | Текущий текст (сокращённо) | Код, который уже это гарантирует | Решение |
|---|---|---|---|
| безопасность-10 | «Никогда не выводите служебные reasoning-теги вроде `<think>`…» | `THINK_BLOCK_RE`/`THINK_TAG_RE` вырезают `<think>…</think>` в `choice_message_text()`, `bot/chat_response_utils.py:10-11,20-21` | **Удалить** |
| безопасность-15 | «ЗАПРЕЩЕНО создавать новые файлы скриптов через codeinterpreter (…fs.writeFile…)» | `SCRIPT_FILE_CREATION_RE` (`bot/skill_script_routing.py:11-16`), проверяется в `_skill_script_routing_error()` (`:126-132`) для `codeinterpreter.deep_analysis` в режиме skills_agent | **Удалить** |
| безопасность-16 | «Схема обязательных параметров tools… нарушение отбивается валидатором…» | `validate_function_args()` через `Draft7Validator`, `bot/validation.py:33-51`, вызывается до исполнения плагина | **Удалить** |
| безопасность-11 | «Никогда не выводите сырые результаты tools вида "Function … returned: …"» | `RAW_TOOL_RESULT_RE`, `bot/chat_response_utils.py:12,23-24` — тоже код-дубль | **Оставить** — явное указание заказчика: этот путь снимает T08, в волне 2 не трогаем (закреплено тестом `tests/test_chat_modes_registry.py:137`, который НЕ должен ломаться) |
| безопасность-14 | «Скрипты, принадлежащие активному skill, ЗАПРЕЩЕНО запускать через codeinterpreter.deep_analysis… Допустимые способы — skills.run_skill_script и terminal.terminal» | Частично код-дубль: `_active_skill_scripts()`+`_refers_to_active_script()` (`bot/skill_script_routing.py:109-115`) блокируют это **безусловно** (не только в skills_agent, в отличие от правила 15) | **Флаг, по умолчанию — оставить** (см. §9, п.1 — не было явно названо в списке заказчика из 3 пунктов; удалять только по отдельному подтверждению) |

Оба явных маркера, от которых зависит код вне владения T06, физически проверены и
сохраняются:
- `prompt_markers: ["локальные skills", "skills."]` (`chat_modes.yml:1799-1801`) — сама
  фраза «локальные skills» остаётся в первом предложении промпта, «skills.» тривиально
  сохраняется (десятки `skills.xxx`-ссылок).
- Подстрока `manage_plan_tasks` в тексте промпта — используется в
  `bot/plugins/agent_tools.py:411` (`if isinstance(content, str) and "manage_plan_tasks" in
  content: mode_prompt_covers_rule = True`), чтобы не задваивать инъекцию правила о плане.
  Остаётся в правиле 2a (`agent_tools.manage_plan_tasks`).

### 6.2 Правки формулировок (без удаления)

**Правило 2** (строка 1758), убрать капс:
```
# было:
2. Если подходящий skill найден, вызовите skills.get_skill. ОБЯЗАТЕЛЬНО изучите раздел "Stage Selection" …

# стало:
2. Если подходящий skill найден, вызовите skills.get_skill. Изучите раздел "Stage Selection" …
```

**Правило 2a** (строка 1759) — порог «больше двух шагов» без «даже один нетривиальный
вызов» (мастер-план шаг 4), убрать капс; подстроки `больше двух шагов` (закреплена тестом
`tests/test_chat_modes_registry.py:125`) и `agent_tools.manage_plan_tasks`
(зависимость `agent_tools.py:411`) сохранены дословно:
```
# было:
2a. Перед первым execution-tool в задаче ОБЯЗАТЕЛЬНО создайте план через agent_tools.manage_plan_tasks (action=add) с полным definition_of_done: goal, success_criteria и verification. Это требование сервера: если задача ожидаемо займёт больше двух шагов (или даже один нетривиальный execution-вызов), без плана первый execution-tool будет принудительно перенаправлен на manage_plan_tasks и итерация потеряется. …

# стало:
2a. Перед первым execution-tool в задаче создайте план через agent_tools.manage_plan_tasks (action=add) с полным definition_of_done: goal, success_criteria и verification. Это требование сервера: если задача ожидаемо займёт больше двух шагов, без плана первый execution-tool будет принудительно перенаправлен на manage_plan_tasks и итерация потеряется. …
```

**Правило 14** (если оставляем по умолчанию, §9), убрать капс:
```
# было:
14. Скрипты, принадлежащие активному skill, ЗАПРЕЩЕНО запускать через codeinterpreter.deep_analysis …

# стало:
14. Скрипты, принадлежащие активному skill, нельзя запускать через codeinterpreter.deep_analysis …
```

### 6.3 Разбивка на 4 раздела (мастер-план шаг 5: «роль / порядок работы / безопасность /
формат ответа»)

Текущая структура — 2 раздела («Основной цикл работы», 15 пунктов + подпункты 2a/2b/2b.1/2c;
«Правила безопасности и качества», 17 пунктов, смешивающих script-safety, response-style и
финальный чек-лист). Новая структура — 4 заголовка, перераспределение существующих пунктов
БЕЗ переписывания их текста (кроме уже перечисленных выше точечных правок и 3 удалений):

1. **«Роль и принцип работы»** — вступительный абзац + абзац про PPTX (без изменений,
   сейчас уже идут первыми, до заголовка «Основной цикл работы»).
2. **«Порядок работы»** — весь текущий список «Основной цикл работы» 1-15 (с 2a/2b/2b.1/2c),
   без изменения состава, только правки 2/2a из §6.2.
3. **«Безопасность»** — из текущего списка «Правила безопасности и качества» пункты
   **1, 2, 3, 4, 12, 13, 14(флаг), 17** (script/install safety + финальный чек-лист перед
   `deliver_to_user`), перенумеровать 1..7(или 8, если 14 остаётся).
4. **«Формат ответа»** — из текущего списка пункты **5, 6, 7, 8, 9, 11** (10 удалено,
   15/16 удалены и не попадают ни в один раздел), перенумеровать 1..6.

Итог: было 15+17=32 пронумерованных пункта в 2 разделах; станет 15 + 7(или 8) + 6 = 28(или
29) пунктов в 4 разделах (минус удалённые 10/15/16, минус возможное удаление 14).

---

## 7. bot/skills/**: правки

### 7.1 sequential-thinking — добавить frontmatter

`bot/skills/sequential-thinking/SKILL.md` (5725 байт, единственный файл в директории) не
имеет YAML frontmatter (файл начинается сразу с `# Sequential Thinking MCP Skill`).
`_parse_skill_markdown()` (`bot/plugins/skills.py:1064-1078`) в этом случае тихо возвращает
`{}` без падения — но каталог skills получает пустое `name`/`description` для этого скилла.
Добавить в начало файла:
```yaml
---
name: sequential-thinking
description: MCP server for structured, step-by-step reasoning through complex problems.
---
```
(текст ниже уже существующего заголовка `# Sequential Thinking MCP Skill` не трогать).

### 7.2 Шесть дублирующихся пар `X` vs `META-SKILLS/X`

Подтверждено `diff -q` (все 6 пар различаются по содержимому) для: `belief-examination`,
`board-of-directors`, `decision-framework`, `multi-agent-brainstorm`, `project-planning`,
`storytelling-structure`. `_iter_skill_paths()` (`bot/plugins/skills.py:1030-1054`) находит
`META-SKILLS/X` как отдельный skill id (id = POSIX-путь относительно `skills_dir`,
`META-SKILLS` не входит в список исключаемых частей пути) — то есть сейчас модель видит **в
каталоге по 2 разных, местами противоречащих друг другу описания** на каждую из 6 тем.

Решение (обоснование — стиль/качество контента у версий META-SKILLS последовательнее,
таймстемпы SKILL.md кластеризуются вокруг 2026-05-10 у всех 6 META-версий против разброса
03-31…05-06 у top-level версий, и как минимум одна reference-пара
(`project-planning/references/linear-project.md` vs
`META-SKILLS/project-planning/references/execution-project.md`) в META-версии специально
переименована, чтобы не путать с реальным Linear.app — то есть META-версии — более поздняя,
осознанная переработка):
1. Взять содержимое `META-SKILLS/X/SKILL.md` (и его `references/`, если есть) как
   каноническое.
2. Физически переместить его в `bot/skills/X/` (заменить старый top-level файл).
3. Перенести файлы, уникальные для top-level версии (если такие есть — проверить `find`
   по каждой паре перед удалением, не предполагать заранее).
4. Удалить `bot/skills/META-SKILLS/X/` целиком.
5. Для `decision-framework` **не переносить** 20 файлов `bot/skills/decision-framework/
   from_*` — они не референсятся из самого `decision-framework/SKILL.md` (проверено:
   `grep -c "from_"` внутри SKILL.md — 0 совпадений) и дублируют темы отдельных
   META-SKILLS-скиллов (`career-strategy`, `negotiation-prep`, `conflict-resolution`,
   `reflection-postmortem`, `root-cause-analysis` — все уже есть как самостоятельные
   `META-SKILLS/*` записи не из дублирующихся 6 пар). Перенос `from_*` воссоздал бы ту же
   проблему дублирования, которую эта задача устраняет.

Это решение — по 6 конкретным парам, основанное на прочитанном содержимом, а не универсальное
правило «наследовать META-SKILLS всегда»; если у ревьюера другие приоритеты по конкретной
паре — пункт 5 (`decision-framework`/`from_*`) самый спорный и стоит подтвердить отдельно
(см. §9).

### 7.3 decision-framework — описание ≤240 символов

Текущее top-level описание — 1421 символ (русское, состоит из длинного списка триггерных
фраз). Текущее META-описание — 553 символа (английское, короче и по существу, но тоже
превышает лимит рантайм-обрезки `bot/plugins/skills.py:179-180,282-283`:
`if len(desc) > 240: desc = desc[:240].rstrip() + "..."`). Раз каноническим контентом
становится META-версия (§7.2), новое описание нужно писать в её духе — короткое, на
английском (весь корпус `bot/skills/*/SKILL.md`, кроме 6 дублирующихся пар, уже пишет
description по-английски — проверено: `arxiv`, `humanizer`, `nano-pdf`,
`research-paper-writing` и др., все в диапазоне 46-274 символов, английский). Черновик (218
символов, проверено `len()`):
```yaml
description: Use this skill when the user must choose: whether to act, which option to take, or how to reason through trade-offs under uncertainty. Not for execution steps, long-horizon planning, bargaining, or relationship repair.
```

---

## 8. Тесты

### 8.1 `tests/test_chat_modes_registry.py` — правки существующего файла

- `test_skills_agent_mode_is_registered()` (строка 110-139): **удалить** строку 136
  (`assert "Никогда не выводите служебные reasoning-теги" in mode["prompt_start"]`) — эта
  фраза удаляется из промпта (§6.1). Строки 119-135, 137-139 не трогать — они пинят
  подстроки, которые остаются дословно (`skills.list_skills`, `agent_tools.deliver_to_user`,
  `больше двух шагов`, `agent_tools.manage_plan_tasks`, `Никогда не выводите сырые
  результаты tools` (безопасность-11, оставлена), `Не выдумывайте абсолютные пути`
  (безопасность-12, не трогается), `не повторяйте тот же вызов` (безопасность-13, не
  трогается) — все проверено чтением текущего файла в этой сессии).
- `test_skills_agent_mode_is_detected_by_prompt_markers()` (строка 152-163) — не зависит от
  реального `prompt_start`, использует синтетический текст с теми же двумя маркерами; не
  трогать.
- Добавить новые тесты:
  - `test_shared_blocks_are_substituted_and_hidden()` — временный `chat_modes.yml` в
    `tmp_path` с `shared_blocks: {foo: "bar"}` и режимом, содержащим `{{shared:foo}}` в
    `prompt_start`; проверить, что `get_mode_by_key(...)["prompt_start"]` содержит `bar`, а
    `all_modes()` / `get_all_modes_list()` не содержат ключа `shared_blocks`.
  - `test_web_search_rules_shared_block_appears_once_in_real_yaml()` — грузит настоящий
    `bot/chat_modes.yml`, считает вхождения канонической фразы из `web_search_rules` (или
    маркер после подстановки) суммарно по всем `prompt_start` — должно быть ровно 23 (по
    числу режимов, где блок использовался), а не 0 и не рассинхрон.
  - `test_real_chat_modes_yaml_has_no_typo()` — грузит настоящий файл, проверяет отсутствие
    подстроки «исползуй» во всех `prompt_start`.

### 8.2 `tests/test_database.py` — новые тесты миграции (только добавление, владение
ограничено «миграция mode_key»)

- Сессия со старым текстом (взятым из `LEGACY_PROMPT_FINGERPRINTS`, например
  `assistant`-текст ДО правки — фикстура должна захардкодить именно старый текст, не читать
  текущий `chat_modes.yml`, иначе тест перестанет проверять миграцию после правки текстов)
  без `mode_key` в БД → после `init_db()`/миграции `context.messages[0].mode_key ==
  "assistant"`, а инструменты/режим по-прежнему резолвятся (через
  `chat_modes_registry.get_mode_by_key`).
- Идемпотентность: повторный вызов миграции (или повторный `init_db()` на той же БД) не
  меняет уже проставленный `mode_key` и не увеличивает `version` повторно.
- Регрессия на §3.3: после применения миграции 3 `schema_version` таблица содержит `3`;
  повторный `_reconcile_schema_version_with_shape()` на этой же БД НЕ удаляет запись версии
  3 (проверяет фикс, иначе миграция запускалась бы на каждом старте).
- Сессия с сообщением не-system первым или с уже существующим `mode_key` — миграция не
  трогает (no-op), контент байт-в-байт не меняется.

### 8.3 Новые тесты skills-метаданных

Новый файл (например `tests/test_skills_metadata.py`, владение — «новые тесты
skills-метаданных» по мастер-плану) или расширение существующего `test_skills_plugin.py`
(вне владения T06 — не трогать, если он уже покрывает это; сначала проверить содержимое
перед принятием решения о новом файле, чтобы не задваивать):
- каждый `SKILL.md` под `bot/skills/**` (после реорганизации §7.2) имеет непустые `name` и
  `description` после парсинга через ту же логику, что `_parse_skill_markdown` /
  `_parse_flat_frontmatter` (или напрямую через `SkillsPlugin`, если тест их создаёт).
- skill id не дублируются (после удаления `META-SKILLS/X` дублей для 6 пар — сет id
  уникален по построению, тест фиксирует регрессию).
- `decision-framework` описание ≤240 символов.

---

## 9. Риски / открытые допущения

1. **Правило безопасности-14 (skills_agent) — не входит в список из 3 пунктов, которые
   явно назвал заказчик** (`<think>`, `fs.writeFile`, «validator rejects»), хотя код
   (`bot/skill_script_routing.py:109-115`) действительно блокирует то же самое безусловно.
   План по умолчанию **оставляет** правило 14 (только снимает капс) — удалять его нужно
   отдельным явным решением, не мимоходом внутри T06.
2. **6 пар skills-дублей**: решение «взять META-SKILLS-версию, перенести на top-level
   путь» — по содержательному чтению, не механический алгоритм; самый спорный подпункт —
   отказ переносить `decision-framework/from_*` (20 файлов). Нужно подтверждение перед
   исполнением, если ревьюер видит причину их сохранить.
3. **decision-framework описание** — переход с русского (1421 симв.) на английский (черновик
   218 симв.) текст меняет язык, а не только длину; обоснование — соответствие остальному
   корпусу `bot/skills/*` (все не-дублирующиеся SKILL.md пишут description по-английски).
   Если у проекта есть непротиворечащая этому политика «весь user-facing текст по-русски»,
   это нужно уточнить — она не была видна в исследованных файлах.
4. `tests/test_openai_helper_tool_calls.py:1503` (`"больше двух шагов" in prompt`,
   тест `test_auto_chat_mode_prompt_routes_by_complexity_not_keywords`,
   строки 1485-1505) — **проверено и безопасно**: этот текст берётся не из
   `chat_modes.yml`, а из отдельного захардкоженного шаблона внутри
   `OpenAIHelper._build_auto_chat_mode_prompt` (`bot/openai_helper.py`, вне владения T06);
   `helper.chat_modes_registry` в этом тесте — `types.SimpleNamespace` со синтетическим
   `get_all_modes_list`. Правки `chat_modes.yml` в T06 этот тест не затрагивают.
5. `assistant` правило 12 — предложенная замена «переключитесь на режим Code Assistant через
   меню /mode» — точная команда/UX переключения режима не проверялась (вне владения T06);
   формулировку нужно свериться с реальным способом смены режима в боте перед коммитом,
   важно только не оставить несуществующий `/reset code_assistant`.
6. Нумерация правил внутри `assistant` и внутри `skills_agent`-разделов после
   удалений/перегруппировки — чисто механическая правка, но именно из-за этого easy to
   mis-count вручную; при реализации стоит сверить финальную нумерацию построчно перед
   коммитом.

---

## 10. Приёмочные команды

```bash
~/.venvs/ctb/bin/python -m pytest tests/test_chat_modes_registry.py tests/test_database.py \
  tests/test_skills_prompt_fragment.py tests/test_skills_plugin.py -q --no-header -p no:cacheprovider
# (+ новый файл skills-метаданных, если создан отдельно)
~/.venvs/ctb/bin/python -m pytest tests/test_openai_helper_tool_calls.py -q --no-header -p no:cacheprovider
# ожидается зелёным без изменений — регрессионная проверка риска §9.4
~/.venvs/ctb/bin/python -m ruff check bot/chat_modes_registry.py bot/database.py
python3 -m mypy bot/chat_modes_registry.py bot/database.py --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports
```
