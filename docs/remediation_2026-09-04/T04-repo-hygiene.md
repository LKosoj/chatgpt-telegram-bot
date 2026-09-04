# T04. Гигиена репозитория

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел «T04. Гигиена
репозитория»). Роль этого документа — план для разработчика; сам код не менялся, только
проверено чтением.

## Цель

1. Убрать из git-индекса файл `bot/plugins/pdf_cache/cache_metadata.json` — это runtime-кэш
   плагина `ask_your_pdf`, который случайно попал в коммит из-за бага в `__init__` (см. п. 2),
   и закрыть класс файлов `bot/plugins/pdf_cache/*` через `.gitignore`, не трогая при этом
   файл на диске и не ломая работу плагина в runtime.
2. Устранить причину появления этого файла: `AskYourPDFPlugin.__init__` создаёт каталоги и
   файл метаданных кэша на диске сразу при инстанцировании класса — ещё до вызова
   `initialize()`, то есть до того, как известен настоящий `storage_root`. Перенести создание
   каталогов/файла метаданных в `initialize()`, сохранив рабочее поведение при всех путях
   вызова (с `storage_root` и без, с `initialize()` и без него).
3. `IMG_3980.jpg` — по итогам проверки **не удалять** (см. «Находка» ниже): пункт исходной
   задачи расходится с фактическим состоянием кода, найдена живая ссылка на файл.

## Находка, меняющая пункт 3 задачи

Задача формулировала пункт 3 как `git rm IMG_3980.jpg (проверь, что на него нет ссылок)`.
Проверка показала обратное — ссылка есть и она рабочая:

- `bot/telegram_bot.py:4837` — `send_inline_query_result()` использует файл как
  `thumbnail_url` инлайн-ответа Telegram (режим `@botusername <query>`):

  ```python
  thumbnail_url='https://github.com/LKosoj/chatgpt-telegram-bot/blob/main/IMG_3980.jpg?raw=true',
  ```

- Это не локальный путь, а сырой (`raw=true`) URL на GitHub, указывающий на файл в ветке
  `main` репозитория `LKosoj/chatgpt-telegram-bot`. Текущий `origin` этого клона —
  `https://github.com/LKosoj/chatgpt-telegram-bot.git` (проверено `git remote -v`), то есть
  это тот же самый репозиторий, а не сторонний форк.
- Такой URL отдаёт содержимое файла из **текущего дерева ветки `main` на GitHub**, а не из
  локальной копии на диске. Поэтому `git rm --cached` (снять с индекса, оставить на диске)
  тут не спасает: если изменение когда-нибудь закоммитят и смёржат/запушат в `main`, файл
  пропадёт из дерева `main`, и URL начнёт отдавать 404 — вне зависимости от того, использован
  ли `--cached` или обычный `git rm`. Разница между ними — только в том, останется ли файл в
  локальной рабочей копии, на судьбу самого GitHub-URL это не влияет.
- Верхняя сводка `docs/audit_remediation_plan_2026-09-04.md` (преамбула) говорит про
  `IMG_3980.jpg` мягче — «убрать... из индекса» — что скорее похоже на `--cached`
  (по аналогии с `cache_metadata.json` из этой же фразы), тогда как раздел T04 требует
  буквально `git rm` (с диска тоже). Формулировки расходятся между собой; ни одна не
  учитывает найденную ссылку.

**Рекомендация:** пункт 3 не выполнять в этом заходе — оставить `IMG_3980.jpg` как есть
(и в индексе, и на диске). Причина: файл маленький (88 КБ), явно используется в проде для
превью инлайн-результата, а его удаление — это уже не «гигиена репозитория», а полноценное
изменение поведения бота (пропадёт картинка в инлайн-ответах), которое требует отдельного
решения владельца и правки `bot/telegram_bot.py:4837`, а не просто `git rm`.

**Альтернатива (если владелец подтвердит, что превью не нужно):** одним связанным изменением
— убрать строку `thumbnail_url=...` (или заменить на другой хостинг картинки) в
`bot/telegram_bot.py:4837` **и** `git rm IMG_3980.jpg`. Это отдельная от T04 по духу правка
(меняет поведение фичи, не только репозиторий), поэтому в план T04 её включать не стал —
предлагаю заводить отдельной задачей, если будет решение убрать превью.

## Точные правки

### 1. `.gitignore` — добавить каталог кэша `ask_your_pdf`

Файл: `/srv/git_projects/chatgpt-telegram-bot/.gitignore`. Сейчас (полностью, 33 строки,
без завершающего перевода строки на 33-й):

```
1  __pycache__
2  /.idea
3  .env
4  .DS_Store
5  /usage_logs
6  venv
7  .venv/
8  /.cache
9  .vscode/lua.booster.lint.json
10 private.txt
11 bot/test.py
12 bot/file.json
13 .vscode/extensions.json
14 bot/plugins/reminders.json
15 bot/.vscode/lua.booster.lint.json
16 bot/plugins/.cursorrules
17 /.specstory
18 bot/user_data.db
19 bot/user_data.db-shm
20 bot/user_data.db-wal
21 /.vscode
22 data/
23 bot/skills/
24 presentation_analysis_*.json
25 presentation_analysis_*.txt
26 # SpecStory explanation file
27 .specstory/.what-is-this.md
28 .cli-proxy/
29 .ai_docs_cache/
30 .ai-docs/
31 .attachments/
32 .cli-proxy/runtime/
```

Правка: вставить новую строку `bot/plugins/pdf_cache/` после строки 16
(`bot/plugins/.cursorrules`), рядом с уже существующими точечными исключениями плагинов
(`bot/plugins/reminders.json`, `bot/plugins/.cursorrules`), например:

```diff
 bot/plugins/reminders.json
 bot/.vscode/lua.booster.lint.json
 bot/plugins/.cursorrules
+bot/plugins/pdf_cache/
 /.specstory
```

Замечание: `data/` (строка 22) уже покрывает реальный `storage_root` в проде
(`PLUGIN_STORAGE_ROOT` или `<repo>/data` по умолчанию, см. `bot/plugin_manager.py:68-70`) —
после правки п. 2 плагин будет писать кэш именно туда. Новый паттерн
`bot/plugins/pdf_cache/` нужен только чтобы закрыть путь `bot/plugins/` (использовался как
резервный путь до `initialize()` и остаётся резервным путём, когда `initialize()` вызывают
без `storage_root`, — см. тест `bot/tests` / контрактный тест ниже).

### 2. `bot/plugins/ask_your_pdf.py` — не создавать каталоги/файл в `__init__`

Файл: `bot/plugins/ask_your_pdf.py`. Текущий код, строки 32–65:

```python
32	    def __init__(self):
33	        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp_pdfs")
34	        self.cache_dir = os.path.join(os.path.dirname(__file__), "pdf_cache")
35	        self.cache_metadata_path = os.path.join(
36	            self.cache_dir,
37	            "cache_metadata.json",
38	        )
39	        self.max_cache_size_mb = 500
40	        self.max_cache_age_days = 10
41	        self.max_extracted_text_chars = 50000
42	
43	        os.makedirs(self.temp_dir, exist_ok=True)
44	        os.makedirs(self.cache_dir, exist_ok=True)
45	        self._init_cache_metadata()
46	
47	    def initialize(
48	        self,
49	        openai=None,
50	        bot=None,
51	        storage_root: str | None = None,
52	    ) -> None:
53	        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
54	        if not hasattr(self, "max_extracted_text_chars"):
55	            self.max_extracted_text_chars = 50000
56	        if storage_root:
57	            self.temp_dir = os.path.join(storage_root, "temp_pdfs")
58	            self.cache_dir = os.path.join(storage_root, "pdf_cache")
59	            self.cache_metadata_path = os.path.join(
60	                self.cache_dir,
61	                "cache_metadata.json",
62	            )
63	            os.makedirs(self.temp_dir, exist_ok=True)
64	            os.makedirs(self.cache_dir, exist_ok=True)
65	            self._init_cache_metadata()
```

Заменить на (только переносим 3 вызова с файловым I/O — `os.makedirs` x2 и
`_init_cache_metadata()` — из `__init__` в конец `initialize()`, без условия `if storage_root:`
вокруг них, чтобы каталоги создавались и при вызове `initialize()` без `storage_root`, как
раньше делал `__init__` по умолчанию):

```python
    def __init__(self):
        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp_pdfs")
        self.cache_dir = os.path.join(os.path.dirname(__file__), "pdf_cache")
        self.cache_metadata_path = os.path.join(
            self.cache_dir,
            "cache_metadata.json",
        )
        self.max_cache_size_mb = 500
        self.max_cache_age_days = 10
        self.max_extracted_text_chars = 50000

    def initialize(
        self,
        openai=None,
        bot=None,
        storage_root: str | None = None,
    ) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        if not hasattr(self, "max_extracted_text_chars"):
            self.max_extracted_text_chars = 50000
        if storage_root:
            self.temp_dir = os.path.join(storage_root, "temp_pdfs")
            self.cache_dir = os.path.join(storage_root, "pdf_cache")
            self.cache_metadata_path = os.path.join(
                self.cache_dir,
                "cache_metadata.json",
            )
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_cache_metadata()
```

Пояснение простыми словами: раньше плагин создавал папки на диске и писал JSON-файл сразу в
момент, когда Python создаёт объект плагина (`__init__` — это метод, который вызывается
первым, при создании объекта). Но на этот момент ещё не известно, где реально должен лежать
кэш (`storage_root` — правильная папка для хранения данных плагина, её сообщает
`PluginManager` позже, через отдельный вызов `initialize()`). Из-за этого папка и файл сначала
создавались в «неправильном» месте — `bot/plugins/pdf_cache/` — и один такой файл случайно
попал в git. Правка убирает файловые операции (создание папок, запись файла) из `__init__` и
переносит их в конец `initialize()`, когда все нужные пути уже посчитаны — там уже стоит
проверка `if storage_root:` для выбора правильной папки, я её не трогаю, а операции создания
файлов ставлю после неё безусловно (совпадает с тем, что раньше делал `__init__` по
умолчанию, до этой правки).

Почему это безопасно (проверено чтением и локальным прогоном на копии в `/tmp`, вне
репозитория — правка в самом репозитории не делалась):

- `PluginManager.storage_root` в реальном рантайме **никогда не бывает пустым**
  (`bot/plugin_manager.py:68-70`: берётся `PLUGIN_STORAGE_ROOT` или `<repo>/data` по
  умолчанию), поэтому боевой путь всегда идёт через ветку `if storage_root:`.
- `PluginManager.get_plugin()` (`bot/plugin_manager.py:683-697` и `:1268-1283`) гарантирует
  вызов `initialize()` (через `_call_initialize`) перед тем, как отдать инстанс плагина на
  `execute()` — либо при первом создании (когда `self.openai or self.storage_root` истинно —
  а `storage_root` всегда истинно), либо при повторной выдаче закэшированного инстанса, если
  `not hasattr(instance, 'openai') or not instance.openai`. То есть в обычном потоке запроса
  `execute()` никогда не вызывается на инстансе, у которого не было `initialize()`.
- Единственное место, где плагин создаётся напрямую через `plugin_class()` без `initialize()`
  — `tests/test_plugin_descriptions_contract.py:127-129` (`_instantiate`) и
  `PluginManager._get_or_create_bare_instance` (`bot/plugin_manager.py:1280-1291`, используется
  для `register_schema()` при старте). Оба места вызывают только `get_spec()` /
  `register_schema()` — ни один не трогает `cache_dir`/`temp_dir`/`cache_metadata_path`, так
  что для них ничего не меняется.
- `tests/test_ask_your_pdf.py` (единственный тест, который реально исполняет `execute()`)
  создаёт плагин через `object.__new__(module.AskYourPDFPlugin)` (то есть `__init__` вообще не
  вызывается — см. `_plugin()`, строки 41-46) и сразу зовёт `plugin.initialize(storage_root=str(tmp_path))`.
  Правка ничего не меняет для этого пути: `initialize()` как и раньше получает непустой
  `storage_root` и создаёт каталоги/файл.
- Прогон на копии репозитория в `/tmp` (не в рабочем дереве) с применённой правкой показал: (а)
  `AskYourPDFPlugin()` без `initialize()` не создаёт директорий на диске; (б) `get_spec()`
  работает без директорий; (в) `initialize(storage_root=...)` создаёт `temp_dir`, `cache_dir`
  и `cache_metadata.json` как раньше; (г) `initialize()` без `storage_root` тоже создаёт
  дефолтные каталоги под `bot/plugins/`, то есть поведение «по умолчанию» не потерялось; (д)
  весь `tests/test_ask_your_pdf.py` (9 тестов) зелёный.

## Команды

Выполнять из корня репозитория `/srv/git_projects/chatgpt-telegram-bot`.

```bash
# 1. Снять cache_metadata.json с индекса, файл остаётся на диске
git rm --cached bot/plugins/pdf_cache/cache_metadata.json

# 2. Добавить исключение в .gitignore (правка редактором/патчем, см. диф выше)
#    — вручную вставить строку `bot/plugins/pdf_cache/` после `bot/plugins/.cursorrules`

# 3. Правка bot/plugins/ask_your_pdf.py — см. точный код выше (перенос 3 строк из __init__
#    в конец initialize(), без условия if storage_root: вокруг них)

# 3.b) IMG_3980.jpg — НЕ трогать (см. «Находка» выше). Никаких git rm по этому файлу.

# Проверка после правок
git status --short
python3 -m pytest tests/test_ask_your_pdf.py -q -p no:cacheprovider
```

`git rm --cached` не запускалось из этого документа — команда для разработчика на следующем
шаге процесса (планировщику менять репозиторий запрещено).

## Проверка

1. `git status --short` — должно показать:
   - `D  bot/plugins/pdf_cache/cache_metadata.json` (staged delete из индекса) **или**
     `bot/plugins/pdf_cache/cache_metadata.json` уже отсутствует в `git ls-files`, при этом
     `ls bot/plugins/pdf_cache/cache_metadata.json` на диске всё ещё находит файл;
   - `M  .gitignore`;
   - `M  bot/plugins/ask_your_pdf.py`;
   - никаких изменений по `IMG_3980.jpg`.
2. `git ls-files | grep pdf_cache` — после `git rm --cached` не должен возвращать
   `bot/plugins/pdf_cache/cache_metadata.json`.
3. `ls -la bot/plugins/pdf_cache/` — файл `cache_metadata.json` остался на диске (рантайм не
   пострадал, следующий импорт плагина найдёт валидный файл или создаст новый через
   `_init_cache_metadata()`, который не перезаписывает существующий файл — см.
   `if not os.path.exists(self.cache_metadata_path):` в `_init_cache_metadata()`).
4. `python3 -m pytest tests/test_ask_your_pdf.py -q -p no:cacheprovider` — все 9 тестов
   зелёные (базовый прогон без правок уже даёт 9 passed).
5. Дополнительно (регрессия по остальным тестам, которые трогают `PluginManager` и общий
   список плагинов, раз меняется класс, инстанциируемый везде): `python3 -m pytest
   tests/test_plugin_manager.py tests/test_plugin_descriptions_contract.py -q
   -p no:cacheprovider`.
6. Ручная проверка (не обязательна, но полезна): удалить `bot/plugins/pdf_cache/` целиком и
   `bot/plugins/temp_pdfs/` на диске, затем в python-консоли создать
   `AskYourPDFPlugin()` — директории **не** должны появиться; затем вызвать
   `.initialize(storage_root='/tmp/whatever')` — директории и `cache_metadata.json` должны
   появиться по новому пути.

## Риски

- **Расхождение с буквальной формулировкой T04 по `IMG_3980.jpg`.** Пункт задачи предполагал
  безусловный `git rm`, но найдена рабочая ссылка (`bot/telegram_bot.py:4837`). Если это
  решение не устроит владельца — можно either (a) оставить как в этом плане (не трогать), либо
  (b) явно попросить подтверждения на комбинированную правку (убрать ссылку в
  `telegram_bot.py` + удалить файл) отдельной задачей.
- **`.gitignore` и уже закоммиченные файлы.** Добавление `bot/plugins/pdf_cache/` в
  `.gitignore` само по себе не убирает `cache_metadata.json` из истории/индекса — это делает
  отдельный `git rm --cached`. Если бы `git rm --cached` забыли выполнить, `.gitignore` не дал
  бы эффекта для уже отслеживаемого файла (git продолжает отслеживать once-tracked файлы,
  игнорируя их только для *новых* попаданий в индекс).
- **`ai_docs_site/` содержит сгенерированную HTML-страницу**
  (`ai_docs_site/configs/files/bot/plugins/pdf_cache/cache_metadata__json.html`) с копией
  содержимого `cache_metadata.json`. Она станет slightly stale (описывает файл, который больше
  не отслеживается), но `ai_docs_site/` явно исключён из T04 верхней сводкой плана
  («не трогаем») — оставляю как есть, только упоминаю здесь, чтобы это не было сюрпризом на
  ревью.
- **Другие потребители `cache_dir`/`temp_dir`/`cache_metadata_path`.** Все места, где эти три
  атрибута читаются или пишутся (`_init_cache_metadata`, `_update_cache_metadata`,
  `generate_file_hash` — не использует их, `_cache_file_path`, `load_cache`, `save_cache`,
  `_is_path_in_controlled_storage`, `execute()`), находятся в том же файле после метода
  `initialize()` и ни один не завязан на то, что каталог создан именно в `__init__` —
  все они просто открывают файлы по путям, которые уже валидны после `initialize()`. Прочитаны
  все вхождения `self.cache_dir` / `self.temp_dir` / `self.cache_metadata_path` в файле —
  других создателей каталогов, кроме двух `os.makedirs` в текущем `__init__`/`initialize()`,
  нет.
- **Плагин с ещё не вызванным `initialize()`, но уже вызванным `execute()`.** Теоретически
  возможно только при обходе `PluginManager.get_plugin()` — в дереве такого пути не найдено
  (командный обработчик и `openai_tool_handler` получают инстанс плагина только через
  `get_plugin()`/аналоги, которые гарантируют `initialize()` — см. «Почему это безопасно»
  выше). Если появится новый способ получить инстанс в обход `PluginManager`, разработчику
  нужно будет либо тоже гарантировать `initialize()`, либо добавить ленивую инициализацию
  каталогов в сами методы (как сделано в `agent_tools.py` через guard
  `if not self.pending_file or not os.path.exists(...)`, `bot/plugins/agent_tools.py:2235`) —
  в этой правке я специально не стал вводить такой guard, потому что для `ask_your_pdf` не
  нашлось ни одного реального пути, где он был бы нужен, а лишний guard — это код без
  доказанной необходимости (см. правило «Simplicity First»).

## Критерии готовности

- `bot/plugins/pdf_cache/cache_metadata.json` не отслеживается git (`git ls-files` его не
  показывает), но физически присутствует на диске без изменений содержимого.
- `.gitignore` содержит строку `bot/plugins/pdf_cache/`.
- `bot/plugins/ask_your_pdf.py`: `__init__` не содержит `os.makedirs` и не вызывает
  `_init_cache_metadata()`; оба `os.makedirs` и вызов `_init_cache_metadata()` выполняются
  из `initialize()` при любом способе его вызова (с `storage_root` и без).
- `IMG_3980.jpg` не тронут: остаётся и в индексе, и на диске, `bot/telegram_bot.py:4837`
  не менялся.
- `python3 -m pytest tests/test_ask_your_pdf.py -q -p no:cacheprovider` — 9 passed.
- `git status --short` показывает только ожидаемые три изменения (delete из индекса
  `cache_metadata.json`, правка `.gitignore`, правка `ask_your_pdf.py`) и ничего лишнего
  (никаких случайно подхваченных файлов).


## Постскриптум после реализации (2026-09-04)

Решение владельца: `IMG_3980.jpg` всё-таки удалён (`git rm`), и одновременно из
`bot/telegram_bot.py` (`send_inline_query_result`) убран аргумент `thumbnail_url`, который
ссылался на этот файл в GitHub. Пункты плана «не трогать IMG_3980.jpg» считать отменёнными.
