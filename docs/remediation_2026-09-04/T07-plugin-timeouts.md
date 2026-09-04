# T07. Блокирующие вызовы (HTTP/subprocess) в плагинах

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел «Волна 1», T07);
исходные находки — `docs/architecture_code_review_2026-09-04.md` §3.9 и §4.2.

Пояснение термина **event loop** (петля событий): это единственный поток, в котором asyncio
по очереди выполняет все `async def`-корутины бота — обработку сообщений всех пользователей,
таймеры, фоновые задачи. Если внутри `async def` вызвать обычную (синхронную) функцию, которая
сама «спит» или ждёт сеть (`requests.get` без `timeout`, `subprocess.run`, `time.sleep`), весь
бот встаёт на это время: ни одно другое сообщение не обработается, пока вызов не завершится (а
если он вообще не завершится — бот зависает навсегда для всех).

Пояснение термина **timeout** (таймаут): максимальное время ожидания ответа; без него запрос
может ждать бесконечно.

Пояснение термина **`asyncio.to_thread`**: способ выполнить синхронную функцию в отдельном
потоке ОС, не блокируя event loop — сам вызов внутри потока может по-прежнему быть медленным
или зависнуть без timeout, поэтому `to_thread` не заменяет timeout, а дополняет его (без
timeout поток просто никогда не освободится и рано или поздно кончится пул потоков).

## Цель

Убрать блокировки event loop и утечку сетевого соединения в плагинах:

1. `bot/plugins/webshot.py:37,40` — `requests.get` без общего таймаута на «прогревочном»
   вызове + сам вызов выполняется прямо на event loop; `except Exception: os.remove(...)`
   может сам бросить исключение и замаскировать исходную ошибку.
2. `bot/plugins/movie_info.py:117,143,169,205` — четыре `requests.get` без `timeout`, вызванные
   синхронно из `async execute`.
3. `bot/plugins/chief.py:191-194` — `aiohttp.ClientSession` создаётся лениво в
   `_ensure_session()` и никогда не закрывается — нет `close_async()`.
4. Дополнительно найдено сканированием всех `bot/plugins/*.py` (см. ниже) —
   `bot/plugins/show_me_diagrams.py:220,261` — `subprocess.run` (запуск `java -jar
   plantuml.jar`) без `timeout`, напрямую в `async def _generate_plantuml`, на пути каждого
   вызова любого типа диаграммы.

Код в этом плане не меняется — правки выполняет следующий агент (разработчик) по этому файлу.

## Как выполнялось сканирование

Скрипт на AST (абстрактное синтаксическое дерево — разбор кода в структуру, а не текстовый
grep, что не путает вызов в закомментированном коде и точно определяет, из какой функции сделан
вызов) прошёл по каждому `bot/plugins/*.py` и:
- нашёл все `async def`;
- для каждого — построил граф вызовов до синхронных методов того же класса (BFS — обход
  вширь), чтобы поймать не только прямые вызовы `requests.get` в `async def`, но и вызовы через
  синхронные хелперы (как в `movie_info.py`, где `requests.get` сидит в отдельных `_get_*`
  методах, вызываемых из `execute`);
- отдельно текстовым grep проверены `urllib.request.urlopen`, `subprocess.Popen/.wait()`,
  `.communicate()`, `time.sleep`.

## Таблица находок (все `bot/plugins/*.py`)

| # | Файл:строка | Вызов | `timeout=`? | Как достигается из `async` | Грузится по умолчанию* | Действие |
|---|---|---|---|---|---|---|
| 1 | `webshot.py:37` | `requests.get(image_url)` | нет | прямо в `execute` | да (нет обязательных ключей) | `timeout=` + `to_thread` |
| 2 | `webshot.py:40` | `requests.get(image_url, timeout=30)` | есть | прямо в `execute` | да | только `to_thread` (таймаут уже есть) |
| 3 | `webshot.py:60-61` | `os.remove(image_file_path)` в `except` | — | — | да | `contextlib.suppress(OSError)` |
| 4 | `movie_info.py:117` | `requests.get` в `_get_new_movies` | нет | через синхронный хелпер из `execute` | требует `TMDB_API_KEY` | `timeout=` + `to_thread` на месте вызова хелпера |
| 5 | `movie_info.py:143` | `requests.get` в `_get_movie_details` | нет | то же | требует `TMDB_API_KEY` | то же |
| 6 | `movie_info.py:169` | `requests.get` в `_get_movie_reviews` | нет | то же | требует `TMDB_API_KEY` | то же |
| 7 | `movie_info.py:205` | `requests.get` в `_discover_movies` | нет | то же | требует `TMDB_API_KEY` | то же |
| 8 | `chief.py:191-194` | `aiohttp.ClientSession(...)` создаётся, не закрывается | н/п | — | требует `EDAMAM_APP_ID/APP_KEY` | добавить `close_async()` |
| 9 | `show_me_diagrams.py:220` | `subprocess.run(['java','-jar',...])` | нет | прямо в `_generate_plantuml`, вызывается из всех 7 типов диаграмм | да (нет обязательных ключей; нужен установленный `java`, но это не мешает загрузке плагина) | `timeout=` + `to_thread` |
| 10 | `show_me_diagrams.py:261` | тот же вызов (повтор после попытки исправить ошибку) | нет | то же | да | то же |

\* «Грузится по умолчанию» = при **пустой/несданной** переменной окружения `PLUGINS`
(`bot/plugin_manager.py:223,234` — тогда загружаются все файлы `bot/plugins/*.py`, кроме
`NON_PLUGIN_MODULES` = `__init__.py, plugin.py, background.py, db_handle.py, hooks.py`,
`bot/plugin_manager.py:34`) и когда `__init__` плагина не бросает исключение. `movie_info.py`
(`raise ValueError` без `TMDB_API_KEY`) и `chief.py` (`raise ValueError` без ключей Edamam)
не проходят регистрацию инстанса без ключей — их найдёт `get_functions_specs`, поймает
исключение и молча пропустит (класс всё равно зарегистрирован, но `get_plugin()` возвращает
`None`). `webshot.py` и `show_me_diagrams.py` ключей не требуют — грузятся всегда.

**Важно для приоритезации, проверено в `.env` этого репозитория (значения ключей не
привожу):** в этом конкретном боте `PLUGINS` задан явно и НЕ включает `webshot`, но включает
`movie_info`, `chief`, `show_me_diagrams` — и `TMDB_API_KEY`/`EDAMAM_APP_ID`/`EDAMAM_APP_KEY`
заданы. То есть прямо сейчас в проде активны находки №4-10 (движок фильмов, рецепты, диаграммы),
а находка №1-3 (`webshot`) сейчас неактивна, но становится актуальной при любом изменении
`PLUGINS` на пустой/расширенный список — чинить всё равно нужно всё, так как это общий код
репозитория, а не разовая правка под один деплой.

### Проверено и НЕ требует правки (чтобы не чинили повторно)

- `skills.py:1975,1998` — `subprocess.run(..., timeout=self.install_timeout, ...)` — таймаут
  уже есть, а вызывающая цепочка (`_install_skill_from_external_source` →
  `_materialize_skill_source` → `_clone_git_source[_branch]`) уже обёрнута в
  `await asyncio.to_thread(self._install_skill_from_external_source, ...)` на
  `skills.py:1521-1527`. Эталонный пример того, как это должно быть сделано у остальных.
- `codeinterpreter.py:287`, `mcp_server.py:497,504`, `terminal.py:255,285`,
  `skills.py:2539,2543,3283,3287` — это `await process.wait()` на `asyncio.subprocess`-объекте
  (асинхронный subprocess API, не блокирует event loop сам по себе), не путать с
  `subprocess.run/Popen` из синхронного модуля `subprocess`.
- Отдельного `urllib.request.urlopen(...)` вызовов в `bot/plugins/*.py` не найдено.
- Отдельного `time.sleep(...)` внутри `async def` в `bot/plugins/*.py` не найдено.

## Существующие тесты (что уже есть, что может сломаться)

- `webshot.py`: своего теста плагина нет. Единственное упоминание — `tests/test_docker_runtime_config.py:32`,
  проверяет только наличие `uploads/webshot` в списке путей Dockerfile — не пересекается с
  правкой.
- `movie_info.py`: своего теста плагина нет. Упоминается в `tests/test_plugin_descriptions_contract.py`
  (входит в `PENDING_AUDIT_PLUGINS`, инстанцирование пропускается, если нет `TMDB_API_KEY`) —
  не пересекается.
- `chief.py`: `tests/test_chief_model_choice.py` тестирует только выбор модели в
  `_parse_with_retry` / `_parse_menu_preferences` / `_enhance_recipe` через `FakeHelper`, сессию
  и HTTP-вызовы к Edamam не трогает — не пересекается с добавлением `close_async()`.
  `tests/test_openai_helper_tool_calls.py` использует имя тула `chief.get_recipe` как
  фикстуру для тестов маршрутизации tool-call — код `chief.py` не импортирует и не исполняет,
  не пересекается.
- `show_me_diagrams.py`: тестов нет вообще.

Вывод: ни одна правка не ломает существующий тест — можно добавлять новые тесты, ничего не
адаптируя в старых.

## Правки по `file:line` с кодом

### 1) `bot/plugins/webshot.py`

Импорты (строки 1-6) — добавить `asyncio` (для `to_thread`) и `contextlib` (для
`suppress` — короткая обёртка «проигнорировать исключения этого типа, если возникнут»):

```python
import asyncio
import contextlib
import os
import requests
import random
import string
from typing import Dict
from .plugin import Plugin
```

`execute` (сейчас строки 32-63):

```python
    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            image_url = f'https://image.thum.io/get/maxAge/12/width/720/{kwargs["url"]}'

            # preload url first
            await asyncio.to_thread(requests.get, image_url, timeout=10)

            # download the actual image
            response = await asyncio.to_thread(requests.get, image_url, timeout=30)

            if response.status_code == 200:
                if not os.path.exists("uploads/webshot"):
                    os.makedirs("uploads/webshot")

                image_file_path = os.path.join("uploads/webshot", f"{self.generate_random_string(15)}.png")
                with open(image_file_path, "wb") as f:
                    f.write(response.content)

                return {
                    'direct_result': {
                        'kind': 'photo',
                        'format': 'path',
                        'value': image_file_path
                    }
                }
            else:
                return {'result': 'Unable to screenshot website'}
        except Exception:
            if 'image_file_path' in locals():
                with contextlib.suppress(OSError):
                    os.remove(image_file_path)

            return {'result': 'Unable to screenshot website'}
```

Изменения: строка 37 и 40 — `await asyncio.to_thread(requests.get, image_url, timeout=...)`
вместо прямого `requests.get(...)`; строка 37 получает новый `timeout=10` (раньше не было
вообще никакого); строки 60-61 — `os.remove` обёрнут в `contextlib.suppress(OSError)`.

**Компромисс по значению 10 секунд на «прогревочном» вызове (строка 37) — решение нужно
подтвердить перед реализацией:**
- **Рекомендация (минимальный риск):** `timeout=10`, как сделано выше. Если прогрев не
  успеет за 10 с, сработает общий `except Exception` — плагин просто ответит «не удалось
  сделать скриншот», как и сегодня при любой сетевой ошибке. Это не хуже текущего поведения
  (сейчас прогрев может зависнуть навсегда) и не требует новой ветки кода.
- **Альтернатива:** обернуть только прогревочный вызов в свой `try/except: pass`, чтобы его
  таймаут не отменял основную загрузку (сервис `thum.io` часто отвечает на прогрев с
  задержкой, но продолжает рендерить скриншот в фоне независимо от того, дождался клиент
  ответа или нет). Более корректно с точки зрения UX, но это уже не «минимальная правка» из
  формулировки T07 — отдельная доработка, если увидим в проде ложные отказы.

### 2) `bot/plugins/movie_info.py`

Импорты (строка 1-6) — добавить `asyncio`:

```python
from typing import Dict, Optional, List
import asyncio
import os
import requests
import logging
import random
from .plugin import Plugin
```

Добавить `timeout=10` (используется в проекте как типовое значение для похожих внешних JSON
API — `weather.py:19`, `pravo_gov_ru_api.py:115`, `iplocation.py:17`, `crypto.py:18`) в каждый
из четырёх `requests.get`:

- `movie_info.py:117` (в `_get_new_movies`):
  `response = requests.get(url, params=params, timeout=10)`
- `movie_info.py:143` (в `_get_movie_details`):
  `response = requests.get(url, params=params, timeout=10)`
- `movie_info.py:169` (в `_get_movie_reviews`):
  `response = requests.get(url, params=params, timeout=10)`
- `movie_info.py:205` (в `_discover_movies`):
  `response = requests.get(url, params=params, timeout=10)`

`execute` (строки 215-247, ветка `get_new_movies`) — обернуть вызов синхронного метода в
`asyncio.to_thread`:

```python
            if function_name == 'get_new_movies':
                genre = kwargs.get('genre')
                count = kwargs.get('count', 30)

                movies = await asyncio.to_thread(self._get_new_movies, genre=genre, count=count)
                return {
                    'movies': movies,
                    'genre_filter': genre or 'Все жанры'
                }
```

`execute` (ветка `get_movie_recommendations`, строки 228-247+) — то же для остальных трёх
хелперов:

```python
            elif function_name == 'get_movie_recommendations':
                genre = kwargs.get('genre')
                count = kwargs.get('count', 10)

                movies = await asyncio.to_thread(self._get_new_movies, genre=genre, count=count)
                movies.extend(await asyncio.to_thread(self._discover_movies, genre=genre, count=count))

                movie_data = []
                for movie in movies:
                    movie_id = movie.get('id')
                    if not movie_id:
                        continue

                    details = await asyncio.to_thread(self._get_movie_details, movie_id) or {}
                    reviews = await asyncio.to_thread(self._get_movie_reviews, movie_id)
                    ...
```

(Остальное тело цикла — форматирование `critic_reviews`/`genres`/`movie_data.append(...)` —
не трогать, оно уже чисто синтетическое и не блокирует.)

**Компромисс (по каждому вызову отдельный `to_thread` vs один `to_thread` на весь цикл) —
выбрана более точечная правка:**
- **Рекомендация (сделано выше):** каждый вызов сети оборачивается отдельно. Плюс — минимальный
  диф, структура кода не меняется. Минус — при большом `count` создаётся много коротких
  потоков подряд (по умолчанию `count=10`, то есть до ~21 вызова `to_thread` на одну
  рекомендацию — не критично, пул потоков `asyncio` держит их по умолчанию с запасом).
- **Альтернатива:** вынести тело `for movie in movies: ...` в отдельный синхронный метод
  `_build_movie_data(movies)` и обернуть его целиком в один `await asyncio.to_thread(...)`.
  Меньше потоков, но требует выделения новой функции — больше похоже на рефакторинг, чем на
  точечную правку. Не выбрано, чтобы не расширять диф без необходимости.

### 3) `bot/plugins/chief.py`

Добавить `close_async()` сразу после `_ensure_session` (после строки 194), по образцу
`bot/plugins/hindsight_memory.py:807-821` (тот же паттерн: `getattr` с default, проверка,
что объект существует и не закрыт, try/except на закрытие с логированием):

```python
    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self._api_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_async(self) -> None:
        """Закрывает aiohttp-сессию, лениво созданную `_ensure_session()`.

        Без этого при остановке бота в логах видно предупреждение aiohttp
        "Unclosed client session" — сессия держит открытый TCP-сокет до сборки мусора.
        """
        session = getattr(self, "session", None)
        if session is None or session.closed:
            return
        try:
            await session.close()
        except Exception:
            logging.exception("Failed to close Chief aiohttp session")
```

`close_async()` уже вызывается фреймворком — `bot/plugin_manager.py:745`
(`close_all_async()`), которая, в свою очередь, вызывается на остановке бота
(`bot/telegram_bot.py:5926`). Никаких изменений в `plugin_manager.py`/`telegram_bot.py` не
требуется — правка только в `chief.py`.

### 4) `bot/plugins/show_me_diagrams.py` (дополнительно найдено сканированием)

Импорты (строка 20-27) — добавить `asyncio`:

```python
import os
import asyncio
import tempfile
from typing import Dict, List
import uuid
import subprocess
import logging
from pathlib import Path
```

`_generate_plantuml` (строки 207-268) — оба вызова `subprocess.run` (строка 220 и строка 261)
получают `timeout=` и оборачиваются в `asyncio.to_thread`:

```python
        result = await asyncio.to_thread(
            subprocess.run,
            ['java', '-jar', self.plantuml_jar, '-tpng', puml_file, '-o', temp_dir],
            capture_output=True, text=True, timeout=60, check=False,
        )
```

(идентичная правка для второго вызова на строке 261, внутри цикла повторных попыток).

`subprocess.run(..., timeout=60)` при истечении таймаута бросает
`subprocess.TimeoutExpired` — это новое исключение, которого раньше не было; оно всплывёт из
`_generate_plantuml` в `execute` (строка ~204: `except Exception as e: return {"result":
f"Error generating diagram: {str(e)}"}`), то есть уже перехватывается существующим кодом —
дополнительных `try/except` не требуется.

Значение `60` секунд подобрано по аналогии (запуск JVM + рендер небольшой PNG-диаграммы
обычно укладывается в несколько секунд; проектных констант для таймаута subprocess в
плагинах нет — единственный похожий пример, `skills.py`, использует настраиваемый
`self.install_timeout`, но заводить отдельную конфигурацию под один плагин ради двух вызовов
избыточно для этой правки).

## Тесты

Все — новые файлы (для затронутых плагинов тестов на HTTP/подпроцесс сейчас нет вообще).
Общий паттерн проверки «код действительно уходит через `asyncio.to_thread`, а не блокирует
event loop напрямую» уже есть в проекте — `tests/test_utils_send_long_response_file.py`
(`fake_to_thread`, подменяет `<module>.asyncio.to_thread` на функцию-шпион, которая
записывает вызов и тут же выполняет обёрнутую функцию синхронно). Использовать тот же приём.

### `tests/test_webshot_plugin.py` (новый)

- `test_execute_passes_timeout_and_uses_to_thread`: `monkeypatch.setattr(webshot, "requests",
  fake_requests)` с `fake_requests.get` — шпион, возвращающий `SimpleNamespace(status_code=200,
  content=b"...")`; `monkeypatch.setattr(webshot.asyncio, "to_thread", fake_to_thread)`.
  После `await plugin.execute("screenshot_website", helper=None, url="https://example.com")`
  проверить: `fake_requests.get` вызван дважды; оба вызова прошли через `fake_to_thread`
  (список записанных имён функций содержит `"get"` дважды); у первого вызова
  `kwargs["timeout"] == 10`, у второго `kwargs["timeout"] == 30`.
- `test_write_failure_does_not_raise` (регрессия на баг из аудита): замокать `requests.get`
  на успешный ответ (`status_code=200`), но `open(...)` (или `f.write`) поднять исключение
  ДО того как файл реально появился на диске — например, `monkeypatch.setattr(webshot, "open",
  lambda *a, **kw: (_ for _ in ()).throw(PermissionError()))`. Проверить: `execute` возвращает
  `{'result': 'Unable to screenshot website'}` и не бросает исключение (сейчас — до правки —
  тест должен падать, потому что `os.remove` на несуществующий файл бросает
  `FileNotFoundError`, которая улетает из `execute` наружу).

### `tests/test_movie_info_plugin.py` (новый)

- Фикстура плагина: `monkeypatch.setenv("TMDB_API_KEY", "test-key")` → `MovieInfoPlugin()`.
- `test_get_new_movies_passes_timeout_and_uses_to_thread`: замокать `movie_info.requests.get`
  шпионом, возвращающим объект с `.raise_for_status()` (no-op) и `.json()` →
  `{"results": []}`; замокать `movie_info.asyncio.to_thread` шпионом. Вызвать
  `await plugin.execute("get_new_movies", helper=None, genre=None, count=5)`. Проверить:
  `requests.get` вызван с `timeout=10`; вызов прошёл через `to_thread` (шпион зафиксировал имя
  `_get_new_movies`).
- `test_get_movie_recommendations_uses_to_thread_for_all_sync_calls`: аналогично, но с
  фейковым `helper.ask` (как в `FakeHelper` из `tests/test_chief_model_choice.py`) и
  непустым списком фильмов от `_get_new_movies`/`_discover_movies`, чтобы дойти до цикла с
  `_get_movie_details`/`_get_movie_reviews`. Проверить, что шпион `to_thread` зафиксировал все
  четыре имени функций (`_get_new_movies`, `_discover_movies`, `_get_movie_details`,
  `_get_movie_reviews`).

### `tests/test_chief_close_async.py` (новый, либо новый класс тестов в
`tests/test_chief_model_choice.py` — на усмотрение разработчика, т.к. в существующем файле уже
есть фикстура `plugin` с нужными `monkeypatch.setenv`)

- `test_close_async_closes_open_session`: `await plugin._ensure_session()` (создаёт реальную
  `aiohttp.ClientSession`, сеть не трогает), затем `await plugin.close_async()`; проверить
  `plugin.session.closed is True`.
- `test_close_async_noop_when_session_never_created`: не вызывать `_ensure_session`; проверить,
  что `await plugin.close_async()` не бросает исключение (сейчас `self.session = None` из
  `__init__`).

### `tests/test_show_me_diagrams_plantuml.py` (новый)

- `test_generate_plantuml_passes_timeout_and_uses_to_thread`: замокать
  `show_me_diagrams.subprocess.run` шпионом, который создаёт файл-заглушку по пути
  `output_file` (иначе код упадёт на проверке `os.path.exists(output_file)`, `show_me_diagrams.py`
  строка ~271) и возвращает объект с `returncode=0`; замокать
  `show_me_diagrams.asyncio.to_thread` шпионом. Вызвать `await
  plugin._generate_plantuml("@startuml\n@enduml", helper=None, user_id=1)`. Проверить:
  `subprocess.run` вызван с `timeout=60`; вызов прошёл через `to_thread`.

## Команды проверки

Окружение — `~/.venvs/ctb` (см. `docs/audit_remediation_plan_2026-09-04.md`: `.venv` проекта
недоступен по правам). На момент планирования в нём не установлен `pytest` — разработчику
сначала поставить зависимости:

```bash
~/.venvs/ctb/bin/python3 -m pip install -r requirements.txt
```

Целевые тесты:

```bash
~/.venvs/ctb/bin/python3 -m pytest \
  tests/test_webshot_plugin.py \
  tests/test_movie_info_plugin.py \
  tests/test_chief_close_async.py \
  tests/test_chief_model_choice.py \
  tests/test_show_me_diagrams_plantuml.py \
  tests/test_plugin_close_async.py \
  tests/test_plugin_descriptions_contract.py \
  tests/test_docker_runtime_config.py \
  -v
```

Полный прогон (проверить отсутствие побочных эффектов за пределами затронутых плагинов):

```bash
~/.venvs/ctb/bin/python3 -m pytest -q
```

Внешние сервисы не вызывать — все `requests.get`/`subprocess.run` в тестах замоканы.

## Риски

- **`os.getenv` при импорте модуля**: `movie_info.py`/`chief.py` бросают `ValueError` в
  `__init__`, если ключи не заданы — тесты обязаны ставить `monkeypatch.setenv(...)` до
  создания инстанса плагина (как уже делает `tests/test_chief_model_choice.py`), иначе тест
  упадёт с ошибкой, не связанной с правкой.
- **`asyncio.to_thread` в Python 3.8 недоступен** (появился в 3.9) — проверить целевую версию
  Python проекта; `pyproject`/`requirements.txt`/CI уже используют более новый Python (в этой
  среде `python3 --version` = 3.12), риска нет, но стоит перепроверить перед мержем, если в
  проекте где-то зафиксирован минимум 3.8.
- **`subprocess.TimeoutExpired`** после правки в `show_me_diagrams.py` — новое исключение,
  которое раньше не могло возникнуть (raньше `subprocess.run` мог просто виснуть вечно).
  Перепроверено: перехватывается существующим `except Exception` в `execute` — новых сбоев
  наружу не всплывёт, но сообщение об ошибке пользователю изменится с «зависло молча» на
  «Error generating diagram: Command '...' timed out after 60 seconds» — это улучшение, но
  стоит упомянуть в PR-описании как поведенческое изменение.
- **`webshot.py`: смена таймаута прогревочного вызова** с «нет лимита» на `timeout=10` —
  теоретически может увеличить долю неудачных скриншотов, если `thum.io` в среднем отвечает
  на прогрев дольше 10 с (сервис бесплатный, гарантий SLA нет). Так как в этом деплое `webshot`
  сейчас не входит в `PLUGINS`, живого трафика для проверки нет — если после включения плагина
  в проде увидим рост `Unable to screenshot website`, поднять до `timeout=20-30` или перейти на
  альтернативу из раздела «Компромисс» выше.
- **`chief.py` в этом деплое активен и уже создавал незакрытые сессии** — до правки на каждом
  вызове `get_recipe`/`plan_menu` через `_search_recipes` могла плодиться новая
  `aiohttp.ClientSession`, если предыдущая протухла (`session.closed` после долгого простоя) —
  сама утечка сокетов эту правку не устраняет (просто закрывает сессию на shutdown бота), это
  вне рамок T07 и не описано в аудите как отдельная находка — не трогать, только добавить
  `close_async`.

## Критерии готовности

- [ ] `webshot.py`: оба `requests.get` — через `asyncio.to_thread` с `timeout=`; `os.remove`
      в `except` — под `contextlib.suppress(OSError)`.
- [ ] `movie_info.py`: все 4 `requests.get` — с `timeout=10`; все 4 точки вызова из `execute`
      (`_get_new_movies` ×2, `_discover_movies`, `_get_movie_details`, `_get_movie_reviews`) —
      через `asyncio.to_thread`.
- [ ] `chief.py`: добавлен `close_async()`, закрывающий `self.session`, если он существует и
      не закрыт; исключение при закрытии — логируется, не пробрасывается (как у остальных
      плагинов из `AGENTS.md`).
- [ ] `show_me_diagrams.py`: оба `subprocess.run` в `_generate_plantuml` — с `timeout=60`,
      через `asyncio.to_thread`.
- [ ] Новые тесты (`tests/test_webshot_plugin.py`, `tests/test_movie_info_plugin.py`,
      `tests/test_chief_close_async.py` или расширение `tests/test_chief_model_choice.py`,
      `tests/test_show_me_diagrams_plantuml.py`) добавлены и проходят.
- [ ] Существующие `tests/test_plugin_close_async.py`, `tests/test_chief_model_choice.py`,
      `tests/test_plugin_descriptions_contract.py`, `tests/test_docker_runtime_config.py`
      по-прежнему проходят (не пересекаются по коду, но затрагивают те же файлы).
- [ ] Полный `pytest -q` зелёный.
- [ ] Внешние сервисы (thum.io, TMDb, Edamam, реальный `java`) во время проверки не
      вызывались — всё замокано.

## Постскриптум после ревью (2026-09-04)

Реализовано буква в букву по плану: блокирующие сетевые/subprocess-вызовы в `bot/plugins/webshot.py`,
`bot/plugins/movie_info.py`, `bot/plugins/show_me_diagrams.py` вынесены в `asyncio.to_thread` с
таймаутами; в `bot/plugins/chief.py` добавлен `close_async()`. Ревью (Sonnet, persona reviewer):
ошибок и предупреждений нет. Исключения из потоков по-прежнему ловятся внешними `except` в
`execute()`; `subprocess.TimeoutExpired` не уходит пользователю трейсом. Предполагаемый флак
`test_get_movie_recommendations_uses_to_thread_for_all_sync_calls` не воспроизвёлся (5+5+3 прогонов).
