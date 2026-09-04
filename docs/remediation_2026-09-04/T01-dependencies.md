# T01. Зависимости: поднять версии и привести код

Источники задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел «Волна 1», пункт T01) и
`docs/architecture_code_review_2026-09-04.md` §3.5 («зависимости без верхних границ и с
открытыми CVE» — CVE это публично известная уязвимость программы с номером вида
CVE-2025-xxxxx) и §6 (таблица «что стоит сейчас → что актуально → какой риск»).

Термины, которые встретятся ниже:
- **wheel** — предсобранный бинарный пакет Python; если для пакета есть wheel под вашу ОС и
  версию Python, `pip`/`uv` его просто скачивает и распаковывает, компилятор не нужен.
- **sdist** — исходный пакет; если wheel нет, `pip` пытается собрать его из исходников, и вот
  тогда нужен компилятор (`g++`) и системные заголовки (`libc6-dev`).
- **транзитивная зависимость** — пакет B, который устанавливается не потому, что мы сами его
  просим, а потому что его требует пакет A, который мы просим напрямую.
- **AST-скрипт** — скрипт, который не запускает код, а разбирает текст `.py`-файла в дерево
  (Abstract Syntax Tree) и по нему ищет строки `import X`/`from X import ...`. Так можно найти
  «этот пакет реально импортируется в коде» без risk запускать сам код.
- **guarded import** — `try: import X \n except ImportError: X = None`, то есть код готов
  работать и без пакета X (просто отключает одну из веток функциональности).

Все версии/сигнатуры ниже проверены руками через `~/.local/bin/uv venv /tmp/probe` с реально
установленными `openai==3.8.0`, `mcp==2.1.1`, `python-telegram-bot==22.8`, `httpx2==2.12.0`,
`ddgs==9.16.0`, `pytubefix==10.11.0`, а не взяты из документации на веру.


## Цель

1. Поднять версии зависимостей в `requirements.txt` до актуальных на 2026-09-04 (список в
   `audit_remediation_plan_2026-09-04.md:20-23`), закрыть открытые CVE, убрать пакеты, которые
   реально не используются кодом бота.
2. Отделить dev-инструменты (`pytest`, `pytest-asyncio`, `ruff`, `pip-audit`) от runtime-списка в
   новый `requirements-dev.txt`.
3. Поправить код, который перестанет работать под новыми мажорными версиями: конструктор клиента
   `openai` (v3 сменил HTTP-библиотеку под капотом), сборку Telegram `ApplicationBuilder`
   (добавить поддержку прокси, которой сейчас нет вообще), два плагина с переименованными
   пакетами (`duckduckgo_search`→`ddgs`, `pytube`→`pytubefix`).
4. Убрать из `Dockerfile` компилятор `g++`/`libc6-dev`, если после чистки зависимостей сборка
   действительно не просит собирать что-то из исходников.
5. НЕ трогать миграцию `bot/plugins/mcp_server.py` на API `mcp` 2.x — это отдельная задача T02
   (`docs/remediation_2026-09-04/T02-mcp-v2.md`, уже написана и явно ждёт, что T01 поднимет пин
   `mcp` в `requirements.txt` первым). Здесь только меняем версию пакета в `requirements.txt`.


## Допущения

- Прокси (`PROXY`/`OPENAI_PROXY`/`TELEGRAM_PROXY` в `.env`) должен реально работать, если задан
  — это явное требование владельца, а не опция. Сейчас `bot/telegram_bot.py` вообще не читает
  `config['proxy']` в `run()`, хотя переменная давно прокидывается в конфиг
  (`bot/__main__.py:319`) — это скрытый, ранее не описанный баг, который эта задача заодно
  чинит попутно с миграцией на новый httpx.
- `deep_analysis`/`codeinterpreter` не трогаем и не отключаем (отдельное указание владельца в
  `audit_remediation_plan_2026-09-04.md:7`). Единственное, что нас касается — четыре пакета,
  которые `codeinterpreter` даёт пользовательскому коду через `exec()`
  (`numpy`, `pandas`, `matplotlib`, `plotly`, `sympy`), должны остаться в `requirements.txt`
  даже там, где AST-скрипт их не видит (пояснение — в разделе «Расхождения» ниже).
- `plugin_tool_adapter.py` и `_guard_tool_call` не удаляем — они не связаны с этой задачей вообще,
  упоминаю только чтобы не зацепить случайно.
- Разработчик, который будет реализовывать этот план, работает на системном `python3` через
  `uv venv ~/.venvs/ctb` (у `.venv` внутри репозитория нет прав на исполнение — это подтверждено:
  `chmod`/владелец файла не тот, `Permission denied` при попытке запустить
  `.venv/bin/python3`). Не пытаться чинить права на `.venv` — просто работать через новый venv.
- `git rm`/коммиты не делаются в рамках этой задачи (только правки рабочего дерева).


## Расхождения с ориентировочным списком плана (важно прочитать перед началом)

Формулировка задачи прямо говорит: «список в плане ориентировочный, перепроверь каждый». Проверка
была сделана AST-скриптом по всем `.py` под `bot/`, отдельно для «прод»-кода (`bot/*.py`,
`bot/plugins/*.py`, `bot/ai_providers/*.py` и т.д., **исключая** `bot/tests/` и `bot/skills/`) и
отдельно для тестов (`tests/`, `bot/tests/`). Три находки идут вразрез с тем, что явно написано
в `audit_remediation_plan_2026-09-04.md:34-40`, и их нужно учесть, а не копировать список слепо:

1. **`google-api-python-client` — НЕ удалять.** План перечисляет его как кандидат на удаление
   (`audit_remediation_plan_2026-09-04.md:36`). Это ошибка списка: `bot/plugins/google_web_search.py:5-6`
   делает `from googleapiclient.discovery import build` и
   `from googleapiclient.errors import HttpError` без `try/except` — обычный, боевой, ничем не
   защищённый импорт в плагине google-поиска. Удаление пакета сломает загрузку этого плагина
   при старте бота (плагин не сможет импортироваться — `PluginManager` его либо пропустит с
   ошибкой в логе, либо, в зависимости от строгости валидации, это будет замечено только в
   рантайме). Оставить пакет как есть, версию не трогать (`google-api-python-client==2.140.0`
   не входит в список «Целевые версии» в шапке плана, апгрейдить не просят).

2. **`sympy`, `lxml`, `lxml-html-clean` — AST-скрипт помечает их «неиспользуемыми», но их нужно
   оставить (первый — без изменений, два других — с указанным в плане апгрейдом).** Все три
   реально нужны, просто не через обычный `import` на верхнем уровне файла:
   - `sympy`: `bot/plugins/codeinterpreter.py:127` рекламирует пользователю в описании тула
     «(pandas, numpy, matplotlib, plotly, sympy)», а сам код исполняет пользовательский текст
     через `exec(code, exec_globals, exec_globals)` (`bot/plugins/codeinterpreter.py:421`) —
     `exec_globals` не блокирует обычный `import`, поэтому код, который сгенерирует модель,
     может сам написать `import sympy` и это отработает, если пакет стоит в окружении. AST не
     видит такого использования, потому что физически строки `import sympy` в файле плагина
     нет — она есть только в тексте, который будет исполнен во время работы бота.
   - `lxml` / `lxml-html-clean`: `bot/plugins/text_summarizer.py:56` вызывает
     `readability.Document(...)` (пакет `readability-lxml`). Внутри `readability-lxml` сам
     импортирует `lxml` и (на Python ≥3.11, через extra `lxml[html-clean]`) `lxml_html_clean` —
     наш код никогда не пишет `import lxml` напрямую, поэтому AST считает пакет «неиспользуемым».
     На деле это транзитивная, но обязательная зависимость. Более того: `readability-lxml==0.9`
     требует лишь `lxml-html-clean>=0.4.2` (проверено `pypi.org/pypi/readability-lxml/0.9/json`)
     — то есть **без явного пина `lxml-html-clean>=0.4.5` в `requirements.txt` резолвер вправе
     поставить 0.4.2 или 0.4.4, которые всё ещё содержат CVE-2024-52595/CVE-2026-28348/28350.**
     Явный пин — не просто гигиена, а единственный способ реально закрыть эту уязвимость.

3. **`ddg_translate.py` уже сломан сегодня, и замена импорта это не чинит.** Проверено в двух
   изолированных venv: `duckduckgo_search==8.1.1` (последняя версия на PyPI, `requirements.txt`
   её вообще не пинит, значит именно она стоит сейчас) и `ddgs==9.16.0` — у обоих класс `DDGS`
   **не имеет метода `.translate()`** (реальные методы: `books, extract, images, news, text,
   threads, videos` — проверено `dir(DDGS)` в `/tmp/probe`). `bot/plugins/ddg_translate.py:31`
   вызывает `ddgs.translate(...)`, значит вызов тула `ddg_translate.translate` уже сегодня падает
   с `AttributeError` в проде, до всякого апгрейда. Соседние плагины `ddg_web_search.py` и
   `ddg_image_search.py` это уже пережили — оба переписаны на LLMGateway (см. docstring
   `bot/plugins/ddg_image_search.py:9`: «Backward-compatible image search plugin backed by
   LLMGateway web search»), а `ddg_translate.py` — забыли мигрировать. Задача T01 по своему духу
   («поднять версии») не должна чинить эту логическую дыру — это отдельная фича-задача (переписать
   `execute()` на LLMGateway так же, как соседей). Что нужно сделать в T01: **только** поменять
   импорт (`from ddgs import DDGS`), чтобы код продолжал грузиться и вести себя ровно так же
   (падать с той же ошибкой на вызове, а не с `ModuleNotFoundError` на импорте пакета, которого
   скоро не будет на PyPI под старым именем). Ниже в разделе про этот файл — обязательный
   комментарий-TODO, который надо оставить в коде, чтобы находка не потерялась.


## Точный список изменений по файлам

### 1. `requirements.txt` (полная замена содержимого)

Текущий файл (61 строка, без пустых) — `requirements.txt:1-67`. Ниже — новое содержимое.
Управляющий принцип: каждая оставленная строка либо `USED` по AST-проверке кода в `bot/`
(исключая `bot/tests/`, `bot/skills/` — те две папки живут в других процессах, см. ниже), либо
явно нужна `codeinterpreter`/`readability-lxml` по причинам из раздела «Расхождения».

```
python-dotenv~=1.0.0
pydub~=0.25.1
tiktoken>=0.14.0
openai>=3.8,<4
httpx2>=2.7,<3
python-telegram-bot>=22.8,<23
httpx>=0.27,<0.29
requests>=2.32.4,<3
wolframalpha~=5.0.0
ddgs>=9.16
spotipy>=2.26
pytubefix>=10.11
Pillow>=12.3
readability-lxml>=0.9
lxml>=6.1.3
lxml-html-clean>=0.4.5
google-api-python-client==2.140.0
plotly
numpy
sympy
aiohttp>=3.8.0
matplotlib
PyYAML~=6.0
pandas
jsonschema>=4.0.0
telegramify-markdown>=1.2
beautifulsoup4>=4.12.2
Pygments>=2.21
PyPDF2
pdfminer.six
markdown2
mcp>=2.1,<3
# Optional: only needed when SESSION_LOG_OTEL_ENDPOINT is set. Left uninstalled by
# default because the grpc exporter pulls ~19 MB of wheels that nothing imports while
# the endpoint is empty; without them the bot logs sessions to JSONL as usual and
# build_otel_bridge() warns once and falls back to the no-op bridge. To enable:
#   pip install 'opentelemetry-api>=1.27,<2' 'opentelemetry-sdk>=1.27,<2' \
#       'opentelemetry-exporter-otlp-proto-grpc>=1.27,<2'
```

Хвостовой комментарий про opentelemetry (последние 5 строк) — переносится как есть, без
изменений (`requirements.txt:62-67` в текущем файле). Не трогать: `mcp>=2.1,<3` теперь тянет
`opentelemetry-api>=1.28.0` транзитивно в любом случае (проверено `pypi.org/pypi/mcp/2.1.1/json`
→ `requires_dist`), но это только `-api`, не `-sdk`/`-exporter-otlp-proto-grpc` — комментарий
и `bot/session_otel.py`, который сам делает `try/except ImportError` на `opentelemetry.sdk...`,
продолжают работать ровно как раньше.

**Убрано полностью (импорт не найден нигде в `bot/`, не guarded, не используется в
пользовательском exec-коде — 28 пакетов):** `chardet`, `tenacity`, `youtube-transcript-api`,
`asyncpg`, `pyswisseph`, `opencv-python`, `assemblyai`, `telethon`, `pygame`, `pytz`,
`SpeechRecognition`, `cryptography`, `jinja2`, `Markdown`, `deep-translator`, `countryinfo`,
`forex_python`, `qrcode`, `nltk`, `typing-extensions`, `ratelimit`, `ffmpeg-python`,
`requests-toolbelt`, `fastapi`, `uvicorn`, `moviepy`, `scikit-learn`, `trafilatura`.
Детали по паре из них, чтобы при код-ревью не переспрашивали:
- `chardet`: не импортируется в `bot/`, но `readability-lxml==0.9` сам требует
  `chardet>=5.2.0,<6.0.0` — после удаления явной строки пакет всё равно встанет транзитивно,
  просто мы больше не диктуем его версию напрямую.
- `assemblyai`: пакет не импортируется, но ключ `assemblyai_api_key` в конфиге
  (`bot/__main__.py:247,336`) читается из `ASSEMBLYAI_API_KEY` и никем не используется дальше —
  мёртвый конфиг-ключ. Сам ключ конфига не в скоупе этой задачи (это не про зависимости), просто
  фиксирую находку.
- `cryptography`: единственное использование в дереве — `bot/skills/excalidraw/scripts/upload.py:29`
  (`from cryptography.hazmat.primitives.ciphers.aead import AESGCM`, с собственным `except
  ImportError` и подсказкой `pip install cryptography` пользователю). Это skill-скрипт, который
  запускается не в процессе бота, а `asyncio.create_subprocess_exec`
  (`bot/plugins/skills.py:3267`) в отдельном интерпретаторе/venv — см. следующий пункт.
- `uvicorn`/`fastapi`: `uvicorn` всё равно останется установлен — это транзитивная зависимость
  `mcp>=2.1` (`requires_dist` пакета `mcp` включает `uvicorn>=0.31.1`), просто не по нашей прямой
  строке. `fastapi` уходит полностью, у `mcp` его в зависимостях нет.

**Заменено (переименованный/устаревший пакет → актуальный, тот же смысл):**
- `duckduckgo_search` → `ddgs>=9.16` (пакет переименован апстримом).
- `pytube~=15.0.0` → `pytubefix>=10.11` (форк, `pytube` не работает с YouTube с 2023 года;
  `pytubefix.YouTube` — проверенный drop-in: тот же конструктор `YouTube(url)`, тот же
  `.streams.filter(...).first()`, `Stream.download(filename=...)`).

**Добавлено явно (раньше приезжало транзитивно или не было объявлено вовсе):**
- `Pygments>=2.21` — используется в `bot/plugins/github_analysis.py:7-8`
  (`from pygments.lexers import get_lexer_for_filename`, `from pygments.util import
  ClassNotFound`), безусловный импорт без `try/except`. Раньше в `requirements.txt` не было явной
  строки — пакет приезжал только потому, что его тянет `ipython>=8.16.0` (сам `ipython` нигде не
  импортируется в `bot/`, это чистый вспомогательный «извозчик» для `pygments`). Теперь `ipython`
  убираем совсем (он не входит даже в список dev-инструментов ниже), значит `pygments` обязательно
  нужно прописать напрямую — иначе `github_analysis.py` перестанет загружаться при `pip install`
  с нуля (например, в свежей Docker-сборке).
- `PyPDF2` — используется в `bot/plugins/ask_your_pdf.py:9-11` (`try: import PyPDF2 / except
  ImportError: PyPDF2 = None`), один из трёх способов извлечь текст из PDF (наравне с
  `pdfminer.six`, который уже объявлен, и `textract`, который в requirements не входит и в эту
  задачу не добавляется — см. ниже). Guarded-импорт, но раз он есть и активно используется как
  часть публичной функциональности плагина — явно объявляем, как и просит план
  (`audit_remediation_plan_2026-09-04.md:34`, «PyPDF2 (если ask_your_pdf его использует)» — да,
  использует).

**Находка вне скоупа (без действия, для протокола):** `bot/plugins/ask_your_pdf.py:15-17` также
делает `try: import textract / except ImportError: textract = None` — это третий бэкенд
извлечения текста из PDF, тоже guarded, но пакета `textract` нет ни в старом, ни в новом
`requirements.txt`, и план явно просит добавить только `PyPDF2`. Оставляю как есть (сейчас этот
бэкенд молча выключен — `textract is None` в рантайме, `_extract_text_with_textract` возвращает
пусто и код переходит к следующему бэкенду, см. `bot/plugins/ask_your_pdf.py:257-276`); если
владелец захочет включить и его — это отдельное решение, не часть апгрейда зависимостей.

**Дедуп:** `beautifulsoup4` была объявлена дважды (текущие строки `requirements.txt:39` и `:58`)
— в новом списке одна строка `beautifulsoup4>=4.12.2`.

**`codeinterpreter`-пакеты (оставлены без изменений версии, кроме уже перечисленных выше через
`lxml`):** `numpy`, `pandas`, `matplotlib`, `plotly`, `sympy` — без пинов, как было. План явно
просит их не трогать (`audit_remediation_plan_2026-09-04.md:40`).

**Пакеты вне скоупа апгрейда, оставлены как есть:** `python-dotenv~=1.0.0`, `pydub~=0.25.1`,
`requests>=2.32.4,<3`, `wolframalpha~=5.0.0`, `PyYAML~=6.0`, `aiohttp>=3.8.0`,
`jsonschema>=4.0.0`, `google-api-python-client==2.140.0` (см. «Расхождения», п.1),
`beautifulsoup4>=4.12.2`, `pdfminer.six`, `markdown2` — ни один не входит в список «Целевые
версии» в шапке `audit_remediation_plan_2026-09-04.md:20-23`, версии не менять.


### 2. `requirements-dev.txt` (новый файл)

```
-r requirements.txt
pytest>=7.4.2
pytest-asyncio>=0.21.1
ruff
pip-audit
```

`pytest`/`pytest-asyncio` — перенесены из `requirements.txt` (были там строками `:54-55`, версии
сохранены как есть, просто сменили файл). `ruff`/`pip-audit` — новые, без пина (задача T03,
`docs/remediation_2026-09-04/T03-ci-infra.md`, ставит под них `pyproject.toml`/CI — версию `ruff`
там не диктуют, значит здесь тоже без пина, чтобы не создать двух источников правды).
`ipython>=8.16.0` (была строка `:56`) **не переносится ни сюда, ни в `requirements.txt`** — она
нужна была только транзитивно ради `pygments`, который теперь объявлен явно в runtime-списке
(см. выше); в `bot/`, `bot/tests/`, `tests/` пакет `IPython` не импортируется ни разу.


### 3. `bot/openai_helper.py` — конструктор `OpenAIHelper.__init__`

Текущий код, `bot/openai_helper.py:261-276`:

```python
        # http_client = httpx.AsyncClient(proxies=config['proxy']) if 'proxy' in config else None
        self._http_client = httpx.AsyncClient()

        if config['openai_base'] != '' :
            openai.api_base = config['openai_base']
        self.api_key = config['api_key']
        client_kwargs = {
            "api_key": config["api_key"],
            "http_client": self._http_client,
            "timeout": 300.0,
            "max_retries": 3,
        }
        if config["openai_base"]:
            client_kwargs["base_url"] = config["openai_base"]
        self.client = openai.AsyncOpenAI(**client_kwargs)
        self.gateway_client = LLMGatewayClient(config.get("openai_base", ""), config["api_key"])
```

Заменить на:

```python
        proxy = config.get('proxy') or None
        self._http_client = httpx2.AsyncClient(proxy=proxy)

        self.api_key = config['api_key']
        client_kwargs = {
            "api_key": config["api_key"],
            "http_client": self._http_client,
            "timeout": 300.0,
            "max_retries": 3,
        }
        if config["openai_base"]:
            client_kwargs["base_url"] = config["openai_base"]
        self.client = openai.AsyncOpenAI(**client_kwargs)
        self.gateway_client = LLMGatewayClient(config.get("openai_base", ""), config["api_key"])
```

Что именно меняется и почему:
- `openai.AsyncOpenAI.__init__` в 3.8.0 принимает `http_client: httpx2.AsyncClient | None`, а не
  `httpx.AsyncClient` (проверено `inspect.signature` на реально установленном `openai==3.8.0` —
  openai v3 полностью перешёл на форк `httpx2`, это не опечатка и не косметика). Передавать туда
  старый `httpx.AsyncClient()` можно (Python это не запретит на уровне типов, isinstance-проверки
  внутри SDK нет по умолчанию), но тогда обычный `httpx.AsyncClient` не будет понимать формат
  запросов/исключений, которые ждёт `openai` v3 внутри — рабочий вариант только через `httpx2`.
- `httpx2.AsyncClient` принимает `proxy=` (единственное число — один прокси на клиент), не
  `proxies=` (устаревший в самом `httpx` ещё до форка); в закомментированной строке-заглушке
  `bot/openai_helper.py:261` было использовано старое `proxies=`, которое давно не работает даже
  под `httpx` 0.27+ — она была мёртвой ещё до этой задачи и её строку просто убираем вместе с
  реализацией.
- `config.get('proxy') or None` — специально `.get()`, а не `config['proxy']`: в двух местах
  тестов (`tests/test_hindsight_memory.py:165`, `tests/test_openai_helper_tool_calls.py:446`)
  ключ `proxy` есть и равен `None`, но чтобы не полагаться на то, что каждый вызывающий код всегда
  кладёт этот ключ, безопаснее `.get()`. `or None` — на случай пустой строки из `.env`
  (`os.environ.get('PROXY', None) or os.environ.get('OPENAI_PROXY', None)` в
  `bot/__main__.py:210` и так уже даёт `None` при отсутствии, но `or None` не помешает, если
  когда-то конфиг соберут иначе).
- `if config['openai_base'] != '': openai.api_base = config['openai_base']` — удаляется целиком.
  `openai.api_base` — атрибут из API v0 (до `openai>=1.0`, вышедшего в конце 2023); в v1+ SDK его
  никто не читает, строка ничего не делает даже сегодня на `openai==2.20.0` (это подтверждено
  чтением кода `openai.AsyncOpenAI`/`openai._client` — базовый URL берётся только из
  `base_url=` в конструкторе клиента, который уже передаётся четырьмя строками ниже через
  `client_kwargs["base_url"]`). В `openai==3.8.0` атрибута `api_base` вообще нет в модуле
  (проверено: `'api_base' in dir(openai)` → `False`), но само присваивание всё равно не упадёт
  (Python разрешает произвольные атрибуты модуля) — убираем не потому что сломается, а потому что
  это мёртвый код, который вводит в заблуждение (аудит справедливо называет её так,
  `docs/architecture_code_review_2026-09-04.md:157`).
- Закрытие клиента (`bot/openai_helper.py:4321-4323`, метод `close_async`/`close` — не путать с
  номером строки конструктора) **не меняется**: `self._http_client.aclose()` работает одинаково
  что на `httpx.AsyncClient`, что на `httpx2.AsyncClient` (проверено —
  `hasattr(httpx2.AsyncClient, 'aclose')` → `True`), потому что `httpx2` — байт-в-байт совместимый
  по публичному API форк `httpx`, только под новым именем пакета.

Импорт вверху файла — `bot/openai_helper.py:20`:

```python
import httpx
```

заменить на:

```python
import httpx2
```

Проверено: `httpx` в этом файле после правки конструктора нигде больше не используется (был
только в строке-комментарии и в самой строке 262 — оба места правятся). Если бы `httpx` тип
использовался ещё где-то (например, `except httpx.HTTPError`), импорт нужно было бы оставить —
но в `bot/openai_helper.py` таких мест нет (проверено скринингом всего файла на `\bhttpx\b`, кроме
двух уже упомянутых строк — совпадений нет).

**Отдельно НЕ трогать:** `bot/llm_gateway_client.py` (весь файл, включая `httpx.AsyncClient(...)`
на строке 39 и `except httpx.HTTPError` на строке 47) — этот клиент ходит в отдельный сервис
LLMGateway напрямую через `requests`/`httpx`-совместимый интерфейс, никак не завязан на SDK
`openai`, и `python-telegram-bot` в этом же процессе требует классический `httpx>=0.27,<0.29`
(проверено `requires_dist` пакета `python-telegram-bot==22.8` на PyPI — там `httpx<0.29,>=0.27`,
`httpx2` не упоминается вовсе).两 экосистемы (`httpx` для PTB и `bot/llm_gateway_client.py`,
`httpx2` для `openai`/`mcp`) в одном процессе — это нормально и ожидаемо, они не конфликтуют
(разные имена пакетов на PyPI, разные модули при импорте). Переводить `llm_gateway_client.py` на
`httpx2` не нужно и не входит в задачу — он и так работает, трогать без причины запрещает
AGENTS.md («surgical changes»). Прокси для LLMGateway-клиента этот файл сегодня не поддерживает
вообще (ни `config['proxy']`, ни `config['proxy_web']` в него не попадают) — это существующее
поведение, не регрессия от T01, чинить не в этой задаче (нет явного указания владельца
чинить прокси именно здесь, а список файлов T01 этот файл не включает).


### 4. `bot/telegram_bot.py` — `run()`, поддержка прокси в `ApplicationBuilder`

Текущий код, `bot/telegram_bot.py:6287-6302`:

```python
            builder = ApplicationBuilder() \
                .token(self.config['token']) \
                .post_init(self.post_init) \
                .post_shutdown(self._post_shutdown) \
                .concurrent_updates(True)

            telegram_local_mode = self.config.get('telegram_local_mode', True)
            telegram_base_url = self.config.get(
                'telegram_base_url',
                DEFAULT_TELEGRAM_BASE_URL
            )
            builder = builder.local_mode(telegram_local_mode)
            if telegram_local_mode and telegram_base_url:
                builder = builder.base_url(telegram_base_url)

            application = builder.build()
```

Заменить на:

```python
            builder = ApplicationBuilder() \
                .token(self.config['token']) \
                .post_init(self.post_init) \
                .post_shutdown(self._post_shutdown) \
                .concurrent_updates(True)

            telegram_local_mode = self.config.get('telegram_local_mode', True)
            telegram_base_url = self.config.get(
                'telegram_base_url',
                DEFAULT_TELEGRAM_BASE_URL
            )
            builder = builder.local_mode(telegram_local_mode)
            if telegram_local_mode and telegram_base_url:
                builder = builder.base_url(telegram_base_url)

            telegram_proxy = self.config.get('proxy')
            if telegram_proxy:
                builder = builder.proxy(telegram_proxy)
                builder = builder.get_updates_proxy(telegram_proxy)

            application = builder.build()
```

Почему нужны оба вызова, а не один:
- PTB различает два независимых HTTP-клиента внутри одного бота: обычный (`sendMessage` и т.п.,
  настраивается через `.proxy()`) и отдельный — только для long polling, то есть постоянного
  опроса `getUpdates` (настраивается через `.get_updates_proxy()`). Проверено чтением исходников
  `telegram.ext._applicationbuilder.ApplicationBuilder._build_request`/`._build_ext_bot` в
  установленном PTB 22.8: если задать только `.proxy()`, атрибут `_get_updates_proxy` остаётся
  `DEFAULT_NONE`, и `getUpdates`-соединение уходит без прокси. Бот работает через `run_polling`,
  то есть именно `getUpdates`-соединение и есть основной канал получения сообщений — без второго
  вызова прокси половину смысла настройки теряет.
- `.proxy(...)`/`.get_updates_proxy(...)` не конфликтуют по порядку вызова с уже существующими
  `.local_mode()`/`.base_url()` — единственные вызовы, с которыми они несовместимы, это
  `.request()`/`.bot()`/`.updater()` (проверено чтением `_request_param_check` в исходниках PTB),
  которых в этом коде нет.
- `self.config.get('proxy')`, не `self.config['proxy']` — обязательно `.get()`. Два существующих
  фейковых билдера в тестах не кладут ключ `proxy` в `bot.config` вовсе:
  `tests/test_telegram_builder_config.py:262-268` (`_make_bot`) и
  `tests/test_plugin_handlers_registration.py` (там конфиг вообще не содержит слова «proxy» — проверено
  полнотекстовым поиском). При прямой индексации `self.config['proxy']` оба этих теста упадут с
  `KeyError` на ровном месте. `.get()` + `if telegram_proxy:` — единственный вариант, который не
  трогает существующие тесты и не требует их менять только ради этой строчки.
- Если прокси не задан (`None`/пустая строка из `.env`), блок `if telegram_proxy:` целиком
  пропускается, `.proxy()`/`.get_updates_proxy()` не вызываются вообще — второй фейковый билдер
  (`tests/test_plugin_handlers_registration.py:110-132`), у которого этих методов нет вовсе,
  продолжает работать без изменений (не входит в список файлов T01, трогать не нужно).

Локальный режим и прокси одновременно — не наша забота решать здесь: если оператор одновременно
включит `TELEGRAM_LOCAL_MODE=true` (трафик идёт на `localhost:8081`, локальный Telegram Bot API
сервер) и `PROXY=...`, то через прокси пойдёт трафик к `localhost` — это осознанный выбор
конфигурации оператора, а не то, что код должен запрещать; сама переменная `PROXY` уже
документирована в `.env.example:30` как общая настройка, без привязки к local-mode.


### 5. `bot/plugins/ddg_translate.py`

Текущая строка `bot/plugins/ddg_translate.py:3`:

```python
from duckduckgo_search import DDGS
```

Заменить на:

```python
from ddgs import DDGS
```

Больше в файле ничего не менять. Дополнительно — добавить короткий комментарий прямо над
`execute()` (`bot/plugins/ddg_translate.py:29`), чтобы находка из раздела «Расхождения» (п.3) не
потерялась при код-ревью:

```python
    async def execute(self, function_name, helper, **kwargs) -> Dict:
        # TODO: DDGS().translate(...) отсутствует и в duckduckgo_search 8.x, и в ddgs 9.x —
        # этот тул уже не работает (AttributeError), независимо от миграции пакета T01.
        # Соседи ddg_web_search.py/ddg_image_search.py уже переписаны на LLMGateway;
        # ddg_translate.py — нет. Чинить нужно отдельной задачей, не апгрейдом зависимостей.
        with DDGS() as ddgs:
            return ddgs.translate(kwargs['text'], to=kwargs['to_language'])
```

Это не меняет поведение (тул как падал с `AttributeError` при вызове, так и продолжит падать —
plugin-исполнитель ловит исключения из `execute()` и превращает их в текст ошибки для модели, это
не приводит к падению бота), только фиксирует находку в коде рядом с местом бага.


### 6. `bot/plugins/youtube_audio_extractor.py`

Текущая строка `bot/plugins/youtube_audio_extractor.py:5`:

```python
from pytube import YouTube
```

Заменить на:

```python
from pytubefix import YouTube
```

Больше в файле ничего не менять — `pytubefix.YouTube(link)`, `.streams.filter(only_audio=True,
file_extension='mp4').first()`, `Stream.download(filename=output)` — сигнатуры и поведение
проверены на установленном `pytubefix==10.11.0` как прямая замена (в отличие от `ddg_translate`,
здесь реальной поломки функциональности нет, только замена мёртвого пакета на поддерживаемый
форк с тем же API).


### 7. `Dockerfile`

Текущая строка сборки, `Dockerfile:17-21`:

```dockerfile
RUN apt-get update \
     && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg default-jre-headless graphviz g++ libc6-dev \
     && pip install -r requirements.txt --no-cache-dir \
     && apt-get purge -y --auto-remove g++ libc6-dev \
     && rm -rf /var/lib/apt/lists/* \
```

Изменить на:

```dockerfile
RUN apt-get update \
     && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg default-jre-headless graphviz \
     && pip install -r requirements.txt --no-cache-dir \
     && rm -rf /var/lib/apt/lists/* \
```

Условие — «если сборка проходит без них» (формулировка из плана), не «удалить безусловно».
Проверить обязательно реальной сборкой (команда — в разделе «Команды проверки» ниже) **после**
того, как обновлён `requirements.txt`: причина, по которой раньше был нужен компилятор — почти
наверняка старые/безверсийные пакеты без `manylinux`-wheel под Python 3.12 (`opencv-python`,
`scikit-learn`, `asyncpg`, `pygame`, `SpeechRecognition`, `moviepy` — все теперь удалены). Все
пакеты, которые остаются после чистки (`numpy`, `pandas`, `matplotlib`, `lxml`, `Pillow`,
`tiktoken`, `pydantic`-ядро через `mcp`/`openai`) на 2026 год публикуют wheels под
`cp312-manylinux`/`cp312-musllinux`, но это нужно подтвердить именно сборкой образа, а не
предположением — если после чистки `pip install -r requirements.txt` внутри контейнера всё же
попросит компилятор на какой-то пакет, вернуть `g++ libc6-dev` (и `apt-get purge` их обратно, как
было) и явно написать в результате задачи, из-за какого пакета.


### 8. `tests/test_telegram_builder_config.py`

Файл — 383+ строк, `FakeApplicationBuilder` определён дважды: первый раз на
`tests/test_telegram_builder_config.py:205-233`, используется в `_run_bot_with_fake_builder`
(`:292-315`) и повторно создаётся в `test_run_post_shutdown_cleanup_not_repeated_by_finally`
(`:754`). Нужно добавить туда трекинг вызовов `.proxy()`/`.get_updates_proxy()` — по аналогии с
уже существующими `local_mode_calls`/`base_url_calls`.

В `__init__` класса `FakeApplicationBuilder` (`tests/test_telegram_builder_config.py:206-209`):

```python
    def __init__(self, application):
        self.application = application
        self.token_calls = []
        self.local_mode_calls = []
        self.base_url_calls = []
```

добавить две строки:

```python
    def __init__(self, application):
        self.application = application
        self.token_calls = []
        self.local_mode_calls = []
        self.base_url_calls = []
        self.proxy_calls = []
        self.get_updates_proxy_calls = []
```

После метода `base_url` (`tests/test_telegram_builder_config.py:225-227`):

```python
    def base_url(self, url):
        self.base_url_calls.append(url)
        return self
```

добавить:

```python
    def base_url(self, url):
        self.base_url_calls.append(url)
        return self

    def proxy(self, proxy):
        self.proxy_calls.append(proxy)
        return self

    def get_updates_proxy(self, proxy):
        self.get_updates_proxy_calls.append(proxy)
        return self
```

Новые тесты — добавить сразу после `test_telegram_builder_skips_base_url_when_local_mode_disabled`
(заканчивается на `tests/test_telegram_builder_config.py:749`, перед
`def test_run_post_shutdown_cleanup_not_repeated_by_finally`):

```python
def test_telegram_builder_sets_proxy_when_configured(monkeypatch):
    builder, application = _run_bot_with_fake_builder(
        monkeypatch,
        {"proxy": "http://proxy.local:8080"},
    )

    assert builder.proxy_calls == ["http://proxy.local:8080"]
    assert builder.get_updates_proxy_calls == ["http://proxy.local:8080"]
    assert application.run_polling_calls == 1


def test_telegram_builder_skips_proxy_when_not_configured(monkeypatch):
    builder, _application = _run_bot_with_fake_builder(monkeypatch)

    assert builder.proxy_calls == []
    assert builder.get_updates_proxy_calls == []
```

Второй тест дублирует часть проверки `test_default_telegram_builder_uses_local_bot_api`, но
явно фиксирует «прокси не задан → билдер не трогается» как отдельный сценарий, а не побочный
эффект — так его проще найти при будущей регрессии именно по прокси.

Файл `tests/test_plugin_handlers_registration.py` **не менять** — его `FakeApplicationBuilder`
(`:110-132`) не имеет методов `proxy`/`get_updates_proxy`, но конфиг в этом файле не содержит
ключа `proxy` ни разу (проверено полнотекстовым поиском), поэтому новый код в `run()`
(`if telegram_proxy: ...`) туда просто не зайдёт — тест как проходил, так и будет проходить.


## Проверка типов SDK `openai`/`httpx` — что явно НЕ нужно менять

Задание отдельно просит найти все места, зависящие от `CompletionUsage`, `.construct()`,
`ChatCompletionChunk`, `openai.types...` и от связки `httpx`+`openai`. Полнотекстовый поиск по
всему `bot/` и `tests/`/`bot/tests/` (не только по `grep`, а по чтению файлов построчно, т.к.
`rg`/`grep` в этом окружении местами искажают вывод — см. заметку в самом задании) дал:

- **`CompletionUsage`, `.construct(`, `ChatCompletionChunk`, `openai.types.*` — не встречаются
  нигде** в `bot/` или в `tests/`/`bot/tests/`. Это НЕ значит, что тема неактуальна — T06
  (`docs/remediation_2026-09-04/T06-usage-none.md`) как раз работает с полем `usage` ответа
  модели, но делает это через `getattr(response, "usage", None)` /
  `getattr(usage, "total_tokens", 0)` (`bot/ai_providers/openai_compatible.py:254-265`, функции
  `_usage`/`_int_or_zero`) — то есть по «утиной типизации» (duck typing: не важно, объект какого
  класса пришёл, важно что у него есть нужный атрибут), без импорта самого класса
  `CompletionUsage`. Такой код одинаково переживёт что `openai` v2, что v3 — атрибуты `usage`,
  `prompt_tokens`, `completion_tokens`, `total_tokens` на объекте ответа не переименовывались.
  **Действие: не требуется**, только подтверждено, что T01 не создаёт новых проблем для T06 (T06
  — отдельная, уже написанная задача, за неё эта задача не отвечает).
- **`openai.APIError`/`openai.RateLimitError`/`openai.BadRequestError`** — используются в
  `bot/openai_helper.py:490,1676,1680,2544,2547` и `bot/openai_tool_handler.py:1197,1257`. Все
  четыре класса присутствуют в `openai==3.8.0` без изменений (проверено `dir(openai)` и MRO
  `openai.APIError.__mro__` на установленном пакете). Во всех восьми местах код делает только
  `str(exc)`/`log_exception_shape(exc)` (`bot/utils.py:45-47`, который тоже просто делает
  `f"{type(exc).__name__}: {message}"`) — никаких обращений к версионно-специфичным атрибутам
  вроде `.response.status_code`/`.body`. **Действие: не требуется.**
- **`httpx.AsyncClient` вне `openai_helper.py`** — используется в `bot/llm_gateway_client.py:39`,
  `bot/telegram_bot.py:1301`, и в нескольких плагинах (`crypto.py`, `iplocation.py`, `vkusvill.py`,
  `weather.py`, `hindsight_memory.py`, `pravo_gov_ru_api.py`, `jina_web_search.py`,
  `codeinterpreter.py`, `text_document_qa.py`, `text_summarizer.py`, `mcp_server.py`). Ни один из
  них не получает клиент из `OpenAIHelper` — каждый создаёт свой `httpx.AsyncClient()` независимо
  для обращения к стороннему HTTP API (не к `openai`). Замена типа http-клиента в конструкторе
  `OpenAIHelper` их не касается. **Действие: не требуется.**
- **Мок `httpx.AsyncClient` в тестах** — единственное место:
  `bot/tests/test_mcp_server.py:230` (`patch('httpx.AsyncClient', return_value=mock_client)`),
  проверяет HTTP-транспорт `mcp_server.py` (не stdio, не SDK `mcp`, а свой `httpx`-запрос к
  удалённому MCP-серверу). Это T02-территория (`bot/plugins/mcp_server.py`), но конкретно этот
  тест **не трогает** SDK `mcp` вообще (`import mcp` в файле нет — весь файл использует только
  `pytest.importorskip("mcp")` как страховку и стандартный `httpx`) — апгрейд `mcp` до 2.1.1 этот
  тест не задевает. **Действие: не требуется** (и не в T01, и не будет задето T02 тоже — сверено
  по факту отсутствия импортов `mcp.*` в этом тестовом файле).


## Дополнительная проверка: не сломает ли апгрейд `mcp` до 2.1 модуль `mcp_server.py` до T02

T01 поднимает пин `mcp` до `>=2.1,<3`, но код `bot/plugins/mcp_server.py` **не трогает** — миграция
API это T02. Важно было убедиться, что между T01 и T02 бот не окажется в состоянии, когда плагин
вообще не загружается (это уронило бы куда больше, чем один плагин — `PluginManager` при падении
модуля на `exec_module` логирует и пропускает конкретно этот плагин, не весь бот, но лучше знать
заранее, что именно сломается). Проверено на `mcp==2.1.1` в `/tmp/probe`:

- `from mcp.client.stdio import stdio_client, StdioServerParameters` и `from mcp import
  ClientSession` (`bot/plugins/mcp_server.py:12-13`) — оба импорта работают без изменений.
- `StdioServerParameters(command=..., args=..., env=...)` (`bot/plugins/mcp_server.py:475-478`) —
  все три параметра именованные, совпадают с сигнатурой в 2.1.1.
- `ClientSession(read_stream, write_stream)` + `await session.initialize()`
  (`bot/plugins/mcp_server.py:490-494`) и `await session.call_tool(function_name,
  arguments=kwargs)` (`bot/plugins/mcp_server.py:689`) — сигнатуры конструктора/методов в 2.1.1
  остаются позиционно- и именовано-совместимыми.
- Единственное, что уже сломано (и остаётся сломанным, без ухудшения) — `_fetch_stdio_tools`
  (`bot/plugins/mcp_server.py:364-386`), задокументировано как CERTAIN-находка в
  `docs/architecture_code_review_2026-09-04.md:194-202` и является предметом T02. Апгрейд пина
  T01 не делает эту находку хуже — она одинаково не работает что на `mcp` 1.x, что на 2.1.1,
  потому что баг не в версии SDK, а в неверном разборе `ListToolsResult`
  (`tool.parameters` вместо `tool.inputSchema`).

Вывод: бампать `mcp` в `requirements.txt` в T01 безопасно и не создаёт окна, в котором плагин
перестаёт грузиться — можно мержить T01 и T02 в любом порядке относительно этого файла (T02 сам
уже написан с расчётом, что T01 будет первым, но технической жёсткой зависимости в эту сторону
нет).


## Порядок шагов

1. `requirements.txt` — переписать содержимое (раздел 1).
2. `requirements-dev.txt` — создать (раздел 2).
3. `bot/openai_helper.py` — константа `import httpx` → `import httpx2` (строка 20) и правка
   конструктора (строки 261-276) (раздел 3).
4. `bot/telegram_bot.py` — добавить блок прокси в `run()` (раздел 4).
5. `bot/plugins/ddg_translate.py` — смена импорта + TODO-комментарий (раздел 5).
6. `bot/plugins/youtube_audio_extractor.py` — смена импорта (раздел 6).
7. `tests/test_telegram_builder_config.py` — трекинг прокси в фейковом билдере + два новых теста
   (раздел 8).
8. `Dockerfile` — убрать `g++ libc6-dev` из `apt-get install` и удалить строку `apt-get purge`
   (раздел 7) — делать последним шагом, чтобы проверять сборкой уже финальный `requirements.txt`.
9. Прогнать полный набор команд проверки (следующий раздел). Если что-то из шагов 3-6 не проходит
   тест — чинить на месте, не откатывать весь план: проблемные точки заранее описаны выше с тем,
   что именно проверять.


## Команды проверки

Создание чистого окружения (не трогать `.venv` в репозитории — на него нет прав на исполнение):

```bash
~/.local/bin/uv venv ~/.venvs/ctb
~/.local/bin/uv pip install --python ~/.venvs/ctb/bin/python -r requirements-dev.txt
```

Полный прогон тестов (per AGENTS.md — весь `tests/` + `bot/tests/`, не только затронутые файлы,
т.к. `PluginManager` грузит все плагины при старте любого теста, который создаёт `PluginManager`):

```bash
~/.venvs/ctb/bin/python -m pytest -q
```

Прицельно — файлы из списка изменений (быстрее, для итерации на месте до полного прогона):

```bash
~/.venvs/ctb/bin/python -m pytest tests/test_telegram_builder_config.py -q
~/.venvs/ctb/bin/python -m pytest bot/tests/test_mcp_server.py -q
~/.venvs/ctb/bin/python -m pytest tests/test_plugin_manager.py tests/test_plugin_handlers_registration.py -q
```

Проверка, что клиент `openai` реально строится новым способом (быстрый ручной smoke-тест без
сети — только конструктор, не реальный запрос):

```bash
~/.venvs/ctb/bin/python -c "
import os
os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'x')
os.environ.setdefault('OPENAI_API_KEY', 'x')
from bot.openai_helper import OpenAIHelper
import httpx2
print('httpx2 OK:', httpx2.__version__)
"
```

CVE-аудит по нашим пинам (pip-audit не вызывает внешние сервисы по расшифровке кода — только
сверяет версии пакетов с публичной базой уязвимостей, сетевой запрос к базе допустим, задача
явно просит этот шаг):

```bash
~/.venvs/ctb/bin/python -m pip_audit -r requirements.txt
```

Проверка Docker-сборки без компилятора (раздел 7) — реальная сборка образа, не имитация:

```bash
docker build -t ctb-t01-check /srv/git_projects/chatgpt-telegram-bot
```

Если сборка падает на каком-то пакете без wheel — увидеть в логе `error: command 'g++' failed`
или похожее, вернуть `g++ libc6-dev` в `Dockerfile` и явно указать при сдаче задачи, из-за какого
пакета они снова нужны.


## Риски и откат

- **`ddg_translate.translate` остаётся сломан** (см. «Расхождения», п.3) — это не регрессия от
  T01 (сломано уже сегодня на `duckduckgo_search==8.1.1`), но при код-ревью это может выглядеть
  как «T01 не починил очевидный баг». Явно оставленный TODO-комментарий в коде — способ не дать
  этому потеряться и не выдать замену импорта за исправление функциональности.
- **Docker-сборка без `g++`/`libc6-dev` может не собраться**, если какой-то из оставшихся пакетов
  (маловероятно, но не проверено фактической сборкой на момент написания этого плана) не публикует
  wheel под используемую платформу/архитектуру. Откат — вернуть обе пакетные строки и
  `apt-get purge` в `Dockerfile`, задача по остальным пунктам (requirements/код) не зависит от
  этого решения.
- **`mcp>=2.1,<3` тянет `starlette`/`sse-starlette`/`uvicorn`/`pyjwt`/`python-multipart`
  транзитивно** (проверено `requires_dist` пакета `mcp==2.1.1` на PyPI) — образ станет тяжелее,
  чем можно было бы ожидать от «просто подняли пин», даже при том что мы одновременно убираем
  прямые строки `fastapi`/`uvicorn`. Это ожидаемая, неизбежная цена перехода на `mcp` 2.x (сам
  SDK теперь тянет свой HTTP/SSE-сервер в зависимостях), не ошибка плана — фиксирую, чтобы не
  удивлялись при ревью `pip list`.
- **`spotipy>=2.26`, `telegramify-markdown>=1.2`, `tiktoken>=0.14.0`** — версии подняты по
  минимальному использованному API (`spotipy.Spotify(...)` в `bot/plugins/spotify.py:21`,
  `telegramify_markdown.convert`/`.split_entities` в `bot/utils.py:437-438`,
  `tiktoken.encoding_for_model`/`.get_encoding`, судя по коду вокруг
  `bot/openai_helper.py:3947`), но не проверены построчным диффом чейнджлогов между текущей и
  целевой версией (не входило в объём проверки — задача просила проверить SDK-типы `openai`
  подробно, остальные пакеты по минимуму). Если полный прогон тестов (обязательный шаг проверки)
  пройдёт зелёным — этого достаточно для этой задачи; если нет — ошибка будет видна в конкретном
  тесте, не «где-то в проде».
- **Откат целиком:** все изменения — это правки существующих файлов + один новый файл
  (`requirements-dev.txt`), без удаления файлов и без БД-миграций. `git checkout --
  requirements.txt bot/openai_helper.py bot/telegram_bot.py bot/plugins/ddg_translate.py
  bot/plugins/youtube_audio_extractor.py Dockerfile tests/test_telegram_builder_config.py &&
  rm requirements-dev.txt` откатывает всё разом.


## Критерии готовности

- [ ] `requirements.txt` содержит только пакеты, реально используемые в `bot/` (прод-код,
  исключая `bot/tests/`/`bot/skills/`) либо явно нужные `codeinterpreter`/`readability-lxml` по
  задокументированным выше причинам; версии соответствуют «Целевые версии» в шапке
  `audit_remediation_plan_2026-09-04.md:20-23`; `mcp` пин — `>=2.1,<3`.
- [ ] `requirements-dev.txt` существует, содержит `-r requirements.txt` + `pytest`,
  `pytest-asyncio`, `ruff`, `pip-audit`; `ipython` нигде не упоминается.
- [ ] `bot/openai_helper.py`: конструктор строит `self._http_client` через `httpx2.AsyncClient`,
  прокидывает `config.get('proxy')`; строка `openai.api_base = ...` удалена; `import httpx`
  заменён на `import httpx2`.
- [ ] `bot/telegram_bot.py`: `run()` вызывает `.proxy()`/`.get_updates_proxy()` на билдере, когда
  `self.config.get('proxy')` не пусто, и не вызывает их вовсе, когда пусто.
- [ ] `bot/plugins/ddg_translate.py` импортирует `from ddgs import DDGS`, с TODO-комментарием
  над `execute()`.
- [ ] `bot/plugins/youtube_audio_extractor.py` импортирует `from pytubefix import YouTube`.
- [ ] `Dockerfile` либо не содержит `g++ libc6-dev` (если сборка прошла), либо содержит — с явной
  пометкой в результатах задачи, из-за какого пакета они остались нужны.
- [ ] `~/.venvs/ctb/bin/python -m pytest -q` — полностью зелёный (весь `tests/` + `bot/tests/`).
- [ ] `~/.venvs/ctb/bin/python -m pip_audit -r requirements.txt` — без findings по пакетам,
  версии которых мы явно пинили в этой задаче (находки по пакетам вне нашего контроля версии,
  например транзитивным — не блокер этой задачи, зафиксировать отдельно).
- [ ] `docker build` образа проходит до конца.
- [ ] Новые тесты `test_telegram_builder_sets_proxy_when_configured` и
  `test_telegram_builder_skips_proxy_when_not_configured` добавлены и проходят;
  `tests/test_plugin_handlers_registration.py` не изменён и продолжает проходить.


## Постскриптум после ревью (2026-09-04)

- `python-dotenv` поднят до `>=1.2.2,<2` (PYSEC-2026-2270).
- `PyPDF2` заменён на `pypdf>=6.0` (PyPDF2 заморожен на 3.0.1 с открытой CVE-2023-36464);
  `bot/plugins/ask_your_pdf.py` и `tests/test_ask_your_pdf.py` переведены на `import pypdf`,
  API `PdfReader`/`pages`/`extract_text` совпадает.
- `Dockerfile`: `g++ libc6-dev` оставлены с поясняющим комментарием — реальную `docker build`
  в этой среде запустить нельзя (нет доступа к docker-сокету); удаление — отдельным шагом
  после проверки сборки.
- `bot/tests/test_mcp_server.py` импортирует `from mcp import types` (в плане сказано
  обратное) — на результат не влияет, mcp 2.1.1 установлен.
