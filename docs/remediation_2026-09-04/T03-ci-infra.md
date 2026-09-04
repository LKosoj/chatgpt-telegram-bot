# T03. CI и инфраструктура

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел «T03. CI и инфраструктура»),
подкреплено находками из `docs/architecture_code_review_2026-09-04.md` §3.4 («CI не запускает
тесты и работает на неподдерживаемом Python») и §4.5 («Инфраструктура и гигиена репозитория»).
Роль этого документа — план для разработчика; код не менялся, только прочитан и проверен
инструментами (`ruff`, `docker compose config`, GitHub API) без коммитов.

Термины, которые встречаются ниже: **CI** (Continuous Integration — автоматическая проверка
кода на сервере GitHub при каждом push/PR), **linter/линтер** (`ruff` — инструмент, который
статически ищет ошибки и плохой стиль в коде без его запуска), **F401/E501 и т.п.** — это коды
конкретных правил `ruff`; F401 значит «импортировали, но не используем», E501 — «слишком
длинная строка». **dockerignore** — файл, который говорит `docker build`, какие файлы НЕ
копировать внутрь образа (аналог `.gitignore`, но для Docker).

## Цель

1. `.github/workflows/ci.yml` — реально гонять весь набор тестов на Python 3.12 плюс `ruff` и
   `pip-audit` (проверку зависимостей на известные уязвимости); удалить нерабочий
   `python-package-conda.yml`; точечно обновить версии actions в `publish.yaml`.
2. Завести `pyproject.toml` с конфигом `ruff`, который фиксирует то, что фактически уже
   применяется по умолчанию, плюс закрыть 2 реальных найденных замечания (F401).
3. Расширить `.dockerignore`, чтобы образ не тащил тесты/доки/агентские артефакты — но не
   сломать при этом файл, который код реально читает в рантайме.
4. `docker-compose.yml` — сделать `docker compose up` рабочим «из коробки» без сети до
   несуществующего local Bot API сервиса; сверить README на соответствие новому поведению.
5. Команда `chmod 600 .env` — закрыть groupwrite-доступ к секретам на диске.
6. Поправить в README.md/README.ru.md реальный минимум Python (3.9+ → 3.11+).

## Допущения

- `requirements-dev.txt` создаёт параллельная задача T01 (той же волны, независимые файлы);
  этот план ссылается на файл, предполагая, что он появится с составом `-r requirements.txt` +
  `pytest`, `pytest-asyncio`, `ruff`, `pip-audit` — как написано в `audit_remediation_plan`. Пока
  T01 не смёржен, шаг `pip install -r requirements-dev.txt` в новом `ci.yml` будет падать —
  это ожидаемо, не баг этого плана (см. «Порядок»).
- `bot/tests/` (вложенный в `bot/`) НЕ добавляю в `.dockerignore` — в задаче явно назван только
  верхнеуровневый `tests/`. Он тоже тестовый и тоже не нужен в образе, но раз не попросили —
  оставляю как есть; при желании добавить по аналогии одной строкой `bot/tests`.
- `pyproject.toml` не получает `[build-system]` — проект не паковается через `pip install .`
  (запускается как `python -m bot` из `requirements.txt`), поэтому секция не нужна и её
  отсутствие ничего не ломает.
- Версии GitHub Actions (`checkout`, `setup-python`, `docker/*`) проверены живым запросом к
  `api.github.com` 2026-09-04 (см. «Находки»), а не по памяти — совпадает с правилом AGENTS.md
  не делать runtime-заявления без проверки.

## Находки, которые меняют дословную формулировку задачи

### 1. Реальные текущие версии actions новее, чем предполагала задача

Задача предлагала «checkout v4/v5, setup-python v5». Прямой запрос к GitHub API
(`curl https://api.github.com/repos/actions/checkout/releases/latest` и аналогично для других)
показал на 2026-09-04:

| Action | В файлах сейчас | Реальный текущий major |
|---|---|---|
| `actions/checkout` | `v3` (ci.yml), `v3` (publish.yaml) | **v7** (v7.0.1) |
| `actions/setup-python` | `v4` (ci.yml) | **v7** (v7.0.0) |
| `docker/setup-qemu-action` | `v2` | **v4** (v4.3.0) |
| `docker/setup-buildx-action` | `v2` | **v4** (v4.3.0) |

Проверил, что мажорные плавающие теги (`v7`, `v4` и т.д.) реально существуют как git-теги в
этих репозиториях (`GET /repos/<r>/git/ref/tags/<tag>` → 200), и что changelog версий не
содержит несовместимых с текущим использованием изменений (ни один из используемых inputs —
`python-version`, `cache`, `registry`/`username`/`password`, `images`, `context`/`platforms`/
`push`/`tags`/`labels` — не переименован и не удалён в новых мажорах). Использую реальные
текущие мажоры, а не более старые из формулировки задачи.

### 2. SHA-пины в `publish.yaml` — не просто «старые теги», а версии 2021 года

`docker/login-action`, `docker/metadata-action`, `docker/build-push-action` в `publish.yaml`
запинены на конкретные git-коммиты (SHA), а не на теги — это правильная практика с точки зрения
безопасности (тег можно переписать, коммит — нет), но сами закреплённые коммиты оказались очень
старыми. Проверка через `GET /repos/<repo>/tags` (сравнение SHA):

| Action | Текущий SHA-пин | Реальная версия | Дата |
|---|---|---|---|
| `docker/login-action` | `f054a8b5...` | **v1.10.0** | 2021-06-22 |
| `docker/metadata-action` | `98669ae8...` | **v3.3.0** | 2021-05-25 |
| `docker/build-push-action` | `ad44023a...` | **v2.5.0** | 2021-05-26 |

Это отстаёт от текущих `v4.6.0` / `v6.2.0` / `v7.3.0` на 3–5 лет. Задача просила по
`publish.yaml` только «минимальные правки, не переписывать» — оставляю тот же стиль
(SHA-пины, тот же набор шагов, тот же реестр), но обновляю сами закреплённые коммиты, потому
что «пин на 2021 год» — это не консервативность, а просто забытое обновление.

### 3. Критично: блок `*.md` в `.dockerignore` задел бы реально нужный в рантайме файл

Задача просила исключить `*.md`, «кроме нужных», и явно спросить, не читает ли код
README/переводы в рантайме. Проверил grep'ом по всем `.py` в `bot/`:

- `translations.json` (`bot/i18n.py:46`) — это `.json`, не `.md`, `*.md` его не заденет.
  Ложная тревога, но проверить стоило.
- **`bot/prompts/subagent_system.md`** — читается по прямому пути в
  `bot/plugins/agent_tools.py:172`:
  `_SUBAGENT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "subagent_system.md"`.
  Файл отслеживается git'ом (`git ls-files` подтверждает). Если добавить `*.md` в
  `.dockerignore` без исключения, этот файл не попадёт в образ и Docker-сборка молча сломает
  системный промпт саб-агентов в проде (упадёт при первом обращении к файлу или тихо отдаст
  старое поведение — зависит от того, есть ли try/except вокруг чтения; проверять не стал,
  это не тот путь, который стоит проверять чтением ошибки в проде).
  **Решение:** после строки `*.md` добавить строку-исключение (negation-pattern, начинается
  с `!` и возвращает файл обратно после того как его исключили) `!bot/prompts/*.md`.
- `bot/README_MCP.md` — тоже `.md`, но нигде не читается кодом (только для людей) — можно
  спокойно исключать вместе со всеми остальными `.md`.

### 4. `ruff check` без конфига уже сегодня не 100% чист — 9 замечаний, не 2, если смотреть шире `bot tests`

Задача говорит «2 × F401 в тестах» — это верно **только** для `bot tests bot/tests`. Прогнал
`ruff check . --select E4,E7,E9,F` (весь репозиторий) — нашлось **9** F401, из них 7 в
`examples/mcp_server_example.py`, `mcp_stdio_client.py`, `mcp_stdio_server.py` (эти файлы вне
`bot`/`tests`, поэтому CI их не увидит, но голый `ruff check .` в IDE или будущем pre-commit —
увидит). Поэтому в `pyproject.toml` в `exclude` добавляю `examples` (в задаче было только
«ai_docs_site, .ai-docs, evals?» через вопросительный знак) — иначе конфиг с `select` без
`exclude` даёт разный результат в зависимости от того, что именно указано в командной строке,
а это ломает саму идею «зафиксировать текущее состояние». `docs/`, `ai_docs_site/`, `.ai-docs`
не содержат `.py` файлов вообще (проверено `find ... -name '*.py'`), поэтому их наличие или
отсутствие в `exclude` для `ruff` практически не важно — включаю `ai_docs_site` и `.ai-docs`
в `exclude` только для явности (что каталог не предназначен для линтера), `evals` — исключаю
и по факту (0 замечаний), и по духу правила AGENTS.md «evals никогда не часть routine
verification» (см. раздел Testing And Verification: evals гарантированно не должен попадать ни
в pytest, ни в CI — то же самое разумно применить к ruff, третьей защитой к уже существующим
трём).

### 5. `line-length`: измеренный максимум (632) больше жёсткого потолка ruff (320)

Задача просила задать `line-length` «по факту кода — измерь». Измерил (Python-скрипт по всем
`.py` в `bot/`, `tests/`, `bot/tests/`): максимальная строка — **632 символа**
(`bot/plugins/show_me_diagrams.py:295`), из них **12 строк длиннее 320** символов ещё в 4
файлах (`bot/plugins/codeinterpreter.py:255`, `bot/plugins/github_analysis.py:127`,
`bot/html_utils.py` — 8 строк, `bot/openai_helper.py:4283`). У `ruff` `line-length` — жёсткий
потолок в 320 (`--line-length` со значением больше 320 сам `ruff` отклоняет с ошибкой), то есть
задать «фактический максимум» технически невозможно.

Проверил, из-за чего фактически 0 замечаний сегодня: правило `E501` (длина строки) НЕ входит
в набор правил, которые `ruff` включает по умолчанию без конфига (`E4, E7, E9, F` — это именно
то, что означает фраза аудита «pyflakes/ruff check bot/ дают 0 замечаний, но это ничем не
закреплено»). Значит буквальная просьба задачи «правила E/F» — это про то, чтобы явно
прописать в файле именно этот набор (иначе он молча меняется вместе с версией `ruff`), а не
про включение полного E1-E9 (что означало бы 3857 новых E501 при `line-length=88` или минимум
12 при `line-length=320` — то и другое противоречит фразе задачи «2 × F401» как единственному
ожидаемому результату). Проверил все три варианта командой `ruff check bot tests bot/tests
--select ... --no-cache`.

**Решение:** `select = ["E4", "E7", "E9", "F"]` (буквально текущее поведение `ruff` по
умолчанию, зафиксированное явно) и `line-length = 320` — не как работающее ограничение (при
таком `select` оно ни на что не влияет), а как задокументированный потолок «где ruff вообще
способен считать» с комментарием в файле, объясняющим, почему число не 632. Проверено
финальной командой (см. «Команды проверки») — даёт ровно 2 замечания, оба уже известные F401.

### 6. `docker-compose.yml`: смена дефолта — это поведенческое изменение, не только «правка YAML»

Сейчас `docker-compose.yml` не задаёт `TELEGRAM_LOCAL_MODE` вообще, значит внутри контейнера
действует дефолт из кода — `True` (`bot/__main__.py:293`:
`parse_bool_env('TELEGRAM_LOCAL_MODE', True)`), то есть бот пытается стучаться на
`http://localhost:8081/bot` **внутри своего же контейнера**, где такого сервиса нет — сеть
недоступна, `docker compose up` не работает «из коробки», как и написано в аудите §4.5.

Проверил механику `${VAR:-default}` реальным `docker compose config` (не по памяти): Compose
подставляет значение из того же файла `.env`, который лежит в корне проекта (того же самого,
что подключён через `env_file: .env`), а если переменной там нет (как в `.env.example:45` —
`# TELEGRAM_LOCAL_MODE=true`, закомментировано) — подставляет дефолт `false`. Прогнал на
реальных `docker-compose.yml` + `.env.example` (переименованном в `.env`) — результат
`docker compose config` подтвердил: `TELEGRAM_LOCAL_MODE: "false"`.

Это значит: после правки `docker compose up` с чистым `.env.example` по умолчанию пойдёт на
**хостируемый** Telegram API (без local-mode), а не будет пытаться достучаться до
несуществующего сервиса. Для тех, кто раньше руками не прописывал `TELEGRAM_LOCAL_MODE` в
`.env`, но параллельно поднимал свой local Bot API сервис (например, через сеть Docker) —
поведение при следующем `docker compose up` изменится (было true по умолчанию, станет false).
Это стоит явно отметить как поведенческое изменение (см. «Риски»), а не просто «правка
инфраструктуры».

## Правки по файлам

### 1. `.github/workflows/ci.yml` — полный итоговый текст

Текущий файл (не менялся с 2025-05-24, больше года, Python 3.10, гоняет только
`bot/tests/test_mcp_server.py` дважды, не трогает 1364 теста из `tests/`) заменяется целиком:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - uses: actions/checkout@v7
      - name: Set up Python
        uses: actions/setup-python@v7
        with:
          python-version: '3.12'
          cache: 'pip'
          cache-dependency-path: requirements-dev.txt
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      - name: Lint with ruff
        run: ruff check bot tests
      - name: Run tests
        run: python -m pytest -q
      - name: Audit dependencies
        run: pip-audit -r requirements.txt
```

Пояснения к решениям:
- `python -m pytest -q` без явных путей — подхватит `testpaths = tests bot/tests` из
  `pytest.ini` (уже настроено, не мой файл, не трогаю). Так одна команда покрывает и
  `tests/`, и `bot/tests/`, вместо старого дублирования одного файла дважды.
- Порядок шагов — сначала быстрый линтер, потом тесты, потом `pip-audit` (обращается к
  внешней базе уязвимостей, потенциально медленнее и менее стабилен по сети) — чтобы дешёвые
  проверки падали быстрее дорогих.
- `cache-dependency-path: requirements-dev.txt` в шаге `setup-python` — явно указываю файл,
  который реально устанавливается (`pip install -r requirements-dev.txt`), а не полагаюсь на
  автопоиск `setup-python` по маске `**/requirements*.txt` (нашёл бы и `requirements.txt`, и
  `requirements-dev.txt`, но ключ кэша тогда зависел бы от обоих файлов сразу, включая тот, что
  напрямую не ставится).
- `timeout-minutes: 10` — не менял. Замерил локально: `python3 -m pytest -q` полного набора
  (1450 тестов) занимает **50.10 сек** на этой машине (`1449 passed, 1 skipped`). Даже с
  запасом на более медленный раннер GitHub и время установки зависимостей (~1-2 мин на `pip
  install` из ~30 пакетов) 10 минут — комфортный запас; если после первого реального прогона
  CI оказалось тесно — поднять до 15.
- Никаких `permissions:` не добавлял — job ничего не пушит и не создаёт релизы, дефолтные
  read-only права GITHUB_TOKEN достаточны, как и в исходном файле.

### 2. Удалить `.github/workflows/python-package-conda.yml`

Команда: `git rm .github/workflows/python-package-conda.yml` (или обычное удаление файла +
`git add` при коммите — коммитить в рамках этого плана не нужно, планировщик не коммитит).
Причина — из аудита §3.4: ставит `environment.yml` без `pytest-asyncio`, поэтому async-тесты
не выполняются вообще; единственная проверка — `flake8 --select=E9,F63,F7,F82` (даже уже, чем
дефолт `ruff`). Дублирует и хуже нового `ci.yml`.

### 3. `.github/workflows/publish.yaml` — точечные правки (не переписывать)

Три изменения версий actions в существующих строках, остальной файл (шаги, реестры, теги,
секреты) не трогаем:

```diff
       - name: Check out the repo
-        uses: actions/checkout@v3
+        uses: actions/checkout@v7

       - name: Set up QEMU
-        uses: docker/setup-qemu-action@v2
+        uses: docker/setup-qemu-action@v4

       - name: Set up Docker Buildx
-        uses: docker/setup-buildx-action@v2
+        uses: docker/setup-buildx-action@v4

       - name: Log in to Docker Hub
-        uses: docker/login-action@f054a8b539a109f9f41c372932f1ae047eff08c9
+        uses: docker/login-action@dbcb813823bdd20940b903addbd779551569679f

       - name: Log in to the Container registry
-        uses: docker/login-action@f054a8b539a109f9f41c372932f1ae047eff08c9
+        uses: docker/login-action@dbcb813823bdd20940b903addbd779551569679f

       - name: Extract metadata (tags, labels) for Docker
         id: meta
-        uses: docker/metadata-action@98669ae865ea3cffbcbaa878cf57c20bbf1c6c38
+        uses: docker/metadata-action@dc802804100637a589fabce1cb79ff13a1411302

       - name: Build and push Docker images
-        uses: docker/build-push-action@ad44023a93711e3deb337508980b4b5e9bcdc5dc
+        uses: docker/build-push-action@53b7df96c91f9c12dcc8a07bcb9ccacbed38856a
```

Новые SHA соответствуют тегам `docker/login-action@v4.6.0`, `docker/metadata-action@v6.2.0`,
`docker/build-push-action@v7.3.0` (проверено через GitHub API — SHA коммита совпадает с тегом).
Сохраняю существующий стиль (пин на коммит, не на плавающий тег) — это правильнее с точки
зрения безопасности, просто обновляю сами коммиты.

**Не трогаю, но флагаю отдельно (нужно решение владельца, не инженерный факт):**
`publish.yaml:44-46` пушит образ в `n3d1117/chatgpt-telegram-bot` — это неймспейс DockerHub
исходного (upstream) проекта `n3d1117`, а не форка. Секреты `DOCKER_USERNAME`/
`DOCKER_PASSWORD` в этом репозитории (судя по `origin` — `LKosoj/chatgpt-telegram-bot`,
см. находку в `T04-repo-hygiene.md`) почти наверняка принадлежат не аккаунту `n3d1117`, то
есть либо этот шаг реально не работает (нет доступа), либо (хуже) кто-то залил свои секреты
от чужого DockerHub-неймспейса. Задача просила «минимальные правки, не переписывать» — менять
эту строку без подтверждения владельца не стал; это решение про то, куда вообще публиковать
образ, а не про версии actions.

### 4. `pyproject.toml` — новый файл, полный текст

```toml
[tool.ruff]
# line-length измерен по факту кода (bot/, tests/, bot/tests/): максимальная строка —
# 632 символа (bot/plugins/show_me_diagrams.py:295), 12 строк длиннее 320 в ещё 4 файлах.
# 320 — это жёсткий потолок самого ruff (--line-length не принимает больше), поэтому точно
# повторить факт (632) невозможно технически. E501 (правило "слишком длинная строка")
# сознательно НЕ включено в select ниже, поэтому это значение сейчас ни на что не влияет
# при `ruff check` — это просто задокументированный потолок на будущее.
target-version = "py312"
line-length = 320
exclude = [
    "ai_docs_site",
    ".ai-docs",
    "evals",
    "examples",
]

[tool.ruff.lint]
# E4 (import), E7 (statement), E9 (синтаксис/runtime-ошибки) + F (pyflakes) — это ровно
# набор, который ruff применяет по умолчанию без всякого конфига. Фиксируем его явно, чтобы
# поведение линтера не менялось молча при обновлении ruff (см. docs/architecture_code_review
# _2026-09-04.md §3.4).
select = ["E4", "E7", "E9", "F"]
```

Не конфликтует с `pytest.ini`: у pytest `pytest.ini` в корне имеет приоритет над
`[tool.pytest.ini_options]` в `pyproject.toml`, а этот `pyproject.toml` такую секцию и не
объявляет — проверено чтением `pytest.ini` (существует, `testpaths = tests bot/tests`).

### 5. Закрыть 2 × F401, которые ловит `ruff check bot tests bot/tests`

`tests/test_hindsight_burst_buffer.py:13`:
```diff
-from unittest.mock import AsyncMock, MagicMock
+from unittest.mock import AsyncMock
```
(`MagicMock` больше нигде в файле не встречается — проверено `grep -n MagicMock`.)

`tests/test_stream_usage.py:11`:
```diff
 import types

-import pytest

 from tests.test_openai_helper_tool_calls import (
```
(`pytest.` нигде в файле не вызывается — проверено `grep -n "pytest\."`, файл полагается на
глобальный `asyncio_mode = auto` из `pytest.ini`, явного использования модуля `pytest` нет.)

### 6. `.dockerignore` — добавить блок в конец файла (существующие 25 строк не трогаем)

```
bot/skills/
ai_docs_site
tests
.attachments
docs
evals
examples
.ruff_cache
*.md
!bot/prompts/*.md
```

Обоснование по каждой строке — в разделе «Находки» выше (`.ai-docs` уже есть в файле без
слэша — не дублирую и не трогаю). `!bot/prompts/*.md` обязателен после `*.md` и должен идти
именно после неё, иначе `bot/plugins/agent_tools.py:172` не найдёт файл в собранном образе.

### 7. `docker-compose.yml` — добавить одну переменную окружения

```diff
     environment:
+      TELEGRAM_LOCAL_MODE: ${TELEGRAM_LOCAL_MODE:-false}
       DB_PATH: ${DB_PATH:-/app/data/user_data.db}
       PLUGIN_STORAGE_ROOT: ${PLUGIN_STORAGE_ROOT:-/app/data}
       SESSION_LOG_DIR: ${SESSION_LOG_DIR:-/app/log}
       BOT_DATA_DIR: ${BOT_DATA_DIR:-/app/data}
       BOT_OUTPUT_DIR: ${BOT_OUTPUT_DIR:-/app/output}
       BOT_PLOTS_DIR: ${BOT_PLOTS_DIR:-/app/plots}
       SKILLS_DIR: /app/data/skills
       SKILLS_WORKDIR: /app/data/skill_workdir
```

Проверено `docker compose config` на реальных `docker-compose.yml` + `.env.example`
(переименованном в `.env` во временной копии — исходные файлы репозитория не трогал):
с закомментированной `# TELEGRAM_LOCAL_MODE=true` в `.env.example` результат —
`TELEGRAM_LOCAL_MODE: "false"`; если пользователь раскомментирует и поставит `true` в своём
`.env` — так и подставится `"true"`.

### 8. README.md / README.ru.md

**Python версия.** `README.md:194`:
```diff
-- **Python 3.9+** (3.12 is what the project is currently developed against).
+- **Python 3.11+** (3.12 is what the project is currently developed against).
```
`README.ru.md:201`:
```diff
-- **Python 3.9+** (текущая разработка идёт на 3.12).
+- **Python 3.11+** (текущая разработка идёт на 3.12).
```
Причина минимума именно 3.11, не выдумана: `bot/utils.py` использует `asyncio.timeout`
(появился в 3.11), `bot/plugins/plugin.py:11` использует синтаксис `str | None` (3.10+) — из
двух ограничений старше 3.11.

**Docker/local Bot API раздел.** Добавить одно уточняющее предложение перед уже существующим
`README.md:233-235` (не убирая и не переписывая существующий текст — он остаётся верным
для тех, кто вручную включит local-mode):

```diff
 If you used the old full-repository bind mount, copy any existing
-`bot/user_data.db*` files into the new data volume before switching. If a local
+`bot/user_data.db*` files into the new data volume before switching. By default,
+`docker-compose.yml` sets `TELEGRAM_LOCAL_MODE=false` (there is no bundled local Bot API
+service in Compose), so a fresh `docker compose up` talks to Telegram's hosted API without
+extra setup; set `TELEGRAM_LOCAL_MODE=true` in `.env` only if you also run your own reachable
+local Bot API server. If a local
 Telegram Bot API server is not reachable from inside the container, set
 `TELEGRAM_LOCAL_MODE=false` or point `TELEGRAM_BASE_URL` at a reachable host.
```

`README.ru.md:241-243` — тот же смысл по-русски:
```diff
 использовался старый bind-mount всего репозитория, перед переходом скопируй
-существующие `bot/user_data.db*` в новый data-volume. Если локальный Telegram
+существующие `bot/user_data.db*` в новый data-volume. По умолчанию `docker-compose.yml`
+задаёт `TELEGRAM_LOCAL_MODE=false` (в Compose нет своего сервиса local Bot API), поэтому
+`docker compose up` из коробки идёт на хостируемое Telegram API без дополнительной
+настройки; включай `TELEGRAM_LOCAL_MODE=true` в `.env`, только если параллельно поднимаешь
+свой достижимый local Bot API сервер. Если локальный Telegram
 Bot API сервер недоступен из контейнера, задай `TELEGRAM_LOCAL_MODE=false` или
 укажи достижимый `TELEGRAM_BASE_URL`.
```

Раздел «Requirements» (`README.md:197-199` / `README.ru.md:203-205`, «the bot defaults to
`TELEGRAM_LOCAL_MODE=true`») не трогаю — это верно для обычного `python -m bot` запуска
(дефолт в `bot/__main__.py:293` не меняется, меняется только Compose-специфичный дефолт).

### 9. Права на `.env`

Проверено на диске: `.env` сейчас `-rwxrwx---` (770 — группа может читать/писать/исполнять,
включая произвольных пользователей в группе `cli-proxy-workgroup`). Команда:

```bash
chmod 600 .env
```

Это не правка кода, выполняется локально на машине/сервере, где лежит рабочий `.env` (и
отдельно на каждом окружении, где он есть — в самом репозитории `.env` не коммитится).

## Порядок выполнения

1. `.dockerignore`, `docker-compose.yml`, README-правки, `chmod 600 .env` — независимы друг от
   друга и от остального, можно делать в любом порядке первыми.
2. `pyproject.toml` + 2 × F401 fix — тоже независимы, но их стоит сделать до/вместе с `ci.yml`,
   раз `ci.yml` начинает вызывать `ruff check bot tests` (иначе первый же CI-прогон падает на
   уже известных 2 замечаниях).
3. `.github/workflows/ci.yml` — зависит от того, что `requirements-dev.txt` из T01 уже
   существует в ветке/PR к моменту, когда CI реально запустится. Если T01 и T03 идут отдельными
   PR — либо мёржить T01 первым, либо держать оба изменения в одном PR/ветке волны 1, как и
   описано в `audit_remediation_plan_2026-09-04.md` (волна 1, независимые файлы, но общий
   зелёный CI нужен на объединённом дереве).
4. `.github/workflows/python-package-conda.yml` (удаление) и `publish.yaml` (правки версий) —
   независимы от всего остального, любой момент.

## Команды проверки

```bash
# 1. Синтаксис YAML (не через rg/grep — по правилам окружения используем python3 -c)
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ci.yml OK')"
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yaml')); print('publish.yaml OK')"

# 2. Синтаксис TOML
python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb')); print('pyproject.toml OK')"

# 3. ruff — должно остаться ровно 0 замечаний после фикса 2 × F401
ruff check bot tests bot/tests --no-cache

# 4. Полный набор тестов (как и будет делать CI после T01)
python3 -m pytest -q

# 5. docker-compose.yml — валидность конфигурации (нужен .env; временно cp .env.example .env,
#    если .env ещё не создан)
docker compose config

# 6. Опционально (требует полной пересборки образа, дольше): проверить, что нужный .md-файл
#    реально попал в образ несмотря на общее исключение "*.md"
docker build -t ctb-t03-check . \
  && docker run --rm ctb-t03-check python -c \
     "from pathlib import Path; assert Path('/app/bot/prompts/subagent_system.md').exists()"
```

## Риски

- **Поведенческое изменение в `docker-compose.yml`.** Кто-то, кто уже поднял отдельный local
  Bot API сервис и запускал `docker compose up` без явного `TELEGRAM_LOCAL_MODE` в `.env`,
  после этой правки получит `false` вместо прежнего неявного `true` — бот начнёт идти на
  хостируемое API. Нужно явно упомянуть в PR/changelog, не только в README.
- **`ci.yml` временно красный до мёржа T01.** Шаг `pip install -r requirements-dev.txt`
  сломается, пока файла нет. Это ожидаемо (см. «Порядок»), но если T03 попадёт в `main` раньше
  T01 — CI будет виден как падающий, это может смутить смотрящих на бейдж.
- **`pip-audit` может найти новые CVE после публикации этого плана** (база уязвимостей
  обновляется постоянно) — команда написана так, что упадёт на любом найденном advisory по
  текущим пинам; это ожидаемое поведение шага, а не баг конфигурации.
- **Скачок мажоров actions на 3-5 лет** (`docker/login-action` v1→v4, `metadata-action` v3→v6,
  `build-push-action` v2→v7). Проверил, что используемые input-имена не изменились, но полный
  список поведенческих изменений между этими версиями не вычитывал построчно (не входит в
  разумный объём для «минимальной правки инфраструктуры»). Отдельно: `build-push-action` v5+
  по умолчанию включает provenance-attestation (`provenance: true` по умолчанию) — в v2.5.0 этого
  не было; для DockerHub/GHCR это обычно безвредно, но если после мёржа `publish.yaml` реально
  запустится (нужны настоящие secrets) и что-то не так с публикацией — первое, что проверить:
  `provenance: false` в `with:` шага build-push-action.
- **`line-length = 320` в `pyproject.toml` бездействует.** Явно объяснено комментарием в файле
  и в этом плане, но при беглом просмотра конфига можно ошибочно решить, что строки длиннее 320
  символов сейчас ловятся — это не так, пока `E501`/`E5` не добавлены в `select`.
- **Namespace в `publish.yaml`** (`n3d1117/chatgpt-telegram-bot`) — сознательно не менял,
  но это открытый вопрос, требующий решения владельца репозитория, не инженерного факта.

## Критерии готовности

- `ruff check bot tests bot/tests --no-cache` — 0 замечаний.
- `python3 -m pytest -q` — все тесты проходят (сейчас: 1449 passed, 1 skipped, ни один тест
  этим планом не менялся, значит число не должно измениться, кроме двух тестовых файлов, где
  правится только импорт).
- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` и то же для
  `publish.yaml` — без исключений.
- `python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"` — без исключений.
- `docker compose config` (с любым валидным `.env`) печатает
  `TELEGRAM_LOCAL_MODE: "false"`, если переменная не задана в `.env`, и реальное значение
  переменной, если она задана.
- `.github/workflows/python-package-conda.yml` отсутствует в дереве.
- `bot/prompts/subagent_system.md` присутствует в собранном Docker-образе (см. команда 6 в
  «Команды проверки»).
- `.env` на диске — права `600` (`ls -la .env` начинается с `-rw-------`).
- В README.md и README.ru.md больше нет строки «Python 3.9+».


## Постскриптум после ревью (2026-09-04)

- `bot/skills/` в `.dockerignore`: безопасно. `bot/plugins/skills.py` берёт каталог скиллов из
  `SKILLS_DIR` (в docker-compose — `/app/data/skills`, volume) либо `<storage_root>/skills` и
  никогда не читает `bot/skills/` по жёсткому пути; каталог и так был в `.gitignore`.
- Комментарии в `pyproject.toml` про «дефолтный набор ruff» и «потолок 320» были неверны и
  исправлены: ruff 0.16 без конфига включает сотни правил, `select` здесь — сознательно узкий.
- В `ci.yml` линтер запускается на `bot tests bot/tests`.
