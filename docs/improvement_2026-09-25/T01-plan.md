# T01. Инфраструктура mypy + мёртвый код — план реализации

Источник: `docs/improvement_2026-09-25/00-master-plan.md`, раздел «T01» (строки 58-88).
Проверено против кода на HEAD `08bc457` (git status на момент планирования: только
untracked `docs/improvement_2026-09-25/`, файлы задачи не тронуты).

## Владение файлами (не менять ничего другого)

- `pyproject.toml`
- `.github/workflows/ci.yml`
- `scripts/mypy_baseline.py` (новый)
- `mypy_baseline.json` (новый, корень репозитория)
- `.gitignore`
- `bot/plugin_manager.py` — только удаление 3 методов, больше ничего
- `tests/test_mypy_baseline_script.py` (новый)
- `.cli-proxy/.codebase_map/api/bot/plugin_manager-py.md`

`requirements-dev.txt` в списке владения НЕТ — mypy в CI ставится отдельной командой
`pip install mypy` внутри шага workflow, а не добавлением строки в requirements-dev.txt
(см. «Отклонения» в конце).

## Проверка баз для плана

- `pyproject.toml` сейчас содержит только `[tool.ruff]`/`[tool.ruff.lint]` (21 строка),
  секции `[tool.mypy]` нет.
- `.github/workflows/ci.yml` (38 строк): шаг `Lint with ruff` — строки 29-30, шаг
  `Run tests` — строки 31-32. Между ними и будет вставлен шаг mypy.
- `.gitignore` (33 строки): ни `.mypy_cache/`, ни `.coverage` не перечислены.
- `scripts/` в репозитории не существует (создаётся с нуля, `__init__.py` не нужен —
  проверено: `python -m pytest` добавляет корень репо в `sys.path`, `from scripts import
  mypy_baseline` резолвится как implicit namespace package, эмпирически проверено в
  `/tmp/pkgtest`).
- Методы для удаления в `bot/plugin_manager.py` (1369 строк всего):
  - `is_subagent_function_allowed` — строки **732-741** (мастер-план указывает 732-740;
    фактически последняя строка тела — 741, `return self.is_function_allowed(...)`;
    строка 742 пустая, далее `__get_plugin_by_function_name`). Разница на 1 строку —
    считать источником истины код, не число из мастер-плана.
  - `get_all_plugin_descriptions` — строки **892-920** (совпадает с мастер-планом).
  - `get_plugin_spec` — строки **922-933** (совпадает с мастер-планом).
  - Вызовов этих трёх методов нигде в репозитории (кроме определений и упоминаний в
    `docs/`, `.ai-docs/`, `.ai_docs_cache/`, которые не входят во владение) не найдено —
    проверено рекурсивным поиском по `bot/`, `tests/`, `bot/tests/`.
  - `List`/`Dict` из `typing` (импорт `bot/plugin_manager.py:14`) остаются нужны — в файле
    ещё 18 (`List`) и 16 (`Dict`) других употреблений после удаления методов, импорт не
    трогать.
- В `/tmp/impl/mypy_before.keep` (718 строк, 585 строк `error:`, 133 строки `note:`,
  без итоговой строки `Found N errors...`) воспроизведена эмпирически: команда
  `python3 -m mypy --config-file <toml с [tool.mypy] из шага 1> --python-executable
  ~/.venvs/ctb/bin/python` (без явных путей на файлы, полагаясь на `files = ["bot"]` из
  конфига) без корневой директории `cwd=repo_root` даёт **те же 718 строк** (набор
  идентичен, порядок обхода файлов у mypy отличается от прогона к прогону — это
  не влияет на подсчёт по `(file, code)`). Итог: `Found 585 errors in 53 files (checked
  81 source files)`.
- Среди этих 585 ошибок **4 ошибки лежат внутри удаляемых методов**:
  `bot/plugin_manager.py:920,925,930,933: error: ... [return-value]` — все относятся к
  `get_all_plugin_descriptions`/`get_plugin_spec`, других `[return-value]`-ошибок в
  `bot/plugin_manager.py` нет. Значит после удаления методов итоговый локальный прогон
  mypy даст **581** ошибку, а не 585 — это важно для шага "сгенерировать
  `mypy_baseline.json`" (см. шаг 7 ниже и раздел «Отклонения»).

## Шаги реализации

### 1. `pyproject.toml` — добавить `[tool.mypy]`

Добавить в конец файла (после `[tool.ruff.lint]`, строка 21) новую секцию:

```toml

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
exclude = ["bot/tests/", "bot/skills/"]
warn_unused_ignores = true
files = ["bot"]
```

`check_untyped_defs` не включать (явно оговорено в мастер-плане). Проверено на реальном
прогоне: с этой секцией (`--config-file`, без списка файлов в команде) mypy проверяет
именно `bot/**`, кроме `bot/tests/` и `bot/skills/`, и не задевает `bot/plugins/pdf_cache/`
и т.п. (там нет `.py`).

### 2. `scripts/mypy_baseline.py` — новый файл

CLI: `python3 scripts/mypy_baseline.py update|check`. Ничего, кроме stdlib (`argparse` не
использовать — в `bot/` нет устоявшегося паттерна CLI-скриптов с `argparse`, хватает
`sys.argv`).

Публичные функции (чтобы `tests/test_mypy_baseline_script.py` могли тестировать их без
реального запуска mypy):

```python
MYPY_ERROR_RE = re.compile(
    r'^(?P<file>[^:]+):\d+: error: .*\[(?P<code>[\w-]+)\]\s*$'
)
MYPY_ERROR_NO_CODE_RE = re.compile(r'^(?P<file>[^:]+):\d+: error: .*$')

def parse_mypy_output(output: str) -> dict[tuple[str, str], int]:
    """Count mypy `error:` lines by (file, code); `note:` lines are ignored.
    Errors without a trailing `[code]` (rare) are bucketed under code "no-code"."""
    counts: dict[tuple[str, str], int] = {}
    for line in output.splitlines():
        m = MYPY_ERROR_RE.match(line)
        if m:
            key = (m.group("file"), m.group("code"))
        else:
            m2 = MYPY_ERROR_NO_CODE_RE.match(line)
            if not m2:
                continue
            key = (m2.group("file"), "no-code")
        counts[key] = counts.get(key, 0) + 1
    return counts

def load_baseline(path: Path) -> dict[tuple[str, str], int]:
    """Missing file -> {} (empty baseline, not an error)."""
    ...

def save_baseline(path: Path, counts: dict[tuple[str, str], int]) -> None:
    """Nested JSON {file: {code: count}}, sorted keys, trailing newline."""
    ...

def diff_counts(
    baseline: dict[tuple[str, str], int], current: dict[tuple[str, str], int]
) -> tuple[list[str], list[str]]:
    """Returns (regressions, improvements) as printable "file [code]: N -> M" lines.
    Regression: current > baseline (includes brand-new pairs, baseline=0).
    Improvement: current < baseline (includes pairs that disappeared, current=0)."""
    ...

def run_mypy() -> str:
    """Invoke mypy via subprocess, return combined stdout (ignore returncode: mypy
    exits 1 whenever there is at least one error, which is expected/normal here)."""
    repo_root = Path(__file__).resolve().parent.parent
    config_file = repo_root / "pyproject.toml"
    python_executable = os.environ.get("MYPY_PYTHON", sys.executable)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(config_file),
         "--python-executable", python_executable],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.stdout

def main(argv: list[str]) -> int:
    """argv[0] must be "update" or "check"; anything else -> usage on stderr, exit 2."""
    ...
```

`update`: считает `current = parse_mypy_output(run_mypy())`, пишет
`mypy_baseline.json` через `save_baseline`, печатает
`"mypy_baseline.json updated: {sum} errors across {len} (file, code) pairs."`, `return 0`.

`check`: считает `current`, грузит `baseline = load_baseline(mypy_baseline.json)`,
`regressions, improvements = diff_counts(baseline, current)`.
- `regressions` не пусто → печатает `"mypy baseline regressions:"` + построчно
  `f"  {line}"`, `return 1`.
- иначе, если `improvements` не пусто → печатает
  `"mypy error counts decreased; consider running: python scripts/mypy_baseline.py update"`
  + построчно `improvements`, затем `"mypy baseline check passed."`, `return 0`.
- иначе → просто `"mypy baseline check passed."`, `return 0`.

Формат `mypy_baseline.json` (реальный пример по текущему состоянию репозитория ДО
удаления методов, посчитано из `/tmp/impl/mypy_before.keep`: 585 ошибок, 53 файла, 121
пара `(file, code)`):

```json
{
  "bot/agent_delivery.py": {
    "assignment": 1,
    "misc": 1
  },
  "bot/ai_events.py": {
    "arg-type": 1
  },
  "bot/plugin_manager.py": {
    "return-value": 4
  }
}
```

(После удаления 3 методов запись `"bot/plugin_manager.py": {"return-value": 4}` уйдёт
целиком — см. шаг 7.)

`if __name__ == "__main__": sys.exit(main(sys.argv[1:]))`.

### 3. `.github/workflows/ci.yml` — шаг mypy после ruff

Вставить между текущими строками 30 (`run: ruff check bot tests bot/tests`) и 31
(`- name: Run tests`):

```yaml
      - name: Type check with mypy
        run: |
          pip install mypy
          python scripts/mypy_baseline.py check
```

`mypy` ставится отдельно от `requirements-dev.txt` (см. «Отклонения»). Файл должен
остаться валидным YAML — проверка: `python3 -c "import yaml;
yaml.safe_load(open('.github/workflows/ci.yml'))"`.

### 4. `.gitignore` — добавить 2 строки

В конец файла (после текущей строки 33 `.cli-proxy/runtime/`) добавить:

```
.mypy_cache/
.coverage
```

### 5. Удалить 3 метода в `bot/plugin_manager.py`

Удалить целиком, включая закрывающую пустую строку после каждого (чтобы не оставить
двойной пробел между соседними методами):
- строки 732-742 (метод `is_subagent_function_allowed` + 1 пустая строка) — после
  удаления `filter_allowed_plugins`... нет, после этого метода идёт
  `__get_plugin_by_function_name`; проверить визуально diff, что между
  `get_functions_specs`-блоком (кончается на строке 730 `return filtered,
  allowed_function_names`) и `def __get_plugin_by_function_name` остаётся ровно одна
  пустая строка.
- строки 892-921 (метод `get_all_plugin_descriptions` + 1 пустая строка).
- строки 922-934 (метод `get_plugin_spec` + 1 пустая строка) — после удаления
  `filter_allowed_plugins` (строка 890, `return [p for p in allowed_plugins if p in
  self.plugins]`) должна сразу (через одну пустую строку) идти `def
  has_plugin(self, plugin_name: str) -> bool:`.

Практически: сначала удалить `get_all_plugin_descriptions`+`get_plugin_spec` одним блоком
(строки 892-934, они соседние), затем `is_subagent_function_allowed` (строки 732-742)
отдельной правкой — так номера строк второй правки не съедут после первой (нижний блок
удаляется первым).

Импорт `List`, `Dict` (`bot/plugin_manager.py:14`) не трогать — используются в других
местах файла.

### 6. `.cli-proxy/.codebase_map/api/bot/plugin_manager-py.md`

Удалить строки 31-34 (записи `get_all_plugin_descriptions` и `get_plugin_spec`, каждая —
заголовок + строка описания):

```
- `def get_all_plugin_descriptions()` (line 348)
  - *Get all plugin descriptions from their get_spec methods.*
- `def get_plugin_spec(plugin_name)` (line 378)
  - *Возвращает спецификацию плагина по имени*
```

`is_subagent_function_allowed` в этом файле не упомянут (карта уже устарела и не
отражает добавленные после мая методы) — удалять нечего, это не пропуск с нашей стороны.

Остальное (заголовок `Generated: ...`, номера строк у прочих методов, которые и так уже
разошлись с кодом) не трогать — правка хирургическая, обновление всей карты не входит
в задачу T01. Поле `Last reviewed` в этом файле не существует ни у одного `api/*.md`
(проверено по всем 27 файлам в `api/`) — этот формат используется только в
`nodes/*.md`, которого нет во владении T01 (см. «Отклонения»).

### 7. Сгенерировать `mypy_baseline.json`

После шагов 1, 2, 5 выполнить локально (из корня репозитория):

```bash
MYPY_PYTHON=~/.venvs/ctb/bin/python python3 scripts/mypy_baseline.py update
```

Это создаст `mypy_baseline.json`, отражающий состояние ПОСЛЕ удаления 3 методов —
ожидаемо **581** ошибка (585 минус 4 `[return-value]`, которые лежали внутри удалённых
методов), 52 файла с ошибками в `bot/plugin_manager.py`-записи не будет вовсе.
Это нужно, чтобы выполнить критерий готовности «`python3 scripts/mypy_baseline.py check`
проходит» локально для T01. Координатор, согласно мастер-плану (шаг 6 раздела T01),
перегенерирует этот файл в конце волны 1 после слияния T02-T04 (у них тоже параллельно
меняются файлы в том же рабочем дереве, значит текущий локальный прогон mypy может
отражать частично готовые правки соседних задач — это ожидаемо и не является ошибкой
T01).

## Тесты — `tests/test_mypy_baseline_script.py` (новый файл)

Стиль — как в `tests/test_command_policy.py`: плоские функции, `# --- секция ---`
разделители, `from scripts import mypy_baseline`. Мокать `mypy_baseline.run_mypy` через
`monkeypatch.setattr`, никогда не запускать реальный mypy. Для `update`/`check` через
`main()` — использовать `tmp_path` и `monkeypatch.chdir` или прямую передачу пути
(`main()` сам вычисляет `mypy_baseline.json` рядом со `scripts/../`, поэтому тестам
удобнее тестировать `save_baseline`/`load_baseline`/`diff_counts` как отдельные функции
с явным `tmp_path / "mypy_baseline.json"`, а `main()` — через monkeypatch
`mypy_baseline.__file__`... проще: вынести путь к baseline в отдельный аргумент/константу
`BASELINE_PATH = Path(__file__).resolve().parent.parent / "mypy_baseline.json"` и в тестах
`monkeypatch.setattr(mypy_baseline, "BASELINE_PATH", tmp_path / "mypy_baseline.json")`).

Список тестов:
1. `test_parse_mypy_output_counts_errors_by_file_and_code` — на фикстуре из нескольких
   строк вида `bot/x.py:10: error: msg  [arg-type]` (две одинаковые пары → count=2,
   разные файлы/коды → отдельные ключи) — проверяет словарь `{(file, code): n}`.
2. `test_parse_mypy_output_ignores_note_lines` — вход с `note:`-строками (в т.ч.
   `annotation-unchecked`) вперемешку с `error:` — в результате только error-пары.
3. `test_parse_mypy_output_ignores_summary_line` — строка `Found 2 errors in 1 file
   (checked 1 source file)` не создаёт ложную пару.
4. `test_parse_mypy_output_handles_missing_code` — строка `error:` без `[code]` в конце
   → попадает под ключ с кодом `"no-code"`, не падает и не теряется молча.
5. `test_save_and_load_baseline_roundtrip` — `save_baseline` в `tmp_path`, затем
   `load_baseline` того же пути → тот же словарь `(file, code) -> n`.
6. `test_load_baseline_missing_file_returns_empty` — `load_baseline(tmp_path /
   "nope.json")` → `{}`.
7. `test_diff_counts_detects_growth` — `baseline={(f,c):1}`, `current={(f,c):2}` →
   `regressions` содержит строку с `f`/`c`, `improvements` пуст.
8. `test_diff_counts_detects_new_pair` — пары нет в `baseline`, есть в `current` →
   попадает в `regressions` (это и есть "новая пара" из мастер-плана).
9. `test_diff_counts_detects_decrease` — `baseline={(f,c):3}`, `current={(f,c):1}` →
   `improvements` содержит строку, `regressions` пуст.
10. `test_diff_counts_no_change_is_empty_both` — `baseline == current` → оба списка пусты.
11. `test_main_check_exits_1_on_growth` — `monkeypatch` `run_mypy` на функцию,
    возвращающую текст с большим числом ошибок по паре, чем в baseline-файле (записанном
    заранее в `tmp_path`); `capsys` проверяет код возврата `1` и что вывод содержит
    "regressions".
12. `test_main_check_exits_1_on_new_pair` — аналогично, но пара отсутствует в baseline.
13. `test_main_check_exits_0_and_suggests_update_on_decrease` — baseline с большим
    числом, `run_mypy` возвращает меньше → код `0`, в stdout — подсказка про `update`.
14. `test_main_check_exits_0_when_unchanged` — baseline == current → код `0`, без
    подсказки про `update`.
15. `test_main_update_writes_expected_json` — вызывает `main(["update"])` с
    замоканным `run_mypy`, читает файл, сверяет точное содержимое (структуру
    `{file: {code: n}}`).
16. `test_main_rejects_unknown_command` — `main(["bogus"])` → код `2`, сообщение в
    stderr/stdout содержит `"update|check"` или usage-текст.

Дополнительно (не юнит-тест скрипта, а regression-проверка факта удаления методов) —
можно добавить в тот же файл или отдельно:
`test_deleted_methods_are_gone` — импортирует `PluginManager` из `bot.plugin_manager` и
`assert not hasattr(PluginManager, "is_subagent_function_allowed")` (и два других имени).
Опционально — по вкусу разработчика; явно не требуется мастер-планом, но дёшево и ловит
регресс, если кто-то случайно вернёт метод.

## Критерии готовности (точные команды)

1. `~/.venvs/ctb/bin/python -m pytest tests/test_mypy_baseline_script.py -q --no-header
   -p no:cacheprovider` — зелёный.
2. `~/.venvs/ctb/bin/python -m pytest -q --no-header -p no:cacheprovider` — весь набор
   зелёный (1650+ тестов, ни один не завязан на удалённые методы — проверено, вызовов
   нет).
3. `MYPY_PYTHON=~/.venvs/ctb/bin/python python3 scripts/mypy_baseline.py check` — код
   возврата `0`.
4. `~/.venvs/ctb/bin/python -m ruff check bot tests` — без новых ошибок (скрипт и тест —
   новые файлы, должны пройти под текущий `select = ["E4", "E7", "E9", "F"]`).
5. `python3 -c "import ast; ast.parse(open('bot/plugin_manager.py').read())"` — синтаксис
   не сломан (быстрая проверка перед mypy/pytest).
6. `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` — не
   бросает исключение (валидный YAML).
7. Убедиться, что в `bot/plugin_manager.py` нет вхождений `is_subagent_function_allowed`,
   `get_all_plugin_descriptions`, `get_plugin_spec` (grep/поиск по файлу).

## Риски

- **Версия mypy в CI не пиновалась.** CI ставит `pip install mypy` без версии — если
  апстрим выпустит mypy с новыми проверками по умолчанию, число ошибок может вырасти
  само по себе и уронить `check` без изменений в коде. Мастер-план версию не требует;
  если станет проблемой — пиновать в отдельной задаче.
- **Параллельные задачи волны 1 меняют тот же `bot/`.** Baseline, сгенерированный T01
  локально (шаг 7), отражает срез рабочего дерева на момент генерации; T02-T04 могут
  поменять типы ошибок в своих файлах. Мастер-план явно перекладывает финальную
  генерацию на координатора (шаг 6 раздела T01) — T01 не должен пытаться синхронизировать
  baseline с чужими незавершёнными правками.
- **`.mypy_cache/` может уже существовать локально** от предыдущих ручных прогонов mypy
  вне git — не отслеживается git, но стоит убедиться, что `git status --short` не
  показывает такую директорию как untracked после прогонов (добавление в `.gitignore`
  в шаге 4 это закрывает).
- **`get_plugin_spec` возвращал `None` при ошибке, хотя тип — `List[Dict]` без
  `Optional`.** Это тот самый источник 4 удаляемых mypy-ошибок; поскольку метод целиком
  удаляется, а не чинится, это не противоречит правилу "не менять tool specs" — метод не
  является `get_spec()` плагина, это внутренний хелпер `PluginManager`.

## Отклонения от буквального текста мастер-плана (для финального ответа)

1. `requirements-dev.txt` не во владении T01 → mypy ставится в CI отдельной командой
   `pip install mypy`, а не строкой в `requirements-dev.txt`.
2. Точные границы `is_subagent_function_allowed` — строки 732-741 (не 732-740, как в
   мастер-плане); код — источник истины.
3. Фраза мастер-плана «обновить узел карты кода (убрать методы, `Last reviewed:
   2026-09-25`)» относится к формату `nodes/*.md` (поле `Last reviewed` есть только там,
   например `nodes/bot.md`), а владение T01 ограничено `api/bot/plugin_manager-py.md`, у
   которого такого поля нет ни у одного файла в `api/`. План ограничивается правкой
   владеемого api-файла (удалить 2 устаревшие записи) и не трогает `nodes/bot.md`.
4. `mypy_baseline.json`, сгенерированный в шаге 7, будет отражать 581 ошибку (не 585) —
   удаление 3 методов само устраняет 4 mypy-ошибки внутри них; это ожидаемо и не
   является отдельной "починкой багов".
