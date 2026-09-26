# T01 — ревью реализации

## Раунд 1

**Проверено:** `git diff HEAD` для `pyproject.toml`, `.github/workflows/ci.yml`,
`.gitignore`, `bot/plugin_manager.py`; новые файлы `scripts/mypy_baseline.py`,
`mypy_baseline.json`, `tests/test_mypy_baseline_script.py`; untracked (не в git, но во
владении задачи) `.cli-proxy/.codebase_map/api/bot/plugin_manager-py.md`. Прогнаны:
`pytest tests/test_mypy_baseline_script.py` (17 passed), `pytest tests/test_plugin_manager.py`
(35 passed), полный `pytest -q` (1734 passed, 0 failed), `ruff check scripts
tests/test_mypy_baseline_script.py` (чисто), `MYPY_PYTHON=~/.venvs/ctb/bin/python python3
scripts/mypy_baseline.py check` (код 0), `mypy bot/plugin_manager.py` напрямую и сверка с
`/tmp/impl/mypy_before.keep`.

**Соответствие плану.**
- `pyproject.toml`: секция `[tool.mypy]` — точно как в плане (`python_version = "3.12"`,
  `ignore_missing_imports = true`, `exclude = ["bot/tests/", "bot/skills/"]`,
  `warn_unused_ignores = true`, `files = ["bot"]`), `check_untyped_defs` не включён — ок.
- `.github/workflows/ci.yml`: шаг `Type check with mypy` вставлен между `Lint with ruff` и
  `Run tests`, `pip install mypy` + `python scripts/mypy_baseline.py check`, YAML валиден
  (`yaml.safe_load` проходит без исключения).
- `.gitignore`: добавлены `.mypy_cache/` и `.coverage`, ничего лишнего не тронуто.
- `scripts/mypy_baseline.py`: публичные функции соответствуют сигнатурам из плана
  (`parse_mypy_output`, `load_baseline`, `save_baseline`, `diff_counts`, `run_mypy`, `main`);
  логика `check`/`update` — regressions → код 1 и печать построчно; improvements без
  regressions → код 0 + подсказка `update`; без изменений → код 0 без подсказки — всё
  проверено вручную прогоном и совпадает с реальным поведением (see «баланс» ниже). Формат
  `mypy_baseline.json` (вложенный `{file: {code: n}}`, отсортированные ключи, `indent=2`,
  завершающий `\n`) подтверждён чтением файла.
- `bot/plugin_manager.py`: 3 метода (`is_subagent_function_allowed`,
  `get_all_plugin_descriptions`, `get_plugin_spec`) удалены целиком, ровно одна пустая
  строка между соседними методами в обоих местах удаления — двойных пустых строк нет.
  Поиск по всему репозиторию (кроме `docs/`, `.ai-docs/`, `.ai_docs_cache/` — вне владения)
  не находит других мест, вызывающих эти методы, кроме regression-теста
  `test_deleted_methods_are_gone`. Импорты `List`/`Dict` не тронуты, используются и дальше
  в файле.
- `.cli-proxy/.codebase_map/api/bot/plugin_manager-py.md`: файл вне git (`.cli-proxy/`
  целиком в `.gitignore`, `git show HEAD:...` подтверждает — путь не в HEAD), поэтому
  `git diff` его не показывает; по mtime виден как отредактированный в то же время, что и
  `plugin_manager.py`. Текущее содержимое уже не содержит записей
  `get_all_plugin_descriptions`/`get_plugin_spec` — соответствует шагу 6 плана.
  `is_subagent_function_allowed` в файле и не был упомянут (план это отдельно оговаривает
  как ранее устаревшую карту) — ничего не пропущено.
- `mypy` напрямую на `bot/plugin_manager.py`: 4 ошибки `[return-value]` на строках 920, 925,
  930, 933 из `/tmp/impl/mypy_before.keep` исчезли вместе с удалёнными методами, новых
  ошибок в файле не появилось.
- `mypy_baseline.json`: формат корректен (проверено по заданию — не сверяю точные счётчики,
  они будут перегенерированы координатором в конце волны).

**Тесты `tests/test_mypy_baseline_script.py`.** Все 16+1 тестов из плана присутствуют и по
существу проверяют заявленное поведение (парсинг ошибок/`note`-строк/итоговой строки/строк
без кода; roundtrip `save`/`load`; рост/новая пара/уменьшение/без изменений в `diff_counts`;
`main` — коды возврата и текст вывода для `check`/`update`/неизвестной команды; regression-
тест на отсутствие удалённых методов). Тесты используют `monkeypatch.setattr(mypy_baseline,
"run_mypy", ...)` и `monkeypatch.setattr(mypy_baseline, "BASELINE_PATH", ...)` — реальный
mypy не запускается, что соответствует требованию «мокать `run_mypy`».

**Находки.** Ничего, что требовало бы исправления. Один теоретический момент — не баг,
привожу для полноты: `MYPY_ERROR_RE`/`MYPY_ERROR_NO_CODE_RE` требуют `:\d+:` после имени
файла, поэтому гипотетическая строка mypy без номера строки (напр. `file.py: error: ...`,
у mypy такое изредка бывает для ошибок уровня модуля) была бы молча проигнорирована и не
попала бы в счётчики. Проверено по `/tmp/impl/mypy_before.keep` (718 строк) — таких строк в
реальном выводе на этой кодовой базе нет, а план такой формат явно не оговаривал. Не поднимаю
до WARNING, так как не воспроизводится и не входит в описанный в плане набор форматов.

**Итог:** 0 ERROR, 0 WARNING, 0 NIT (наблюдение выше — информационное, не квалифицируется
как находка, требующая действия).
