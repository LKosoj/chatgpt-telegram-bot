# T06 — ревью

## Раунд 1

**Проверено:** `git diff HEAD` для `bot/chat_modes_registry.py`, `bot/database.py`,
`tests/test_chat_modes_registry.py`, `tests/test_database.py`; прямой осмотр (не через git,
файлы в `.gitignore`) `bot/chat_modes.yml` и всего `bot/skills/**`; новый файл
`tests/test_skills_metadata.py`.

**Тесты:**
`~/.venvs/ctb/bin/python -m pytest tests/test_chat_modes_registry.py tests/test_database.py
tests/test_skills_prompt_fragment.py tests/test_skills_plugin.py tests/test_skills_metadata.py
tests/test_openai_helper_tool_calls.py -q` → 292 passed (168 + 124), 0 failed.
`ruff check bot/chat_modes_registry.py bot/database.py tests/test_chat_modes_registry.py
tests/test_database.py tests/test_skills_metadata.py` → чисто.
`mypy bot/chat_modes_registry.py bot/database.py` → 42 ошибки, все в `bot/database.py` на
строках вне диапазонов, изменённых T06 (305, 329-457, 1059-2001 — существовавший до T06 долг
типов `_local`/`_executor`/`Optional`-параметров); ни одна ошибка не попадает в добавленный
код (105-137, 213-218, 647, 668, 680, 759-796) — новых ошибок нет.

**Проверка HARD-требования «tools: без изменений»:** скриптом на PyYAML сравнил
`bot/chat_modes.yml` с `git show HEAD:bot/chat_modes.yml` — 25/25 режимов, `tools:` побайтово
идентичны; `parse_mode`, `welcome_message`, `prompt_markers` тоже не изменились. `get_spec()`
нигде не тронут (T06 не редактирует `bot/plugins/*`).

**Проверка LEGACY_PROMPT_FINGERPRINTS:** пересчитал `sha256(prompt_start.strip())` по
`git show HEAD:bot/chat_modes.yml` — все 25 хэшей совпадают с константой в `database.py`
1-в-1 (включая тестовую фикстуру `LEGACY_ASSISTANT_PROMPT` в `test_database.py`, её хэш даёт
именно `b2c1c8ab...` → `assistant`).

**Проверка reconcile/миграции (§3.3 плана):** прочитал `_apply_schema_migrations` /
`_reconcile_schema_version_with_shape` целиком, прошёл руками fresh-install / v1→v2 / legacy
(без session_id) / уже-мигрированный-до-v3 сценарии — во всех `SHAPE_VERIFIABLE_SCHEMA_VERSION
= 2` работает как задумано (для v3 reconcile теперь no-op вместо отката). Новые тесты
(`test_migration_backfills_mode_key_for_legacy_assistant_session`,
`test_migration_backfill_mode_key_is_idempotent`,
`test_reconcile_does_not_roll_back_schema_version_after_migration_3`,
`test_migration_backfill_mode_key_is_noop_for_non_system_or_already_tagged`) реально
воспроизводят баг из §3.3 и падали бы без фикса — проверено логикой (не запускал diff
до/после патча, но код `_reconcile...` однозначно даёт ошибочный результат без нового
условия `recorded_version > SHAPE_VERIFIABLE_SCHEMA_VERSION`).

**Проверка shared_blocks:** код в `chat_modes_registry.py` соответствует плану 1-в-1. В
реальном YAML — 23 маркера `{{shared:web_search_rules}}`, после подстановки ни один не виден
снаружи (тест `test_web_search_rules_shared_block_appears_once_in_real_yaml`,
`test_shared_blocks_are_substituted_and_hidden`) — подтверждено и вручную скриптом:
0 вхождений `{{shared:` в `all_modes()`. `manage_plan_tasks`, `больше двух шагов`,
`prompt_markers: ["локальные skills", "skills."]` — все физически на месте.

**Проверка skills-реорганизации (§7.2):** `bot/skills/META-SKILLS/` больше не содержит 6
дублирующихся пар (`belief-examination`, `board-of-directors`, `decision-framework`,
`multi-agent-brainstorm`, `project-planning`, `storytelling-structure`), но сохранил
уникальные записи (`career-strategy`, `conflict-resolution`, `hypothesis-design`,
`negotiation-prep`, `reflection-postmortem`, `research-synthesis`, `root-cause-analysis`,
`thinking-in-systems`, `_shared`). Top-level версии этих 6 skills содержат META-контент
(проверил `project-planning/references/execution-project.md` — переименование от
linear-project.md подтверждено текстом файла, как и описано в плане). Относительные ссылки
вида `META-SKILLS/_shared/safety-gate.md` внутри перенесённых SKILL.md по-прежнему резолвятся
— проверил вживую через `SkillsPlugin._get_skill_reference('belief-examination',
'META-SKILLS/_shared/core-principles.md')` → `success: True`, т.к. `_resolve_skill_file_path`
ищет и относительно `skill_path`, и относительно `skills_dir` (`bot/plugins/skills.py:1331-1334`),
а `META-SKILLS/_shared/` физически остался на месте. `decision-framework/from_*` (20 файлов)
отсутствуют — соответствует плану. Полный скан через реальный `SkillsPlugin._scan_skills()`
(30 skills): пустых name/description нет, дублей id нет, `decision-framework.description` —
238 символов (по решению координатора — остаётся русским, ≤240; проверено программным
парсингом frontmatter, не вручную). `sequential-thinking/SKILL.md` получил frontmatter.

---

### ERROR

1. **Гитигнорд-контент `bot/skills/**` удалён/перезаписан без обязательного бэкапа
   (нарушение HARD-правила `common.txt`).**
   `common.txt`: «NEVER delete... files or directories that git does not track... If a task
   requires removing such a file, first copy it to `/tmp/impl/backup/<same relative path>`».
   T06 §7.2 требует физически заменить `bot/skills/{belief-examination,board-of-directors,
   decision-framework,multi-agent-brainstorm,project-planning,storytelling-structure}/`
   содержимым `META-SKILLS/X` (старый top-level файл при этом уничтожается), удалить
   `bot/skills/META-SKILLS/{те же 6}/` целиком и НЕ переносить 20 файлов
   `decision-framework/from_*`. Все эти данные негитованы (`.gitignore:24`) — то есть
   безвозвратны без ручного бэкапа. Проверил `/tmp/impl/backup/` и весь `/tmp` на признаки
   бэкапа (`find / -iname "*decision-framework*from_*"`, `ls /tmp/impl`) — бэкапа нет нигде,
   в финальном отчёте разработчика (если такой был) бэкап не упомянут по факту его
   отсутствия на диске.
   Последствие: невозможно проверить, действительно ли перед удалением были перенесены
   «уникальные для top-level версии» файлы (план §7.2, п.3: «если такие есть — проверить find
   по каждой паре перед удалением, не предполагать заранее») — если что-то уникальное было и
   осталось не перенесено, оно потеряно без возможности восстановления.
   **Исправление:** для этого раунда факт уже необратим (файлов, чтобы сделать бэкап задним
   числом, не существует). Разработчику нужно явно подтвердить в отчёте, что шаг «проверить
   find по каждой паре перед удалением» был выполнен ДО удаления (а не пропущен), и что
   уникальных файлов не было ни в одной из 6 пар. Если подтвердить нельзя — эскалировать
   координатору как реальную потерю данных, а не молчаливо закрывать раунд.

### WARNING

1. **«НЕ ПЫТАЙТЕСЬ...» не смягчено ни в одном из 7 мест, хотя план явно показывает
   ожидаемый результат.** Мастер-план шаг 7 требует заменить
   «ОБЯЗАТЕЛЬНО/ЗАПРЕЩЕНО/НЕ ПЫТАЙТЕСЬ» на спокойные формулировки. `ОБЯЗАТЕЛЬНО` и
   `ЗАПРЕЩЕНО` действительно убраны везде (0 вхождений). Но `T06-plan.md` §4.3, пример 3
   (`technical_writer`) явно показывает «стало»: `НЕ ПЫТАЙТЕСЬ оптимизировать уже
   оптимизированный промпт` → `Не оптимизируйте уже оптимизированный промпт`. В реальном
   файле это НЕ применено — фраза `НЕ ПЫТАЙТЕСЬ оптимизировать уже оптимизированный промпт`
   осталась дословно (только `optimize_prompt`→`prompt_perfect` переименован) в 7 режимах:
   `travel_guide` (строка 144), `content_creator` (226), `technical_writer` (282, тот самый
   пример из плана), `medical_assistant` (588), `legal_assistant_ru` (657), `school_tutor`
   (1141), `personal_finance_planner` (1640). Проверено: `grep -c "НЕ ПЫТАЙТЕСЬ"` → 7,
   `grep -c "ОБЯЗАТЕЛЬНО\|ЗАПРЕЩЕНО"` → 0.
   **Исправление:** заменить все 7 вхождений `НЕ ПЫТАЙТЕСЬ оптимизировать уже оптимизированный
   промпт` на `Не оптимизируйте уже оптимизированный промпт` (ровно как показано в плане).

### NIT

1. Два из 23 мест подстановки `{{shared:web_search_rules}}` (`assistant` — строка 28,
   `text_improver` — строка 91) оставляют 2 пустые строки перед `parse_mode:` вместо одной;
   остальные 21 — ровно одну. План (§5) просил «не более одной» как чистую косметику
   («косметика, не отдельная задача»). Не блокирует, но раз задача явно называлась —
   несложно поправить заодно с WARNING-пунктом.
2. `tests/test_skills_metadata.py::test_real_skill_ids_are_unique` проверяет
   `len(ids) == len(set(ids))` на ключах `available_skills` — это `dict`, дублирующиеся ключи
   в нём структурно невозможны, поэтому тест не может упасть ни при какой регрессии (сам
   docstring теста это признаёт: «ids уже уникальны по построению»). Как документирующий
   тест — ок, как регрессионная защита — не работает. Не блокирует (тест не заявлен как
   единственная защита от дублей — реальную защиту даёт сам факт использования dict), но
   стоит иметь в виду при следующей правке `_scan_skills()`.

---

**Итог раунда 1:** 1 ERROR (процессное нарушение HARD-правила о бэкапе гитигнорд-файлов,
требует подтверждения от разработчика/эскалации, не технический баг), 1 WARNING (7 мест
`НЕ ПЫТАЙТЕСЬ` не смягчены вопреки явному примеру в плане), 2 NIT (двойной перенос строки в
2/23 местах; слабый тест уникальности id). Вся остальная проверенная область (миграция
mode_key, reconcile-фикс, shared_blocks, tools:-списки, маркеры детекции режима, тесты,
ruff/mypy) — без замечаний.

## Раунд 2

**ERROR раунда 1 (гитигнорд-контент `bot/skills/**` удалён/перезаписан без бэкапа) —
ACCEPTED координатором, необратим, будет сообщён пользователю отдельно.** Не переоткрываю и
не переоцениваю в этом раунде; ниже — только проверка WARNING/NIT раунда 1 и повторный скан
diff на новые проблемы.

**Проверено:** `git diff HEAD` для `bot/chat_modes.yml`, `bot/chat_modes_registry.py`,
`bot/database.py`, `tests/test_chat_modes_registry.py`, `tests/test_database.py`,
`tests/test_skills_metadata.py`; текущее состояние `bot/skills/**` (не через git,
`.gitignore:24`).

**WARNING раунда 1 (7× «НЕ ПЫТАЙТЕСЬ оптимизировать уже оптимизированный промпт» не
смягчено) — ИСПРАВЛЕНО.** `grep`-эквивалент по всему файлу: `"НЕ ПЫТАЙТЕСЬ"` → 0 вхождений,
`"ОБЯЗАТЕЛЬНО"` → 0, `"ЗАПРЕЩЕНО"` → 0. Все 7 мест (`travel_guide:142`, `content_creator:224`,
`technical_writer:280`, `medical_assistant:586`, `legal_assistant_ru:655`, `school_tutor:1139`,
`personal_finance_planner:1638`) теперь читают ровно «Не оптимизируйте уже оптимизированный
промпт», как показывал план (§4.3, пример 3). Подтверждено.

**NIT 1 раунда 1 (2/23 места оставляли двойной перенос строки перед `parse_mode:`) —
ИСПРАВЛЕНО.** Скриптом проверено количество пустых строк сразу после каждого из 23
`{{shared:web_search_rules}}`-маркеров — везде ровно 1 (`assistant:28`, `text_improver:90` —
места, где раньше было 2, теперь тоже 1). Проверил также отсутствие где-либо в файле трёх и
более подряд идущих пустых строк — 0 совпадений.

**NIT 2 раунда 1 (`test_real_skill_ids_are_unique` не мог упасть ни при какой регрессии,
т.к. сравнивал ключи `dict`) — ИСПРАВЛЕНО ПО СУЩЕСТВУ.** Тест переписан: теперь сравнивает
frontmatter `name` разных `skill_id` (`tests/test_skills_metadata.py:45-54`) — это реальная
регрессия, которую стоит ловить (модель видит `name`, не `id`; повторный дубль
META-SKILLS/top-level с разными id, но одинаковым `name`, был бы неотличим для модели).
Комментарий в тесте прямо ссылается на находку раунда 1. Прогнал вручную через реальный
`SkillsPlugin._scan_skills()` — 30 skills, дублей `name` нет, тест зелёный.

**Повторная проверка HARD-требования «tools:/markers без изменений» (PyYAML-скрипт,
все 25 режимов против `git show HEAD:bot/chat_modes.yml`):** `tools`, `parse_mode`,
`welcome_message`, `prompt_markers`, `defer_direct_results`, `max_tokens_percent` —
побайтово идентичны HEAD во всех режимах. `get_spec()` по-прежнему не тронут (T06 не
редактирует `bot/plugins/*`).

**Повторный скан всего T06-диффа на новые проблемы:**
- `bot/chat_modes_registry.py`, `bot/database.py`, `tests/test_chat_modes_registry.py`,
  `tests/test_database.py` — побайтово идентичны состоянию, проверенному в раунде 1 (diff
  не менялся); замечаний нет.
- `bot/chat_modes.yml` — полный diff против HEAD перечитан целиком построчно: помимо
  подтверждённого фикса WARNING/NIT1, содержимое совпадает с раунд-1-проверенным (замена
  `optimize_prompt`→`prompt_perfect`, удаление правил 9/12/14 `assistant` с корректной
  перенумерацией 1-13, замена `/reset code_assistant` на `/reset («Изменить режим текущей
  сессии»)` — сверено с реальной локализацией: `translations.json` ключ
  `session_change_mode`, `ru` = «📝 Изменить режим текущей сессии», текст режима цитирует её
  без эмодзи корректно; `skills_agent` разбит на 4 раздела с сохранением состава пунктов
  1:1 плану §6.3, `shared_blocks:` блок в самом низу файла). Новых проблем не найдено.
- `tests/test_skills_metadata.py` (untracked) — только правка теста `test_real_skill_ids_are_unique`
  (см. NIT2 выше), остальные тесты файла не изменились.
- `bot/skills/**` — состояние идентично зафиксированному в раунде 1 (30 skills, без
  дублей `name`/`id`, `decision-framework` описание 238 символов, `sequential-thinking`
  frontmatter на месте, `decision-framework/from_*` отсутствуют, `META-SKILLS/_shared`
  физически на месте). Файлы не тронуты в этом раунде — согласуется с тем, что ERROR
  раунда 1 ACCEPTED и не требует новых действий с файлами.

**Тесты:**
`~/.venvs/ctb/bin/python -m pytest tests/test_chat_modes_registry.py tests/test_database.py
tests/test_skills_prompt_fragment.py tests/test_skills_plugin.py tests/test_skills_metadata.py
tests/test_openai_helper_tool_calls.py -q` → 292 passed, 0 failed (то же число, что в раунде 1).
`ruff check bot/chat_modes_registry.py bot/database.py tests/test_chat_modes_registry.py
tests/test_database.py tests/test_skills_metadata.py` → чисто.
`mypy bot/chat_modes_registry.py bot/database.py` → 42 ошибки, все вне диапазонов, изменённых
T06 (та же картина, что в раунде 1: `_local`/`_executor`/implicit-Optional долг в
`bot/database.py`); новых ошибок в добавленном T06 коде нет.

### ERROR

Нет новых. 1 ERROR раунда 1 — **ACCEPTED координатором** (необратимая потеря негитованного
`bot/skills/**`-контента при реорганизации META-SKILLS; будет отдельно сообщено пользователю
координатором). Не переоткрывается в этом раунде.

### WARNING

Нет.

### NIT

Нет новых (оба NIT раунда 1 закрыты, см. выше).

---

**Итог раунда 2:** 0 новых ERROR/WARNING/NIT. ERROR раунда 1 — ACCEPTED (необратим,
эскалация не требуется, координатор сообщит пользователю). WARNING раунда 1 (7×
«НЕ ПЫТАЙТЕСЬ») — исправлено и подтверждено. NIT1 (двойной перенос строки) — исправлено и
подтверждено. NIT2 (слабый тест уникальности) — исправлено по существу и подтверждено. Тесты
(292), ruff, mypy — без регрессий. T06 готов к закрытию.
