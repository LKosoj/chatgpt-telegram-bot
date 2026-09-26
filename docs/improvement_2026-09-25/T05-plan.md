# T05 — Отправка файлов 1.5: `bot/artifact_paths.py` и три точки проверки

## 0. Критические отклонения и находки (прочитать до реализации)

1. **`request_started_at` сейчас недостижим.** `RequestContext` (`bot/request_context.py`,
   24 строки, читан целиком) не содержит поля времени. Ни один из трёх интеграционных вызовов
   (`agent_tools._deliver_to_user`, `utils.handle_direct_result`, `agent_delivery._send_direct_payload`)
   не имеет доступа к моменту начала запроса и не должен получить его в рамках T05 —
   `RequestContext` не входит во владение этой задачи. `is_deliverable(..., request_started_at=None)`
   принимает параметр по мастер-плану, но на практике он всегда `None`; ветка recency в §2 существует
   только как задел на будущее и тестируется отдельно синтетическим вызовом.

2. **Temp-правило не может быть «список префиксов» — это ломает реальный поток.** Проверено 5
   продюсеров, пишущих в `direct_result`/`artifacts` файлы во временный каталог:
   - `bot/plugins/auto_tts.py:36` — голый `tempfile.NamedTemporaryFile(delete=False, suffix=...)`,
     без `dir=`/`prefix=` → верхний уровень `tempfile.gettempdir()`, имя вида `tmpXXXXXXXX.wav`.
   - `bot/plugins/haiper_image_to_video.py:71,733` — то же самое (голый `NamedTemporaryFile`).
   - `bot/plugins/show_me_diagrams.py:200` — `tempfile.gettempdir()` + `diagram_<uuid>.png`, тоже
     верхний уровень.
   - `bot/llm_gateway_client.py:237` (`_write_base64_image`, вызывается из
     `bot/plugins/stable_diffusion.py:59,67` через `extract_image_result`) — `tempfile.mkdtemp(prefix=
     "llmgateway_images_")`, файл кладётся ВЛОЖЕННО: `/tmp/llmgateway_images_<hex>/<uuid>.png`.
   - `bot/plugins/terminal.py:18` — `DEFAULT_CWD = "/tmp"`, шелл-команда модели создаёт файлы
     произвольной вложенности и имени под `/tmp`. У terminal.py **нет** `direct_result`
     (`grep direct_result bot/plugins/terminal.py` — пусто): единственный путь доставить такой файл
     пользователю — передать его путь в `deliver_to_user(artifacts=[...])`.
   Единого префикса нет (у 3 из 5 продюсеров — случайное имя `tmpXXXXXXXX` от стандартной
   библиotеки), а terminal.py в принципе не ограничен никаким префиксом. Узкий allow-list префиксов
   сломал бы terminal→deliver_to_user поток. Решение: fallback-правило — «путь лежит где угодно
   внутри `tempfile.gettempdir()` (resolved), на любой глубине». Это НЕ расширение текущего
   поведения для `deliver_to_user`: `_allowed_artifact_roots` уже сегодня безусловно даёт
   `Path("/tmp")` (`bot/plugins/agent_tools.py:2906`) без каких-либо ограничений. Для
   `handle_direct_result`/`_send_direct_payload` это, наоборот, ужесточение — сейчас там нет
   вообще никакой проверки пути.

3. **`cleanup_intermediate_files` — обязательная часть работы, не входит в 3 шага мастер-плана
   буквально, но названа в исходном брифе задачи и необходима для корректности.** Если
   `handle_direct_result` мягко отклонит путь (текстовое сообщение вместо отправки) и продолжит
   выполнение, строка `bot/utils.py:1434-1435` (`if result_format == 'path':
   cleanup_intermediate_files(response)`) всё равно выполнится и удалит файл — включая путь к БД,
   если бы плагин (гипотетически, до T05) его подсунул. Без правки `cleanup_intermediate_files`
   политика «отказ в отправке» превращается в «отказ в отправке, но файл всё равно уничтожен» для
   чувствительных путей. Добавляется охранная проверка через новую `is_protected_path()`
   (переиспользует deny-логику `is_deliverable`, без привязки к scope — она и не нужна для
   deny-списка). Это единственное изменение в `cleanup_intermediate_files`, финализирующее
   собственно требование мастер-плана «файл не отправляется» → «и не удаляется вместо этого».

4. **Источник `storage_root` для `_deliver_to_user` меняется с `helper.plugin_manager.storage_root`
   на `self.storage_root`.** Текущий `_allowed_artifact_roots(helper)` читает
   `getattr(helper, "plugin_manager", None)` (`bot/plugins/agent_tools.py:2907-2910`). Во ВСЕХ
   8 существующих тестах `deliver_to_user` (`tests/test_agent_tools_plugin.py:2221-2385`) `helper =
   SimpleNamespace()` — без `plugin_manager` — а `storage_root` передаётся напрямую через
   `plugin.initialize(storage_root=str(tmp_path))`, что кладёт значение в `self.storage_root`
   (`bot/plugins/plugin.py:21-25`, база `AgentToolsPlugin.initialize` вызывает `super().initialize(...)`
   на `bot/plugins/agent_tools.py:266` первой строкой). В продакшене оба атрибута всегда совпадают
   (оба в конечном счёте берутся из `PluginManager.storage_root`), поэтому переключение источника не
   меняет поведение в проде, но делает проверку рабочей в существующих unit-тестах без обращения к
   `/tmp`-fallback как единственной подстраховке. Это сознательное, обоснованное отклонение от
   буквального повторения старого `_allowed_artifact_roots`, а не рефactor ради рефактора.

5. **`storage_root`/`skills_dir`/`skills_workdir_root`/`db_path` в `artifact_paths.py` вычисляются
   независимо от живых объектов (`PluginManager`, `Database()`, skills-плагина) — через
   репликацию тех же переменных окружения.** Причина: `handle_direct_result` и
   `_send_direct_payload` физически не имеют доступа ни к `PluginManager`, ни к
   skills-инстансу (проверено: `handle_direct_result(config, update, response, *, bot=None)`
   получает `config` — плоский словарь без `plugin_manager`, и `bot` — сырой `telegram.Bot`
   (`bot/telegram_bot.py:669` `application_bot = getattr(getattr(self, "application", None), "bot",
   None)`); `_send_direct_payload(bot, *, chat_id, payload, ...)` не получает вообще ничего
   похожего). Значения:
   - `storage_root` ⇔ `os.getenv("PLUGIN_STORAGE_ROOT")` или `<repo_root>/data`
     (`bot/plugin_manager.py:67-69`).
   - `db_path` ⇔ `os.getenv("DB_PATH")` или `<repo_root>/bot/user_data.db`
     (`bot/database.py:216-220`). Точно совпадает с продакшен-значением: `bot/__main__.py:384-385`
     вызывает `Database.configure(db_path=os.environ.get('DB_PATH'), ...)` — то есть
     `Database._configured['db_path']` и `os.getenv("DB_PATH")` в проде всегда идентичны;
     расхождение возможно только если тест напрямую зовёт `Database.configure(db_path=<строка не
     из env>)`, что вне зоны действия доставки файлов.
   - `skills_dir` ⇔ `os.getenv("SKILLS_DIR")` или `<storage_root>/skills`
     (`bot/plugins/skills.py:119-120`).
   - `skills_workdir_root` ⇔ `os.getenv("SKILLS_WORKDIR")` или `<storage_root>/skill_workdir`
     (`bot/plugins/skills.py:125-130`).
   `artifact_paths.py` не импортирует `Database` (нет побочных эффектов `init_db()`) и не делает
   `plugin_manager.get_plugin("skills")`. `agent_tools._deliver_to_user` при этом ПЕРЕДАЁТ свой
   реальный `self.storage_root` явно (см. п.4) — он у неё уже есть и точнее совпадает с тем, что
   реально сконфигурировано, чем повторное чтение env. Остальные два вызова полагаются на
   auto-resolve.

6. **`.env` / `usage_logs/` / `uploads/webshot` — CWD-relative, не REPO_ROOT-relative.** Это не
   новое допущение T05, а точное повторение того, как эти пути уже резолвятся в существующем коде:
   `bot/__main__.py:189` — `load_dotenv()` без аргумента (ищет `.env` от текущей директории
   запуска); `bot/utils.py:848` — `def make_usage_tracker(config, user_id, user_name,
   logs_dir="usage_logs")` — относительный путь; `bot/plugins/webshot.py:54-57` —
   `os.path.join("uploads/webshot", ...)` — тоже относительный. Ни один `os.chdir()` не встречен
   в `bot/` (репозиторий проверен целиком), поэтому `Path.cwd()` на протяжении жизни процесса
   стабилен и эквивалентен директории запуска. `artifact_paths.py` использует `Path.cwd()` для этих
   трёх правил — не переизобретает допущение, а согласуется с уже действующим.

7. **`stable_diffusion.py`** (генерация изображений через LLM Gateway) — продюсер, не входивший в
   исходный список файлов задачи, но реально пишущий `direct_result{kind=photo, format=path}` с
   путём внутри `tempfile.mkdtemp(prefix="llmgateway_images_")`
   (`bot/plugins/stable_diffusion.py:59,67,86-92` → `bot/llm_gateway_client.py:234-245,283`).
   Учтён в п.2 fallback-правиле.

8. **`haiper_image_to_video.py` — вне охвата T05, подтверждено заново.** `grep direct_result
   bot/plugins/haiper_image_to_video.py` — пусто. Готовое видео отправляется напрямую
   `message.reply_video(...)` (строки ~744-747), минуя `direct_result`/`handle_direct_result`
   /`_send_direct_payload` целиком. Ни одна из трёх точек интеграции его не видит — значит и
   ограничить нечем; остаётся как есть (см. §6 риски).

## 1. `bot/artifact_paths.py` — API

Новый файл, без внешних побочных эффектов при импорте (не трогает `Database`, не создаёт
директорий).

```python
def artifact_workspace(storage_root: str, scope: str) -> Path:
    """<storage_root>/artifacts/<safe_scope>. safe_scope — та же трансформация, что и
    skills._ensure_skill_workdir (bot/plugins/skills.py:3614):
    scope.replace(":", "_").replace("/", "_") or "global"."""

def is_protected_path(path: str, *, storage_root: str | None = None) -> bool:
    """True, если resolved(path) попадает в жёсткий deny-список (БД + -wal/-shm/-journal,
    .env, usage_logs/, голый *.json|*.jsonl прямо в корне storage_root, исходники skills).
    Не требует scope — используется как для is_deliverable, так и для cleanup_intermediate_files
    (защита от удаления после отказа в доставке)."""

def is_deliverable(
    path: str,
    *,
    scope: str,
    request_started_at: float | None = None,
    storage_root: str | None = None,
) -> tuple[bool, str | None]:
    """(allowed, reason). reason всегда None при allowed=True."""
```

`storage_root=None` → внутри резолвится как в п.5 §0. Приватные помощники (без публичного API):
`_safe_scope`, `_default_storage_root`, `_default_db_path`, `_default_skills_dir`,
`_default_skills_workdir_root`.

## 2. Алгоритм `is_deliverable` (порядок важен)

```
resolved = Path(os.path.realpath(os.path.expanduser(path)))   # символические ссылки раскрыты
effective_storage_root = Path(storage_root).resolve() if storage_root else _default_storage_root()

# --- DENY (безусловно, не зависит от scope) ---
db = _default_db_path()
if resolved in {db, db.with_name(db.name + "-wal"),
                db.with_name(db.name + "-shm"), db.with_name(db.name + "-journal")}:
    return False, "database file"
if resolved == (Path.cwd() / ".env").resolve():
    return False, ".env"
if _is_relative_to(resolved, (Path.cwd() / "usage_logs").resolve()):
    return False, "usage_logs directory"
if resolved.parent == effective_storage_root and resolved.suffix.lower() in {".json", ".jsonl"}:
    return False, "bare json/jsonl in storage root"
skills_dir = _default_skills_dir(effective_storage_root)
if _is_relative_to(resolved, skills_dir):
    return False, "skills source directory"

# --- ALLOW: scope-namespaced деревья — здесь же явный DENY для чужого scope ---
artifacts_root = artifact_workspace(str(effective_storage_root), scope).parent  # .../artifacts
if _is_relative_to(resolved, artifacts_root):
    expected = artifact_workspace(str(effective_storage_root), scope)
    return (True, None) if resolved == expected or _is_relative_to(resolved, expected) \
        else (False, "artifact belongs to a different delivery scope")

workdir_root = _default_skills_workdir_root(effective_storage_root)
if _is_relative_to(resolved, workdir_root):
    parts = resolved.relative_to(workdir_root).parts
    safe = _safe_scope(scope)
    return (True, None) if len(parts) >= 2 and parts[1] == safe \
        else (False, "artifact belongs to a different skills workdir scope")

# --- ALLOW: широкие правила ---
if _is_relative_to(resolved, effective_storage_root):
    return True, None
if _is_relative_to(resolved, runtime_output_dir()) or _is_relative_to(resolved, runtime_plots_dir()):
    return True, None
if request_started_at is not None:
    try:
        if resolved.stat().st_mtime >= request_started_at - 2:
            return True, None
    except OSError:
        return True, None   # файла нет — не наша забота, downstream даст свою ошибку
if _is_relative_to(resolved, Path(tempfile.gettempdir()).resolve()):
    return True, None
if _is_relative_to(resolved, (Path.cwd() / "uploads" / "webshot").resolve()):
    return True, None

return False, "outside allowed delivery locations"
```

`_is_relative_to(a, b)` — обёртка над `a == b or a.is_relative_to(b)` (Python 3.12 в venv
`~/.venvs/ctb`, `Path.is_relative_to` доступен с 3.9). Deny-блок выполняется первым и безусловно —
даже путь внутри `tempfile.gettempdir()`, случайно совпавший с БД (при экзотической конфигурации
`DB_PATH` внутри `/tmp`), будет отклонён раньше, чем дойдёт до temp-fallback.

## 3. Точки интеграции

### 3.1 `bot/plugins/agent_tools.py`

Удалить `_allowed_artifact_roots` целиком (`bot/plugins/agent_tools.py:2904-2931`).

`_deliver_to_user` (было `bot/plugins/agent_tools.py:2803-2806`):
```python
allowed_roots = self._allowed_artifact_roots(helper)
artifact_items, error = self._normalize_delivery_artifacts(
    kwargs.get("artifacts"), allowed_roots=allowed_roots,
)
```
→
```python
artifact_items, error = self._normalize_delivery_artifacts(
    kwargs.get("artifacts"), scope=scope, storage_root=self.storage_root,
)
```
(`scope` уже вычислен строкой выше — `bot/plugins/agent_tools.py:2799`.)

`_normalize_delivery_artifacts` (было `bot/plugins/agent_tools.py:2933-2939` сигнатура,
`2973-2982` тело проверки):
```python
@classmethod
def _normalize_delivery_artifacts(
    cls, artifacts: Any, *, allowed_roots: List[str] | None = None,
) -> tuple[List[Dict[str, Any]], str | None]:
    ...
    if allowed_roots:
        in_allowed = any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in allowed_roots
        )
        if not in_allowed:
            return [], (
                f"Artifact path '{file_path}' is outside allowed roots "
                f"({', '.join(allowed_roots)})"
            )
```
→
```python
@classmethod
def _normalize_delivery_artifacts(
    cls, artifacts: Any, *, scope: str, storage_root: str | None = None,
) -> tuple[List[Dict[str, Any]], str | None]:
    ...
    allowed, reason = artifact_paths.is_deliverable(
        resolved, scope=scope, storage_root=storage_root,
    )
    if not allowed:
        return [], f"Artifact path '{file_path}' is outside allowed delivery locations ({reason})"
```
Место в теле функции — сразу после текущей проверки `os.path.isfile(resolved)`
(`bot/plugins/agent_tools.py:2971-2972`), перед проверкой размера. Добавить
`from .. import artifact_paths` (или `from ..artifact_paths import is_deliverable`) в блок импортов
(`bot/plugins/agent_tools.py:1-23`).

### 3.2 `bot/utils.py`

Добавить импорт `from .artifact_paths import is_deliverable` рядом с существующими
относительными импортами (`bot/utils.py:19-25`).

Новый приватный помощник рядом с `compute_scope_key` (`bot/utils.py:1037`, сама функция не
меняется):
```python
def _artifact_scope_for_update(update: Update) -> str:
    chat_id = getattr(getattr(update, "effective_chat", None), "id", None)
    user_id = getattr(getattr(update, "effective_user", None), "id", None)
    return compute_scope_key(chat_id, user_id)
```

В `handle_direct_result` три ветки с `format == 'path'`:

- **photo** (`bot/utils.py:1296-1313`, внутри `elif result_format == 'path':`) — проверка
  вставляется ДО текущего `try:` блока (`get_image_size`), чтобы не трогать существующую
  fallback-логику через `resize_image_if_needed` (её покрывает
  `test_handle_direct_result_photo_path_failure_logs_values`,
  `tests/test_plugin_direct_results.py:408-440`):
  ```python
  elif result_format == 'path':
      allowed, reason = is_deliverable(value, scope=_artifact_scope_for_update(update))
      if not allowed:
          logging.warning("Rejected direct_result photo path=%s reason=%s", value, reason)
          sent_messages.append(await message.reply_text(
              **common_args,
              text=f"Artifact path is unavailable: {os.path.basename(value)}",
              parse_mode=None,
          ))
      else:
          try:
              ...  # существующий блок без изменений
  ```
- **gif** (`bot/utils.py:1317-1319`) и **file** (`bot/utils.py:1320-1325`) получают тот же паттерн:
  проверка перед `open(value, 'rb')`, при отказе — тот же текст в `reply_text`, без исключения.

`value` в `is_deliverable(value, ...)` передаётся ДО `os.path.realpath` — сама функция делает
`realpath`/`expanduser` внутри, повторно резолвить снаружи не нужно (симметрично с тем, что делает
`agent_delivery._send_direct_payload` уже сегодня на строке 205, см. 3.3).

`cleanup_intermediate_files` (`bot/utils.py:1462-1464`):
```python
if format == 'path' and value and not result.get("preserve_after_delivery"):
    if os.path.exists(value):
        os.remove(value)
```
→
```python
if format == 'path' and value and not result.get("preserve_after_delivery"):
    if os.path.exists(value) and not is_protected_path(value):
        os.remove(value)
```
(добавить `is_protected_path` к импорту из `.artifact_paths`).

### 3.3 `bot/agent_delivery.py`

`_send_direct_payload` (`bot/agent_delivery.py:204-215`):
```python
if kind in {"file", "photo", "gif"} and result_format == "path":
    path = os.path.realpath(os.path.expanduser(str(value)))
    if not os.path.isfile(path):
        await send_text_chunks(..., text=f"Artifact path is unavailable: {os.path.basename(path)}", ...)
        return sent
```
→
```python
if kind in {"file", "photo", "gif"} and result_format == "path":
    path = os.path.realpath(os.path.expanduser(str(value)))
    allowed, reason = is_deliverable(path, scope=compute_scope_key(chat_id))
    if not allowed:
        logger.warning("Rejected direct_result artifact path=%s reason=%s", path, reason)
        await send_text_chunks(..., text=f"Artifact path is unavailable: {os.path.basename(path)}", ...)
        return sent
    if not os.path.isfile(path):
        await send_text_chunks(..., text=f"Artifact path is unavailable: {os.path.basename(path)}", ...)
        return sent
```
`compute_scope_key(chat_id)` (без `user_id`, которого здесь нет в сигнатуре) даёт `"chat:{chat_id}"`
— тот же scope, что вычисляет `_deliver_to_user` для того же чата, так как `compute_scope_key`
отдаёт приоритет `chat_id` (`bot/utils.py:1051-1053`). Импорт: добавить `is_deliverable` к
существующему `from .utils import (cleanup_intermediate_files, is_direct_result, ...)`
(`bot/agent_delivery.py:18-23`); `compute_scope_key` уже нужно туда же добавить — сейчас не
импортирован.

## 4. Таблица легитимных потоков

| Продюсер | Путь | Проходит через | Правило allow |
|---|---|---|---|
| `deliver_to_user`, тест-фикстура `storage_root=tmp_path`, файл в корне | `<storage_root>/report.txt` | `_normalize_delivery_artifacts` | широкое "внутри storage_root" |
| `codeinterpreter.py` | `runtime_output_dir()`/`runtime_plots_dir()` | оба пути (deliver_to_user и handle_direct_result) | `runtime_output_dir()`/`runtime_plots_dir()` |
| `terminal.py` → модель зовёт `deliver_to_user(artifacts=[...])` на созданный в `/tmp` файл (любая вложенность) | `/tmp/**` | `_normalize_delivery_artifacts` | temp-fallback |
| `auto_tts.py` | `/tmp/tmpXXXXXXXX.wav` | `handle_direct_result` (file) | temp-fallback |
| `show_me_diagrams.py` | `/tmp/diagram_<uuid>.png` | `handle_direct_result` (photo) | temp-fallback |
| `stable_diffusion.py` | `/tmp/llmgateway_images_<hex>/<uuid>.png` | `handle_direct_result` (photo) | temp-fallback |
| `webshot.py` | `<cwd>/uploads/webshot/<rand>.png` | `handle_direct_result` (photo/file) | uploads/webshot allow |
| skills workdir этого scope | `<skills_workdir_root>/<skill_id>/<safe_scope>/...` | `deliver_to_user` | skills-workdir-this-scope |
| новая `artifact_workspace` (пока не используется ни одним продюсером — инфраструктура на будущее) | `<storage_root>/artifacts/<safe_scope>/...` | `deliver_to_user` | artifacts-this-scope |
| `haiper_image_to_video.py` | — | не проходит ни через одну из 3 точек (`message.reply_video` напрямую) | вне охвата T05 |

## 5. Тесты

### 5.1 Новый `tests/test_artifact_paths.py`

- `test_is_deliverable_allows_inside_storage_root` — файл прямо в `storage_root`, `scope="chat:1"`.
- `test_is_deliverable_allows_runtime_output_and_plots_dir` — monkeypatch `BOT_OUTPUT_DIR`/
  `BOT_PLOTS_DIR` на `tmp_path`, файл внутри.
- `test_is_deliverable_allows_anywhere_under_tempdir` — monkeypatch `tempfile.gettempdir` (через
  `tmp_path` + `monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))`), файл
  вложенный на 2 уровня — allowed.
- `test_is_deliverable_allows_uploads_webshot_relative_to_cwd` —
  `monkeypatch.chdir(tmp_path)`, файл в `tmp_path/uploads/webshot/shot.png`.
- `test_is_deliverable_rejects_env_file` — `monkeypatch.chdir(tmp_path)`, `tmp_path/".env"`.
- `test_is_deliverable_rejects_usage_logs` — `monkeypatch.chdir(tmp_path)`,
  `tmp_path/"usage_logs"/"1.json"`.
- `test_is_deliverable_rejects_bare_json_in_storage_root_root` — `storage_root=tmp_path`,
  `tmp_path/"mcp_servers.json"` и `tmp_path/"reminders.json"` — оба rejected; тот же json НЕ в
  корне (`tmp_path/"sub"/"reminders.json"`) — allowed (широкое правило).
- `test_is_deliverable_rejects_skills_source_dir` — `SKILLS_DIR=tmp_path/"skills"` через
  monkeypatch, файл внутри — rejected.
- `test_is_deliverable_rejects_db_path_and_wal_shm_journal` — `DB_PATH=tmp_path/"user_data.db"`,
  проверить сам файл и `+"-wal"/"-shm"/"-journal"` — все 4 rejected.
- `test_is_deliverable_rejects_db_path_via_symlink` — реальный файл `tmp_path/"real.db"` с
  `DB_PATH` на него, symlink `tmp_path/"link.db" -> real.db`, вызов с путём симлинка — rejected
  (realpath раскрывает ссылку).
- `test_is_deliverable_deny_wins_over_tempdir_fallback` — `DB_PATH` указывает ВНУТРЬ
  `tempfile.gettempdir()` (через monkeypatch обоих) — всё равно rejected (порядок: deny раньше
  temp-fallback).
- `test_is_deliverable_allows_matching_scope_artifact_workspace` /
  `test_is_deliverable_rejects_other_scope_artifact_workspace` — файл в
  `artifact_workspace(storage_root, "chat:1")`, запрос с `scope="chat:1"` → allowed,
  `scope="chat:2"` → rejected с текстом про "different delivery scope".
- `test_is_deliverable_allows_matching_scope_skills_workdir` /
  `test_is_deliverable_rejects_other_scope_skills_workdir` — путь вида
  `<workdir_root>/<skill_id>/<safe_scope>/out.txt`, аналогично allow/deny по scope.
- `test_is_deliverable_request_started_at_allows_recent_file_outside_known_roots` — файл ВНЕ
  storage_root/tempdir (например, отдельный `tmp_path2` не связанный с default temp), но
  `request_started_at` = время до записи файла минус запас — allowed именно через эту ветку
  (используется для документирования готовности API, а не текущего рантайм-пути).
- `test_is_deliverable_missing_file_inside_known_root_is_allowed` — путь внутри
  `tempfile.gettempdir()`, файла физически нет — allowed=True (не должен требовать `stat()`
  успеха для path-membership правил).
- `test_artifact_workspace_path_shape` — `artifact_workspace("/data", "chat:10")` ==
  `Path("/data/artifacts/chat_10")` (двоеточие → `_`, как в
  `bot/plugins/skills.py:3614`).
- `test_is_protected_path_matches_is_deliverable_deny_set` — параметризованный тест: каждый
  сценарий из deny-тестов выше также даёт `is_protected_path(path) is True`; произвольный
  allowed-путь даёт `False`.

### 5.2 `tests/test_agent_tools_plugin.py` — существующие (не менять, должны остаться зелёными)

`test_deliver_to_user_returns_final_direct_result`, `test_deliver_to_user_records_blocked_status`,
`test_deliver_to_user_rejects_blocked_without_reason`,
`test_deliver_to_user_deduplicates_only_within_same_request`,
`test_deliver_to_user_requires_text_or_artifacts`,
`test_deliver_to_user_rejects_missing_or_empty_files` (`tests/test_agent_tools_plugin.py:2221-2379`)
— все используют `storage_root=str(tmp_path)` и файлы прямо в его корне → покрыты широким
"внутри storage_root" правилом через `self.storage_root`, без изменений в самих тестах.

Новые в этом файле:
- `test_deliver_to_user_rejects_artifact_outside_storage_root_and_temp` — артефакт по пути вне
  `storage_root`, вне `tempfile.gettempdir()` (например, соседний `tmp_path_factory` каталог) →
  `result["success"] is False`, `"outside allowed delivery locations"` в `result["error"]`.
- `test_deliver_to_user_rejects_db_path_artifact` — `artifacts=[{"file_path": <DB_PATH из env>}]`
  (создать реальный файл через monkeypatch `DB_PATH`) → rejected.

### 5.3 `tests/test_plugin_direct_results.py` — существующие (не менять)

`test_handle_direct_result_photo_path_failure_logs_values`,
`test_handle_direct_result_file_path_read_failure_logs_value`
(`tests/test_plugin_direct_results.py:408-465`) используют `tmp_path`-based несуществующие пути.
Подтверждено эмпирически (`tempfile.gettempdir() == "/tmp"`, pytest `tmp_path` реально создаётся
под `/tmp/pytest-of-.../...` в этом окружении — проверено прогоном) — оба пути проходят
temp-fallback правило (чистая проверка принадлежности пути, без обращения к `stat()`), поэтому
`is_deliverable` возвращает `allowed=True` и downstream-код (`get_image_size`/`open()`) кидает
исходные исключения без изменений.

Новые:
- `test_handle_direct_result_rejects_photo_path_outside_allowed_locations` — путь вне
  storage_root/tempdir/runtime dirs (monkeypatch `tempfile.gettempdir` на другой каталог, файл
  реально существует в третьем месте) → `message.reply_text` вызван с текстом
  `"Artifact path is unavailable: ..."`, `message.reply_photo`/`reply_document` НЕ вызваны.
- `test_handle_direct_result_rejects_file_path_without_raising` — аналогично для kind=file:
  отклонённый путь → текстовый ответ, БЕЗ `FileNotFoundError`/иного исключения (в отличие от
  `test_handle_direct_result_file_path_read_failure_logs_value`, где путь разрешён, но
  физически отсутствует).
- `test_cleanup_intermediate_files_skips_protected_path` — вызвать
  `cleanup_intermediate_files({"direct_result": {"format": "path", "value": <DB_PATH>}})` на
  реально существующий файл по этому пути → файл НЕ удалён.

### 5.4 `tests/test_agent_delivery.py` — существующие (не менять)

`test_send_agent_response_final_sends_artifacts_before_text`,
`test_send_agent_response_missing_path_reports_unavailable`,
`test_send_agent_response_file_path_cleans_up_after_delivery`,
`test_send_agent_response_file_path_cleans_up_expanded_home_path` — все на `tmp_path`-путях
(включая `~`-раскрытие через `HOME=tmp_path/"home"`), все под `/tmp` → temp-fallback,
поведение не меняется, включая точный текст `"Artifact path is unavailable: missing.txt"` для
отсутствующего файла (эта проверка идёт ПОСЛЕ is_deliverable и не переопределяется).

Новый:
- `test_send_agent_response_rejects_path_outside_allowed_locations` — путь вне temp/storage_root
  (создать реальный файл в изолированном каталоге, замоканном как НЕ являющийся gettempdir()) →
  тот же текст `"Artifact path is unavailable: ..."`, `bot.send_document`/`send_photo` не
  вызваны.

## 6. Приёмочные команды

```
~/.venvs/ctb/bin/python -m pytest tests/test_artifact_paths.py tests/test_agent_tools_plugin.py \
  tests/test_plugin_direct_results.py tests/test_agent_delivery.py -q --no-header -p no:cacheprovider
~/.venvs/ctb/bin/python -m ruff check bot/artifact_paths.py bot/plugins/agent_tools.py bot/utils.py \
  bot/agent_delivery.py tests/test_artifact_paths.py
python3 -m mypy bot/artifact_paths.py bot/plugins/agent_tools.py bot/utils.py bot/agent_delivery.py \
  --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports
```
Полный `tests/` прогон — задача координатора волны 2, не только этой задачи.

## 7. Риски / открытые допущения

1. **CWD-стабильность.** `.env`/`usage_logs`/`uploads/webshot` резолвятся от `Path.cwd()`.
   Верно только пока процесс не делает `os.chdir()` (подтверждено — таких вызовов в `bot/` нет) и
   запускается из корня репозитория (текущее соглашение проекта). Если это когда-то изменится —
   правила деградируют тихо (allow может не сработать для `uploads/webshot`, deny может не
   сработать для `usage_logs`/`.env`) без явной ошибки. Не хуже текущего поведения этих путей.
2. **`request_started_at` ветка не покрыта реальным вызовом.** Код существует и тестируется
   синтетически (§5.1), но ни один из трёх интеграционных сайтов сегодня не передаёт значение —
   вся защита для temp-путей идёт через `tempfile.gettempdir()`-fallback. Если позже добавят
   timestamp в `RequestContext`, сужение станет автоматическим без правок `artifact_paths.py`.
3. **`storage_root`/`skills_dir`/`skills_workdir_root` эвристика реплицирует env-логику, а не
   спрашивает живой `PluginManager`/skills-плагин.** Совпадает с продакшеном, пока
   `PLUGIN_STORAGE_ROOT`/`SKILLS_DIR`/`SKILLS_WORKDIR` не переопределены ПОСЛЕ старта процесса
   динамически (в коде такого не происходит — читаются один раз при `initialize()`).
4. **`haiper_image_to_video.py`** отправляет видео напрямую через `message.reply_video`, минуя
   все 3 точки интеграции — остаётся полностью незащищённым T05 (было так и до задачи). Не
   исправляется в этом тикете — вне заявленного владения файлами.
5. **`text_document_qa.py`** (`bot/plugins/text_document_qa.py:793,914`) зовёт
   `handle_direct_result` — вне владения T05, не трогается, но автоматически получает защиту,
   так как это внутренняя правка `handle_direct_result`, а не его сигнатуры.
6. **`cleanup_intermediate_files` правка технически выходит за буквальный список шагов
   мастер-плана** (там перечислены только 3 send-точки), но без неё deny-политика частично
   бессмысленна для отклонённых, но существующих чувствительных файлов (см. §0.3). Явно
   выделено как необходимое, а не самовольное расширение.
7. **Отсутствие проверки владельца scope для широкого "внутри storage_root" правила.** Любой файл
   в `storage_root`, но НЕ во scope-именованных поддеревьях (`artifacts/<scope>`,
   `skill_workdir/<skill>/<scope>`), доступен для доставки из любого scope — это сохранение
   ТЕКУЩЕГО поведения (`_allowed_artifact_roots` сегодня даёт полный `storage_root` без scope-
   привязки вообще), не новая дыра, но и не полная scope-изоляция. Дальнейшее ужесточение (если
   потребуется) — отдельная задача.
