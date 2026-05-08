import asyncio
import json
from types import SimpleNamespace

import pytest

from bot.plugin_manager import PluginManager
from bot.plugins.agent_tools import AgentToolsPlugin


class FakeBot:
    def __init__(self):
        self.messages = []
        self.markup_edits = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.messages))

    async def edit_message_reply_markup(self, **kwargs):
        self.markup_edits.append(kwargs)


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools") and not any(message.get("role") == "tool" for message in kwargs["messages"]):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="tool_1",
                                    function=SimpleNamespace(
                                        name="skills.list_skills",
                                        arguments="{}",
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        content = kwargs["messages"][-1]["content"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"done: {content.splitlines()[0]}", tool_calls=None)
                )
            ]
        )


class FakePluginManager:
    def __init__(self):
        self.calls = []

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        return [
            {
                "type": "function",
                "function": {
                    "name": "skills.list_skills",
                    "description": "list skills",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "skills.publish_result",
                    "description": "publish final result",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "agent_tools.run_subagents",
                    "description": "nested subagents",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    async def call_function(self, function_name, helper, arguments, request_context=None):
        self.calls.append((function_name, json.loads(arguments)))
        return json.dumps({"success": True, "skills": ["demo"]}, ensure_ascii=False)


class FakeLLMHelper:
    def __init__(self):
        self.completions = FakeCompletions()
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))
        self.plugin_manager = FakePluginManager()

    def get_current_model(self, user_id):
        return "llmgateway/high"


class FakeMessage:
    def __init__(self, chat_id: int, text: str, user_id: int = 42):
        self.chat = SimpleNamespace(id=chat_id)
        self.from_user = SimpleNamespace(id=user_id)
        self.text = text
        self.replies = []

    async def reply_text(self, text: str):
        self.replies.append(text)


def test_agent_tools_registers_specs_and_handlers():
    pm = PluginManager(config={"plugins": ["agent_tools"]})

    specs = pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["agent_tools"])
    names = {spec["function"]["name"] for spec in specs}
    assert names == {
        "agent_tools.manage_plan_tasks",
        "agent_tools.ask_telegram_user",
        "agent_tools.cancel_pending_question",
        "agent_tools.run_subagents",
    }
    ask_spec = next(spec["function"] for spec in specs if spec["function"]["name"] == "agent_tools.ask_telegram_user")
    assert ask_spec["parameters"]["required"] == ["question", "options"]
    assert ask_spec["parameters"]["properties"]["options"]["minItems"] == 1

    commands = pm.get_plugin_commands()
    assert any(command.get("callback_pattern") == "^agentask:" for command in commands)
    assert pm.get_message_handlers()


def test_agent_tools_preserves_full_option_text():
    plugin = AgentToolsPlugin()
    long_option = "A" * 120

    assert plugin._normalize_options([long_option]) == [long_option]


@pytest.mark.asyncio
async def test_manage_plan_tasks_tracks_progress(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = SimpleNamespace(user_id=42)

    added = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[
            {"id": "T1", "content": "Inspect implementation", "status": "completed"},
            {"id": "T2", "content": "Add Telegram ask tool", "status": "in_progress"},
        ],
    )

    assert added["success"] is True
    assert added["plan_tasks"]["progress"] == {"total": 2, "closed": 1, "open": 1}

    updated = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="update",
        tasks=[{"id": "T2", "status": "completed"}],
    )

    assert updated["success"] is True
    assert updated["plan_tasks"]["progress"] == {"total": 2, "closed": 2, "open": 0}

    cleared = await plugin.execute("manage_plan_tasks", helper, chat_id=10, action="clear")
    assert cleared["success"] is True
    assert cleared["plan_tasks"]["tasks"] == []


@pytest.mark.asyncio
async def test_run_subagents_runs_tool_capable_workers(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper()

    result = await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        shared_context="Shared facts",
        subagents=[
            {"id": "a1", "role": "reviewer", "task": "Check assumptions"},
            {"id": "a2", "role": "tester", "task": "Find test cases", "context": "Focus on files"},
        ],
    )

    assert result["success"] is True
    assert [item["id"] for item in result["subagents"]] == ["a1", "a2"]
    assert [item["status"] for item in result["subagents"]] == ["completed", "completed"]
    assert len(helper.completions.calls) == 4
    assert all(call["model"] == "llmgateway/high" for call in helper.completions.calls)
    assert all("max_tokens" not in call for call in helper.completions.calls)
    tool_names = {
        tool["function"]["name"]
        for call in helper.completions.calls
        for tool in call.get("tools", [])
    }
    assert "skills.list_skills" in tool_names
    assert "skills.publish_result" not in tool_names
    assert "agent_tools.run_subagents" not in tool_names
    assert [call[0] for call in helper.plugin_manager.calls] == ["skills.list_skills", "skills.list_skills"]
    assert any(
        "Shared facts" in message["content"]
        for message in helper.completions.calls[0]["messages"]
        if message.get("role") == "user"
    )


@pytest.mark.asyncio
async def test_run_subagents_applies_per_subagent_overrides(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper()

    await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        max_rounds=4,
        subagents=[
            {"id": "a1", "role": "r", "task": "t", "temperature": 0.7, "max_rounds": 25},
        ],
    )

    assert helper.completions.calls
    assert helper.completions.calls[0]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_run_subagents_floors_max_rounds_to_minimum(tmp_path):
    from bot.plugins.agent_tools import MIN_SUBAGENT_TOOL_ROUNDS, AgentToolsPlugin as _Plugin

    assert _Plugin._normalize_max_rounds(1) == MIN_SUBAGENT_TOOL_ROUNDS
    assert _Plugin._normalize_max_rounds(None) == MIN_SUBAGENT_TOOL_ROUNDS
    assert _Plugin._normalize_max_rounds(MIN_SUBAGENT_TOOL_ROUNDS + 5) == MIN_SUBAGENT_TOOL_ROUNDS + 5
    assert _Plugin._normalize_max_rounds("garbage") == MIN_SUBAGENT_TOOL_ROUNDS
    assert _Plugin._normalize_max_rounds(None, default=15) == 15
    assert _Plugin._normalize_max_rounds(5, default=15) == MIN_SUBAGENT_TOOL_ROUNDS
    assert _Plugin._normalize_max_rounds(20, default=15) == 20


@pytest.mark.asyncio
async def test_run_subagents_rejects_too_many_workers(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper()

    result = await plugin.execute(
        "run_subagents",
        helper,
        subagents=[
            {"id": f"a{i}", "role": "worker", "task": "work"}
            for i in range(6)
        ],
    )

    assert result["success"] is False
    assert "At most" in result["error"]


@pytest.mark.asyncio
async def test_ask_telegram_user_resolves_from_text_answer(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick an approach",
            options=["Minimal", "Full port"],
            timeout_seconds=1,
        )
    )

    for _ in range(10):
        if plugin.pending_by_chat:
            break
        await asyncio.sleep(0)

    assert bot.messages
    assert plugin.is_waiting_for_text(10, 42) is True

    other_user_message = FakeMessage(chat_id=10, text="Full port", user_id=99)
    await plugin.handle_text_answer(SimpleNamespace(effective_message=other_user_message), SimpleNamespace())
    assert task.done() is False

    message = FakeMessage(chat_id=10, text="Minimal", user_id=42)
    update = SimpleNamespace(effective_message=message)
    await plugin.handle_text_answer(update, SimpleNamespace())
    result = await task

    assert result["success"] is True
    assert result["answer"] == "Minimal"
    assert message.replies == ["Answer received."]
    assert plugin.pending_by_chat == {}


class FakeSkillsAwarePluginManager(FakePluginManager):
    def __init__(self):
        super().__init__()
        self._skills_plugin = SimpleNamespace(
            active_skills={"chat:10": {"pptx": {}}},
            available_skills={"pptx": {"scripts": ["build.py"]}},
            _scope_key=lambda kwargs: f"chat:{kwargs.get('chat_id')}",
        )

    def get_plugin(self, name: str):
        return self._skills_plugin if name == "skills" else None

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        specs = super().get_functions_specs(helper, model_to_use, allowed_plugins)
        return specs + [
            {
                "type": "function",
                "function": {
                    "name": "codeinterpreter.deep_analysis",
                    "description": "run code",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]


class _DeepAnalysisCompletions(FakeCompletions):
    def __init__(self, code_prompt: str):
        super().__init__()
        self._code_prompt = code_prompt

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools") and not any(message.get("role") == "tool" for message in kwargs["messages"]):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="ci_call",
                    function=SimpleNamespace(
                        name="codeinterpreter.deep_analysis",
                        arguments=json.dumps({"code_prompt": self._code_prompt}),
                    ),
                )],
            ))])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="acknowledged", tool_calls=None),
        )])


class _DeepAnalysisHelper:
    def __init__(self, code_prompt: str):
        self.completions = _DeepAnalysisCompletions(code_prompt)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))
        self.plugin_manager = FakeSkillsAwarePluginManager()

    def get_current_model(self, user_id):
        return "llmgateway/high"


@pytest.mark.asyncio
async def test_subagent_routing_guard_blocks_active_skill_script_via_codeinterpreter(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = _DeepAnalysisHelper("import subprocess; subprocess.run(['python3', 'build.py'])")

    result = await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        subagents=[{"id": "a1", "role": "worker", "task": "do work"}],
    )

    assert result["success"] is True
    called = [name for name, _args in helper.plugin_manager.calls]
    assert "codeinterpreter.deep_analysis" not in called
    tool_messages = [
        message["content"]
        for message in helper.completions.calls[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert tool_messages
    assert any("skills.run_skill_script" in content for content in tool_messages)
    assert any('"script_name": "build.py"' in content for content in tool_messages)


@pytest.mark.asyncio
async def test_subagent_routing_guard_allows_unrelated_codeinterpreter_call(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = _DeepAnalysisHelper("print(6 * 7)")

    result = await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        subagents=[{"id": "a1", "role": "worker", "task": "do work"}],
    )

    assert result["success"] is True
    called = [name for name, _args in helper.plugin_manager.calls]
    assert "codeinterpreter.deep_analysis" in called


@pytest.mark.asyncio
async def test_ask_telegram_user_timeout_clears_markup_and_notifies(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    result = await plugin.execute(
        "ask_telegram_user",
        helper,
        chat_id=10,
        user_id=42,
        question="Pick an approach",
        options=["Yes", "No"],
        allow_free_text=False,
        timeout_seconds=1,
    )

    assert result["success"] is False
    assert "Timed out" in result["error"]
    assert bot.markup_edits and bot.markup_edits[0]["reply_markup"] is None
    assert bot.markup_edits[0]["chat_id"] == 10
    notification = next(
        (m for m in bot.messages if "истекло" in str(m.get("text", ""))),
        None,
    )
    assert notification is not None
    assert plugin.pending_by_chat == {}


@pytest.mark.asyncio
async def test_close_clears_inline_markup_for_pending_questions(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick an approach",
            options=["Yes", "No"],
            timeout_seconds=10,
        )
    )
    for _ in range(20):
        if plugin.pending_by_chat and any(
            pending.get("message_id") for pending in plugin.pending_questions.values()
        ):
            break
        await asyncio.sleep(0)
    assert plugin.pending_by_chat

    plugin.close()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await task

    assert bot.markup_edits, "Expected close() to schedule markup cleanup"
    assert bot.markup_edits[0]["reply_markup"] is None
    assert plugin.pending_by_chat == {}


@pytest.mark.asyncio
async def test_ask_telegram_user_requires_options(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    result = await plugin.execute(
        "ask_telegram_user",
        helper,
        chat_id=10,
        user_id=42,
        question=(
            "Как получить файл?\n"
            "- Попробовать другой способ отправки\n"
            "- Предоставить путь к файлу"
        ),
        allow_free_text=True,
        timeout_seconds=1,
    )

    assert result["success"] is False
    assert "options" in result["error"]
    assert bot.messages == []


@pytest.mark.asyncio
async def test_cancel_pending_question_clears_markup_and_resolves(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick",
            options=["A", "B"],
            timeout_seconds=10,
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
        if plugin.pending_by_chat and any(
            pending.get("message_id") for pending in plugin.pending_questions.values()
        ):
            break
    assert plugin.pending_by_chat

    cancel_result = await plugin.execute(
        "cancel_pending_question",
        helper,
        chat_id=10,
        user_id=42,
        reason="Решили иначе",
    )
    assert cancel_result["success"] is True
    assert plugin.pending_by_chat == {}

    cancel_notice = next(
        (m for m in bot.messages if "отменён" in str(m.get("text", ""))),
        None,
    )
    assert cancel_notice is not None
    assert any(edit.get("reply_markup") is None for edit in bot.markup_edits)

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_pending_question_without_pending_returns_error(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    result = await plugin.execute(
        "cancel_pending_question",
        helper,
        chat_id=10,
        user_id=42,
    )
    assert result["success"] is False
    assert "No pending question" in result["error"]


@pytest.mark.asyncio
async def test_ask_telegram_user_multi_select_force_disables_free_text(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick all that apply",
            options=["A", "B", "C"],
            multi_select=True,
            allow_free_text=True,
            timeout_seconds=10,
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
        if plugin.pending_by_chat and any(
            pending.get("message_id") for pending in plugin.pending_questions.values()
        ):
            break
    assert plugin.pending_by_chat

    pending = next(iter(plugin.pending_questions.values()))
    assert pending["multi_select"] is True
    assert pending["allow_free_text"] is False

    markup = bot.messages[0]["reply_markup"]
    button_labels = [btn.text for row in markup.inline_keyboard for btn in row]
    assert any(label.startswith("☐ ") for label in button_labels)
    assert "Confirm" in button_labels

    plugin.close()
    await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await task


def test_subagent_internal_publish_collects_items_for_parent():
    plugin = AgentToolsPlugin()
    published: list = []

    response = plugin._handle_internal_publish(
        {"kind": "text", "value": "found something"},
        published,
    )
    assert "stored" in response
    response2 = plugin._handle_internal_publish(
        {"kind": "file", "value": "/tmp/result.csv", "caption": "csv"},
        published,
    )
    assert "stored" in response2

    assert len(published) == 2
    assert published[0] == {"kind": "text", "value": "found something"}
    assert published[1] == {"kind": "file", "value": "/tmp/result.csv", "caption": "csv"}


def test_internal_publish_outside_subagent_returns_error():
    plugin = AgentToolsPlugin()
    response = plugin._handle_internal_publish(
        {"kind": "text", "value": "hi"},
        None,
    )
    assert "only available inside a subagent" in response


def test_internal_publish_validates_kind():
    plugin = AgentToolsPlugin()
    published: list = []
    response = plugin._handle_internal_publish(
        {"kind": "weird", "value": "x"},
        published,
    )
    assert "kind must be one of" in response
    assert published == []


def test_question_markup_renders_selected_marks(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    markup = plugin._question_markup(
        "qid",
        ["A", "B", "C"],
        allow_free_text=False,
        multi_select=True,
        selected_indices={0, 2},
    )
    labels = [btn.text for row in markup.inline_keyboard for btn in row]
    assert labels == ["✅ A", "☐ B", "✅ C", "Confirm"]


@pytest.mark.asyncio
async def test_pending_question_persisted_and_recovered_on_startup(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick",
            options=["A", "B"],
            timeout_seconds=10,
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
        if any(p.get("message_id") for p in plugin.pending_questions.values()):
            break

    pending_file = tmp_path / "agent_pending_questions.json"
    assert pending_file.exists()
    snapshot = json.loads(pending_file.read_text())
    assert len(snapshot) == 1
    assert snapshot[0]["chat_id"] == 10
    assert snapshot[0]["message_id"] == 1

    plugin.close()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await task

    bot.messages.clear()
    bot.markup_edits.clear()

    new_plugin = AgentToolsPlugin()
    new_plugin.initialize(storage_root=str(tmp_path))
    assert new_plugin._orphaned_pending and len(new_plugin._orphaned_pending) == 1

    await new_plugin.on_startup(SimpleNamespace(bot=bot))

    assert new_plugin._orphaned_pending == []
    assert any(edit.get("reply_markup") is None for edit in bot.markup_edits)
    notice = next((m for m in bot.messages if "перезапущен" in str(m.get("text", ""))), None)
    assert notice is not None
    assert not pending_file.exists()
