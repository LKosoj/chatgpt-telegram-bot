from types import SimpleNamespace

from bot.chat_response_utils import finalize_chat_answer, leading_system_count


# ---------------------------------------------------------------------------
# leading_system_count
# ---------------------------------------------------------------------------


def test_leading_system_count_empty_list():
    assert leading_system_count([]) == 0


def test_leading_system_count_no_system_messages():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert leading_system_count(messages) == 0


def test_leading_system_count_all_leading_system():
    messages = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert leading_system_count(messages) == 2


def test_leading_system_count_stops_at_first_non_system():
    messages = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "system", "content": "c"},
    ]
    # A system message after a non-system message does not count.
    assert leading_system_count(messages) == 1


def test_leading_system_count_ignores_non_dict_entries_as_break():
    messages = [{"role": "system", "content": "a"}, "not-a-dict", {"role": "system", "content": "b"}]
    assert leading_system_count(messages) == 1


# ---------------------------------------------------------------------------
# finalize_chat_answer
# ---------------------------------------------------------------------------


class FakePluginManager:
    def get_plugin_source_name(self, function_name):
        return function_name.split(".", 1)[0]


class FakeHelper:
    def __init__(self, config):
        self.config = config
        self.plugin_manager = FakePluginManager()
        self.history_calls = []

    async def _add_to_history(self, chat_id, *, role, content, session_id=None):
        self.history_calls.append((chat_id, role, content, session_id))


def _config(**overrides):
    base = {
        "n_choices": 1,
        "bot_language": "en",
        "show_usage": False,
        "show_plugins_used": False,
    }
    base.update(overrides)
    return base


def _choice(content):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")


def _usage(total, prompt=None, completion=None):
    return SimpleNamespace(total_tokens=total, prompt_tokens=prompt, completion_tokens=completion)


async def test_finalize_chat_answer_single_choice_no_usage_no_plugins():
    helper = FakeHelper(_config())
    response = SimpleNamespace(choices=[_choice("hello")], usage=_usage(5))

    answer, total_tokens = await finalize_chat_answer(helper, chat_id=1, response=response)

    assert answer == "hello"
    assert total_tokens == 5
    assert helper.history_calls == [(1, "assistant", "hello", None)]


async def test_finalize_chat_answer_multiple_choices_numbered():
    helper = FakeHelper(_config(n_choices=2))
    response = SimpleNamespace(choices=[_choice("first"), _choice("second")], usage=_usage(9))

    answer, total_tokens = await finalize_chat_answer(helper, chat_id=1, response=response, session_id="s1")

    assert "1⃣\nfirst" in answer
    assert "2⃣\nsecond" in answer
    assert total_tokens == 9
    # Only the first choice is written to history.
    assert helper.history_calls == [(1, "assistant", "first", "s1")]


async def test_finalize_chat_answer_show_usage_with_full_split():
    helper = FakeHelper(_config(show_usage=True))
    response = SimpleNamespace(choices=[_choice("hi")], usage=_usage(7, prompt=3, completion=4))

    answer, total_tokens = await finalize_chat_answer(helper, chat_id=1, response=response)

    assert total_tokens == 7
    assert "\U0001f4b0 7 tokens" in answer
    assert "3 prompt" in answer
    assert "4 completion" in answer


async def test_finalize_chat_answer_show_usage_without_matching_split_omits_parens():
    helper = FakeHelper(_config(show_usage=True))
    token_accumulator = [3, 4]  # total (7) != usage_tokens (5) -> parenthetical omitted
    response = SimpleNamespace(choices=[_choice("hi")], usage=_usage(5, prompt=2, completion=3))

    answer, total_tokens = await finalize_chat_answer(
        helper, chat_id=1, response=response, token_accumulator=token_accumulator,
    )

    assert total_tokens == 7
    assert "\U0001f4b0 7 tokens" in answer
    assert "prompt" not in answer


async def test_finalize_chat_answer_show_plugins_used_with_show_usage():
    helper = FakeHelper(_config(show_usage=True, show_plugins_used=True))
    response = SimpleNamespace(choices=[_choice("hi")], usage=_usage(5, prompt=2, completion=3))

    answer, total_tokens = await finalize_chat_answer(
        helper, chat_id=1, response=response, plugins_used=("web_search.search",),
    )

    assert "\U0001f50c web_search" in answer


async def test_finalize_chat_answer_show_plugins_used_without_show_usage():
    helper = FakeHelper(_config(show_usage=False, show_plugins_used=True))
    response = SimpleNamespace(choices=[_choice("hi")], usage=_usage(5))

    answer, total_tokens = await finalize_chat_answer(
        helper, chat_id=1, response=response, plugins_used=("web_search.search",),
    )

    assert answer.endswith("\U0001f50c web_search")


async def test_finalize_chat_answer_no_plugins_used_no_footer_when_show_usage_false():
    helper = FakeHelper(_config())
    response = SimpleNamespace(choices=[_choice("hi")], usage=_usage(5))

    answer, total_tokens = await finalize_chat_answer(helper, chat_id=1, response=response)

    assert answer == "hi"
