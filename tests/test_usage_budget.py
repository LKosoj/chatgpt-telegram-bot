import importlib.machinery
import importlib.util
import sys
import types
from types import SimpleNamespace

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.usage_tracker import UsageTracker
from bot.utils import get_remaining_budget


def test_weekly_budget_uses_week_cost(tmp_path):
    tracker = UsageTracker(42, "user", logs_dir=str(tmp_path))
    tracker.add_current_costs(2.5)
    usage = {42: tracker}
    update = SimpleNamespace(
        message=SimpleNamespace(from_user=SimpleNamespace(id=42, name="user")),
        callback_query=None,
        inline_query=None,
        effective_user=SimpleNamespace(id=42, name="user"),
    )
    config = {
        "admin_user_ids": "-",
        "allowed_user_ids": "42",
        "user_budgets": "10",
        "budget_period": "weekly",
        "guest_budget": 0,
    }

    assert get_remaining_budget(config, usage, update) == 7.5
