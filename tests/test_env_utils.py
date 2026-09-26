from bot.env_utils import env_bool


def test_env_bool_unset_returns_default(monkeypatch):
    monkeypatch.delenv("SOME_FLAG", raising=False)
    assert env_bool("SOME_FLAG", True) is True
    assert env_bool("SOME_FLAG", False) is False


def test_env_bool_true_is_case_insensitive(monkeypatch):
    for value in ("true", "True", "TRUE", "tRuE"):
        monkeypatch.setenv("SOME_FLAG", value)
        assert env_bool("SOME_FLAG", False) is True


def test_env_bool_anything_else_is_false(monkeypatch):
    for value in ("false", "0", "1", "yes", "no", "on", "garbage", ""):
        monkeypatch.setenv("SOME_FLAG", value)
        assert env_bool("SOME_FLAG", True) is False
