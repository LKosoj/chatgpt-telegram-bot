from bot.utils import parse_model_choices


def test_parse_model_choices_none_raw_returns_default_only():
    assert parse_model_choices(None, "default") == ["default"]


def test_parse_model_choices_string_does_not_dedupe_and_prepends_default():
    assert parse_model_choices("a,b, a", "default") == ["default", "a", "b", "a"]


def test_parse_model_choices_list_prepends_default():
    assert parse_model_choices(["a", "b"], "default") == ["default", "a", "b"]


def test_parse_model_choices_default_already_present_not_duplicated():
    assert parse_model_choices("a,default,b", "default") == ["a", "default", "b"]


def test_parse_model_choices_empty_string_raw():
    assert parse_model_choices("", "default") == ["default"]


def test_parse_model_choices_list_strips_whitespace():
    assert parse_model_choices([" a ", "b "], "default") == ["default", "a", "b"]


def test_parse_model_choices_no_default_model():
    assert parse_model_choices("a,b", "") == ["a", "b"]
