import re

from bot.i18n import (
    configured_language,
    localized_text,
    normalize_language,
    reset_current_language,
    set_current_language,
    supported_languages,
    translations,
)


PLACEHOLDER_PATTERN = re.compile(r"\{[^{}]+\}")


def test_normalize_language_accepts_supported_codes_and_aliases():
    assert normalize_language("pt_BR") == "pt-br"
    assert normalize_language("zh_CN") == "zh-cn"
    assert normalize_language("de-DE") == "de"
    assert normalize_language("zh-Hans") == "zh-cn"


def test_normalize_language_falls_back_to_english_for_unknown_codes():
    assert normalize_language("missing-language") == "en"


def test_all_languages_have_same_translation_keys():
    expected_keys = set(translations["en"])
    for language in supported_languages():
        assert set(translations[language]) == expected_keys


def test_all_languages_keep_english_format_placeholders():
    for key, english_value in translations["en"].items():
        if not isinstance(english_value, str):
            continue

        expected_placeholders = set(PLACEHOLDER_PATTERN.findall(english_value))
        for language in supported_languages():
            value = translations[language][key]
            if isinstance(value, str):
                assert set(PLACEHOLDER_PATTERN.findall(value)) == expected_placeholders


def test_new_agent_tool_strings_are_localized():
    assert localized_text("busy_status_preparing", "ru") == "Готовлю ответ..."
    assert localized_text("agent_tools_answer_received", "es") == "Respuesta recibida."


def test_configured_language_preserves_auto_mode():
    assert configured_language(None) == "auto"
    assert configured_language("") == "auto"
    assert configured_language("auto") == "auto"
    assert configured_language("pt_BR") == "pt-br"


def test_localized_text_uses_current_language_in_auto_mode():
    token = set_current_language("ru")
    try:
        assert localized_text("settings_title", "auto") == "Настройки"
        assert localized_text("settings_title", "en") == "Настройки"
    finally:
        reset_current_language(token)
