from __future__ import annotations

import pytest

pytest.importorskip("tiktoken")

from bot.plugins.hindsight_memory import (
    ConsolidationAction,
    apply_consolidation_actions,
    parse_consolidation_actions,
)


# --- parse_consolidation_actions -------------------------------------------------


def test_parse_update_delete_add():
    out = "UPDATE 1: revised fact\nDELETE 2\nADD: new fact"
    actions = parse_consolidation_actions(out)
    assert actions == [
        ConsolidationAction(kind="update", index=1, text="revised fact"),
        ConsolidationAction(kind="delete", index=2),
        ConsolidationAction(kind="add", text="new fact"),
    ]


def test_parse_is_case_insensitive():
    out = "update 1: revised fact\ndelete 2\nadd: new fact\nnone"
    actions = parse_consolidation_actions(out)
    assert actions == [
        ConsolidationAction(kind="update", index=1, text="revised fact"),
        ConsolidationAction(kind="delete", index=2),
        ConsolidationAction(kind="add", text="new fact"),
    ]


def test_parse_ignores_garbage_lines():
    out = "banana\nUPDATE abc: not a number\nDELETE\nADDnocolon\n\nUPDATE 1: kept"
    actions = parse_consolidation_actions(out)
    assert actions == [ConsolidationAction(kind="update", index=1, text="kept")]


def test_parse_none_alone_returns_empty():
    assert parse_consolidation_actions("NONE") == []
    assert parse_consolidation_actions("none") == []
    assert parse_consolidation_actions("  None  \n") == []


def test_parse_none_mixed_with_garbage_returns_empty():
    out = "NONE\nnot a real action\n   \n"
    assert parse_consolidation_actions(out) == []


def test_parse_never_raises_on_arbitrary_input():
    # Should not raise for any of these, and should return a list.
    for bad in ("", None, "\x00\x01", "UPDATE " + "9" * 50 + ": x", "🤖 emoji only"):
        assert isinstance(parse_consolidation_actions(bad), list)


# --- apply_consolidation_actions -------------------------------------------------


def test_apply_update():
    body = "- fact one\n- fact two"
    actions = [ConsolidationAction(kind="update", index=2, text="fact two revised")]
    assert apply_consolidation_actions(body, actions) == "- fact one\n- fact two revised"


def test_apply_delete():
    body = "- fact one\n- fact two"
    actions = [ConsolidationAction(kind="delete", index=1)]
    assert apply_consolidation_actions(body, actions) == "- fact two"


def test_apply_add_appends_new_bullet():
    body = "- fact one\n- fact two"
    actions = [ConsolidationAction(kind="add", text="fact three")]
    assert apply_consolidation_actions(body, actions) == "- fact one\n- fact two\n- fact three"


def test_apply_multiple_adds_appended_in_order():
    body = "- fact one"
    actions = [
        ConsolidationAction(kind="add", text="fact two"),
        ConsolidationAction(kind="add", text="fact three"),
    ]
    assert apply_consolidation_actions(body, actions) == "- fact one\n- fact two\n- fact three"


def test_apply_empty_actions_returns_body_unchanged_byte_for_byte():
    body = "- fact one\n\n- fact two  \n### heading"
    assert apply_consolidation_actions(body, []) == body


def test_apply_out_of_range_index_is_noop():
    body = "- fact one\n- fact two"
    actions = [ConsolidationAction(kind="update", index=5, text="ignored"), ConsolidationAction(kind="delete", index=99)]
    assert apply_consolidation_actions(body, actions) == body


def test_apply_delete_wins_over_update_on_same_index():
    body = "- fact one\n- fact two"
    actions = [
        ConsolidationAction(kind="update", index=1, text="should not appear"),
        ConsolidationAction(kind="delete", index=1),
    ]
    assert apply_consolidation_actions(body, actions) == "- fact two"


def test_apply_unmentioned_bullets_survive_partial_response_byte_for_byte():
    """Key regression: a partial/garbage-heavy model response must not touch
    bullets it didn't mention."""
    body = "- fact one\n- fact two  (weird trailing spaces)\n- fact three"
    actions = parse_consolidation_actions("UPDATE 2: fact two revised\nnonsense line\nDELETE 99")
    result = apply_consolidation_actions(body, actions)
    lines = result.split("\n")
    assert lines[0] == "- fact one"
    assert lines[1] == "- fact two revised"
    assert lines[2] == "- fact three"


def test_apply_headings_and_blank_lines_are_preserved_and_not_numbered():
    body = "### Preferences\n\n- fact one\n- fact two"
    actions = [ConsolidationAction(kind="delete", index=1)]
    result = apply_consolidation_actions(body, actions)
    assert result == "### Preferences\n\n- fact two"


def test_apply_delete_of_explicit_remember_marked_bullet_is_noop():
    body = "- User explicitly asked to remember their birthday is in March.\n- other fact"
    actions = [ConsolidationAction(kind="delete", index=1)]
    assert apply_consolidation_actions(body, actions) == body


def test_apply_delete_of_explicit_remember_marked_bullet_is_noop_ru():
    body = "- запомни день рождения в марте\n- other fact"
    actions = [ConsolidationAction(kind="delete", index=1)]
    assert apply_consolidation_actions(body, actions) == body
