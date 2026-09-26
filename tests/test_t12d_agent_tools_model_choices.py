"""T12d W2: `_model_choices_for_helper`'s fallback block (no ``get_model_choices``
on the helper) now delegates to ``bot.utils.parse_model_choices`` instead of
duplicating its own parsing loop. This test proves the new call is
byte-for-byte equivalent to the removed manual block on the edge cases that
matter (string vs list input, missing/falsy ``model_choices``, default
already present, whitespace, empty entries) by re-implementing the removed
block verbatim as ``_legacy_model_choices`` and comparing outputs.
"""
from types import SimpleNamespace

import pytest

from bot.plugins.agent_tools import _model_choices_for_helper


def _legacy_model_choices(config: dict) -> list[str]:
    """Verbatim copy of the fallback block removed from
    ``_model_choices_for_helper`` in T12d W2 (pre-``parse_model_choices``)."""
    choices = config.get("model_choices") or []
    if isinstance(choices, str):
        models = [model.strip() for model in choices.split(",") if model.strip()]
    else:
        models = [str(model).strip() for model in choices if str(model).strip()]

    default_model = str(config.get("model") or "").strip()
    if default_model and default_model not in models:
        models.insert(0, default_model)
    return models


EDGE_CASE_CONFIGS = [
    {"model_choices": "a,b,c", "model": "default"},
    {"model_choices": " a , b ,, c ", "model": "default"},
    {"model_choices": "a,b,default", "model": "default"},
    {"model_choices": ["a", "b"], "model": "default"},
    {"model_choices": ["a", "", " ", "b"], "model": "default"},
    {"model_choices": [], "model": "default"},
    {"model_choices": "", "model": "default"},
    {"model_choices": None, "model": "default"},
    {"model": "default"},
    {"model_choices": "a,b", "model": ""},
    {"model_choices": "a,b", "model": None},
    {},
]


@pytest.mark.parametrize("config", EDGE_CASE_CONFIGS)
def test_model_choices_for_helper_matches_legacy_block(config):
    helper = SimpleNamespace(config=config)
    assert _model_choices_for_helper(helper) == _legacy_model_choices(config)
