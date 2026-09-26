"""Shared env-var parsing helpers.

Extracted in T12a to remove copy-pasted parsing loops (see
docs/improvement_2026-09-25/T12-plan.md, section T12a). Behavior must stay
byte-for-byte identical to the historical call sites it replaces -- see the
docstrings below for which call site each function was extracted from.
"""
from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    """Soft boolean env parsing -- replaces the inline ``.lower() == 'true'`` idiom.

    Byte-for-byte equivalent of the historical per-call-site expression
    (moved from ``bot/__main__.py``): unset env -> ``default``; **any**
    other value (including recognizable synonyms like ``'1'``/``'yes'``) ->
    ``False`` unless it is exactly ``'true'`` case-insensitively. Never
    raises. Do not use this for env vars whose invalid-value handling must
    reject startup -- those keep using the stricter ``parse_bool_env`` in
    ``bot/__main__.py``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() == "true"
