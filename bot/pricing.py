"""Per-model chat token pricing.

Cost is still computed and stored at write time (see
``UsageTracker.add_chat_tokens`` / ``bot.utils.record_chat_tokens``), not
derived at read time -- a full derive-at-read-time redesign would touch the
budget hot path (``get_remaining_budget`` bot/utils.py:742 /
``is_within_budget`` bot/utils.py:788). This module only resolves the cost of a single chat
completion given a (possibly per-model) price table; callers keep storing
the resulting cost in usage_events.cost, plus enough metadata (model,
prompt/completion tokens, price_source) for a future re-derivation of
history.

Prices are USD per 1000 tokens, matching the existing ``TOKEN_PRICE``
convention (not per 1M tokens), to avoid mixing units.
"""
from __future__ import annotations

import logging

# Populated only via the MODEL_TOKEN_PRICES env var (see
# load_model_token_prices / bot/__main__.py). Left empty here on purpose:
# our models are gateway aliases (see bot/model_constants.py -- llmgateway/high,
# llmgateway/light_model, llmgateway/big_context), so their real prices are
# only known to whoever owns the installation.
DEFAULT_MODEL_TOKEN_PRICES: dict[str, tuple[float, float]] = {}


def load_model_token_prices(env_value: str | None) -> dict[str, tuple[float, float]]:
    """Parse a ``MODEL_TOKEN_PRICES``-style env value into a price table.

    Format: ``model=in:out,model2=in:out`` where ``in``/``out`` are USD per
    1000 tokens for prompt/completion tokens respectively. Invalid entries
    are skipped with a warning (mirrors ``parse_model_context_windows_env``
    in bot/__main__.py). Starts from ``DEFAULT_MODEL_TOKEN_PRICES`` so env
    entries override (currently empty) built-in defaults.
    """
    prices = dict(DEFAULT_MODEL_TOKEN_PRICES)
    # Tracks models seen in *this env value* only, so overriding a built-in
    # default from DEFAULT_MODEL_TOKEN_PRICES stays silent while a genuine
    # duplicate inside MODEL_TOKEN_PRICES gets flagged.
    seen: set[str] = set()
    raw = env_value or ''
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        model, sep, value = item.partition('=')
        model = model.strip()
        value = value.strip()
        if not sep or not model or not value:
            logging.warning('Skipping invalid MODEL_TOKEN_PRICES entry=%r', item)
            continue
        price_in_raw, price_sep, price_out_raw = value.partition(':')
        if not price_sep:
            logging.warning(
                'Skipping invalid MODEL_TOKEN_PRICES value for %s value=%r', model, value,
            )
            continue
        try:
            price_in = float(price_in_raw.strip())
            price_out = float(price_out_raw.strip())
        except ValueError:
            logging.warning(
                'Skipping invalid MODEL_TOKEN_PRICES value for %s value=%r', model, value,
            )
            continue
        if price_in < 0 or price_out < 0:
            logging.warning(
                'Skipping negative MODEL_TOKEN_PRICES value for %s value=%r', model, value,
            )
            continue
        if model in seen:
            logging.warning(
                'Duplicate MODEL_TOKEN_PRICES entry for %s, using the last one value=%r',
                model, value,
            )
        seen.add(model)
        prices[model] = (price_in, price_out)
    return prices


def resolve_chat_cost(
    *,
    model,
    total_tokens,
    prompt_tokens=None,
    completion_tokens=None,
    fallback_price_per_1k,
    table,
):
    """Resolve the USD cost of one chat completion and how it was priced.

    :param model: model actually used for the turn, or None/unknown.
    :param total_tokens: total tokens for the turn (used for the fallback
        and blended paths).
    :param prompt_tokens: prompt-token count for the turn, if known.
    :param completion_tokens: completion-token count for the turn, if known.
    :param fallback_price_per_1k: legacy flat USD/1k-token price (TOKEN_PRICE).
    :param table: ``{model: (price_in, price_out)}`` USD-per-1k-token table.
    :return: ``(cost, price_source)`` where ``price_source`` is one of:
        - ``'model_split'``: model is in ``table`` and both prompt/completion
          token counts are known -- cost priced per direction.
        - ``'model_blended'``: model is in ``table`` but only total_tokens is
          known -- cost uses the average of price_in/price_out.
        - ``'legacy_fallback'``: model missing from ``table`` -- cost uses
          ``fallback_price_per_1k`` against total_tokens (matches the
          pre-existing single-price behaviour).
    """
    entry = table.get(model) if table and model else None
    if entry is not None and prompt_tokens is not None and completion_tokens is not None:
        price_in, price_out = entry
        cost = prompt_tokens * price_in / 1000 + completion_tokens * price_out / 1000
        return round(cost, 6), 'model_split'
    if entry is not None:
        price_in, price_out = entry
        blended = (price_in + price_out) / 2
        cost = total_tokens * blended / 1000
        return round(cost, 6), 'model_blended'
    cost = total_tokens * fallback_price_per_1k / 1000
    return round(cost, 6), 'legacy_fallback'
