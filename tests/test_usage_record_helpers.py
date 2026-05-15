import importlib.machinery
import importlib.util
import sys
import types

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.usage_tracker import UsageTracker
from bot.utils import (
    make_usage_tracker,
    record_chat_tokens,
    record_image_request,
    record_tts_request,
    record_transcription_seconds,
    record_vision_tokens,
)


def _make_config(allowed="42"):
    return {
        "allowed_user_ids": allowed,
        "token_price": 0.002,
        "image_prices": [0.016, 0.018, 0.02],
        "vision_token_price": 0.01,
        "tts_prices": [0.015, 0.030],
        "transcription_price": 0.006,
    }


def _make_tracker(tmp_path, user_id, name):
    return UsageTracker(user_id, name, logs_dir=str(tmp_path))


def test_record_chat_tokens_charges_guest_for_non_allowed(tmp_path):
    user = _make_tracker(tmp_path, 99, "stranger")
    guests = _make_tracker(tmp_path, "guests", "guests")
    usage = {99: user, "guests": guests}

    assert record_chat_tokens(usage, _make_config(allowed="42"), 99, 1000) is True

    assert sum(user.usage["usage_history"]["chat_tokens"].values()) == 1000
    assert sum(guests.usage["usage_history"]["chat_tokens"].values()) == 1000


def test_record_chat_tokens_skips_guest_for_allowed(tmp_path):
    user = _make_tracker(tmp_path, 42, "owner")
    guests = _make_tracker(tmp_path, "guests", "guests")
    usage = {42: user, "guests": guests}

    assert record_chat_tokens(usage, _make_config(allowed="42"), 42, 500) is True

    assert sum(user.usage["usage_history"]["chat_tokens"].values()) == 500
    assert guests.usage["usage_history"]["chat_tokens"] == {}


def test_record_chat_tokens_returns_false_on_zero(tmp_path):
    user = _make_tracker(tmp_path, 42, "owner")
    usage = {42: user}
    assert record_chat_tokens(usage, _make_config(), 42, 0) is False


def test_record_chat_tokens_rejects_negative_and_bool(tmp_path):
    user = _make_tracker(tmp_path, 42, "owner")
    usage = {42: user}

    assert record_chat_tokens(usage, _make_config(), 42, -1) is False
    assert record_chat_tokens(usage, _make_config(), 42, True) is False
    assert user.usage["usage_history"]["chat_tokens"] == {}


def test_record_chat_tokens_returns_false_for_unknown_user(tmp_path):
    usage = {}
    assert record_chat_tokens(usage, _make_config(), 7, 10) is False


def test_usage_tracker_uses_init_prices_when_add_called_without_price(tmp_path):
    tracker = UsageTracker(
        42, "user", logs_dir=str(tmp_path),
        token_price=0.5,
    )
    tracker.add_chat_tokens(1000)
    assert tracker.usage["current_cost"]["day"] == 0.5  # 1000 * 0.5 / 1000


def test_record_image_request_uses_init_image_prices(tmp_path):
    tracker = UsageTracker(
        42, "user", logs_dir=str(tmp_path),
        image_prices=[1.0, 2.0, 3.0],
    )
    usage = {42: tracker}
    assert record_image_request(usage, _make_config(allowed="42"), 42, "256x256") is True
    assert tracker.usage["current_cost"]["day"] == 1.0


def test_record_vision_tokens_charges_user_and_guest(tmp_path):
    user = UsageTracker(99, "stranger", logs_dir=str(tmp_path), vision_token_price=0.5)
    guests = UsageTracker("guests", "guests", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {99: user, "guests": guests}
    assert record_vision_tokens(usage, _make_config(allowed="42"), 99, 2000) is True
    assert sum(user.usage["usage_history"]["vision_tokens"].values()) == 2000
    assert sum(guests.usage["usage_history"]["vision_tokens"].values()) == 2000
    assert user.usage["current_cost"]["day"] == 1.0
    assert guests.usage["current_cost"]["day"] == 1.0


def test_record_vision_tokens_accepts_numeric_string(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {42: user}

    assert record_vision_tokens(usage, _make_config(allowed="42"), 42, "2000") is True

    assert sum(user.usage["usage_history"]["vision_tokens"].values()) == 2000
    assert user.usage["current_cost"]["day"] == 1.0


def test_record_vision_tokens_skips_zero_string(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {42: user}

    assert record_vision_tokens(usage, _make_config(allowed="42"), 42, "0") is False

    assert user.usage["usage_history"]["vision_tokens"] == {}
    assert user.usage["current_cost"]["day"] == 0.0


def test_record_vision_tokens_rejects_negative_and_bool(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), vision_token_price=0.5)
    usage = {42: user}

    assert record_vision_tokens(usage, _make_config(allowed="42"), 42, -1) is False
    assert record_vision_tokens(usage, _make_config(allowed="42"), 42, False) is False
    assert user.usage["usage_history"]["vision_tokens"] == {}
    assert user.usage["current_cost"]["day"] == 0.0


def test_record_tts_request_charges_user(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), tts_prices=[0.5, 0.75])
    usage = {42: user}
    assert record_tts_request(usage, _make_config(allowed="42"), 42, 1000, "tts-1-hd") is True
    model_history = user.usage["usage_history"]["tts_characters"]["tts-1-hd"]
    assert sum(model_history.values()) == 1000
    assert user.usage["current_cost"]["day"] == 0.75


def test_record_transcription_seconds_charges_user(tmp_path):
    user = UsageTracker(42, "owner", logs_dir=str(tmp_path), transcription_price=0.5)
    usage = {42: user}
    assert record_transcription_seconds(usage, _make_config(allowed="42"), 42, 60) is True
    assert sum(user.usage["usage_history"]["transcription_seconds"].values()) == 60
    assert user.usage["current_cost"]["day"] == 0.5


def test_record_tts_and_transcription_reject_negative_and_bool(tmp_path):
    tts_user = UsageTracker(42, "owner", logs_dir=str(tmp_path / "tts"), tts_prices=[0.5, 0.75])
    transcription_user = UsageTracker(
        42,
        "owner",
        logs_dir=str(tmp_path / "transcription"),
        transcription_price=0.5,
    )

    assert record_tts_request({42: tts_user}, _make_config(allowed="42"), 42, -100, "tts-1") is False
    assert record_tts_request({42: tts_user}, _make_config(allowed="42"), 42, True, "tts-1") is False
    assert tts_user.usage["usage_history"]["tts_characters"] == {}

    assert record_transcription_seconds({42: transcription_user}, _make_config(allowed="42"), 42, -10) is False
    assert record_transcription_seconds({42: transcription_user}, _make_config(allowed="42"), 42, False) is False
    assert transcription_user.usage["usage_history"]["transcription_seconds"] == {}


def test_make_usage_tracker_threads_config_prices(tmp_path):
    config = _make_config()
    config["token_price"] = 0.123
    config["image_prices"] = [9.0, 9.0, 9.0]
    config["vision_token_price"] = 0.456
    config["tts_prices"] = [0.7, 0.8]
    config["transcription_price"] = 0.321
    tracker = make_usage_tracker(config, 42, "owner", logs_dir=str(tmp_path))
    assert tracker.prices["token_price"] == 0.123
    assert tracker.prices["image_prices"] == [9.0, 9.0, 9.0]
    assert tracker.prices["vision_token_price"] == 0.456
    assert tracker.prices["tts_prices"] == [0.7, 0.8]
    assert tracker.prices["transcription_price"] == 0.321
