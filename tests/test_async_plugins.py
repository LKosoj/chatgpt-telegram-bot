import asyncio
import time

import pytest
import requests

from bot.plugins import crypto as crypto_module
from bot.plugins import iplocation as iplocation_module
from bot.plugins import weather as weather_module
from bot.plugins.crypto import CryptoPlugin
from bot.plugins.iplocation import IpLocationPlugin
from bot.plugins.weather import WeatherPlugin


class FakeResponse:
    def __init__(self, payload=None, json_error=None):
        self.payload = payload
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


def _fake_get(calls, payload=None, error=None, json_error=None):
    def fake_get(url, *args, **kwargs):
        calls.append(url)
        if error:
            raise error
        return FakeResponse(payload=payload, json_error=json_error)

    return fake_get


def _assert_controlled_error(result, expected_message):
    assert isinstance(result, dict)
    error = result.get("error") or result.get("Error")
    assert error
    assert expected_message in error


SUCCESS_CASES = [
    pytest.param(
        weather_module,
        WeatherPlugin,
        "get_current_weather",
        {"latitude": "52.52", "longitude": "13.41", "unit": "celsius"},
        {"current_weather": {"temperature": 21.5}},
        {"current_weather": {"temperature": 21.5}},
        id="weather",
    ),
    pytest.param(
        crypto_module,
        CryptoPlugin,
        "get_crypto_rate",
        {"asset": "bitcoin"},
        {"data": {"symbol": "BTC", "rateUsd": "60000.00"}},
        {"data": {"symbol": "BTC", "rateUsd": "60000.00"}},
        id="crypto",
    ),
    pytest.param(
        iplocation_module,
        IpLocationPlugin,
        "iplocation",
        {"ip": "8.8.8.8"},
        {
            "data": {
                "country": "US",
                "subdivisions": "California",
                "city": "Mountain View",
                "asn": 15169,
                "as_name": "Google LLC",
                "as_domain": "google.com",
            }
        },
        {
            "Location": "US, California, Mountain View",
            "ASN": 15169,
            "AS Name": "Google LLC",
            "AS Domain": "google.com",
        },
        id="iplocation",
    ),
]


ERROR_CASES = [
    pytest.param(
        weather_module,
        WeatherPlugin,
        "get_current_weather",
        {"latitude": "52.52", "longitude": "13.41", "unit": "celsius"},
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "WeatherPlugin lets requests errors escape "
                "from async execute"
            ),
        ),
        id="weather",
    ),
    pytest.param(
        crypto_module,
        CryptoPlugin,
        "get_crypto_rate",
        {"asset": "bitcoin"},
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "CryptoPlugin lets requests errors escape "
                "from async execute"
            ),
        ),
        id="crypto",
    ),
    pytest.param(
        iplocation_module,
        IpLocationPlugin,
        "iplocation",
        {"ip": "8.8.8.8"},
        id="iplocation",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module, plugin_cls, function_name, kwargs, payload, expected",
    SUCCESS_CASES,
)
async def test_async_http_plugins_use_mocked_successful_response(
    monkeypatch,
    module,
    plugin_cls,
    function_name,
    kwargs,
    payload,
    expected,
):
    calls = []
    monkeypatch.setattr(
        module.requests,
        "get",
        _fake_get(calls, payload=payload),
    )

    result = await plugin_cls().execute(function_name, helper=None, **kwargs)

    assert result == expected
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module, plugin_cls, function_name, kwargs",
    ERROR_CASES,
)
async def test_async_http_plugins_return_controlled_error_on_network_error(
    monkeypatch,
    module,
    plugin_cls,
    function_name,
    kwargs,
):
    calls = []
    monkeypatch.setattr(
        module.requests,
        "get",
        _fake_get(calls, error=requests.Timeout("timed out")),
    )

    result = await plugin_cls().execute(function_name, helper=None, **kwargs)

    _assert_controlled_error(result, "timed out")
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module, plugin_cls, function_name, kwargs",
    ERROR_CASES,
)
async def test_async_http_plugins_handle_json_parse_error_separately(
    monkeypatch,
    module,
    plugin_cls,
    function_name,
    kwargs,
):
    calls = []
    monkeypatch.setattr(
        module.requests,
        "get",
        _fake_get(calls, json_error=ValueError("invalid json")),
    )

    result = await plugin_cls().execute(function_name, helper=None, **kwargs)

    _assert_controlled_error(result, "invalid json")
    assert len(calls) == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CryptoPlugin calls blocking requests.get directly "
        "inside async execute"
    ),
)
@pytest.mark.asyncio
async def test_slow_http_does_not_block_parallel_async_task(monkeypatch):
    events = []

    def slow_get(url, *args, **kwargs):
        time.sleep(0.05)
        return FakeResponse({"data": {"symbol": "BTC", "rateUsd": "60000.00"}})

    monkeypatch.setattr(crypto_module.requests, "get", slow_get)

    async def parallel_task():
        await asyncio.sleep(0.01)
        events.append("parallel")

    task = asyncio.create_task(parallel_task())
    await CryptoPlugin().execute(
        "get_crypto_rate",
        helper=None,
        asset="bitcoin",
    )
    events.append("plugin_done")
    await task

    assert events == ["parallel", "plugin_done"]
