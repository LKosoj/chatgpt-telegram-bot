import asyncio
import socket

import httpx
import pytest

from bot import net_safety
from bot.net_safety import (
    ResponseTooLargeError,
    SafeHTTPStatusError,
    UnsafeURLError,
    _ByteCappedResponse,
    resolve_public_ip,
    resolve_public_ip_async,
    safe_get,
    safe_request,
    safe_urlopen,
    validate_public_url,
    validate_public_url_async,
)


def _fake_getaddrinfo(ips):
    def fake(host, *_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]
    return fake


def _gaierror_getaddrinfo(host, *_args, **_kwargs):
    raise socket.gaierror("name or service not known")


def _unicode_error_getaddrinfo(host, *_args, **_kwargs):
    # A hostname label longer than 63 octets fails IDNA encoding inside
    # socket.getaddrinfo before any actual DNS lookup, raising UnicodeError
    # instead of socket.gaierror (T03-review.md round 1, WARNING 2).
    raise UnicodeError("label empty or too long")


def _value_error_getaddrinfo(host, *_args, **_kwargs):
    raise ValueError("Int or String expected")


# --- resolve_public_ip / resolve_public_ip_async -----------------------------------


def test_resolve_public_ip_skips_private_and_returns_first_global(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["127.0.0.1", "8.8.8.8"]))
    assert resolve_public_ip("example.com") == "8.8.8.8"


def test_resolve_public_ip_returns_none_when_all_private(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["127.0.0.1", "10.0.0.1"]))
    assert resolve_public_ip("example.com") is None


def test_resolve_public_ip_returns_none_on_gaierror(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _gaierror_getaddrinfo)
    assert resolve_public_ip("nonexistent.invalid") is None


def test_resolve_public_ip_returns_none_on_unicode_error(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _unicode_error_getaddrinfo)
    assert resolve_public_ip("a" * 300 + ".example") is None


def test_resolve_public_ip_returns_none_on_value_error(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _value_error_getaddrinfo)
    assert resolve_public_ip("example.com") is None


def test_resolve_public_ip_rejects_nat64_wellknown_embedded_private(monkeypatch):
    # 64:ff9b::169.254.169.254 is a NAT64 (RFC 6052) address embedding the cloud
    # metadata endpoint; ipaddress.IPv6Address.is_global returns True for it even
    # though the real destination after translation is link-local (WARNING 1).
    monkeypatch.setattr(
        net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["64:ff9b::169.254.169.254"])
    )
    assert resolve_public_ip("example.com") is None


def test_resolve_public_ip_accepts_nat64_wellknown_embedded_public(monkeypatch):
    monkeypatch.setattr(
        net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["64:ff9b::8.8.8.8"])
    )
    assert resolve_public_ip("example.com") == "64:ff9b::8.8.8.8"


@pytest.mark.asyncio
async def test_resolve_public_ip_async_uses_to_thread(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["127.0.0.1", "8.8.8.8"]))
    calls = []
    real_to_thread = asyncio.to_thread

    async def counting_to_thread(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(net_safety.asyncio, "to_thread", counting_to_thread)

    result = await resolve_public_ip_async("example.com")

    assert result == "8.8.8.8"
    assert len(calls) == 1
    assert calls[0][0] is resolve_public_ip


# --- validate_public_url / validate_public_url_async --------------------------------


def test_validate_public_url_rejects_non_http_scheme():
    assert validate_public_url("file:///etc/passwd") is not None
    assert validate_public_url("ftp://x/") is not None


def test_validate_public_url_rejects_missing_hostname():
    assert validate_public_url("http:///path") is not None


def test_validate_public_url_rejects_userinfo():
    assert validate_public_url("http://user:pass@example.com/") is not None


def test_validate_public_url_rejects_loopback_v4(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["127.0.0.1"]))
    assert validate_public_url("http://example.com/") is not None


def test_validate_public_url_rejects_loopback_v6(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["::1"]))
    assert validate_public_url("http://example.com/") is not None


def test_validate_public_url_rejects_link_local_metadata(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["169.254.169.254"]))
    assert validate_public_url("http://example.com/") is not None


def test_validate_public_url_rejects_cgnat(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["100.64.0.1"]))
    assert validate_public_url("http://example.com/") is not None


def test_validate_public_url_rejects_private_v4(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["10.0.0.1"]))
    assert validate_public_url("http://example.com/") is not None


def test_validate_public_url_returns_reason_on_unicode_error(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _unicode_error_getaddrinfo)
    assert validate_public_url("http://" + "a" * 300 + ".example/x") is not None


def test_validate_public_url_returns_reason_on_value_error(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _value_error_getaddrinfo)
    assert validate_public_url("http://example.com/") is not None


# --- NAT64 / 6to4 / Teredo embedded-IPv4 checks (T03-review.md round 1, WARNING 1) --
#
# These IPv6 transition mechanisms embed an IPv4 address inside the low-order bits.
# ipaddress.IPv6Address.is_global classifies the whole prefix (or, for the NAT64
# well-known prefix, always) as global without looking at the embedded address, so a
# malicious DNS answer can smuggle a private/loopback/link-local IPv4 target past the
# plain is_global check. The addresses below are precomputed embeddings (see
# T03-review.md for the construction) - not generated in the test itself, to keep the
# test focused on the validation outcome rather than address-building code.


def test_validate_public_url_rejects_nat64_wellknown_embedded_metadata(monkeypatch):
    monkeypatch.setattr(
        net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["64:ff9b::169.254.169.254"])
    )
    assert validate_public_url("http://example.com/") is not None


def test_validate_public_url_accepts_nat64_wellknown_embedded_public(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["64:ff9b::8.8.8.8"]))
    assert validate_public_url("http://example.com/") is None


def test_validate_public_url_rejects_nat64_local_use_embedded_private(monkeypatch):
    # 64:ff9b:1::/48 (RFC 8215 local-use NAT64 prefix), embedding 169.254.169.254.
    monkeypatch.setattr(
        net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["64:ff9b:1:a9fe:a9:fe00::"])
    )
    assert validate_public_url("http://example.com/") is not None


def test_validate_public_url_accepts_nat64_local_use_embedded_public(monkeypatch):
    # Same prefix, embedding 8.8.8.8.
    monkeypatch.setattr(
        net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["64:ff9b:1:808:8:800::"])
    )
    assert validate_public_url("http://example.com/") is None


def test_validate_public_url_rejects_sixtofour_embedded_private(monkeypatch):
    # 2002::/16 (6to4), embedding 169.254.169.254.
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["2002:a9fe:a9fe::"]))
    assert validate_public_url("http://example.com/") is not None


def test_validate_public_url_accepts_sixtofour_embedded_public(monkeypatch):
    # Same prefix, embedding 8.8.8.8.
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["2002:808:808::"]))
    assert validate_public_url("http://example.com/") is None


def test_validate_public_url_rejects_teredo_embedded_private_client(monkeypatch):
    # 2001::/32 (Teredo), server 8.8.8.8, client 169.254.169.254.
    monkeypatch.setattr(
        net_safety.socket,
        "getaddrinfo",
        _fake_getaddrinfo(["2001:0:808:808:0:ffff:5601:5601"]),
    )
    assert validate_public_url("http://example.com/") is not None


def test_validate_public_url_accepts_teredo_embedded_public(monkeypatch):
    # Same prefix, server 8.8.8.8, client 8.8.4.4.
    monkeypatch.setattr(
        net_safety.socket,
        "getaddrinfo",
        _fake_getaddrinfo(["2001:0:808:808:0:ffff:f7f7:fbfb"]),
    )
    assert validate_public_url("http://example.com/") is None


def test_validate_public_url_accepts_public_host(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["8.8.8.8"]))
    assert validate_public_url("http://example.com/") is None


def test_validate_public_url_allow_private_bypasses_ip_check_but_not_scheme(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["127.0.0.1"]))
    assert validate_public_url("http://example.com/", allow_private=True) is None
    assert validate_public_url("ftp://example.com/", allow_private=True) is not None


@pytest.mark.asyncio
async def test_validate_public_url_async_matches_sync_semantics(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["127.0.0.1"]))
    assert await validate_public_url_async("http://example.com/") is not None
    assert await validate_public_url_async("http://example.com/", allow_private=True) is None
    assert await validate_public_url_async("ftp://example.com/") is not None

    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["8.8.8.8"]))
    assert await validate_public_url_async("http://example.com/") is None


@pytest.mark.asyncio
async def test_validate_public_url_async_returns_reason_on_unicode_error(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _unicode_error_getaddrinfo)
    assert await validate_public_url_async("http://" + "a" * 300 + ".example/x") is not None


# --- synchronous path: safe_urlopen / _ByteCappedResponse ---------------------------


def test_safe_urlopen_rejects_unsafe_initial_url(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["127.0.0.1"]))

    def spy_create_connection(*_args, **_kwargs):
        raise AssertionError("must not connect for an unsafe URL")

    monkeypatch.setattr(net_safety.socket, "create_connection", spy_create_connection)

    with pytest.raises(UnsafeURLError):
        safe_urlopen("http://169.254.169.254/x")


class _FakeRawResponse:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0
        self.closed = False

    def read(self, amt=None):
        if amt is None:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos:self._pos + amt]
        self._pos += len(chunk)
        return chunk

    def close(self):
        self.closed = True


def test_safe_urlopen_enforces_max_bytes_on_read_with_amt():
    raw = _FakeRawResponse(b"x" * 100)
    capped = _ByteCappedResponse(raw, max_bytes=50)

    with pytest.raises(ResponseTooLargeError):
        while True:
            chunk = capped.read(8192)
            if not chunk:
                break


def test_safe_urlopen_enforces_max_bytes_on_read_without_amt():
    calls = []
    real_read = _FakeRawResponse.read

    class _CountingRaw(_FakeRawResponse):
        def read(self, amt=None):
            calls.append(amt)
            return real_read(self, amt)

    raw = _CountingRaw(b"x" * 1_000_000)
    capped = _ByteCappedResponse(raw, max_bytes=50)

    with pytest.raises(ResponseTooLargeError):
        capped.read()

    assert calls, "expected chunked reads, not a single unbounded read"
    assert all(amt == net_safety._CHUNK for amt in calls)


def test_safe_urlopen_is_context_manager():
    raw = _FakeRawResponse(b"hello")
    with _ByteCappedResponse(raw, max_bytes=1000) as response:
        assert response.read() == b"hello"
    assert raw.closed is True


# --- asynchronous path: safe_get / safe_request --------------------------------------


@pytest.mark.asyncio
async def test_safe_get_returns_content_and_status(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["8.8.8.8"]))

    def handler(request):
        return httpx.Response(200, json={"ok": True})

    response = await safe_get("http://example.com/", transport=httpx.MockTransport(handler))

    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.asyncio
async def test_safe_get_raise_for_status_matches_error_shape(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["8.8.8.8"]))

    def handler(request):
        return httpx.Response(500, text="server exploded")

    response = await safe_get("http://example.com/", transport=httpx.MockTransport(handler))

    with pytest.raises(SafeHTTPStatusError) as exc_info:
        response.raise_for_status()
    assert exc_info.value.response.status_code == 500
    assert "server exploded" in exc_info.value.response.text


@pytest.mark.asyncio
async def test_safe_get_rejects_unsafe_initial_url_without_network_call(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["127.0.0.1"]))

    def handler(request):
        raise AssertionError("must not perform the request for an unsafe URL")

    with pytest.raises(UnsafeURLError):
        await safe_get("http://169.254.169.254/", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_safe_get_enforces_max_bytes(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["8.8.8.8"]))

    def handler(request):
        return httpx.Response(200, content=b"x" * 1000)

    with pytest.raises(ResponseTooLargeError):
        await safe_get("http://example.com/", max_bytes=10, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_safe_get_rejects_redirect_to_private_ip(monkeypatch):
    def fake_getaddrinfo(host, *_args, **_kwargs):
        ip = "127.0.0.1" if host == "internal.example" else "8.8.8.8"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(net_safety.socket, "getaddrinfo", fake_getaddrinfo)

    def handler(request):
        if request.url.host == "internal.example":
            raise AssertionError("must not follow the redirect to a private host")
        return httpx.Response(302, headers={"location": "http://internal.example/secret"})

    with pytest.raises(UnsafeURLError):
        await safe_get("http://public.example/", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_safe_get_follows_redirect_to_public_target(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["8.8.8.8"]))

    def handler(request):
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://public-target.example/final"})
        return httpx.Response(200, json={"from": "final target"})

    response = await safe_get("http://public.example/start", transport=httpx.MockTransport(handler))

    assert response.status_code == 200
    assert response.json() == {"from": "final target"}


@pytest.mark.asyncio
async def test_safe_get_too_many_redirects_raises(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["8.8.8.8"]))

    def handler(request):
        return httpx.Response(302, headers={"location": "http://example.com/"})

    with pytest.raises(UnsafeURLError):
        await safe_get("http://example.com/", max_redirects=2, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_safe_request_post_sends_json_body(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["8.8.8.8"]))

    def handler(request):
        assert request.method == "POST"
        import json as _json
        assert _json.loads(request.content) == {"name": "test_function", "arguments": {"x": 1}}
        return httpx.Response(200, json={"result": "success"})

    response = await safe_request(
        "POST",
        "http://example.com/execute",
        json={"name": "test_function", "arguments": {"x": 1}},
        transport=httpx.MockTransport(handler),
    )

    assert response.json() == {"result": "success"}


@pytest.mark.asyncio
async def test_safe_request_allow_private_permits_loopback(monkeypatch):
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", _fake_getaddrinfo(["127.0.0.1"]))

    def handler(request):
        return httpx.Response(200, json={"ok": True})

    response = await safe_request(
        "GET",
        "http://127.0.0.1:9999/tools",
        allow_private=True,
        transport=httpx.MockTransport(handler),
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
