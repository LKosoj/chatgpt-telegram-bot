"""Общие помощники для защиты исходящих HTTP-запросов от SSRF (Server-Side Request
Forgery — когда сервер по указке пользователя обращается к внутреннему/приватному
адресу, до которого снаружи достучаться нельзя, например к облачному
metadata-эндпоинту 169.254.169.254).

Модуль не читает переменные окружения сам — лимиты/флаги (max_bytes, timeout,
allow_private и т.п.) передают вызывающие плагины явно через параметры функций;
каждый плагин продолжает сам читать свой os.getenv.

Два независимых пути:
- Синхронный (``safe_urlopen``, поверх ``urllib``) закрепляет IP перед подключением:
  резолвит hostname, проверяет его один раз и физически соединяется именно с этим
  IP (кастомные ``http.client.HTTPConnection``/``HTTPSConnection``), а не даёт
  библиотеке резолвить домен повторно перед connect. Для HTTPS SNI и проверка
  сертификата всё равно идут по настоящему имени хоста
  (``ssl_context.wrap_socket(sock, server_hostname=host)``).
- Асинхронный (``safe_request``/``safe_get``, поверх ``httpx``) закрепления IP не
  делает: httpx/httpcore резолвят hostname самостоятельно перед TCP-подключением.
  Вместо этого URL (и цель каждого редиректа) проверяется непосредственно перед
  запросом. Между этой проверкой и реальным connect остаётся окно на DNS rebinding
  (домен с очень коротким TTL меняет ответ между проверкой и подключением) — это
  осознанное ограничение этой задачи, а не забытый случай: устойчивое закрепление
  IP для httpx потребовало бы либо собственного ``httpx.AsyncBaseTransport``, либо
  завязки на приватные атрибуты httpcore, без покрытия тестами. Если требования
  изменятся, направление — кастомный ``httpcore.AsyncNetworkBackend``, передаваемый
  в ``httpcore.AsyncConnectionPool(network_backend=...)``.
"""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB

_CHUNK = 65536
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}

# NAT64 (RFC 6052) prefixes: an IPv6 address in either range embeds an IPv4 address in
# its low-order bits. ipaddress.IPv6Address.is_global does not look at that embedded
# address (it classifies 64:ff9b::/96 as always-global, and 64:ff9b:1::/48 as
# always-non-global), so a DNS answer in either prefix can smuggle a private/loopback/
# link-local IPv4 target past a plain is_global check (T03-review.md round 1, WARNING 1).
_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_NAT64_LOCAL_USE_PREFIX = ipaddress.ip_network("64:ff9b:1::/48")


class UnsafeURLError(Exception):
    """URL (или цель редиректа) отклонён: недопустимая схема/userinfo, приватный IP,
    DNS не резолвится, либо превышен лимит редиректов."""


class ResponseTooLargeError(Exception):
    """Тело ответа превысило лимит max_bytes."""


class SafeHTTPStatusError(Exception):
    """Аналог httpx.HTTPStatusError для safe_request/safe_get: статус ответа >= 400.

    .response указывает на self, чтобы код вида exc.response.status_code /
    exc.response.text (уже используемый в mcp_server.py) продолжал работать без
    переписывания.
    """

    def __init__(self, status_code: int, text: str):
        super().__init__(f"HTTP {status_code}: {text[:200]}")
        self.status_code = status_code
        self.text = text
        self.response = self


def _nat64_embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """Достаёт embedded IPv4 из NAT64-адреса (RFC 6052 §2.2), если ip попадает в
    известный NAT64-префикс, иначе None.

    64:ff9b::/96 (well-known prefix, PL=96): IPv4 занимает последние 32 бита целиком.
    64:ff9b:1::/48 (RFC 8215 local-use prefix, PL=48): по таблице RFC 6052 §2.2 первые
    16 бит IPv4 идут сразу за 48-битным префиксом, затем 8-битный зарезервированный
    байт "u", затем оставшиеся 16 бит IPv4.
    """
    packed = ip.packed
    if ip in _NAT64_WELL_KNOWN_PREFIX:
        return ipaddress.IPv4Address(packed[12:16])
    if ip in _NAT64_LOCAL_USE_PREFIX:
        return ipaddress.IPv4Address(packed[6:8] + packed[9:11])
    return None


def _is_global_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Как ip.is_global, но для IPv6-адресов, разворачивающих IPv4 (NAT64/6to4/Teredo),
    решение принимается по embedded IPv4 — реальной цели после трансляции/туннеля —
    а не по классификации самой IPv6-обёртки (см. _nat64_embedded_ipv4 выше и
    T03-review.md round 1, WARNING 1)."""
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _nat64_embedded_ipv4(ip)
        if embedded is not None:
            return embedded.is_global
        if ip.sixtofour is not None:
            return ip.sixtofour.is_global
        if ip.teredo is not None:
            server_ip, client_ip = ip.teredo
            return server_ip.is_global and client_ip.is_global
    return ip.is_global


def resolve_public_ip(host: str) -> str | None:
    """Первый IP из getaddrinfo(host) с _is_global_ip==True, иначе None.

    Если в ответе DNS есть и приватные, и публичные адреса — берём первый публичный,
    не отказываем целиком (это НЕ то же самое, что validate_public_url, которая
    отказывает URL целиком при наличии хотя бы одного не-публичного адреса).
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        # UnicodeError/ValueError: например, хост с меткой длиннее 63 октетов падает
        # на IDNA-кодировании раньше системного резолва (T03-review.md round 1,
        # WARNING 2) — трактуем так же, как "не резолвится".
        return None
    for info in infos:
        ip_str = str(info[4][0])
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_global_ip(ip):
            return ip_str
    return None


async def resolve_public_ip_async(host: str) -> str | None:
    """То же, что resolve_public_ip, но getaddrinfo (блокирующий) выполняется в
    asyncio.to_thread, чтобы не блокировать event loop."""
    return await asyncio.to_thread(resolve_public_ip, host)


def _check_url_shape(url: str) -> tuple[str | None, str | None]:
    """Разбирает URL и проверяет схему/userinfo/hostname без обращения к DNS.

    Возвращает (reason, host): reason is None, если форма URL допустима, иначе
    строка с причиной отказа (host в этом случае не имеет значения).
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:  # pragma: no cover - urlparse редко падает
        return f"invalid URL: {exc}", None
    if parsed.scheme not in {"http", "https"}:
        return f"URL scheme '{parsed.scheme}' is not allowed; use http(s)://", None
    if parsed.username is not None or parsed.password is not None:
        return "URL must not contain userinfo (user:pass@host)", None
    host = parsed.hostname
    if not host:
        return "URL has no hostname", None
    return None, host


def _all_global(infos) -> bool:
    """True, если среди распознанных IP из getaddrinfo(...) нет ни одного
    не-глобального (private/loopback/link-local/CGNAT/...). Записи, не разобравшиеся
    как IP, пропускаются (как и раньше в SkillsPlugin._validate_external_url)."""
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if not _is_global_ip(ip):
            return False
    return True


def validate_public_url(url: str, *, allow_private: bool = False) -> str | None:
    """Возвращает None, если URL безопасен, иначе строку с причиной отказа.

    Проверяет: схема http/https; отсутствие userinfo (user:pass@host); hostname
    присутствует; ВСЕ IP из getaddrinfo(host) — is_global (в отличие от
    resolve_public_ip: если среди ответов есть хоть один не-глобальный IP, весь URL
    отклоняется). allow_private=True пропускает только эту последнюю проверку
    приватности IP — схему и userinfo проверяет всегда; используется MCP-плагином
    для локальных серверов администратора.
    """
    reason, host = _check_url_shape(url)
    if reason is not None:
        return reason
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError) as exc:
        # UnicodeError/ValueError: см. resolve_public_ip выше (T03-review.md round 1,
        # WARNING 2) — трактуем как ошибку резолва, а не даём вылететь наружу.
        return f"DNS resolution failed for {host}: {exc}"
    if not allow_private and not _all_global(infos):
        return f"URL host {host} resolves to a non-public address"
    return None


async def validate_public_url_async(url: str, *, allow_private: bool = False) -> str | None:
    """То же, что validate_public_url, но DNS-резолв идёт через asyncio.to_thread —
    для вызовов из async-кода (safe_request), чтобы не блокировать event loop
    блокирующим getaddrinfo."""
    reason, host = _check_url_shape(url)
    if reason is not None:
        return reason
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except (socket.gaierror, UnicodeError, ValueError) as exc:
        # UnicodeError/ValueError: см. resolve_public_ip выше (T03-review.md round 1,
        # WARNING 2) — трактуем как ошибку резолва, а не даём вылететь наружу.
        return f"DNS resolution failed for {host}: {exc}"
    if not allow_private and not _all_global(infos):
        return f"URL host {host} resolves to a non-public address"
    return None


class _ByteCappedResponse:
    """Оборачивает сырой http.client-подобный response (то, что отдаёт
    urllib.request.OpenerDirector.open) и обрывает чтение ResponseTooLargeError, как
    только суммарно прочитано больше max_bytes — не дожидаясь EOF, т.е. потоково, а
    не пост-фактум после того, как всё уже скачано."""

    def __init__(self, raw, max_bytes: int):
        self._raw = raw
        self._max_bytes = max_bytes
        self._read_bytes = 0

    def read(self, amt=None):
        if amt is None:
            chunks = []
            while True:
                chunk = self._raw.read(_CHUNK)
                if not chunk:
                    break
                chunks.append(chunk)
                self._read_bytes += len(chunk)
                if self._read_bytes > self._max_bytes:
                    raise ResponseTooLargeError(f"response exceeded {self._max_bytes} bytes")
            return b"".join(chunks)
        data = self._raw.read(amt)
        if data:
            self._read_bytes += len(data)
            if self._read_bytes > self._max_bytes:
                raise ResponseTooLargeError(f"response exceeded {self._max_bytes} bytes")
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        self._raw.close()


def safe_urlopen(
    url_or_request: str | urllib.request.Request,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
):
    """Безопасная замена urllib.request.urlopen: закрывает две дыры.

    1) urlopen по умолчанию слепо следует за редиректами -> атакующий отдаёт 302 на
       http://169.254.169.254/. Свой HTTPRedirectHandler пере-валидирует каждый
       новый URL (validate_public_url).
    2) DNS rebinding: между проверкой URL и реальным connect стандартный
       HTTPSConnection делает ещё один getaddrinfo, который атакующий может
       отравить через short-TTL DNS. Пин IP перед connect устраняет это TOCTOU-окно
       (Time-Of-Check-Time-Of-Use — проверили одно, подключились уже к другому).

    Исходный URL проверяется до открытия соединения — при отказе поднимается
    UnsafeURLError. Ошибки на редиректе/резолве конкретного хоста внутри
    opener'а остаются urllib.error.HTTPError/URLError как в оригинальной
    реализации (часть контракта urllib HTTPRedirectHandler).
    """
    url = (
        url_or_request.full_url
        if isinstance(url_or_request, urllib.request.Request)
        else url_or_request
    )
    validation_error = validate_public_url(url)
    if validation_error is not None:
        raise UnsafeURLError(validation_error)

    class _PinnedHTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

    class _PinnedHTTPConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)

    def _build_conn(scheme, host, **kw):
        # urllib передаёт host вида "example.com:8443" или "[::1]:443" — getaddrinfo
        # не принимает host:port, поэтому вытаскиваем чистый hostname для resolve.
        hostname = urllib.parse.urlparse(f"//{host}").hostname or host
        ip = resolve_public_ip(hostname)
        if ip is None:
            raise urllib.error.URLError(f"refused: {hostname} has no public IP")
        cls = _PinnedHTTPSConnection if scheme == "https" else _PinnedHTTPConnection
        conn = cls(host, **kw)
        conn._pinned_ip = ip
        return conn

    class _PinnedHTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(lambda h, **kw: _build_conn("http", h, **kw), req)

    class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(
                lambda h, **kw: _build_conn("https", h, **kw),
                req,
                context=self._context,
                check_hostname=self._check_hostname,
            )

    class _Validating(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            err = validate_public_url(newurl)
            if err is not None:
                raise urllib.error.HTTPError(
                    newurl, code, f"redirect refused: {err}", headers, fp,
                )
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    # Why: max_redirections/max_repeats are read from the HTTPRedirectHandler
    # instance itself in http_error_302 (self.max_redirections), not from the
    # OpenerDirector, so the limit has to be set on the handler.
    validating_handler = _Validating()
    setattr(validating_handler, "max_redirections", max_redirects)
    opener = urllib.request.build_opener(
        _PinnedHTTPHandler(), _PinnedHTTPSHandler(), validating_handler
    )
    raw = opener.open(url_or_request, timeout=timeout)
    return _ByteCappedResponse(raw, max_bytes)


@dataclass(frozen=True)
class SafeResponse:
    """Результат safe_request/safe_get — минимальный аналог httpx.Response, только
    с уже вычитанным (и ограниченным по размеру) телом."""

    status_code: int
    headers: httpx.Headers
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise SafeHTTPStatusError(self.status_code, self.text)


async def safe_request(
    method: str,
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    headers: dict | None = None,
    json: Any = None,
    allow_private: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SafeResponse:
    """Безопасная замена httpx.AsyncClient().request(...): перед КАЖДЫМ запросом
    (первым и каждой целью редиректа) проверяет URL через validate_public_url_async
    и отклоняет UnsafeURLError, если он небезопасен. Редиректы обрабатываются вручную
    (follow_redirects=False), а не через httpx — иначе httpx подключился бы к цели
    редиректа до какой-либо проверки. Тело читается потоково (aiter_bytes) с обрывом
    ResponseTooLargeError, как только пройден max_bytes — не через response.content,
    который вычитывает всё тело до какой-либо проверки размера.

    Известное упрощение: метод и JSON-тело при редиректе не меняются (не
    реализуем понижение POST->GET для 301/302, как это делают браузеры/requests) —
    в вызовах этого проекта редиректы на POST не ожидаются.

    transport=... — только для тестов (httpx.MockTransport), в проде не передаётся.
    """
    current_url = url
    redirects_left = max_redirects
    async with httpx.AsyncClient(transport=transport, follow_redirects=False, timeout=timeout) as client:
        while True:
            validation_error = await validate_public_url_async(current_url, allow_private=allow_private)
            if validation_error is not None:
                raise UnsafeURLError(validation_error)

            content = bytearray()
            async with client.stream(method, current_url, headers=headers, json=json) as response:
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > max_bytes:
                        raise ResponseTooLargeError(f"response exceeded {max_bytes} bytes")
                status_code = response.status_code
                response_headers = response.headers
                location = response.headers.get("location")

            if status_code in _REDIRECT_STATUS_CODES and location:
                if redirects_left <= 0:
                    raise UnsafeURLError(f"too many redirects (limit {max_redirects})")
                redirects_left -= 1
                current_url = str(httpx.URL(current_url).join(location))
                continue

            return SafeResponse(
                status_code=status_code,
                headers=response_headers,
                content=bytes(content),
            )


async def safe_get(url: str, **kwargs: Any) -> SafeResponse:
    return await safe_request("GET", url, **kwargs)
