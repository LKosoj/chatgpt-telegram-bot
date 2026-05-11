from __future__ import annotations

import base64
import tempfile
from uuid import uuid4
from pathlib import Path
from typing import Any

import httpx

from .model_constants import (
    LLMGATEWAY_WEB_DEEP_RESEARCH_MODEL,
    LLMGATEWAY_WEB_READ_MODEL,
    LLMGATEWAY_WEB_RESEARCH_MODEL,
    LLMGATEWAY_WEB_SEARCH_MODEL,
)


class LLMGatewayError(RuntimeError):
    pass


class LLMGatewayClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def get_json(self, path: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        if not self.base_url:
            raise LLMGatewayError("OPENAI_BASE_URL is not configured for LLMGateway requests.")

        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": "tgBot",
        }
        response = await self._client.get(url, headers=headers, params=params, timeout=timeout)
        if response.status_code >= 400:
            detail = response.text
            raise LLMGatewayError(f"LLMGateway request failed: {response.status_code} {detail}")
        try:
            return response.json()
        except ValueError as exc:
            raise LLMGatewayError("LLMGateway returned a non-JSON response.") from exc

    async def post_json(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if not self.base_url:
            raise LLMGatewayError("OPENAI_BASE_URL is not configured for LLMGateway requests.")

        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "tgBot",
        }
        response = await self._client.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code >= 400:
            detail = response.text
            raise LLMGatewayError(f"LLMGateway request failed: {response.status_code} {detail}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMGatewayError("LLMGateway returned a non-JSON response.") from exc
        if not isinstance(data, dict):
            raise LLMGatewayError("LLMGateway returned an unexpected response shape.")
        return data

    async def post_multipart(
        self,
        path: str,
        *,
        data: dict[str, Any],
        files: list[tuple[str, tuple[str, bytes, str]]],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not self.base_url:
            raise LLMGatewayError("OPENAI_BASE_URL is not configured for LLMGateway requests.")

        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": "tgBot",
        }
        response = await self._client.post(url, headers=headers, data=data, files=files, timeout=timeout)
        if response.status_code >= 400:
            detail = response.text
            raise LLMGatewayError(f"LLMGateway request failed: {response.status_code} {detail}")
        try:
            response_data = response.json()
        except ValueError as exc:
            raise LLMGatewayError("LLMGateway returned a non-JSON response.") from exc
        if not isinstance(response_data, dict):
            raise LLMGatewayError("LLMGateway returned an unexpected response shape.")
        return response_data

    async def web_search(
        self,
        query: str,
        *,
        max_results: int = 5,
        language: str = "ru",
        include_images: bool = False,
    ) -> dict[str, Any]:
        return await self.post_json(
            "/web/search",
            {
                "model": LLMGATEWAY_WEB_SEARCH_MODEL,
                "query": query,
                "max_results": max_results,
                "language": language,
                "include_images": include_images,
            },
            timeout=90.0,
        )

    async def web_read(self, url: str, *, output_format: str = "markdown") -> dict[str, Any]:
        return await self.post_json(
            "/web/read",
            {
                "model": LLMGATEWAY_WEB_READ_MODEL,
                "url": url,
                "format": output_format,
            },
            timeout=120.0,
        )

    async def web_research(
        self,
        query: str,
        *,
        max_results_per_lang: int = 10,
        output_language: str = "ru",
    ) -> dict[str, Any]:
        return await self.post_json(
            "/web/research",
            {
                "model": LLMGATEWAY_WEB_RESEARCH_MODEL,
                "query": query,
                "max_results_per_lang": max_results_per_lang,
                "output_language": output_language,
                "format": "markdown",
            },
            timeout=300.0,
        )

    async def web_deep_research(
        self,
        query: str,
        *,
        max_words: int = 2500,
        breadth: int = 4,
        depth: int = 2,
        concurrency: int = 4,
        language: str = "ru",
    ) -> dict[str, Any]:
        return await self.post_json(
            "/web/deep-research",
            {
                "model": LLMGATEWAY_WEB_DEEP_RESEARCH_MODEL,
                "query": query,
                "max_words": max_words,
                "breadth": breadth,
                "depth": depth,
                "concurrency": concurrency,
                "language": language,
                "format": "markdown",
            },
            timeout=1800.0,
        )

    async def image_edit(
        self,
        prompt: str,
        images: list[str | dict[str, Any]],
        *,
        model: str,
    ) -> dict[str, Any]:
        return await self.post_json(
            "/images/edits",
            {
                "model": model,
                "prompt": prompt,
                "images": images,
            },
            timeout=300.0,
        )

    async def audio_voices(self, model: str | None = None) -> Any:
        params = {"model": model} if model else None
        return await self.get_json("/audio/voices", params=params, timeout=30.0)

    async def image_edit_file(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        model: str,
        filename: str = "source.png",
        content_type: str = "image/png",
    ) -> dict[str, Any]:
        return await self.post_multipart(
            "/images/edits",
            data={
                "model": model,
                "prompt": prompt,
            },
            files=[("image", (filename, image_bytes, content_type))],
            timeout=300.0,
        )


def _write_base64_image(encoded: str, suffix: str) -> str:
    output_dir = Path(tempfile.gettempdir()) / "llmgateway_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid4().hex}{suffix}"
    output_path.write_bytes(base64.b64decode("".join(encoded.split()), validate=True))
    return str(output_path)


def _data_image_url_to_path(value: str) -> str | None:
    data_url = value.strip()
    if not data_url.lower().startswith("data:image/"):
        return None
    header, separator, encoded = data_url.partition(",")
    if not separator or ";base64" not in header.lower():
        return None
    mime_type = header[5:].split(";", 1)[0]
    suffix = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(mime_type.lower(), ".png")
    return _write_base64_image(encoded, suffix)


def extract_image_result(response: Any) -> tuple[str, str]:
    data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
    if not data:
        raise LLMGatewayError("Image response contains no data.")

    first = data[0]
    url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
    if url:
        data_url_path = _data_image_url_to_path(str(url))
        if data_url_path:
            return data_url_path, "path"
        return str(url), "url"

    b64_json = first.get("b64_json") if isinstance(first, dict) else getattr(first, "b64_json", None)
    if not b64_json:
        raise LLMGatewayError("Image response contains neither url nor b64_json.")

    return _write_base64_image(str(b64_json), ".png"), "path"
