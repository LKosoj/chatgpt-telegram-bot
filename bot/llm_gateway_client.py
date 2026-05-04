from __future__ import annotations

import base64
import tempfile
from uuid import uuid4
from pathlib import Path
from typing import Any

import httpx

from .model_constants import (
    LLMGATEWAY_IMAGE_EDIT_MODEL,
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

    async def image_edit(self, prompt: str, images: list[str | dict[str, Any]]) -> dict[str, Any]:
        return await self.post_json(
            "/images/edits",
            {
                "model": LLMGATEWAY_IMAGE_EDIT_MODEL,
                "prompt": prompt,
                "images": images,
            },
            timeout=300.0,
        )

    async def image_edit_file(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        filename: str = "source.png",
        content_type: str = "image/png",
    ) -> dict[str, Any]:
        return await self.post_multipart(
            "/images/edits",
            data={
                "model": LLMGATEWAY_IMAGE_EDIT_MODEL,
                "prompt": prompt,
            },
            files=[("image", (filename, image_bytes, content_type))],
            timeout=300.0,
        )


def extract_image_result(response: Any) -> tuple[str, str]:
    data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
    if not data:
        raise LLMGatewayError("Image response contains no data.")

    first = data[0]
    url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
    if url:
        return str(url), "url"

    b64_json = first.get("b64_json") if isinstance(first, dict) else getattr(first, "b64_json", None)
    if not b64_json:
        raise LLMGatewayError("Image response contains neither url nor b64_json.")

    output_dir = Path(tempfile.gettempdir()) / "llmgateway_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid4().hex}.png"
    output_path.write_bytes(base64.b64decode(b64_json))
    return str(output_path), "path"
