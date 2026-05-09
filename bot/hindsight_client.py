from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


class HindsightError(RuntimeError):
    pass


class HindsightClient:
    def __init__(
        self,
        base_url: str,
        api_token: str = "",
        *,
        namespace: str = "default",
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.api_token = api_token
        self.namespace = (namespace or "default").strip("/") or "default"
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _bank_path(self, bank_id: str, suffix: str) -> str:
        bank = quote(str(bank_id), safe="")
        return f"/v1/{self.namespace}/banks/{bank}/{suffix.lstrip('/')}"

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not self.base_url:
            raise HindsightError("HINDSIGHT_BASE_URL is not configured.")

        url = f"{self.base_url}/{path.lstrip('/')}"
        response = await self._client.request(
            method,
            url,
            headers=self._headers(),
            json=json_payload,
            params=params,
            timeout=timeout,
        )
        if response.status_code >= 400:
            detail = response.text
            raise HindsightError(f"Hindsight request failed: {response.status_code} {detail}")
        try:
            data = response.json()
        except ValueError as exc:
            raise HindsightError("Hindsight returned a non-JSON response.") from exc
        if not isinstance(data, dict):
            raise HindsightError("Hindsight returned an unexpected response shape.")
        return data

    async def recall(
        self,
        bank_id: str,
        query: str,
        *,
        budget: str = "mid",
        max_tokens: int = 4096,
        memory_types: list[str] | None = None,
        trace: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "budget": budget,
            "max_tokens": max_tokens,
            "trace": trace,
        }
        if memory_types:
            payload["types"] = memory_types
        return await self.request(
            "POST",
            self._bank_path(bank_id, "memories/recall"),
            json_payload=payload,
            timeout=30.0,
        )

    async def retain_memories(
        self,
        bank_id: str,
        items: list[dict[str, Any]],
        *,
        async_store: bool = True,
    ) -> dict[str, Any]:
        return await self.request(
            "POST",
            self._bank_path(bank_id, "memories"),
            json_payload={"async": async_store, "items": items},
            timeout=60.0,
        )

    async def list_memories(
        self,
        bank_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        query: str | None = None,
        memory_type: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if query:
            params["q"] = query
        if memory_type:
            params["type"] = memory_type
        return await self.request(
            "GET",
            self._bank_path(bank_id, "memories/list"),
            params=params,
            timeout=30.0,
        )

    async def stats(self, bank_id: str) -> dict[str, Any]:
        return await self.request("GET", self._bank_path(bank_id, "stats"), timeout=30.0)

    async def clear_bank(self, bank_id: str) -> dict[str, Any]:
        return await self.request("POST", self._bank_path(bank_id, "clear"), timeout=60.0)


def format_recall_results(data: dict[str, Any], *, max_items: int = 8) -> str:
    results = data.get("results")
    if not isinstance(results, list):
        return ""

    lines = []
    for item in results[:max_items]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        details = []
        memory_type = item.get("type")
        if memory_type:
            details.append(f"type={memory_type}")
        when = item.get("mentioned_at") or item.get("occurred_start")
        if when:
            details.append(f"when={when}")
        context = item.get("context")
        if context:
            details.append(f"context={context}")

        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {text}{suffix}")

    return "\n".join(lines)
