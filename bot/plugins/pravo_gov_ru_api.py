from __future__ import annotations

import json
from datetime import datetime
from typing import Any, ClassVar, Dict, List

import httpx

from .plugin import Plugin

INTERNAL_ARGUMENTS = {"chat_id", "user_id", "message_id", "request_context"}
KNOWN_ARGUMENTS = {
    "query",
    "number",
    "document_type",
    "signatory_authority",
    "block",
    "category",
    "eo_number",
    "page_size",
    "page",
    "extra_params",
}


class PravoGovRuAPIPlugin(Plugin):
    """
    Search official Russian legal acts via publication.pravo.gov.ru API.
    """

    plugin_id = "pravo_gov_ru_api"
    function_prefix = "pravo_gov_ru_api"

    BASE_URL = "http://publication.pravo.gov.ru/api"
    DOCUMENT_URL = "http://publication.pravo.gov.ru/document/{eo_number}"
    _reference_cache: ClassVar[dict[str, list[dict[str, Any]]]] = {}

    def get_source_name(self) -> str:
        return "publication.pravo.gov.ru"

    def get_spec(self) -> List[Dict]:
        return [
            {
                "name": "search_documents",
                "description": (
                    "Query the official Russian publication.pravo.gov.ru API for laws, decrees, orders, "
                    "resolutions, and other normative legal acts, returning titles, dates, and official "
                    "links — or fetch one document's metadata directly via eo_number. Call when answering "
                    "Russian legal questions that require citing the official publication source."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text search over document name/complex name.",
                        },
                        "number": {
                            "type": "string",
                            "description": "Official document number.",
                        },
                        "document_type": {
                            "type": "string",
                            "description": "Document type by name, e.g. 'Федеральный закон'.",
                        },
                        "signatory_authority": {
                            "type": "string",
                            "description": "Issuing/signatory authority by name, e.g. 'Президент Российской Федерации'.",
                        },
                        "block": {
                            "type": "string",
                            "description": "Public block code or name. If a name is provided, the plugin tries to resolve it.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Document category name; resolved through the Categories reference.",
                        },
                        "eo_number": {
                            "type": "string",
                            "description": "Electronic publication number. If provided, fetches that document instead of searching.",
                        },
                        "page_size": {
                            "type": "integer",
                            "description": "Search page size.",
                            "enum": [10, 30, 100, 200],
                            "default": 10,
                        },
                        "page": {
                            "type": "integer",
                            "description": "Search result page index.",
                            "minimum": 1,
                            "default": 1,
                        },
                        "extra_params": {
                            "type": "object",
                            "description": (
                                "Optional raw API parameters from the pravo.gov.ru documentation, "
                                "for example PublishDateFrom or PublishDateTo."
                            ),
                        },
                    },
                    "additionalProperties": {
                        "type": ["string", "number", "integer", "boolean", "null"]
                    },
                    "required": [],
                },
            }
        ]

    async def execute(self, function_name: str, helper: Any, **kwargs: Any) -> Dict:
        if function_name != "search_documents":
            return {"error": f"Unsupported function: {function_name}"}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                eo_number = self._clean(kwargs.get("eo_number"))
                if eo_number:
                    return await self._get_document(client, eo_number)

                params = await self._build_search_params(client, kwargs)
                data = await self._get_json(client, "/Documents", params=params)
        except json.JSONDecodeError as e:
            return {"error": f"pravo.gov.ru response JSON parse error: {e}"}
        except ValueError as e:
            return {"error": str(e)}
        except httpx.TimeoutException as e:
            return {"error": f"pravo.gov.ru request timed out: {e}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"pravo.gov.ru API returned HTTP {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"pravo.gov.ru request failed: {e}"}

        items = self._extract_items(data)
        query = self._clean(kwargs.get("query"))
        if not items:
            return {
                "source": self.get_source_name(),
                "as_of": self._current_date(),
                "result": [],
                "message": f"По запросу «{query}» документы не найдены.",
            }

        return {
            "source": self.get_source_name(),
            "as_of": self._current_date(),
            "result": [self._format_search_item(item) for item in items],
            "search_params": params,
        }

    async def _get_document(self, client: httpx.AsyncClient, eo_number: str) -> Dict:
        data = await self._get_json(client, "/Document", params={"eoNumber": eo_number})
        return {
            "source": self.get_source_name(),
            "as_of": self._current_date(),
            "result": self._format_document(data, eo_number),
        }

    async def _build_search_params(
        self,
        client: httpx.AsyncClient,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        page_size = self._allowed_page_size(kwargs.get("page_size"))
        page = self._bounded_int(kwargs.get("page"), default=1, minimum=1, maximum=1000)

        params: dict[str, Any] = {
            "PageSize": page_size,
            "Index": page,
            "SortedBy": 4,
            "SortDestination": "desc",
        }

        query = self._clean(kwargs.get("query"))
        if query:
            params["Name"] = query

        number = self._clean(kwargs.get("number"))
        if number:
            params["Number"] = number
            params["NumberSearchType"] = 0

        block = self._clean(kwargs.get("block"))
        if block:
            block = await self._resolve_block(client, block)
            params["Block"] = block

        document_type = self._clean(kwargs.get("document_type"))
        if document_type:
            document_type_id = await self._resolve_reference_id(
                client,
                "/DocumentTypes",
                document_type,
                self._reference_params(block),
            )
            if not document_type_id:
                raise ValueError(f"Вид документа «{document_type}» не найден в справочнике.")
            params["DocumentTypeId"] = document_type_id

        signatory_authority = self._clean(kwargs.get("signatory_authority"))
        if signatory_authority:
            authority_id = await self._resolve_reference_id(
                client,
                "/SignatoryAuthorities",
                signatory_authority,
                self._reference_params(block),
            )
            if not authority_id:
                raise ValueError(f"Принявший орган «{signatory_authority}» не найден в справочнике.")
            params["SignatoryAuthorityId"] = authority_id

        category = self._clean(kwargs.get("category"))
        if category:
            category_id = await self._resolve_reference_id(
                client,
                "/Categories",
                category,
                self._reference_params(block),
            )
            if not category_id:
                raise ValueError(f"Категория «{category}» не найдена в справочнике.")
            params["CategoryId"] = category_id

        self._add_extra_params(params, kwargs.get("extra_params"))
        self._add_extra_params(
            params,
            {
                key: value
                for key, value in kwargs.items()
                if key not in KNOWN_ARGUMENTS and key not in INTERNAL_ARGUMENTS
            },
        )
        return params

    async def _resolve_block(self, client: httpx.AsyncClient, block: str) -> str:
        blocks = await self._fetch_reference(client, "/PublicBlocks")
        if not blocks:
            return block
        return self._find_reference_value(
            block,
            blocks,
            value_keys=("code", "id", "value"),
        ) or block

    async def _resolve_reference_id(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        name: str,
        params: dict[str, Any] | None = None,
    ) -> str | None:
        items = await self._fetch_reference(client, endpoint, params=params)
        return self._find_reference_value(name, items, value_keys=("id", "guid", "code"))

    async def _fetch_reference(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        params = params or {}
        cache_key = f"{endpoint}:{json.dumps(params, ensure_ascii=False, sort_keys=True)}"
        if cache_key in self._reference_cache:
            return self._reference_cache[cache_key]

        data = await self._get_json(client, endpoint, params=params)
        items = self._extract_items(data)
        self._reference_cache[cache_key] = items
        return items

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        response = await client.get(f"{self.BASE_URL}{endpoint}", params=params or {})
        response.raise_for_status()
        return response.json()

    def _format_search_item(self, item: dict[str, Any]) -> dict[str, Any]:
        eo_number = self._clean(self._first(item, "eoNumber", "eo_number"))
        return {
            "title": self._first(item, "complexName", "title", "name"),
            "eo_number": eo_number,
            "number": self._first(item, "number", "documentNumber", "docNumber"),
            "document_type": self._first(item, "documentTypeName", "documentType", "typeName"),
            "signatory_authority": self._first(
                item,
                "signatoryAuthorityName",
                "signatoryAuthority",
                "authorityName",
            ),
            "publish_date": self._first(item, "publishDateShort", "publishDate"),
            "link": self.DOCUMENT_URL.format(eo_number=eo_number) if eo_number else None,
        }

    def _format_document(self, data: Any, requested_eo_number: str) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {
                "eo_number": requested_eo_number,
                "link": self.DOCUMENT_URL.format(eo_number=requested_eo_number),
                "raw": data,
            }

        eo_number = self._clean(self._first(data, "eoNumber", "eo_number")) or requested_eo_number
        document = {
            "title": self._first(data, "complexName", "title", "name"),
            "eo_number": eo_number,
            "number": self._first(data, "number", "documentNumber", "docNumber"),
            "publish_date": self._first(data, "publishDateShort", "publishDate"),
            "sign_date": self._first(data, "signDateShort", "signDate"),
            "document_type": self._first(data, "documentTypeName", "documentType", "typeName"),
            "signatory_authority": self._first(
                data,
                "signatoryAuthorityName",
                "signatoryAuthority",
                "authorityName",
            ),
            "link": self.DOCUMENT_URL.format(eo_number=eo_number),
        }
        for text_key in ("text", "content", "body", "html", "documentText"):
            if data.get(text_key):
                document[text_key] = data[text_key]
                break
        return document

    def _find_reference_value(
        self,
        name: str,
        items: list[dict[str, Any]],
        value_keys: tuple[str, ...],
    ) -> str | None:
        needle = name.lower().strip()
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in value_keys:
                value = self._clean(item.get(key))
                if value and value.lower() == needle:
                    return value

            labels = (
                self._clean(item.get(key))
                for key in ("name", "title", "caption", "complexName", "shortName", "code")
            )
            if any(needle in label.lower() for label in labels if label):
                for key in value_keys:
                    value = self._clean(item.get(key))
                    if value:
                        return value
        return None

    @staticmethod
    def _extract_items(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("items", "data", "results"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _reference_params(block: str | None) -> dict[str, Any]:
        return {"block": block} if block else {}

    @staticmethod
    def _add_extra_params(params: dict[str, Any], extra_params: Any) -> None:
        if extra_params is None:
            return
        if not isinstance(extra_params, dict):
            raise ValueError("extra_params must be an object.")
        for key, value in extra_params.items():
            if key and key not in params and value is not None:
                params[str(key)] = value

    @staticmethod
    def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return min(max(parsed, minimum), maximum)

    @staticmethod
    def _allowed_page_size(value: Any) -> int:
        allowed = (10, 30, 100, 200)
        if value is None:
            return 10
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 10
        return parsed if parsed in allowed else 10

    @staticmethod
    def _first(data: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _current_date() -> str:
        return datetime.now().strftime("%d.%m.%Y")
