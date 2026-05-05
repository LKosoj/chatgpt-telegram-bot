from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

from .plugin import Plugin

logger = logging.getLogger(__name__)


class AnythingLLMError(RuntimeError):
    pass


class TextDocumentQAPlugin(Plugin):
    """
    Document Q&A plugin backed by per-chat AnythingLLM workspaces.
    """

    def __init__(self, http_transport: httpx.AsyncBaseTransport | None = None):
        self.metadata_dir = os.path.join(os.path.dirname(__file__), "document_metadata")
        self.workspace_map_path = os.path.join(os.path.dirname(__file__), "anythingllm_workspaces.json")
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.cleanup_task = None
        self.processing_tasks = {}
        self.config = {"enable_quoting": False}
        self.max_document_age = 30 * 24 * 60 * 60
        self.warning_before_delete = 24 * 60 * 60
        self._http_transport = http_transport

    def initialize(self, openai=None, bot=None, storage_root: str | None = None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        if storage_root:
            os.makedirs(storage_root, exist_ok=True)
            self.metadata_dir = os.path.join(storage_root, "document_metadata")
            self.workspace_map_path = os.path.join(storage_root, "anythingllm_workspaces.json")
            os.makedirs(self.metadata_dir, exist_ok=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and not self.cleanup_task:
            self.cleanup_task = loop.create_task(self._cleanup_loop())

    def get_source_name(self) -> str:
        return "TextDocumentQA"

    def get_commands(self) -> List[Dict]:
        return [
            {
                "command": "list_documents",
                "description": self.t("text_doc_command_list_description"),
                "handler": self.execute,
                "handler_kwargs": {"function_name": "list_documents"},
                "plugin_name": "text_document_qa",
                "add_to_menu": True,
            },
            {
                "command": "ask_question",
                "description": self.t("text_doc_command_ask_description"),
                "args": self.t("text_doc_args_ask"),
                "handler": self.execute,
                "handler_kwargs": {"function_name": "ask_question"},
                "plugin_name": "text_document_qa",
            },
            {
                "command": "delete_document",
                "description": self.t("text_doc_command_delete_description"),
                "args": self.t("text_doc_args_delete"),
                "handler": self.execute,
                "handler_kwargs": {"function_name": "delete_document"},
                "plugin_name": "text_document_qa",
            },
        ]

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "upload_document",
            "description": "Загрузить документ в AnythingLLM workspace текущего чата",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_content": {
                        "type": "string",
                        "description": "Содержимое текстового файла",
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Имя файла",
                    },
                },
                "required": ["file_content", "file_name"],
            },
        }, {
            "name": "list_documents",
            "description": "Показать документы AnythingLLM workspace текущего чата",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }, {
            "name": "ask_question",
            "description": "Задать вопрос по документам AnythingLLM workspace текущего чата",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID документа из списка документов",
                    },
                    "query": {
                        "type": "string",
                        "description": "Вопрос к документам",
                    },
                },
                "required": ["document_id", "query"],
            },
        }, {
            "name": "delete_document",
            "description": "Удалить документ из AnythingLLM workspace текущего чата",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID документа для удаления",
                    },
                },
                "required": ["document_id"],
            },
        }]

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        try:
            chat_id = kwargs.get("chat_id")

            if not chat_id:
                return {"error": self.t("text_doc_generic_error", error="chat_id is required")}

            if function_name == "list_documents":
                documents = await self._get_user_documents(chat_id)
                if not documents:
                    return self._direct_text(self.t("text_doc_no_documents"))

                docs_list = [self.t("text_doc_list_title")]
                for doc in documents:
                    docs_list.append(self.t("text_doc_list_item_name", file_name=doc["file_name"]))
                    docs_list.append(self.t("text_doc_list_item_id", doc_id=doc["doc_id"]))
                    docs_list.append(self.t("text_doc_list_item_created", created_at=doc["created_at"]))
                    if doc.get("summary"):
                        docs_list.append(self.t("text_doc_list_item_summary", summary=doc["summary"]))
                    docs_list.append(self.t("text_doc_list_item_commands"))
                    docs_list.append(self.t("text_doc_list_item_command_ask", doc_id=doc["doc_id"]))
                    docs_list.append(self.t("text_doc_list_item_command_delete", doc_id=doc["doc_id"]))
                docs_list.append(self.t("text_doc_list_footer"))

                return self._direct_text("\n".join(docs_list))

            if function_name == "upload_document":
                error = self._configuration_error()
                if error:
                    return {"error": error}

                file_content = kwargs.get("file_content")
                file_name = kwargs.get("file_name")
                if not file_content:
                    return {"error": self.t("text_doc_file_content_missing")}
                if not file_name:
                    return {"error": self.t("text_doc_generic_error", error="file_name is required")}

                temp_id = hashlib.md5(str(time.time()).encode()).hexdigest()
                processing_task = asyncio.create_task(
                    self._process_document(
                        file_content=file_content,
                        file_name=file_name,
                        temp_id=temp_id,
                        chat_id=chat_id,
                        update=kwargs.get("update"),
                    )
                )
                self.processing_tasks[temp_id] = processing_task

                return self._direct_text(
                    self.t("text_doc_processing_started", file_name=file_name, temp_id=temp_id)
                )

            if function_name == "ask_question":
                error = self._configuration_error()
                if error:
                    return {"error": error}

                doc_id = kwargs.get("document_id")
                query = kwargs.get("query")
                if not doc_id or not query:
                    return {"error": self.t("text_doc_generic_error", error="document_id and query are required")}

                metadata = await self._get_document_metadata(doc_id, chat_id)
                if not metadata:
                    return {"error": self.t("text_doc_not_found")}

                await self._update_last_access(doc_id)
                workspace_slug = metadata["workspace_slug"]
                response = await self._chat(workspace_slug, chat_id, self._strip_document_id(query, doc_id))
                return self._direct_text(response)

            if function_name == "delete_document":
                error = self._configuration_error()
                if error:
                    return {"error": error}

                doc_id = kwargs.get("document_id")
                if not doc_id:
                    return {"error": self.t("text_doc_generic_error", error="document_id is required")}

                metadata = await self._get_document_metadata(doc_id, chat_id)
                if not metadata:
                    return {"error": self.t("text_doc_not_found")}

                await self._delete_document(doc_id, metadata)
                return self._direct_text(self.t("text_doc_deleted"))

            return {"error": self.t("text_doc_generic_error", error=f"Unknown function {function_name}")}
        except Exception as e:
            logger.exception("Error in TextDocumentQAPlugin")
            return {"error": self.t("text_doc_generic_error", error=str(e))}

    async def _process_document(self, file_content: Any, file_name: str, temp_id: str, chat_id: str, update=None):
        try:
            workspace_slug = await self._ensure_workspace(chat_id)
            document = await self._upload_document(workspace_slug, file_name, self._coerce_file_content(file_content))
            location = document.get("location")
            if not location:
                raise AnythingLLMError("AnythingLLM upload response did not include document location")

            doc_id = self._document_id(chat_id, location)
            current_time = time.time()
            metadata = {
                "backend": "anythingllm",
                "file_name": document.get("title") or file_name,
                "created_at": current_time,
                "last_accessed": current_time,
                "doc_id": doc_id,
                "owner_chat_id": chat_id,
                "summary": self._document_summary(document),
                "workspace_slug": workspace_slug,
                "anythingllm_location": location,
                "warning_sent": False,
            }
            await self._save_document_metadata(doc_id, metadata)

            response = self._direct_text(
                self.t(
                    "text_doc_processed_message",
                    file_name=file_name,
                    summary=metadata["summary"],
                    doc_id=doc_id,
                )
            )
            await self._send_direct_result(update, response)
        except Exception as e:
            logger.exception("Error processing document with AnythingLLM")
            await self._send_direct_result(
                update,
                self._direct_text(self.t("text_doc_processing_error", file_name=file_name, error=str(e))),
            )
        finally:
            self.processing_tasks.pop(temp_id, None)

    def _configuration_error(self) -> str | None:
        missing = []
        if not self._base_url():
            missing.append("ANYTHINGLLM_BASE_URL")
        if not self._api_key():
            missing.append("ANYTHINGLLM_API_KEY")
        if missing:
            return f"Missing AnythingLLM configuration: {', '.join(missing)}"
        return None

    def _base_url(self) -> str:
        return os.getenv("ANYTHINGLLM_BASE_URL", "").rstrip("/")

    def _api_key(self) -> str:
        return os.getenv("ANYTHINGLLM_API_KEY", "")

    def _timeout(self) -> float:
        return float(os.getenv("ANYTHINGLLM_TIMEOUT", "120"))

    def _chat_mode(self) -> str:
        return os.getenv("ANYTHINGLLM_CHAT_MODE", "query")

    def _top_n(self) -> int:
        return int(os.getenv("ANYTHINGLLM_TOP_N", "6"))

    def _similarity_threshold(self) -> float:
        return float(os.getenv("ANYTHINGLLM_SIMILARITY_THRESHOLD", "0.25"))

    def _vector_search_mode(self) -> str:
        return os.getenv("ANYTHINGLLM_VECTOR_SEARCH_MODE", "rerank")

    def _workspace_prefix(self) -> str:
        return os.getenv("ANYTHINGLLM_WORKSPACE_PREFIX", "telegram-chat")

    def _workspace_name(self, chat_id: str) -> str:
        return f"{self._workspace_prefix()}-{self._chat_hash(chat_id)}"

    def _session_id(self, chat_id: str) -> str:
        return f"{self._workspace_prefix()}-{self._chat_hash(chat_id)}"

    def _chat_hash(self, chat_id: str) -> str:
        return hashlib.sha256(str(chat_id).encode()).hexdigest()[:16]

    def _document_id(self, chat_id: str, location: str) -> str:
        return hashlib.md5(f"{chat_id}:{location}".encode()).hexdigest()

    async def _request(self, method: str, path: str, **kwargs) -> Dict:
        error = self._configuration_error()
        if error:
            raise AnythingLLMError(error)

        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Authorization"] = f"Bearer {self._api_key()}"
        async with httpx.AsyncClient(
            base_url=self._base_url(),
            timeout=self._timeout(),
            transport=self._http_transport,
        ) as client:
            response = await client.request(method, path, headers=headers, **kwargs)

        if response.status_code >= 400:
            raise AnythingLLMError(
                f"AnythingLLM {method} {path} failed with HTTP {response.status_code}: {response.text}"
            )

        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise AnythingLLMError(f"AnythingLLM returned non-JSON response for {method} {path}") from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise AnythingLLMError(str(payload["error"]))
        return payload

    async def _ensure_workspace(self, chat_id: str) -> str:
        workspace_map = self._load_workspace_map()
        cached_slug = workspace_map.get(str(chat_id))
        if cached_slug and await self._workspace_exists(cached_slug):
            return cached_slug

        name = self._workspace_name(chat_id)
        for workspace in await self._list_workspaces():
            if workspace.get("name") == name:
                slug = workspace["slug"]
                workspace_map[str(chat_id)] = slug
                self._save_workspace_map(workspace_map)
                return slug

        payload = {
            "name": name,
            "chatMode": self._chat_mode(),
            "vectorSearchMode": self._vector_search_mode(),
            "topN": self._top_n(),
            "similarityThreshold": self._similarity_threshold(),
        }
        response = await self._request("POST", "/v1/workspace/new", json=payload)
        workspace = response.get("workspace") or {}
        slug = workspace.get("slug")
        if not slug:
            raise AnythingLLMError("AnythingLLM workspace creation response did not include slug")
        workspace_map[str(chat_id)] = slug
        self._save_workspace_map(workspace_map)
        return slug

    async def _workspace_exists(self, slug: str) -> bool:
        try:
            await self._request("GET", f"/v1/workspace/{slug}")
            return True
        except AnythingLLMError:
            return False

    async def _list_workspaces(self) -> List[Dict]:
        response = await self._request("GET", "/v1/workspaces")
        return response.get("workspaces") or []

    async def _upload_document(self, workspace_slug: str, file_name: str, file_content: bytes) -> Dict:
        mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        files = {"file": (file_name, file_content, mime_type)}
        data = {"addToWorkspaces": workspace_slug}
        response = await self._request("POST", "/v1/document/upload", files=files, data=data)
        documents = response.get("documents") or []
        if not documents:
            raise AnythingLLMError("AnythingLLM upload response did not include documents")
        return documents[0]

    async def _chat(self, workspace_slug: str, chat_id: str, query: str) -> str:
        payload = {
            "message": query,
            "mode": self._chat_mode(),
            "sessionId": self._session_id(chat_id),
        }
        response = await self._request("POST", f"/v1/workspace/{workspace_slug}/chat", json=payload)
        if response.get("error"):
            raise AnythingLLMError(str(response["error"]))
        text_response = response.get("textResponse")
        if not text_response:
            raise AnythingLLMError("AnythingLLM chat response did not include textResponse")
        return text_response

    async def _delete_document(self, doc_id: str, metadata: Dict) -> None:
        workspace_slug = metadata.get("workspace_slug")
        location = metadata.get("anythingllm_location")
        if workspace_slug and location:
            await self._request(
                "POST",
                f"/v1/workspace/{workspace_slug}/update-embeddings",
                json={"adds": [], "deletes": [location]},
            )
            await self._request("DELETE", "/v1/system/remove-documents", json={"names": [location]})
        metadata_path = self._metadata_path(doc_id)
        if metadata_path.exists():
            metadata_path.unlink()

    async def _get_user_documents(self, chat_id: str) -> List[Dict]:
        documents = []
        for metadata_path in self._metadata_files():
            try:
                metadata = self._load_json_file(metadata_path)
            except Exception as exc:
                logger.error("Failed to read document metadata %s: %s", metadata_path, exc)
                continue

            if metadata.get("backend") != "anythingllm":
                continue
            if metadata.get("owner_chat_id") != chat_id:
                continue

            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metadata["created_at"]))
            documents.append({
                "doc_id": metadata["doc_id"],
                "file_name": metadata["file_name"],
                "created_at": created_at,
                "summary": metadata.get("summary", ""),
            })

        documents.sort(key=lambda x: x["created_at"], reverse=True)
        return documents

    async def _get_document_metadata(self, doc_id: str, chat_id: str) -> Dict | None:
        metadata_path = self._metadata_path(doc_id)
        if not metadata_path.exists():
            return None
        metadata = self._load_json_file(metadata_path)
        if metadata.get("backend") != "anythingllm":
            return None
        if metadata.get("owner_chat_id") != chat_id:
            return None
        return metadata

    async def _save_document_metadata(self, doc_id: str, metadata: Dict) -> None:
        metadata_path = self._metadata_path(doc_id)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

    async def _update_last_access(self, doc_id: str):
        try:
            metadata_path = self._metadata_path(doc_id)
            if not metadata_path.exists():
                return
            metadata = self._load_json_file(metadata_path)
            metadata["last_accessed"] = time.time()
            metadata["warning_sent"] = False
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to update document access time: %s", e)

    async def _cleanup_loop(self):
        while True:
            try:
                await self._cleanup_old_documents()
            except Exception as e:
                logger.error("Failed to cleanup old documents: %s", e)
            await asyncio.sleep(24 * 60 * 60)

    async def _cleanup_old_documents(self):
        current_time = time.time()
        deleted_count = 0
        for metadata_path in self._metadata_files():
            doc_id = metadata_path.stem
            try:
                metadata = self._load_json_file(metadata_path)
                if metadata.get("backend") != "anythingllm":
                    continue
                last_accessed = metadata.get("last_accessed", metadata["created_at"])
                time_since_last_access = current_time - last_accessed

                if (
                    time_since_last_access > (self.max_document_age - self.warning_before_delete)
                    and time_since_last_access <= self.max_document_age
                    and not metadata.get("warning_sent", False)
                ):
                    await self._send_deletion_warning(metadata, doc_id)
                elif time_since_last_access > self.max_document_age:
                    await self._delete_document(doc_id, metadata)
                    deleted_count += 1
            except Exception as e:
                logger.error("Failed to cleanup document %s: %s", doc_id, e)

        if deleted_count > 0:
            logger.info("Deleted %s expired AnythingLLM documents", deleted_count)

    async def _send_deletion_warning(self, metadata: Dict, doc_id: str):
        try:
            chat_id = metadata.get("owner_chat_id")
            file_name = metadata.get("file_name")
            warning_message = self._direct_text(
                self.t("text_doc_deletion_warning", file_name=file_name, doc_id=doc_id)
            )
            update = type("Update", (), {"effective_chat": type("Chat", (), {"id": chat_id})})()
            from ..utils import handle_direct_result
            await handle_direct_result(self.config, update, warning_message)
            metadata["warning_sent"] = True
            with self._metadata_path(doc_id).open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to send document deletion warning: %s", e)

    def _metadata_path(self, doc_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", doc_id):
            return Path(self.metadata_dir) / "__invalid__.json"
        return Path(self.metadata_dir) / f"{doc_id}.json"

    def _metadata_files(self) -> List[Path]:
        metadata_dir = Path(self.metadata_dir)
        if not metadata_dir.exists():
            return []
        return [path for path in metadata_dir.iterdir() if path.suffix == ".json"]

    def _load_workspace_map(self) -> Dict[str, str]:
        path = Path(self.workspace_map_path)
        if not path.exists():
            return {}
        try:
            data = self._load_json_file(path)
        except Exception as exc:
            logger.error("Failed to read AnythingLLM workspace map: %s", exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _save_workspace_map(self, workspace_map: Dict[str, str]) -> None:
        with Path(self.workspace_map_path).open("w", encoding="utf-8") as f:
            json.dump(workspace_map, f, ensure_ascii=False)

    def _load_json_file(self, path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _coerce_file_content(self, file_content: Any) -> bytes:
        if isinstance(file_content, bytes):
            return file_content
        if isinstance(file_content, bytearray):
            return bytes(file_content)
        if isinstance(file_content, str):
            return file_content.encode("utf-8")
        raise AnythingLLMError(f"Unsupported file_content type: {type(file_content).__name__}")

    def _document_summary(self, document: Dict) -> str:
        details = []
        if document.get("wordCount") is not None:
            details.append(f"{document['wordCount']} words")
        if document.get("token_count_estimate") is not None:
            details.append(f"{document['token_count_estimate']} estimated tokens")
        if document.get("description") and document["description"] != "Unknown":
            details.append(str(document["description"]))
        return ", ".join(details) or "Stored in AnythingLLM"

    def _strip_document_id(self, query: str, doc_id: str) -> str:
        query = query.strip()
        if query.startswith(doc_id):
            return query[len(doc_id):].strip()
        return query

    async def _send_direct_result(self, update, response: Dict) -> None:
        if update is None:
            logger.info("Document processing result: %s", response)
            return
        from ..utils import handle_direct_result
        await handle_direct_result(self.config, update, response)

    def _direct_text(self, value: str) -> Dict:
        return {
            "direct_result": {
                "kind": "text",
                "format": "markdown",
                "value": value,
            }
        }
