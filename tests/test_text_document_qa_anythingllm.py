import json

import httpx
import pytest

from bot.plugins.text_document_qa import TextDocumentQAPlugin


def _plugin(tmp_path, transport):
    plugin = TextDocumentQAPlugin(http_transport=transport)
    metadata_dir = tmp_path / "document_metadata"
    metadata_dir.mkdir()
    plugin.metadata_dir = str(metadata_dir)
    plugin.workspace_map_path = str(tmp_path / "anythingllm_workspaces.json")
    return plugin


def _set_anythingllm_env(monkeypatch):
    monkeypatch.setenv("ANYTHINGLLM_BASE_URL", "http://anything.test/api")
    monkeypatch.setenv("ANYTHINGLLM_API_KEY", "secret")


def _direct_value(result):
    return result["direct_result"]["value"]


@pytest.mark.asyncio
async def test_upload_document_creates_per_chat_workspace_and_metadata(tmp_path, monkeypatch):
    _set_anythingllm_env(monkeypatch)
    requests = []

    async def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer secret"

        if request.method == "GET" and request.url.path == "/api/v1/workspaces":
            return httpx.Response(200, json={"workspaces": []})

        if request.method == "POST" and request.url.path == "/api/v1/workspace/new":
            payload = json.loads(request.content)
            assert payload["name"].startswith("telegram-chat-")
            assert payload["chatMode"] == "query"
            assert payload["vectorSearchMode"] == "rerank"
            assert payload["topN"] == 6
            assert payload["similarityThreshold"] == 0.25
            return httpx.Response(200, json={"workspace": {"slug": "telegram-chat-abc"}})

        if request.method == "POST" and request.url.path == "/api/v1/document/upload":
            body = request.content.decode("utf-8", errors="ignore")
            assert 'name="addToWorkspaces"' in body
            assert "telegram-chat-abc" in body
            assert 'filename="notes.txt"' in body
            return httpx.Response(200, json={
                "success": True,
                "documents": [{
                    "location": "custom-documents/notes.txt-hash.json",
                    "title": "notes.txt",
                    "wordCount": 12,
                    "token_count_estimate": 18,
                }],
            })

        return httpx.Response(404, json={"error": "unexpected"})

    plugin = _plugin(tmp_path, httpx.MockTransport(handler))
    await plugin._process_document(b"hello", "notes.txt", "temp", "123", update=None)

    workspace_map = json.loads((tmp_path / "anythingllm_workspaces.json").read_text())
    assert workspace_map == {"123": "telegram-chat-abc"}

    metadata_files = list((tmp_path / "document_metadata").glob("*.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["backend"] == "anythingllm"
    assert metadata["owner_chat_id"] == "123"
    assert metadata["workspace_slug"] == "telegram-chat-abc"
    assert metadata["anythingllm_location"] == "custom-documents/notes.txt-hash.json"
    assert metadata["summary"] == "12 words, 18 estimated tokens"

    assert [request.url.path for request in requests] == [
        "/api/v1/workspaces",
        "/api/v1/workspace/new",
        "/api/v1/document/upload",
    ]


@pytest.mark.asyncio
async def test_ask_question_uses_anythingllm_workspace_chat(tmp_path, monkeypatch):
    _set_anythingllm_env(monkeypatch)
    plugin = _plugin(tmp_path, httpx.MockTransport(lambda request: httpx.Response(404)))

    doc_id = plugin._document_id("123", "custom-documents/notes.txt-hash.json")
    metadata_path = tmp_path / "document_metadata" / f"{doc_id}.json"
    metadata_path.write_text(json.dumps({
        "backend": "anythingllm",
        "doc_id": doc_id,
        "owner_chat_id": "123",
        "workspace_slug": "telegram-chat-abc",
        "anythingllm_location": "custom-documents/notes.txt-hash.json",
        "file_name": "notes.txt",
        "created_at": 100,
        "last_accessed": 100,
    }))

    async def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/workspace/telegram-chat-abc/chat"
        payload = json.loads(request.content)
        assert payload["message"] == "What is inside?"
        assert payload["mode"] == "query"
        assert payload["sessionId"].startswith("telegram-chat-")
        return httpx.Response(200, json={"textResponse": "Answer from AnythingLLM"})

    plugin._http_transport = httpx.MockTransport(handler)
    result = await plugin.execute(
        "ask_question",
        helper=None,
        chat_id="123",
        document_id=doc_id,
        query=f"{doc_id} What is inside?",
    )

    assert _direct_value(result) == "Answer from AnythingLLM"
    updated_metadata = json.loads(metadata_path.read_text())
    assert updated_metadata["last_accessed"] > 100


@pytest.mark.asyncio
async def test_delete_document_detaches_document_from_workspace(tmp_path, monkeypatch):
    _set_anythingllm_env(monkeypatch)
    plugin = _plugin(tmp_path, httpx.MockTransport(lambda request: httpx.Response(404)))

    location = "custom-documents/notes.txt-hash.json"
    doc_id = plugin._document_id("123", location)
    metadata_path = tmp_path / "document_metadata" / f"{doc_id}.json"
    metadata_path.write_text(json.dumps({
        "backend": "anythingllm",
        "doc_id": doc_id,
        "owner_chat_id": "123",
        "workspace_slug": "telegram-chat-abc",
        "anythingllm_location": location,
        "file_name": "notes.txt",
        "created_at": 100,
        "last_accessed": 100,
    }))
    requests = []

    async def handler(request):
        requests.append(request)
        if request.method == "POST" and request.url.path == "/api/v1/workspace/telegram-chat-abc/update-embeddings":
            assert json.loads(request.content) == {"adds": [], "deletes": [location]}
            return httpx.Response(200, json={"workspace": {"slug": "telegram-chat-abc"}})
        if request.method == "DELETE" and request.url.path == "/api/v1/system/remove-documents":
            assert json.loads(request.content) == {"names": [location]}
            return httpx.Response(200, json={"success": True})
        return httpx.Response(404, json={"error": "unexpected"})

    plugin._http_transport = httpx.MockTransport(handler)
    result = await plugin.execute("delete_document", helper=None, chat_id="123", document_id=doc_id)

    assert "Document deleted successfully" in _direct_value(result)
    assert not metadata_path.exists()
    assert [request.url.path for request in requests] == [
        "/api/v1/workspace/telegram-chat-abc/update-embeddings",
        "/api/v1/system/remove-documents",
    ]


@pytest.mark.asyncio
async def test_ensure_workspace_reuses_cached_slug(tmp_path, monkeypatch):
    _set_anythingllm_env(monkeypatch)
    (tmp_path / "anythingllm_workspaces.json").write_text(json.dumps({"123": "cached-slug"}))
    requests = []

    async def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/v1/workspace/cached-slug"
        return httpx.Response(200, json={"workspace": [{"slug": "cached-slug"}]})

    plugin = _plugin(tmp_path, httpx.MockTransport(handler))
    slug = await plugin._ensure_workspace("123")

    assert slug == "cached-slug"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_missing_anythingllm_configuration_fails_before_upload(tmp_path, monkeypatch):
    monkeypatch.delenv("ANYTHINGLLM_BASE_URL", raising=False)
    monkeypatch.delenv("ANYTHINGLLM_API_KEY", raising=False)

    plugin = _plugin(tmp_path, httpx.MockTransport(lambda request: httpx.Response(500)))
    result = await plugin.execute(
        "upload_document",
        helper=None,
        chat_id="123",
        file_name="notes.txt",
        file_content="hello",
    )

    assert result["error"] == "Missing AnythingLLM configuration: ANYTHINGLLM_BASE_URL, ANYTHINGLLM_API_KEY"
