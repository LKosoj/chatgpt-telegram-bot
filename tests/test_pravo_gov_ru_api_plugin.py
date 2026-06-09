import pytest

from bot.plugin_manager import PluginManager
from bot.plugins import pravo_gov_ru_api as pravo_module
from bot.plugins.pravo_gov_ru_api import PravoGovRuAPIPlugin


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


def _fake_async_client(calls, payloads):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("timeout") == 10.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None):
            params = params or {}
            calls.append((url, dict(params)))
            endpoint = url.rsplit("/api", 1)[-1]
            return FakeResponse(payloads[endpoint])

    return FakeAsyncClient


@pytest.fixture(autouse=True)
def clear_pravo_reference_cache():
    PravoGovRuAPIPlugin._reference_cache.clear()


def test_plugin_manager_exposes_namespaced_pravo_tool():
    manager = PluginManager(config={"plugins": ["pravo_gov_ru_api"]})

    specs = manager.get_functions_specs(
        helper=None,
        model_to_use="llmgateway/high",
        allowed_plugins=["pravo_gov_ru_api"],
    )

    names = [spec["function"]["name"] for spec in specs]
    assert names == ["pravo_gov_ru_api_search_documents"]
    assert manager.to_canonical_function_name("pravo_gov_ru_api_search_documents") == "pravo_gov_ru_api.search_documents"


@pytest.mark.asyncio
async def test_search_resolves_references_and_formats_results(monkeypatch):
    calls = []
    payloads = {
        "/PublicBlocks": [
            {"code": "government", "name": "Органы государственной власти"}
        ],
        "/DocumentTypes": [{"id": "doc-type-id", "name": "Федеральный закон"}],
        "/SignatoryAuthorities": [
            {"id": "authority-id", "name": "Государственная Дума"}
        ],
        "/Categories": [{"id": "category-id", "name": "Конституционный строй"}],
        "/Documents": {
            "items": [
                {
                    "complexName": "Федеральный закон о тестировании",
                    "eoNumber": "0001202605010001",
                    "publishDateShort": "01.05.2026",
                }
            ]
        },
    }
    monkeypatch.setattr(
        pravo_module.httpx,
        "AsyncClient",
        _fake_async_client(calls, payloads),
    )

    result = await PravoGovRuAPIPlugin().execute(
        "search_documents",
        helper=None,
        query="тестирование",
        document_type="Федеральный закон",
        signatory_authority="Государственная Дума",
        block="органы государственной власти",
        category="Конституционный",
        page_size=30,
        PublishDateFrom="01.01.2026",
    )

    assert result["source"] == "publication.pravo.gov.ru"
    assert result["result"] == [
        {
            "title": "Федеральный закон о тестировании",
            "eo_number": "0001202605010001",
            "number": None,
            "document_type": None,
            "signatory_authority": None,
            "publish_date": "01.05.2026",
            "link": "http://publication.pravo.gov.ru/document/0001202605010001",
        }
    ]

    documents_params = [
        params for url, params in calls if url.endswith("/Documents")
    ][0]
    assert documents_params["Name"] == "тестирование"
    assert documents_params["Block"] == "government"
    assert documents_params["DocumentTypeId"] == "doc-type-id"
    assert documents_params["SignatoryAuthorityId"] == "authority-id"
    assert documents_params["CategoryId"] == "category-id"
    assert documents_params["PublishDateFrom"] == "01.01.2026"
    assert documents_params["PageSize"] == 30


@pytest.mark.asyncio
async def test_get_document_by_eo_number(monkeypatch):
    calls = []
    payloads = {
        "/Document": {
            "complexName": "Указ Президента Российской Федерации",
            "eoNumber": "0001202605020002",
            "publishDateShort": "02.05.2026",
            "text": "Официальный текст",
        }
    }
    monkeypatch.setattr(
        pravo_module.httpx,
        "AsyncClient",
        _fake_async_client(calls, payloads),
    )

    result = await PravoGovRuAPIPlugin().execute(
        "search_documents",
        helper=None,
        eo_number="0001202605020002",
    )

    assert calls == [
        (
            "http://publication.pravo.gov.ru/api/Document",
            {"eoNumber": "0001202605020002"},
        )
    ]
    assert result["result"]["title"] == "Указ Президента Российской Федерации"
    assert result["result"]["text"] == "Официальный текст"
    assert (
        result["result"]["link"]
        == "http://publication.pravo.gov.ru/document/0001202605020002"
    )
