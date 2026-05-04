import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _load_ask_your_pdf_module(monkeypatch, extracted_text="Extracted PDF text"):
    fake_pypdf2 = types.ModuleType("PyPDF2")

    class FakePage:
        def extract_text(self):
            return extracted_text

    class FakePdfReader:
        def __init__(self, file_obj):
            self.file_obj = file_obj
            self.pages = [FakePage()]

    fake_pypdf2.PdfReader = FakePdfReader

    fake_textract = types.ModuleType("textract")
    fake_textract.process = lambda file_path: extracted_text.encode("utf-8")

    monkeypatch.setitem(sys.modules, "PyPDF2", fake_pypdf2)
    monkeypatch.setitem(sys.modules, "textract", fake_textract)

    import bot.plugins.ask_your_pdf as ask_your_pdf

    return importlib.reload(ask_your_pdf)


def _create_pdf(path):
    path.write_bytes(b"%PDF-1.4\n" + (b"x" * 4096))
    return path


def _plugin(tmp_path, module):
    plugin = object.__new__(module.AskYourPDFPlugin)
    plugin.max_cache_size_mb = 500
    plugin.max_cache_age_days = 10
    plugin.initialize(storage_root=str(tmp_path))
    return plugin


def _helper(answer="PDF analysis result"):
    return SimpleNamespace(get_chat_response=AsyncMock(return_value=(answer, 123)))


def _helper_prompt(helper):
    call = helper.get_chat_response.await_args
    if "query" in call.kwargs:
        return call.kwargs["query"]
    return call.args[1]


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="ask_your_pdf.analyze_pdf does not yet extract text or define analysis_prompt/result",
)
async def test_analyze_pdf_happy_path_extracts_text_and_returns_result(monkeypatch, tmp_path):
    module = _load_ask_your_pdf_module(monkeypatch, extracted_text="Extracted PDF text")
    plugin = _plugin(tmp_path, module)
    pdf_path = _create_pdf(tmp_path / "source.pdf")
    helper = _helper()

    result = await plugin.execute(
        "analyze_pdf",
        helper,
        file_path=str(pdf_path),
        query="What is this document about?",
    )

    assert "error" not in result
    assert result["result"] == "PDF analysis result"
    assert result["file_hash"] == plugin.generate_file_hash(str(pdf_path))
    helper.get_chat_response.assert_awaited_once()
    prompt = _helper_prompt(helper)
    assert "What is this document about?" in prompt
    assert "Extracted PDF text" in prompt
    assert "source.pdf" in prompt


@pytest.mark.asyncio
async def test_analyze_pdf_cache_hit_returns_cached_result_without_helper(monkeypatch, tmp_path):
    module = _load_ask_your_pdf_module(monkeypatch)
    plugin = _plugin(tmp_path, module)
    pdf_path = _create_pdf(tmp_path / "cached.pdf")
    helper = _helper()
    query = "Summarize cached document"
    file_hash = plugin.generate_file_hash(str(pdf_path))
    cached_result = {"result": "cached answer", "file_hash": file_hash}
    plugin.save_cache(file_hash, query, cached_result)

    result = await plugin.execute(
        "analyze_pdf",
        helper,
        file_path=str(pdf_path),
        query=query,
    )

    assert result == cached_result
    helper.get_chat_response.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_pdf_missing_file_returns_controlled_error_without_helper(monkeypatch, tmp_path):
    module = _load_ask_your_pdf_module(monkeypatch)
    plugin = _plugin(tmp_path, module)
    helper = _helper()

    result = await plugin.execute(
        "analyze_pdf",
        helper,
        file_path=str(tmp_path / "missing.pdf"),
        query="Summarize missing document",
    )

    assert result == {"error": "File not found"}
    helper.get_chat_response.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="ask_your_pdf.analyze_pdf does not yet truncate extracted text before helper calls",
)
async def test_analyze_pdf_truncates_oversized_extracted_text(monkeypatch, tmp_path):
    extracted_text = "BEGIN " + ("x" * 120000) + " SENTINEL_AFTER_LIMIT"
    module = _load_ask_your_pdf_module(monkeypatch, extracted_text=extracted_text)
    plugin = _plugin(tmp_path, module)
    pdf_path = _create_pdf(tmp_path / "large.pdf")
    helper = _helper()

    result = await plugin.execute(
        "analyze_pdf",
        helper,
        file_path=str(pdf_path),
        query="Summarize only the useful parts",
    )

    assert "error" not in result
    helper.get_chat_response.assert_awaited_once()
    prompt = _helper_prompt(helper)
    assert "BEGIN " in prompt
    assert "SENTINEL_AFTER_LIMIT" not in prompt
    assert len(prompt) < 60000


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="ask_your_pdf.upload_pdf currently accepts arbitrary existing paths",
)
async def test_upload_pdf_rejects_path_outside_plugin_storage(monkeypatch, tmp_path):
    module = _load_ask_your_pdf_module(monkeypatch)
    storage_root = tmp_path / "plugin-storage"
    plugin = _plugin(storage_root, module)
    outside_pdf = _create_pdf(tmp_path / "outside.pdf")

    result = await plugin.execute(
        "upload_pdf",
        helper=_helper(),
        file_path=str(outside_pdf),
    )

    assert "error" in result
    assert "direct_result" not in result
