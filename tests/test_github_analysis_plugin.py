import base64
import json

import pytest

import bot.plugins.github_analysis as github_analysis
from bot import net_safety
from bot.plugins.github_analysis import GitHubCodeAnalysisPlugin


@pytest.mark.asyncio
async def test_analyze_github_code_uses_safe_get(monkeypatch):
    plugin = GitHubCodeAnalysisPlugin()

    file_content = base64.b64encode(b"print('hi')").decode("ascii")
    body = json.dumps({
        "name": "main.py",
        "type": "file",
        "encoding": "base64",
        "content": file_content,
    }).encode("utf-8")

    calls = []

    async def fake_safe_get(url, *, max_bytes, timeout):
        calls.append({"url": url, "max_bytes": max_bytes, "timeout": timeout})
        return net_safety.SafeResponse(status_code=200, headers={}, content=body)

    monkeypatch.setattr(github_analysis.net_safety, "safe_get", fake_safe_get)

    async def fake_analyze(code, language, prompt):
        return f"analysis:{language}:{prompt}"

    monkeypatch.setattr(plugin, "analyze_code_with_chatgpt", fake_analyze)

    result = await plugin.analyze_github_code(owner="acme", repo="widget", path="main.py", prompt="explain")

    assert calls == [{
        "url": "https://api.github.com/repos/acme/widget/contents/main.py",
        "max_bytes": plugin.max_response_bytes,
        "timeout": 15.0,
    }]
    assert result["results"] == [{
        "file": "main.py",
        "language": "Python",
        "analysis": "analysis:Python:explain",
    }]


@pytest.mark.asyncio
async def test_analyze_github_code_reports_unsafe_url(monkeypatch):
    plugin = GitHubCodeAnalysisPlugin()

    async def fake_safe_get(url, *, max_bytes, timeout):
        raise net_safety.UnsafeURLError("refused")

    monkeypatch.setattr(github_analysis.net_safety, "safe_get", fake_safe_get)

    called = []

    async def fake_analyze(code, language, prompt):
        called.append(True)
        return "should-not-be-called"

    monkeypatch.setattr(plugin, "analyze_code_with_chatgpt", fake_analyze)

    result = await plugin.analyze_github_code(owner="acme", repo="widget", path="main.py", prompt="explain")

    assert "error" in result
    assert "refused" in result["error"]
    assert called == []
