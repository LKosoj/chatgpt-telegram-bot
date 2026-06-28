from bot.tool_result import normalize_tool_result, tool_result_content


def test_tool_result_normalizes_successful_dict_payload():
    result = normalize_tool_result({"result": "ok"}, tool_name="p.do")

    assert result.success is True
    assert result.error is None
    assert result.content == '{"result": "ok"}'


def test_tool_result_normalizes_error_payloads():
    error_result = normalize_tool_result({"error": "boom"})
    false_result = normalize_tool_result({"success": False, "message": "bad"})
    ok_false_result = normalize_tool_result({"ok": False, "code": "REMOTE_BLOCKED", "error": "blocked"})
    ok_false_code_result = normalize_tool_result({"ok": False, "code": "REMOTE_BLOCKED"})

    assert error_result.success is False
    assert error_result.error == "boom"
    assert false_result.success is False
    assert false_result.error == "bad"
    assert ok_false_result.success is False
    assert ok_false_result.error == "blocked"
    assert ok_false_code_result.success is False
    assert ok_false_code_result.error == "REMOTE_BLOCKED"


def test_tool_result_extracts_direct_result_and_artifacts():
    result = normalize_tool_result({
        "direct_result": {
            "kind": "final",
            "format": "mixed",
            "text": "done",
            "artifacts": [{"kind": "file", "format": "path", "value": "/tmp/out.txt"}],
        }
    }, tool_name="agent_tools.deliver_to_user")

    assert result.direct_result["kind"] == "final"
    assert result.artifacts == ({
        "tool": "agent_tools.deliver_to_user",
        "kind": "file",
        "path": "/tmp/out.txt",
        "format": "path",
        "caption": None,
    },)


def test_tool_result_rejects_malformed_direct_result_without_kind():
    result = normalize_tool_result({
        "direct_result": {
            "format": "text",
            "value": "missing kind",
        }
    }, tool_name="p.do")

    assert result.direct_result is None


def test_tool_result_content_preserves_strings():
    assert tool_result_content("plain") == "plain"
