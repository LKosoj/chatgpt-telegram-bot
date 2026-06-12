from pathlib import Path

from bs4 import BeautifulSoup

from bot.html_utils import HTMLVisualizer


def _style_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return "\n".join(style.get_text() for style in soup.find_all("style"))


def _visualizer() -> HTMLVisualizer:
    return HTMLVisualizer()


def _convert(text: str) -> str:
    """Call _convert_markdown directly and return the HTML string."""
    return _visualizer()._convert_markdown(text)


def test_advanced_visualization_generates_mobile_responsive_shell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    visualizer = HTMLVisualizer()

    output_path = visualizer.advanced_visualization(
        "\n".join(
            [
                "# Report",
                "",
                "| Column A | Column B |",
                "| --- | --- |",
                "| value | https://example.com/really_long_path_with_underscores |",
                "",
                "```python",
                "print('wide code block')",
                "```",
            ]
        ),
        "mobiletest",
    )

    html = Path(output_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    viewport = soup.find("meta", attrs={"name": "viewport"})

    assert viewport is not None
    assert viewport["content"] == "width=device-width, initial-scale=1"

    css = _style_text(html)
    assert "@media (max-width: 640px)" in css
    assert ".markdown-body table" in css
    assert ".mermaid-controls" in css
    assert "-webkit-overflow-scrolling: touch" in css


# --- _convert_markdown code-block escaping fix (5e) ---

def test_fenced_code_block_underscores_not_escaped():
    r"""Underscores inside a fenced code block must not be escaped to \_."""
    md = "```python\nmy_var_1 = x_y\n```"
    result = _convert(md)
    assert "my_var_1" in result
    assert "x_y" in result
    # No backslash-escaped underscores inside the code block output
    assert r"my\_var\_1" not in result
    assert r"x\_y" not in result


def test_fenced_code_block_literal_newline_not_expanded():
    r"""A literal \n in a string literal inside code must not become a real newline."""
    md = "```python\nprint('a\\nb')\n```"
    result = _convert(md)
    # The rendered code element should contain the literal \n sequence, not a real newline
    soup = BeautifulSoup(result, "html.parser")
    code_text = soup.find("code").get_text()
    assert "\\n" in code_text


def test_mixed_text_and_code_code_intact_outside_text_not_mangled():
    """In mixed content: code block underscores preserved; surrounding text not mangled."""
    md = "Normal text_with_underscores.\n\n```python\nmy_var_1 = x_y\n```"
    result = _convert(md)
    soup = BeautifulSoup(result, "html.parser")
    # Code block underscores must not be escaped
    code = soup.find("code")
    assert code is not None
    assert "my_var_1" in code.get_text()
    assert r"my\_var\_1" not in code.get_text()
    # The paragraph text should not have double-escaped underscores
    assert r"text\_with\_underscores" not in result


def test_inline_code_underscores_not_escaped():
    """Underscores inside an inline `code` span must not be escaped."""
    md = "Use `inline_code` here."
    result = _convert(md)
    soup = BeautifulSoup(result, "html.parser")
    code = soup.find("code")
    assert code is not None
    assert "inline_code" in code.get_text()
    assert r"inline\_code" not in code.get_text()
