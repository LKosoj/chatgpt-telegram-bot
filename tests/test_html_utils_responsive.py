from pathlib import Path

from bs4 import BeautifulSoup

from bot.html_utils import HTMLVisualizer


def _style_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return "\n".join(style.get_text() for style in soup.find_all("style"))


def _visualizer() -> HTMLVisualizer:
    return HTMLVisualizer()


def test_plantuml_jar_resolution_finds_bundled_plugin_jar():
    visualizer = HTMLVisualizer()

    assert Path(visualizer.plantuml_jar).name == "plantuml.jar"
    assert Path(visualizer.plantuml_jar).parent.name == "plugins"
    assert Path(visualizer.plantuml_jar).exists()


def _convert(text: str) -> str:
    """Call _convert_markdown directly and return the HTML string."""
    return _visualizer()._convert_markdown(text)


def test_advanced_visualization_generates_mobile_responsive_shell(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    output_dir = tmp_path / "runtime-output"
    data_dir = tmp_path / "runtime-data"
    plots_dir = tmp_path / "runtime-plots"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("BOT_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("BOT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BOT_PLOTS_DIR", str(plots_dir))
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

    assert Path(output_path).parent == output_dir
    assert not (cwd / "output").exists()
    assert not (cwd / "data").exists()
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
    assert 'securityLevel: "strict"' in html
    assert 'securityLevel: "loose"' not in html


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


def test_fenced_code_block_script_remains_text():
    result = _convert("```html\n<script>alert(1)</script>\n```")
    soup = BeautifulSoup(result, "html.parser")

    assert soup.find("script") is None
    assert "<script>alert(1)</script>" in soup.find("code").get_text()
    assert "<script>alert(1)</script>" not in result


def test_mermaid_closing_tag_injection_is_escaped():
    result = _convert(
        "```mermaid\n"
        "graph TD\n"
        'A[ok]</div><script>alert("xss")</script>\n'
        "```"
    )
    soup = BeautifulSoup(result, "html.parser")
    mermaid = soup.find("div", class_="mermaid")

    assert soup.find("script") is None
    assert mermaid is not None
    assert '</div><script>alert("xss")</script>' in mermaid.get_text()


def test_mermaid_container_does_not_decode_encoded_script_to_tag():
    html = _visualizer()._create_mermaid_container(
        "diagram-test",
        "graph TD\nA[&lt;script&gt;alert(1)&lt;/script&gt;]",
        0,
    )
    soup = BeautifulSoup(html, "html.parser")

    assert soup.find("script") is None
    assert "<script>" not in html
    assert "&amp;lt;script" in html


def test_mermaid_container_keeps_safe_line_break_normalization():
    html = _visualizer()._create_mermaid_container(
        "diagram-test",
        "graph TD&lt;br/&gt;&lt;em&gt;A--&gt;B&lt;/em&gt;&lt;br/&gt;C&lt;--D&lt;br/&gt;E&lt;--&gt;F",
        0,
    )
    soup = BeautifulSoup(html, "html.parser")
    mermaid = soup.find("div", class_="mermaid")

    assert mermaid is not None
    assert "graph TD\nA-->B" in mermaid.get_text()
    assert "C<--D" in mermaid.get_text()
    assert "E<-->F" in mermaid.get_text()
    assert "&lt;br" not in mermaid.get_text()
    assert "&lt;em" not in mermaid.get_text()


def test_url_quote_injection_does_not_create_event_handler_attribute():
    result = _convert(
        'Report https://example.com/foo_bar</a><em>baz" onclick="alert(1)</em>'
    )
    soup = BeautifulSoup(result, "html.parser")

    assert soup.find("script") is None
    assert soup.find_all("a")
    for link in soup.find_all("a"):
        assert "onclick" not in link.attrs


def test_raw_script_html_is_removed():
    result = _convert('hello <script>alert("xss")</script> world')
    soup = BeautifulSoup(result, "html.parser")

    assert soup.find("script") is None
    assert "alert" not in soup.get_text()
    assert "hello" in soup.get_text()
    assert "world" in soup.get_text()


def test_markdown_links_with_unsafe_schemes_lose_href():
    result = _convert('[js](javascript:alert(1)) [data](data:text/html,<script>x</script>)')
    soup = BeautifulSoup(result, "html.parser")
    links = soup.find_all("a")

    assert len(links) == 2
    assert all(not link.has_attr("href") for link in links)
    assert soup.find("script") is None


def test_prebuilt_mermaid_container_is_sanitized_before_restore():
    result = _convert(
        '<div class="mermaid-container">'
        '<div class="mermaid">graph TD\\nA-->B</div>'
        '<script>alert("xss")</script>'
        '<img src="javascript:alert(1)" onerror="alert(2)">'
        '</div></div>'
    )
    soup = BeautifulSoup(result, "html.parser")

    assert soup.find("script") is None
    assert "alert" not in str(soup)
    image = soup.find("img")
    assert image is not None
    assert not image.has_attr("src")
    assert not image.has_attr("onerror")


def test_malformed_url_does_not_trigger_raw_html_fallback():
    result = _convert('<a href="http://["><script>alert("xss")</script>bad</a>')
    soup = BeautifulSoup(result, "html.parser")

    assert soup.find("script") is None
    assert "alert" not in str(soup)
    link = soup.find("a")
    assert link is not None
    assert not link.has_attr("href")


def test_form_action_with_unsafe_scheme_is_removed():
    result = _convert('<form action="javascript:alert(1)"><button>send</button></form>')
    soup = BeautifulSoup(result, "html.parser")
    form = soup.find("form")

    assert form is not None
    assert not form.has_attr("action")
