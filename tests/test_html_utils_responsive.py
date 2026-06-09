from pathlib import Path

from bs4 import BeautifulSoup

from bot.html_utils import HTMLVisualizer


def _style_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return "\n".join(style.get_text() for style in soup.find_all("style"))


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
