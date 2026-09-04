"""Tests for ShowMeDiagramsPlugin._generate_plantuml's timeout/to_thread fix (T07).

``subprocess.run`` (launching ``java -jar plantuml.jar``) had no ``timeout`` and
was called directly inside ``async def _generate_plantuml``, blocking the event
loop on every diagram type.
"""

from types import SimpleNamespace

import pytest

import bot.plugins.show_me_diagrams as show_me_diagrams
from bot.plugins.show_me_diagrams import ShowMeDiagramsPlugin


to_thread_calls = []


async def fake_to_thread(func, *args, **kwargs):
    to_thread_calls.append(getattr(func, "__name__", repr(func)))
    return func(*args, **kwargs)


@pytest.fixture(autouse=True)
def _reset_to_thread_calls():
    to_thread_calls.clear()
    yield
    to_thread_calls.clear()


@pytest.mark.asyncio
async def test_generate_plantuml_passes_timeout_and_uses_to_thread(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        # The real subprocess call writes the PNG next to the .puml file;
        # the code checks os.path.exists(output_file) afterwards.
        puml_file = args[4]
        png_path = puml_file.rsplit('.', 1)[0] + '.png'
        with open(png_path, 'wb') as f:
            f.write(b"fake-png")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(show_me_diagrams.subprocess, "run", fake_run)
    monkeypatch.setattr(show_me_diagrams.asyncio, "to_thread", fake_to_thread)

    plugin = ShowMeDiagramsPlugin()
    puml_content, output_file = await plugin._generate_plantuml(
        "@startuml\n@enduml", helper=None, user_id=1
    )

    assert puml_content == "@startuml\n@enduml"
    assert output_file.endswith(".png")
    assert len(calls) == 1
    assert calls[0]["kwargs"]["timeout"] == 60
    assert len(to_thread_calls) == 1
