"""Общие тестовые заглушки, доказанно дублирующиеся в ≥3 файлах.

Смотри docs/remediation_2026-09-04/T20-test-fakes.md — там разбор, почему
именно эти классы, а не «все Fake*/Dummy*/Stub*». Одинаковое имя класса в
разных тестовых файлах в этом проекте почти всегда означает разные, специально
подогнанные под конкретный тест заглушки — не копируй сюда что-то новое, пока
не найдёшь минимум 3 файла с побайтово (или почти побайтово) одинаковым телом.
"""


class FakeEncoding:
    """Заменитель tiktoken.Encoding: encode() считает "токеном" каждый символ.

    Подключается через sys.modules["tiktoken"] только если настоящий пакет не
    установлен (см. _install_module_if_missing в каждом файле, где это
    используется). В .venv этого репозитория tiktoken установлен, поэтому
    здесь класс сейчас не активируется ни в одном тесте — он нужен для сред
    без tiktoken (например, CI-образ без сети).
    """

    def encode(self, value):
        return list(value)


class FakeMessage:
    """Заменитель response.choices[i].message из OpenAI SDK: только
    tool_calls и content — ровно то, что читает bot/openai_tool_handler.py."""

    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls
        self.content = content


class FakeChoice:
    """Заменитель response.choices[i], оборачивает FakeMessage."""

    def __init__(self, tool_calls=None, content=""):
        self.message = FakeMessage(tool_calls=tool_calls, content=content)
        self.delta = None
        self.finish_reason = None
