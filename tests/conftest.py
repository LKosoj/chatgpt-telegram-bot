import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.database import Database  # noqa: E402
from bot.plugins.agent_tools import AgentToolsPlugin  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _close_pytest_asyncio_baseline_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    yield

    if not loop.is_closed():
        loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture()
def agent_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "agent.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        for stmt in AgentToolsPlugin().register_schema():
            conn.execute(stmt)
    yield db
    Database._reset_singleton()
