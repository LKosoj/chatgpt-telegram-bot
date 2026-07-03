import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _close_pytest_asyncio_baseline_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    yield

    if not loop.is_closed():
        loop.close()
    asyncio.set_event_loop(None)
