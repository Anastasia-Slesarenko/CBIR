import asyncio
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from torch import load as torch_model_load
from lib.app import app
from lib.db import Storage
from lib.settings import (
    DATABASE_NAME,
    DEVICE,
    HOSTNAME,
    MODEL_PATH,
    PASSWORD,
    PORT,
    USERNAME,
)


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    app.state.model = torch_model_load(MODEL_PATH).to(DEVICE)
    app.state.storage = Storage(
        host=HOSTNAME,
        user=USERNAME,
        password=PASSWORD,
        database=DATABASE_NAME,
        port=PORT,
    )
    client = TestClient(app, base_url="http://")
    yield client
    app.state.storage.disconnect()
    app.state.model.to("cpu")
    del app.state.model


@pytest.fixture(scope="session")
def event_loop(request):
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def mock_storage():
    storage = Storage(
        host=HOSTNAME,
        user=USERNAME,
        password=PASSWORD,
        database=DATABASE_NAME,
        port=PORT,
    )
    yield storage
    storage.disconnect()
