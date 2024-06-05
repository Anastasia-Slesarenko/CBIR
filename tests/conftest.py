import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from typing import AsyncGenerator
import pytest
from lib.app import app
from lib.db import Storage
from lib.utils import load_torch_model
from lib.settings import (
    HOSTNAME,
    USERNAME,
    PASSWORD,
    DATABASE_NAME,
    PORT,
    VOLUME_DIR,
    MODEL_FILE,
    MODEL_PATH,
    YADISK_API_ENDPOINT,
    MODEL_URL,
)
import os


app.state.storage = Storage(
    host=HOSTNAME,
    user=USERNAME,
    password=PASSWORD,
    database=DATABASE_NAME,
    port=PORT,
)
client = TestClient(app, base_url="http://")

if not os.path.isfile(MODEL_PATH):
    load_torch_model(
        yadisk_model_url=MODEL_URL,
        yadisk_api_endpoint=YADISK_API_ENDPOINT,
        model_dir=VOLUME_DIR,
        file_name=MODEL_FILE,
    )


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
