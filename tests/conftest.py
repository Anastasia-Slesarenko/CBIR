import asyncio
import os
import pytest
from fastapi.testclient import TestClient
from torch import load as torch_model_load
from lib.app import app
from lib.db import Storage
from lib.settings import (
    DATABASE_NAME,
    DEVICE,
    HOSTNAME,
    MODEL_FILE,
    MODEL_PATH,
    MODEL_URL,
    PASSWORD,
    PORT,
    USERNAME,
    VOLUME_DIR,
    YADISK_API_ENDPOINT,
)
from lib.utils import load_torch_model
from lib.utils import load_torch_model as download_model_from_url

if not os.path.isfile(MODEL_PATH):
    download_model_from_url(
        yadisk_model_url=MODEL_URL,
        yadisk_api_endpoint=YADISK_API_ENDPOINT,
        model_dir=VOLUME_DIR,
        file_name=MODEL_FILE,
    )
app.state.model = torch_model_load(MODEL_PATH).to(DEVICE)
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
