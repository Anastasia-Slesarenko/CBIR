import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from typing import AsyncGenerator
import subprocess
import pytest
import time
from app import app
from db import Storage
from settings import (
    HOSTNAME,
    USERNAME,
    PASSWORD,
    DATABASE_NAME,
    PORT,
)


client = TestClient(app, base_url="http://")
compose_file = "docker-compose-dev.yml"
up_cmd = ["docker-compose", "-f", compose_file, "up", "--build", "-d"]
down_cmd = [
    "docker-compose",
    "-f",
    compose_file,
    "down",
    "--remove-orphans",
]


@pytest.fixture(scope="session", autouse=True)
def docker_compose():
    """Pytest fixture to set up and tear down docker-compose services."""
    try:
        print("Starting Docker Compose services...")
        subprocess.check_call(up_cmd)
        time.sleep(5)
        print("Docker Compose services started.")
        yield
    finally:
        subprocess.check_call(down_cmd)
        print("Docker Compose services stopped.")


@pytest.fixture(scope="session")
def event_loop(request):
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def ac() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(app=app, base_url="http://") as ac:
        yield ac


@pytest.fixture()
def mock_storage():
    storage = Storage(
        host=HOSTNAME,
        user=USERNAME,
        password=PASSWORD,
        database=DATABASE_NAME,
        port=PORT,
    )
    yield storage
