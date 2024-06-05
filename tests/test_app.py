import pytest
from httpx import AsyncClient
import os
from bs4 import BeautifulSoup
from lib.settings import (
    IMAGE_FORMAT,
    MODEL_PATH,
    FAISS_INDEX_PATH,
)
from lib.db import Storage
from load_artifacts import prepare_search_db


@pytest.mark.order(1)
async def test_init_db(mock_storage: Storage):
    prepare_search_db(
        storage=mock_storage,
        image_path="./gallery_images_test",
        image_format=IMAGE_FORMAT,
        model_pth=MODEL_PATH,
        csv_path="./test.csv",
        faiss_index_path=FAISS_INDEX_PATH,
        device="cpu",
    )
    # test load to postgresql
    pg_rows = mock_storage.count_rows()
    assert pg_rows == 24, "No data has been added"


@pytest.mark.order(2)
async def test_main_page(ac: AsyncClient):
    response = await ac.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.order(3)
async def test_main_page(ac: AsyncClient):
    response = await ac.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.order(4)
async def test_find_simular_images_invalid_format(ac: AsyncClient):
    file_path = "test.txt"  # invalid format file
    with open(file_path, "w") as f:
        f.write("This is a test file with invalid format")

    with open(file_path, "rb") as f:
        response = await ac.post(
            "/find_simular_images", files={"image": f}
        )

    assert response.status_code == 400, response.status_code
    assert "Неправильный формат картинки" in response.text
    os.remove(file_path)


@pytest.mark.order(5)
async def test_find_simular_images_valid_format(ac: AsyncClient):
    with open("query_image_test.jpg", "rb") as f:
        response = await ac.post(
            "/find_simular_images", files={"image": f}
        )

    assert response.status_code == 200

    assert "text/html" in response.headers["content-type"]

    htmlParse = BeautifulSoup(response.text, "html.parser")
    condidat_list = []
    for para in htmlParse.find_all(
        "p", class_="fw-bold text-center text-muted"
    ):
        condidat_list.append(
            para.get_text().replace("\\", "").replace("\n", "")
        )

    assert (
        "sign_1" in condidat_list[:2]
        or "sign_2" in condidat_list[:2]
        or "sign_3" in condidat_list[:2]
    )
