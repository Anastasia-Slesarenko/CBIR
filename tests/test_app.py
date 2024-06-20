import os
from fastapi.testclient import TestClient
from lib.db import Storage


def test_table_exist(mock_storage: Storage):
    assert mock_storage.check_table_exist() is True, "Table not exist"


def test_table_not_empty(mock_storage: Storage):
    pg_rows = mock_storage.count_rows()
    assert pg_rows != 0, "Table is empty"


def test_main_page(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_images_by_image_invalid_format(client: TestClient):
    file_path = "test.txt"  # invalid format file
    with open(file_path, "w") as f:
        f.write("This is a test file with invalid format")

    with open(file_path, "rb") as f:
        response = client.post(
            "/find_similar_images_by_image", files={"image": f}
        )

    assert response.status_code == 400, response.status_code
    assert "Неправильный формат картинки" in response.text
    os.remove(file_path)


def test_images_by_image_valid_format(client: TestClient):
    with open("./tests/query_image_test.jpg", "rb") as f:
        response = client.post(
            "/find_similar_images_by_image", files={"image": f}
        )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_urls_by_url_invalid_format(client: TestClient):
    image_url = "vhbfhbvhfbvhbfhvbhfbvh"
    response = client.get(
        "/find_similar_image_urls_by_url",
        params={"image_url": image_url},
    )
    assert response.status_code == 400


def test_urls_by_url_valid_format(client: TestClient):
    image_url = (
        "https://www.tennisplaza.com/"
        "prodimages/alt_images/large/FK0761_4.jpg"
    )
    response = client.get(
        "/find_similar_image_urls_by_url",
        params={"image_url": image_url},
    )
    assert response.status_code == 200
    assert len(response.json()) == 8


def test_image_urls_by_image_valid_format(client: TestClient):
    with open("./tests/query_image_test.jpg", "rb") as f:
        response = client.post(
            "/find_similar_image_urls_by_image", files={"image": f}
        )
    assert response.status_code == 200
    assert len(response.json()) == 8
