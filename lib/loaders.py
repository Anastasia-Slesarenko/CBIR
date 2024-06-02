from io import BytesIO
import os
from PIL import Image
from .settings import IMAGE_PATH, IMAGE_FORMAT
import requests
import torch


def get_bytes_image(
    file: int,
    image_path: str = IMAGE_PATH,
) -> BytesIO:
    """Загружает похожие изображения из хранилища и
    переводит их в байты для отображения в html"""
    img = Image.open(os.path.join(image_path, str(file) + IMAGE_FORMAT))
    with BytesIO() as output:
        img.save(output, format="PNG")
        bytes_array = output.getvalue()
    return bytes_array


def read_list_images(
    image_sources: list[int],
    image_path: str = IMAGE_PATH,
) -> list[Image.Image]:
    """Загружает список изображений из хранилища для записи их эмбеддингов"""
    images = []
    for file in image_sources:
        images.append(
            Image.open(os.path.join(image_path, str(file) + IMAGE_FORMAT))
        )
    return images


def load_torch_model(
    yadisk_model_url: str,
    model_dir: str,
    file_name: str,
) -> None:
    """Загружает торчевую модель с яндекс диска"""
    API_ENDPOINT = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}"
    responce = requests.get(API_ENDPOINT.format(yadisk_model_url))
    download_link = responce.json()["href"]
    torch.hub.load_state_dict_from_url(
        url=download_link,
        model_dir=model_dir,
        file_name=file_name,
    )
    return None
