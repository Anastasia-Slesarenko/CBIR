from io import BytesIO
from PIL import Image
import requests


def get_bytes_image(drive_url: str) -> BytesIO:
    """Функция скачивает изображение  и  переводит в байты"""
    file_id = drive_url.split(
        "https://drive.google.com/file/d/",
    )[-1].split(
        "/"
    )[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    response.raise_for_status()
    return BytesIO(response.content)


def read_one_image(url: str) -> Image.Image:
    """Функция открывает изображение"""
    bytes_image = get_bytes_image(url)
    image = Image.open(bytes_image)
    return image


def read_list_images(image_sources: list[str]) -> list[Image.Image]:
    """Функция открывает список изображений"""
    images = []
    for im in image_sources:
        images.append(read_one_image(im))
    return images
