from io import BytesIO
import os
from PIL import Image
from settings import IMAGE_PATH, IMAGE_FORMAT


def get_bytes_image(file: int) -> BytesIO:
    """Загружает похожие изображения из хранилища и
    переводит их в байты для отображения в html"""
    img = Image.open(os.path.join(IMAGE_PATH, str(file) + IMAGE_FORMAT))
    with BytesIO() as output:
        img.save(output, format="PNG")
        bytes_array = output.getvalue()
    return bytes_array


def read_list_images(image_sources: list[int]) -> list[Image.Image]:
    """Загружает список изображений из хранилища для записи их эмбеддингов"""
    images = []
    for file in image_sources:
        images.append(
            Image.open(os.path.join(IMAGE_PATH, str(file) + IMAGE_FORMAT))
        )
    return images
