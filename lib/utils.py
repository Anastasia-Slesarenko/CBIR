from io import BytesIO
import os
from PIL import Image
import requests
import torch


def get_bytes_image(
    file: int,
    image_path: str,
    image_format: str,
) -> BytesIO:
    """
    Loads similar images from storage and converts
    them to bytes for display in HTML.
    """
    img = Image.open(os.path.join(image_path, str(file) + image_format))
    with BytesIO() as output:
        img.save(output, format="PNG")
        bytes_array = output.getvalue()
    return bytes_array


def read_list_images(
    image_sources: list[int],
    image_path: str,
    image_format: str,
) -> list[Image.Image]:
    """
    Loads a list of images from storage to extract their embeddings.
    """
    images = []
    for file in image_sources:
        images.append(
            Image.open(os.path.join(image_path, str(file) + image_format))
        )
    return images


def load_torch_model(
    yadisk_model_url: str,
    yadisk_api_endpoint: str,
    model_dir: str,
    file_name: str,
) -> None:
    """
    Downloads a Torch model from Yandex Disk.
    """
    responce = requests.get(yadisk_api_endpoint.format(yadisk_model_url))
    download_link = responce.json()["href"]
    torch.hub.load_state_dict_from_url(
        url=download_link,
        model_dir=model_dir,
        file_name=file_name,
    )
    return None
