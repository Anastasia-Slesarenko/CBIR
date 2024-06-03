from .loaders import get_bytes_image
from .settings import IMAGE_PATH, IMAGE_FORMAT, FAISS_INDEX_PATH
from PIL import Image
from .model import extract_features_from_image
from .faiss_search import get_similar_images
from .db import Storage
import base64
from fastapi import Request


def build_html(
    image: Image.Image, storage: Storage, request: Request
) -> dict:
    """По входящему изображению находит похожие картинки и собирает бинарники картинок,
    названия и ссылки на объявления в словарь для формирования html"""
    main_image = base64.b64encode(image.getvalue()).decode("utf-8")
    html_data = {
        "request": request,
        "image0": main_image,
    }
    for i in range(1, 9):
        # read image
        html_data[f"image{i}"] = base64.b64encode(
            get_bytes_image(
                file=candidates[i - 1][0],
                image_path=IMAGE_PATH,
                image_format=IMAGE_FORMAT,
            )
        ).decode("utf-8")
        html_data[f"href{i}"] = candidates[i - 1][1]
        html_data[f"title{i}"] = candidates[i - 1][2]
    return html_data
