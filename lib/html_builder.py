import base64
from fastapi import Request
from PIL import Image
from .utils import get_bytes_image


def build_html(
    image: Image.Image,
    image_path: str,
    image_format: str,
    candidates: list,
    request: Request,
) -> dict:
    """
    Generates HTML data for displaying an image and its similar images.
    """
    main_image = base64.b64encode(image.getvalue()).decode("utf-8")
    html_data = {
        "request": request,
        "image0": main_image,
    }
    for i in range(1, 9):
        # read image
        html_data[f"image{i}"] = base64.b64encode(
            get_bytes_image(
                image_id=candidates[i - 1][0],
                image_path=image_path,
                image_format=image_format,
            )
        ).decode("utf-8")
        html_data[f"href{i}"] = candidates[i - 1][1]
        html_data[f"title{i}"] = candidates[i - 1][2]
    return html_data
