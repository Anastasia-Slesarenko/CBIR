import os
import sys
from io import BytesIO
from logging import Logger
from typing import Union
import requests
import torch
from fastapi import HTTPException, Request, UploadFile
from PIL import Image
from starlette.templating import _TemplateResponse


def get_bytes_image_by_url(image_url: str) -> BytesIO:
    """
    Loads query image by url and converts them to bytes.
    """
    response = requests.get(image_url)
    response.raise_for_status()
    return BytesIO(response.content)


def get_bytes_image(
    image_id: int,
    image_path: str,
    image_format: str,
) -> BytesIO:
    """
    Loads similar images from storage and converts
    them to bytes for display in HTML.
    """
    img = Image.open(
        os.path.join(image_path, str(image_id) + image_format)
    )
    with BytesIO() as output:
        img.save(output, format="PNG")
        bytes_array = output.getvalue()
    return bytes_array


def read_list_images(
    image_ids: list[int],
    image_path: str,
    image_format: str,
) -> list[Image.Image]:
    """
    Loads a list of images from storage to extract their embeddings.
    """
    images = []
    for image_id in image_ids:
        images.append(
            Image.open(
                os.path.join(image_path, str(image_id) + image_format)
            )
        )
    return images


def download_torch_model(
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


def check_image_format(
    image: UploadFile,
    logger: Logger,
    templates: _TemplateResponse,
    request: Request,
) -> Union[_TemplateResponse, None]:
    # check file format
    if image.filename.split(".")[-1] not in ["jpg", "png", "jpeg"]:
        logger.error(
            msg=(
                "Неправильный формат картинки. "
                "Введите картину в формате jpg, png, jpeg",
            )
        )
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error_code": "400. Bad Request",
                "error": (
                    "Неправильный формат картинки. "
                    "Введите картину в формате jpg, png, jpeg",
                ),
            },
            status_code=400,
        )


def internal_exception(e: Exception, logger: Logger) -> HTTPException:
    e_type, _, e_traceback = sys.exc_info()
    e_line_number = e_traceback.tb_lineno
    e_filename = os.path.split(e_traceback.tb_frame.f_code.co_filename)[1]
    logger.error(
        msg=f"{e_type} in file {e_filename} line {e_line_number}: {str(e)}"
    )
    raise HTTPException(
        status_code=500,
        detail=[
            "500. Internal server error",
            f"{e_type}: {str(e)}",
        ],
    )
