import logging
import sys
import os
from io import BytesIO
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .db import Storage
from .faiss_search import get_similar_images
from .model import extract_features_from_image
from .html_builder import build_html
from .utils import load_torch_model
from .settings import (
    HOSTNAME,
    USERNAME,
    PASSWORD,
    DATABASE_NAME,
    PORT,
    VOLUME_DIR,
    MODEL_URL,
    ROOT_DIR,
    IMAGE_FORMAT,
    IMAGE_PATH,
    MODEL_PATH,
    MODEL_FILE,
    FAISS_INDEX_PATH,
    DEVICE,
    YADISK_API_ENDPOINT,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Preparing before Starting the app.
    Downloading model, if it doesn't exist and initializing storage.
    """
    # Downloading model
    if not os.path.isfile(MODEL_PATH):
        try:
            load_torch_model(
                yadisk_model_url=MODEL_URL,
                yadisk_api_endpoint=YADISK_API_ENDPOINT,
                model_dir=VOLUME_DIR,
                file_name=MODEL_FILE,
            )
            logger.info("Model downloaded successfully.")
        except Exception as e:
            logger.error("Error downloading model: %s", e)
            logger.info("Shutting down the application...")
            sys.exit(1)
    # Initializing storage
    app.state.storage = Storage(
        host=HOSTNAME,
        user=USERNAME,
        password=PASSWORD,
        database=DATABASE_NAME,
        port=PORT,
    )
    yield
    # Switch off storage
    app.state.storage.disconnect()


app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory=os.path.join(ROOT_DIR, "templates"))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(ROOT_DIR, "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request) -> HTMLResponse:
    """
    Renders the main page of the application.

    :param request: The request object.
    :return: The rendered main page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/find_simular_images", response_class=HTMLResponse)
def find_simular_images(
    request: Request,
    image: UploadFile,
) -> HTMLResponse:
    """
    Finds the top 8 similar images to the input image.

    :param request: The request object.
    :param image: The uploaded image file.
    :param storage: The storage object for database interaction.
    :return: The rendered page with the similar images or an error message.
    """
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
    try:
        image = BytesIO(image.file.read())
        query_emb = extract_features_from_image(
            image=image,
            device=DEVICE,
            model_pth=MODEL_PATH,
        )
        candidates = get_similar_images(
            storage=request.app.state.storage,
            query_emb=query_emb,
            faiss_index_path=FAISS_INDEX_PATH,
        )
        html_data = build_html(
            image=image,
            image_path=IMAGE_PATH,
            image_format=IMAGE_FORMAT,
            candidates=candidates,
            request=request,
        )
        return templates.TemplateResponse("index.html", html_data)
    except Exception as e:
        e_type, _, e_traceback = sys.exc_info()
        e_line_number = e_traceback.tb_lineno
        e_filename = os.path.split(
            e_traceback.tb_frame.f_code.co_filename
        )[1]
        logger.error(
            msg=f"{e_type} in file {e_filename} line {e_line_number}: {str(e)}"
        )
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error_code": "500. Internal server error",
                "error": f"{e_type}: {str(e)}",
            },
            status_code=500,
        )
