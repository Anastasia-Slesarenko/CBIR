import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import AsyncGenerator, Union
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.templating import _TemplateResponse
from torch import load as torch_model_load
from .db import Storage
from .faiss_search import get_similar_images
from .html_builder import build_html
from .model import extract_features_from_image
from .settings import (
    DATABASE_NAME,
    DEVICE,
    FAISS_INDEX_PATH,
    HOSTNAME,
    IMAGE_FORMAT,
    IMAGE_PATH,
    MODEL_PATH,
    PASSWORD,
    PORT,
    ROOT_DIR,
    USERNAME,
)
from .utils import (
    check_image_format,
    get_bytes_image_by_url,
    internal_exception,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Preparing before Starting the app.
    Initializing ML model and storage.
    """
    # Initializing model
    app.state.model = torch_model_load(MODEL_PATH).to(DEVICE)
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


@app.post("/find_similar_images_by_image", response_class=HTMLResponse)
def find_similar_images_by_image(
    request: Request,
    image: UploadFile,
) -> HTMLResponse:
    """
    Finds the top 8 similar images to the input image.

    :param request: The request object.

    :param image: The uploaded image file.

    :return: The rendered page with the similar images or an error message.
    """
    checker = check_image_format(
        image=image,
        logger=logger,
        templates=templates,
        request=request,
    )
    if isinstance(checker, _TemplateResponse):
        return checker
    try:
        image = BytesIO(image.file.read())
        query_emb = extract_features_from_image(
            image=image,
            device=DEVICE,
            model=request.app.state.model,
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
        internal_exception(e=e, logger=logger)


@app.get("/find_similar_image_urls_by_url", response_model=list[str])
def find_similar_image_urls_by_url(
    request: Request,
    image_url: str,
) -> list[str]:
    """
    Finds the top 8 similar item urls to the input image url.

    :param request: The request object.

    :param image: The image url.

    :return: list of item urls with similar images or an error message.
    """
    try:
        image = get_bytes_image_by_url(image_url=image_url)
    except Exception as e:
        logger.error("Invalid image_url [%s]: %s", image_url, str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image_url [{image_url}]: {str(e)}",
        ) from e
    try:
        query_emb = extract_features_from_image(
            image=image,
            device=DEVICE,
            model=request.app.state.model,
        )
        candidates = get_similar_images(
            storage=request.app.state.storage,
            query_emb=query_emb,
            faiss_index_path=FAISS_INDEX_PATH,
        )
        item_urls = [row[1] for row in candidates]
        return item_urls
    except Exception as e:
        internal_exception(e=e, logger=logger)


@app.post("/find_similar_image_urls_by_image", response_model=list[str])
def find_similar_image_urls_by_image(
    request: Request,
    image: UploadFile,
) -> list[str]:
    """
    Finds the top 8 similar item urls to the input image.

    :param request: The request object.

    :param image: The uploaded image file.

    :return: list of item urls with similar images or an error message.
    """
    checker = check_image_format(
        image=image,
        logger=logger,
        templates=templates,
        request=request,
    )
    if isinstance(checker, _TemplateResponse):
        return checker
    try:
        image = BytesIO(image.file.read())
        query_emb = extract_features_from_image(
            image=image,
            device=DEVICE,
            model=request.app.state.model,
        )
        candidates = get_similar_images(
            storage=request.app.state.storage,
            query_emb=query_emb,
            faiss_index_path=FAISS_INDEX_PATH,
        )
        item_urls = [row[1] for row in candidates]
        return item_urls
    except Exception as e:
        internal_exception(e=e, logger=logger)
