import logging
import sys
import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .db import Storage
from .utils import build_html
from .settings import (
    HOSTNAME,
    USERNAME,
    PASSWORD,
    DATABASE_NAME,
    PORT,
    ROOT_DIR,
)

logger = logging.getLogger(__name__)


def init_storage():
    storage = Storage(
        host=HOSTNAME,
        user=USERNAME,
        password=PASSWORD,
        database=DATABASE_NAME,
        port=PORT,
    )
    yield storage
    storage.disconnect()


app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(ROOT_DIR, "static")),
    name="static",
)
templates = Jinja2Templates(directory=os.path.join(ROOT_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/find_simular_images", response_class=HTMLResponse)
def find_simular_images(
    request: Request,
    image: UploadFile,
    storage: Storage = Depends(init_storage),
):
    """Finds the top 8 similar images to the input image"""
    # check file format
    if image.filename.split(".")[-1] not in ["jpg", "png", "jpeg"]:
        logger.error(
            msg="Неправильный формат картинки. Введите картину в формате .jpg, .png, .jpeg",
        )
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error_code": "400. Bad Request",
                "error": "Неправильный формат картинки. Введите картину в формате .jpg, .png, .jpeg",
            },
            status_code=400,
        )
    try:
        image = BytesIO(image.file.read())
        html_data = build_html(
            image=image,
            storage=storage,
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
