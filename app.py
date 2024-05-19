# код фастапи
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from io import BytesIO
import PIL.Image as Image
from PIL import UnidentifiedImageError
from psycopg2 import OperationalError
import pandas as pd
from tqdm import tqdm
import base64
import sys, os
import logging

from model import extract_features_from_images, extract_features_from_image
from db import create_tables_structure, save_embeddings
from faiss_search import train_faiss_index, get_similary_images


logger = logging.getLogger(__name__)
app = FastAPI()
app.mount("/static", StaticFiles(directory="./lib/static"), name="static")
templates = Jinja2Templates(directory="./lib/templates")


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/find_simular_images", response_class=HTMLResponse)
def find_simular_images(
    request: Request,
    image: UploadFile,
    model_pth: str = "extractor.pth",
):
    # check file format
    if image.filename.split(".")[-1] != "jpg":
        logger.error(
            msg="Неправильный формат картинки. Введите картину в формате .jpg",
        )
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error_code": "400. Bad Request",
                "error": "Неправильный формат картинки. Введите картину в формате .jpg",
            },
            status_code=400,
        )

    try:
        image = BytesIO(image.file.read())
        query_feature = extract_features_from_image(image, model_pth)
        image_paths = get_similary_images(query_feature, topk=8)
        main_image = base64.b64encode(image.getvalue()).decode("utf-8")
        html_data = {
            "request": request,
            "image0": main_image,
        }
        for i in range(1, 9):
            with open(image_paths[i - 1][0], "rb") as image_file:
                html_data[f"image{i}"] = base64.b64encode(image_file.read()).decode(
                    "utf-8"
                )
        return templates.TemplateResponse("result.html", html_data)
    except Exception as e:
        e_type, _, e_traceback = sys.exc_info()
        e_line_number = e_traceback.tb_lineno
        e_filename = os.path.split(e_traceback.tb_frame.f_code.co_filename)[1]
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


@app.post("/create_and_insert_pg")
def create_and_insert_pg(
    data: UploadFile = File(...),
    model_pth: str = "extractor.pth",
    batch_size: int = 64,
):
    create_tables_structure()
    df = pd.read_csv(BytesIO(data.file.read()))
    img_paths = df.path.to_list()
    for batch in tqdm(range(0, len(img_paths), batch_size)):
        image_batch = []
        for img_path in img_paths[batch : batch + batch_size]:
            image_batch.append(Image.open(img_path))
        features_galleries = extract_features_from_images(image_batch, model_pth)
        features_galleries = [
            (features_galleries[i].tolist(), img_paths[i])
            for i in range(features_galleries.shape[0])
        ]
        save_embeddings(features_galleries)
    return "ok"


@app.get("/train_faiss")
def train_faiss(batch_size: int = 1000, emb_size: int = 384):
    train_faiss_index(batch_size, emb_size)
    return "ok"
