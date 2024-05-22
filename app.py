import logging
import sys
import os
from io import BytesIO
import base64
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from tqdm import tqdm
from model import extract_features_from_images, extract_features_from_image
from db import create_tables_structure, save_embeddings
from faiss_search import train_faiss_index, get_similary_images
from utils import read_list_images, get_bytes_image


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
):
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
        query_feature = extract_features_from_image(image)
        candidates = get_similary_images(query_feature, topk=8)
        main_image = base64.b64encode(image.getvalue()).decode("utf-8")
        html_data = {
            "request": request,
            "image0": main_image,
        }
        for i in range(1, 9):
            # read image
            html_data[f"image{i}"] = base64.b64encode(
                get_bytes_image(candidates[i - 1][0]).getvalue()
            ).decode("utf-8")
            html_data[f"href{i}"] = candidates[i - 1][1]
            html_data[f"title{i}"] = candidates[i - 1][2]
        return templates.TemplateResponse("index.html", html_data)
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
    batch_size: int = 64,
    col_source: str = "source",
    col_item_url: str = "item_url",
    col_title: str = "title",
):
    create_tables_structure()
    df = pd.read_csv(BytesIO(data.file.read()))
    for batch in tqdm(range(0, df.shape[0], batch_size)):
        image_batch = read_list_images(
            df[col_source].iloc[batch : batch + batch_size].tolist()
        )
        features_galleries = extract_features_from_images(image_batch)
        features_galleries = [
            (
                features_galleries[i].tolist(),
                df[col_source].iloc[i],
                df[col_item_url].iloc[i],
                df[col_title].iloc[i],
            )
            for i in range(features_galleries.shape[0])
        ]
        save_embeddings(features_galleries)
    return {"message": "ok"}


@app.get("/train_faiss")
def train_faiss(batch_size: int = 1000, emb_size: int = 384):
    train_faiss_index(batch_size, emb_size)
    return {"message": "ok"}
