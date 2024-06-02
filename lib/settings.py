import os
from typing import Tuple
from .loaders import load_torch_model


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VOLUME_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data")

HOSTNAME = os.environ.get("HOSTNAME", "localhost")
USERNAME = "postgres"
PASSWORD = "postgres"
DATABASE_NAME = "db"
PORT = 5432

IMAGE_PATH = os.path.join(VOLUME_DIR, "images")
IMAGE_FORMAT = ".jpg"

# ImageNet Params
TNormParam = Tuple[float, float, float]
MEAN: TNormParam = (0.485, 0.456, 0.406)
STD: TNormParam = (0.229, 0.224, 0.225)

FAISS_INDEX_PATH = os.path.join(VOLUME_DIR, "faiss_index.index")
MODEL_URL = "https://disk.yandex.ru/d/K7ozxAlGlPanlw"
MODEL_FILE = "extractor.pth"
MODEL_PATH = os.path.join(VOLUME_DIR, MODEL_FILE)

# загружаем модель из диска если файла еще нет
if not os.path.isfile(MODEL_PATH):
    load_torch_model(
        yadisk_model_url=MODEL_URL,
        model_dir=VOLUME_DIR,
        file_name=MODEL_FILE,
    )
