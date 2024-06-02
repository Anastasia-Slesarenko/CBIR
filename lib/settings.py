import os
from typing import Tuple
from .loaders import load_torch_model


HOSTNAME = os.environ.get("HOSTNAME", "localhost")
USERNAME = "postgres"
PASSWORD = "postgres"
DATABASE_NAME = "db"
PORT = 5432

IMAGE_PATH = "../data/images"
IMAGE_FORMAT = ".jpg"

# ImageNet Params
TNormParam = Tuple[float, float, float]
MEAN: TNormParam = (0.485, 0.456, 0.406)
STD: TNormParam = (0.229, 0.224, 0.225)

FAISS_INDEX_PATH = os.path.join("../data", "faiss_index.index")
MODEL_URL = os.environ.get(
    "MODEL_URL", "https://disk.yandex.ru/d/K7ozxAlGlPanlw"
)
MODEL_FILE = "extractor.pth"
MODEL_PATH = os.path.join("../data", MODEL_FILE)

# загружаем модель из диска если файла еще нет
if not os.path.isfile(MODEL_PATH):
    load_torch_model(
        yadisk_model_url=MODEL_URL,
        model_dir="../data",
        file_name=MODEL_FILE,
    )
