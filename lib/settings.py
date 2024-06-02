import os
from typing import Tuple
from gdown import download


HOSTNAME = (
    os.environ.get("HOSTNAME")
    if os.environ.get("HOSTNAME") is not None
    else "localhost"
)
USERNAME = "postgres"
PASSWORD = "postgres"
DATABASE_NAME = "db"
PORT = 5432

IMAGE_PATH = "D:\DS AAA\Final Project\CBIR\test\gallery_images_test"
IMAGE_FORMAT = ".jpg"

# ImageNet Params
TNormParam = Tuple[float, float, float]
MEAN: TNormParam = (0.485, 0.456, 0.406)
STD: TNormParam = (0.229, 0.224, 0.225)

MODEL_URL = "https://drive.google.com/file/d/18vAIYiyYl5w1RLk5-sJDWc0KG_DcC4Fa/view?usp=sharing"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "extractor.pth")

# загружаем модель из диска если файла еще нет
if not os.path.isfile(MODEL_PATH):
    download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
