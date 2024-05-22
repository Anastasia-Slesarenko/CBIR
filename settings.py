import os
from typing import Tuple
from gdown import download


HOSTNAME = "postgres"
USERNAME = "postgres"
PASSWORD = "postgres"
DATABASE_NAME = "db"
PORT = 5432

# ImageNet Params
TNormParam = Tuple[float, float, float]
MEAN: TNormParam = (0.485, 0.456, 0.406)
STD: TNormParam = (0.229, 0.224, 0.225)

MODEL_URL = (
    "https://drive.google.com/file/d/18vAIYiyYl5w1RLk5-sJDWc0KG_DcC4Fa/view?usp=sharing"
)
MODEL_PATH = "extractor.pth"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# загружаем модель из диска если файла еще нет
if not os.path.isfile(os.path.join(ROOT_DIR, MODEL_PATH)):
    download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
