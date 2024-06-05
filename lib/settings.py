import os
from typing import Tuple
import torch


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VOLUME_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data")

HOSTNAME = os.environ.get("HOSTNAME", "localhost")
USERNAME = "postgres"
PASSWORD = "postgres"
DATABASE_NAME = "db"
PORT = 5432

IMAGE_PATH = os.path.join(VOLUME_DIR, "images")
IMAGE_FORMAT = ".jpg"

# Image Transform Params
TNormParam = Tuple[float, float, float]
MEAN: TNormParam = (0.485, 0.456, 0.406)
STD: TNormParam = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

FAISS_INDEX_PATH = os.path.join(VOLUME_DIR, "faiss_index.index")
MODEL_URL = "https://disk.yandex.ru/d/K7ozxAlGlPanlw"
MODEL_FILE = "extractor.pth"
MODEL_PATH = os.path.join(VOLUME_DIR, MODEL_FILE)
CSV_PATH = os.path.join(VOLUME_DIR, "avito_images.csv")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = "cpu"

YADISK_API_ENDPOINT = (
    "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    "?public_key={}"
)

if not os.path.isdir(VOLUME_DIR):
    os.mkdir(VOLUME_DIR)

TEST_IMAGE = "tests/query_image_test.jpg"
