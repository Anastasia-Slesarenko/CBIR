import sys

sys.path.append("../")

from .utils import prepare_search_db
from lib.db import Storage
from lib.settings import (
    HOSTNAME,
    USERNAME,
    PASSWORD,
    DATABASE_NAME,
    PORT,
    IMAGE_FORMAT,
    IMAGE_PATH,
    MODEL_PATH,
    CSV_PATH,
    FAISS_INDEX_PATH,
    DEVICE,
)


if __name__ == "__main__":

    storage = Storage(
        host=HOSTNAME,
        user=USERNAME,
        password=PASSWORD,
        database=DATABASE_NAME,
        port=PORT,
    )
    prepare_search_db(
        storage=storage,
        image_path=IMAGE_PATH,
        image_format=IMAGE_FORMAT,
        model_pth=MODEL_PATH,
        csv_path=CSV_PATH,
        faiss_index_path=FAISS_INDEX_PATH,
        device=DEVICE,
    )
    print("=" * 52)
    print("The database is ready to search for similar images.")
    print("=" * 52)
