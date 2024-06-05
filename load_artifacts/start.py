import sys

sys.path.append("../")

import os
from utils import prepare_search_db
from lib.utils import load_torch_model
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
    MODEL_URL,
    YADISK_API_ENDPOINT,
    VOLUME_DIR,
    MODEL_FILE
)
from torch import load as torch_model_load


if __name__ == "__main__":
    if not os.path.isfile(MODEL_PATH):
        load_torch_model(
            yadisk_model_url=MODEL_URL,
            yadisk_api_endpoint=YADISK_API_ENDPOINT,
            model_dir=VOLUME_DIR,
            file_name=MODEL_FILE,
        )

    model = torch_model_load(MODEL_PATH).to(DEVICE)
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
        model=model,
        csv_path=CSV_PATH,
        faiss_index_path=FAISS_INDEX_PATH,
        device=DEVICE,
    )
    print("=" * 52)
    print("The database is ready to search for similar images.")
    print("=" * 52)
