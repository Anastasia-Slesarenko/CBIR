import sys

sys.path.append("../")

import os
from torch import load as torch_model_load
from utils import prepare_search_db
from lib.db import Storage
from lib.settings import (
    CSV_PATH,
    DATABASE_NAME,
    DEVICE,
    FAISS_INDEX_PATH,
    HOSTNAME,
    IMAGE_FORMAT,
    IMAGE_PATH,
    MODEL_FILE,
    MODEL_PATH,
    MODEL_URL,
    PASSWORD,
    PORT,
    USERNAME,
    VOLUME_DIR,
    YADISK_API_ENDPOINT,
    EMBED_SIZE,
    MODEL_NAME,
)
from lib.utils import load_torch_model

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
        emb_size=EMBED_SIZE[MODEL_NAME],
    )
    print("=" * 52)
    print("The database is ready to search for similar images.")
    print("=" * 52)
