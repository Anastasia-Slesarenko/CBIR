import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from torch import load as torch_model_load
from lib.db import Storage
from lib.load_artifacts.utils import download_data, prepare_search_db
from lib.settings import (
    CSV_PATH,
    DATA_URL,
    DATABASE_NAME,
    DEVICE,
    EMBED_SIZE,
    FAISS_INDEX_PATH,
    HOSTNAME,
    IMAGE_FORMAT,
    IMAGE_PATH,
    MODEL_FILE,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_URL,
    PASSWORD,
    PORT,
    USERNAME,
    VOLUME_DIR,
    YADISK_API_ENDPOINT,
)
from lib.utils import download_torch_model


if __name__ == "__main__":
    if not os.path.isdir(IMAGE_PATH):
        download_data(
            yadisk_api_endpoint=YADISK_API_ENDPOINT,
            yadisk_data_url=DATA_URL,
            volume_dir=VOLUME_DIR,
        )

    if not os.path.isfile(MODEL_PATH):
        download_torch_model(
            yadisk_model_url=MODEL_URL,
            yadisk_api_endpoint=YADISK_API_ENDPOINT,
            model_dir=VOLUME_DIR,
            file_name=MODEL_FILE,
        )

    storage = Storage(
        host=HOSTNAME,
        user=USERNAME,
        password=PASSWORD,
        database=DATABASE_NAME,
        port=PORT,
    )
    if (not storage.check_table_exist()) or (storage.count_rows() == 0):
        model = torch_model_load(MODEL_PATH).to(DEVICE)
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
        model.to("cpu")
        del model

    storage.disconnect()
    print("=" * 52)
    print("The database is ready to search for similar images.")
    print("=" * 52)
