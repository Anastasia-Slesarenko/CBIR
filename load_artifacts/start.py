import sys

sys.path.append("../")

import pandas as pd
import torch
from tqdm import tqdm
from lib.model import extract_features_from_images
from lib.db import Storage
from lib.faiss_search import train_faiss_index
from lib.loaders import read_list_images
from lib.settings import (
    HOSTNAME,
    USERNAME,
    PASSWORD,
    DATABASE_NAME,
    PORT,
    IMAGE_FORMAT,
    IMAGE_PATH, 
    MODEL_PATH,
    FAISS_INDEX_PATH,
)


def prepare_search_retrieval_db(
    storage: Storage,
    image_path: str,
    image_format: str,
    csv_path: str,
    faiss_index_path: str,
    device: str,
    batch_size: int = 64,
    emb_size: int = 384,
    col_image_id: str = "image_id",
    col_item_url: str = "item_url",
    col_title: str = "title",
) -> None:
    """
    Prepares a database and search indexes for a similar image search application:
    creates a table based on the input data, calculates the embedding of images
    and writes them to the table. After that, it creates faiss indexes for the search.
    """
    # Create table, get embedding and insert data
    storage.create_tables_structure()
    df = pd.read_csv(csv_path)
    for batch in tqdm(range(0, df.shape[0], batch_size)):
        image_batch = read_list_images(
            image_sources=df[col_image_id]
            .iloc[batch : batch + batch_size]
            .tolist(),
            image_path=image_path,
            image_format=image_format,
        )
        features_galleries = extract_features_from_images(
            image_batch, device=device
        )
        features_galleries = [
            (
                int(df[col_image_id].iloc[i]),
                features_galleries[i].tolist(),
                df[col_item_url].iloc[i],
                df[col_title].iloc[i],
            )
            for i in range(features_galleries.shape[0])
        ]
        storage.save_embeddings(features_galleries)
    # Create and train faiss index
    train_faiss_index(
        storage=storage,
        batch_size=batch_size,
        emb_size=emb_size,
        faiss_index_path=faiss_index_path,
    )
    return None


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")

    csv_path = "../data/avito_images.csv"

    storage = Storage(
        host=HOSTNAME,
        user=USERNAME,
        password=PASSWORD,
        database=DATABASE_NAME,
        port=PORT,
    )
    prepare_search_retrieval_db(
        storage=storage,
        image_path=IMAGE_PATH,
        image_format=IMAGE_FORMAT,
        csv_path=csv_path,
        faiss_index_path=FAISS_INDEX_PATH,
        device=device,
    )
    print("=====================================================")
    print("finish")
    print("=====================================================")
