import os
import tarfile
import pandas as pd
import requests
from torch.nn import Module
from tqdm import tqdm
from lib.db import Storage
from lib.faiss_search import train_faiss_index
from lib.model import extract_features_from_images
from lib.utils import read_list_images


def download_data(
    yadisk_data_url: str,
    yadisk_api_endpoint: str,
    volume_dir: str,
) -> None:
    """
    Downloads tar file from URL, extracts its contents to the specified
    directory, and removes the tar file after extraction.
    """
    # Make a request to get the download link
    response = requests.get(yadisk_api_endpoint.format(yadisk_data_url))
    response_data = response.json()
    data_download_link = response_data["href"]
    # Define the paths
    tar_file_path = os.path.join(volume_dir, "data.tar")
    # Download the tar file
    with requests.get(data_download_link, stream=True) as r:
        r.raise_for_status()
        with open(tar_file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    # Unzip the tar file to the specified path
    with tarfile.open(tar_file_path, "r") as tar:
        tar.extractall(path=volume_dir)
    # Remove the tar file
    os.remove(tar_file_path)
    return None


def prepare_search_db(
    storage: Storage,
    image_path: str,
    image_format: str,
    csv_path: str,
    model: Module,
    faiss_index_path: str,
    device: str,
    emb_size: int,
    batch_size: int = 64,
    col_image_id: str = "image_id",
    col_item_url: str = "item_url",
    col_title: str = "title",
) -> None:
    """
    Prepares a database and search indexes for a similar image search
    application: creates a table based on the input data, calculates
    the embedding of images and writes them to the table. After that,
    it creates faiss indexes for the search.
    """
    # Create table, get embedding and insert data
    storage.create_tables_structure()
    df = pd.read_csv(csv_path)
    for batch in tqdm(range(0, df.shape[0], batch_size)):
        image_batch = read_list_images(
            image_ids=df[col_image_id]
            .iloc[batch : batch + batch_size]
            .tolist(),
            image_path=image_path,
            image_format=image_format,
        )
        features_galleries = extract_features_from_images(
            image_batch,
            device=device,
            model=model,
        )
        features_galleries = [
            (
                int(df[col_image_id].iloc[batch + i]),
                features_galleries[i].tolist(),
                df[col_item_url].iloc[batch + i],
                df[col_title].iloc[batch + i],
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
