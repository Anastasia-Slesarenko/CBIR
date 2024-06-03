import faiss
from torch import Tensor
from .db import Storage


def train_faiss_index(
    storage: Storage,
    batch_size: int,
    emb_size: int,
    faiss_index_path: str,
) -> None:
    """Initializes FAISS index and saves it."""
    index = faiss.IndexFlatL2(emb_size)
    index_with_map = faiss.IndexIDMap(index)
    for ids, batch in storage.get_all_emb_from_pg(batch_size):
        index_with_map.add_with_ids(batch, ids)
    faiss.write_index(index_with_map, faiss_index_path)


def get_similar_images(
    storage: Storage,
    query_emb: Tensor,
    topk: int = 8,
    faiss_index_path: str,
) -> list:
    """
    Finds top k nearest image embeddings to the query embedding using FAISS index.
    Return images with urls to ads and titles. 
    """
    index = faiss.read_index(faiss_index_path)
    query_np = query_emb.numpy()
    _, indices = index.search(query_np, topk)
    image_paths = storage.get_image_by_index(indices[0].tolist())
    return image_paths
