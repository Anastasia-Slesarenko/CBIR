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
    faiss_index: faiss.Index,
    topk: int = 8,
) -> list:
    """
    Finds top k nearest image embeddings to the query embedding
    using FAISS index. Return images with urls to ads and titles.
    """
    query_np = query_emb.numpy()
    _, indices = faiss_index.search(query_np, topk)
    ids = [idx for idx in indices[0].tolist() if idx != -1]
    candidates = storage.get_image_by_index(ids)
    return candidates
