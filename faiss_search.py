import faiss
from torch import Tensor
from db import Storage


def train_faiss_index(
    storage: Storage,
    batch_size: int,
    emb_size: int,
    faiss_index_path: str = "train.index",
) -> None:
    """Инициализирует индексы faiss и сохраняет на диск"""
    index = faiss.IndexFlatL2(emb_size)
    index_with_map = faiss.IndexIDMap(index)
    for ids, batch in storage.get_all_emb_from_pg(batch_size):
        index_with_map.add_with_ids(batch, ids)
    faiss.write_index(index_with_map, faiss_index_path)


def get_similar_images(
    storage: Storage,
    query_feature: Tensor,
    topk: int = 8,
    faiss_index_path: str = "train.index",
) -> list:
    """По эмбеддингу изображения запроса находит топ k ближайших
    по индексу faiss, из pg дастает пути к найденным изображениям"""
    index = faiss.read_index(faiss_index_path)
    query_np = query_feature.numpy()
    _, indices = index.search(query_np, topk)
    image_paths = storage.get_image_by_index(indices[0].tolist())
    return image_paths
