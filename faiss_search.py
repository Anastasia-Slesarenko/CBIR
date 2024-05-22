import faiss
from torch import Tensor
from db import get_all_emb_from_pg, get_image_by_index


def train_faiss_index(
    batch_size: int, emb_size: int, faiss_index_path: str = "train.index"
) -> None:
    """Инициализирует индексы faiss и сохраняет на диск"""
    index = faiss.IndexFlatL2(emb_size)
    for batch in get_all_emb_from_pg(batch_size):
        index.add(batch)
    faiss.write_index(index, faiss_index_path)


def get_similary_images(
    query_feature: Tensor,
    topk: int = 8,
    faiss_index_path: str = "train.index",
) -> list:
    """По эмбеддингу изображения запроса находит топ k ближайших
    по индексу faiss, из pg дастает пути к найденным изображениям"""
    index = faiss.read_index(faiss_index_path)
    query_np = query_feature.numpy()
    _, indices = index.search(query_np, topk)
    image_paths = get_image_by_index(indices[0].tolist())
    return image_paths
