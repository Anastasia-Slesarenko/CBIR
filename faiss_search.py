from faiss import Index
import faiss
from db import get_all_emb_from_pg, get_image_by_index
from torch import Tensor


def train_faiss_index(
    batch_size: int, emb_size: int, faiss_index_path: str = "train.index"
) -> None:
    """инициализация и тренировка индексов, сохранение на диск"""
    index = faiss.IndexFlatL2(emb_size)
    for batch in get_all_emb_from_pg(batch_size):
        index.add(batch)
    faiss.write_index(index, faiss_index_path)


# TODO: возвращает сразу бинарники картинок или их расположение
def get_similary_images(
    query_feature: Tensor, topk: int = 9, faiss_index_path: str = "train.index"
) -> list:
    """ест эмбеддинг картинки запроса и выдает топ k ближайших ембеддингов
    по индексу faiss, далее из pg дастает картинки"""
    index = faiss.read_index(faiss_index_path)
    query_np = query_feature.numpy()
    _, indices = index.search(query_np, topk)
    image_paths = get_image_by_index(indices[0].tolist())
    return image_paths
