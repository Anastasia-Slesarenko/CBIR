import faiss
from torch import Tensor
from .db import Storage


def train_faiss_index(
    storage: Storage,
    batch_size: int,
    emb_size: int,
    faiss_index_path: str = "faiss_index.index",
) -> None:
    """Initializes FAISS index on CPU and saves it."""
    index = faiss.IndexFlatL2(emb_size)
    index_with_map = faiss.IndexIDMap(index)
    for ids, batch in storage.get_all_emb_from_pg(batch_size):
        index_with_map.add_with_ids(batch, ids)
    faiss.write_index(index_with_map, faiss_index_path)


def train_faiss_index_gpu(
    storage: Storage,
    batch_size: int,
    emb_size: int,
    faiss_index_path: str = "faiss_index.index",
) -> None:
    """Initializes FAISS index on GPU and saves it."""
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(emb_size)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    index_with_map = faiss.IndexIDMap(gpu_index)
    for ids, batch in storage.get_all_emb_from_pg(batch_size):
        index_with_map.add_with_ids(batch, ids)
    index_cpu = faiss.index_gpu_to_cpu(index_with_map)
    faiss.write_index(index_cpu, faiss_index_path)


def get_similar_images(
    storage: Storage,
    query_feature: Tensor,
    topk: int = 8,
    faiss_index_path: str = "faiss_index.index",
) -> list:
    """Finds top k nearest images to the query feature using FAISS index on CPU."""
    index = faiss.read_index(faiss_index_path)
    query_np = query_feature.numpy()
    _, indices = index.search(query_np, topk)
    image_paths = storage.get_image_by_index(indices[0].tolist())
    return image_paths


def get_similar_images_on_gpu(
    storage: Storage,
    query_feature: Tensor,
    topk: int = 8,
    faiss_index_path: str = "faiss_index.index",
) -> list:
    """Finds top k nearest images to the query feature using FAISS index on GPU."""
    res = faiss.StandardGpuResources()
    index_cpu = faiss.read_index(faiss_index_path)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    query_np = query_feature.numpy()
    _, indices = gpu_index.search(query_np, topk)
    image_paths = storage.get_image_by_index(indices[0].tolist())
    return image_paths
