from io import BytesIO
import torch
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torch import Tensor
from PIL import Image
from .settings import MODEL_PATH, MEAN, STD, TNormParam


def get_normalisation_resize_torch(
    im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD
) -> Compose:
    """Функция трансформирует изображения"""
    return Compose(
        [
            Resize(size=(im_size, im_size), antialias=True),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


def extract_features_from_batch(
    batch: list[Image.Image],
    device: str,
    model_pth: str = MODEL_PATH,
) -> Tensor:
    """Функция возвращает эмбеддинги батча. На инференсе рамер батча 1."""
    model = torch.load(model_pth).to(device)
    transform = get_normalisation_resize_torch(im_size=224)
    batch = torch.cat(
        [transform(image).unsqueeze(0) for image in batch], dim=0
    )
    model.eval()
    with torch.no_grad():
        out = model(batch.to(device))
        out = out.cpu()
    return out


def extract_features_from_image(
    image: BytesIO, device: str = "cpu"
) -> Tensor:
    """Функция возвращает эмбеддинг одного изображения"""
    batch = Image.open(image)
    return extract_features_from_batch(batch=[batch], device=device)


def extract_features_from_images(
    batch: Tensor, device: str = "cpu"
) -> Tensor:
    """Функция возвращает эмбеддинги батча изображений"""
    return extract_features_from_batch(batch=batch, device=device)
